import copy
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.ops import segment
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import softmax as edge_softmax

from layers.GCN.CL_layers import ProjectionHead, simple_cl
from layers.GCN.RGCN import CustomHeteroGCN
from layers.common_layers import MLP, RNNEncoder, GateModule, CapsuleNetwork, PositionalEncodingSC
from utilis.scripts import SimLoss


class DHGCNEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, time_length, ntypes, etypes, start_times, n_layers,
                 encoder_type, special_edges, node_attr_types, time_embs, shared=False, learned_time=False,
                 time_pooling='sum',
                 time_layers=1, snapshot_mode='gate', device=None, mixed=True, updater='normal', **kwargs):
        super(DHGCNEncoder, self).__init__()
        self.start_times = start_times
        self.time_length = time_length
        self.shared = shared
        self.ntypes = ntypes
        if shared:
            self.snapshot_encoders = SnapshotEncoder(embed_dim, hidden_dim, 'is_target', mode=snapshot_mode)
            self.snapshot_embedding = dglnn.HeteroEmbedding({ntype: time_length for ntype in ntypes}, embed_dim)
        else:
            self.snapshot_encoders = nn.ModuleList([SnapshotEncoder(embed_dim, hidden_dim, 'is_target')
                                                    for t in range(time_length)])
        self.layers = nn.ModuleList([SingleDHGCN(embed_dim, hidden_dim, time_length, ntypes, etypes,
                                                 encoder_type, special_edges, node_attr_types,
                                                 time_encoder_layers=time_layers, shared=shared, layer=0, mixed=mixed,
                                                 updater=updater)])
        for i in range(1, n_layers):
            self.layers.append(SingleDHGCN(hidden_dim, hidden_dim, time_length, ntypes, etypes,
                                           encoder_type, special_edges, node_attr_types,
                                           time_encoder_layers=time_layers, shared=shared, layer=i, mixed=mixed,
                                           updater=updater))
        self.time_embs = time_embs
        # self.time_fc = None
        self.time_pooling = time_pooling
        print('>>>snapshot_mode:', snapshot_mode)

    def forward(self, inputs, show_detail=False):
        # graphs, cur_snapshot_types, masks, phase = inputs
        graphs = inputs['graphs']
        cur_snapshot_types = inputs['cur_snapshot_types']
        masks = inputs['time_masks']
        phase = inputs['phase']
        index_dicts = inputs['index_dicts']
        feats = [cur_graph.srcdata['h'] for cur_graph in graphs]
        cur_time_embs = self.time_embs[phase]
        # if self.te:
        #     cur_time_embs = cur_time_embs * self.te
        other_outputs = []

        snapshot_embs = []
        for t in range(self.time_length):
            if self.shared:
                snapshot_time_embs = self.snapshot_embedding(graphs[t].srcdata['snapshot_idx'])
                for ntype in self.ntypes:
                    feats[t][ntype] = feats[t][ntype] + snapshot_time_embs[ntype]
                snapshot_embs.append(self.snapshot_encoders(graphs[t], feats[t],
                                                            cur_time_embs[t]))
            else:
                snapshot_embs.append(self.snapshot_encoders[t](graphs[t], feats[t],
                                                               cur_time_embs[t]))
            # print(snapshot_embs[t].shape)
        snapshot_embs = pad_sequence(snapshot_embs, batch_first=True)
        # print(snapshot_embs.shape)
        # print("before gcn:{:.2f}MB".format(torch.cuda.memory_allocated(0) / 1024 ** 2))

        snapshot_readouts_list = []
        time_lens = (~masks).unsqueeze(dim=-1).sum(dim=-2)
        # print(masks.shape, time_lens.shape)
        # print(time_masks)
        # print(time_lens)
        for layer in self.layers:
            if show_detail:
                feats, snapshot_embs, other_output = layer(graphs, feats, cur_snapshot_types,
                                                           snapshot_embs, index_dicts,
                                                           masks,
                                                           show_detail=show_detail)
                other_outputs.append(other_output)
            else:
                feats, snapshot_embs = layer(graphs, feats, cur_snapshot_types,
                                             snapshot_embs, index_dicts, masks)
            # print("after gcn:{:.2f}MB".format(torch.cuda.memory_allocated(0) / 1024 ** 2))
            #  [T, batch, h]
            if self.time_pooling == 'mean':
                snapshot_readouts_list.append(snapshot_embs.sum(0) / time_lens)  # time mean pooling
            elif self.time_pooling == 'sum':
                snapshot_readouts_list.append(snapshot_embs.sum(0))  # time sum pooling
            elif self.time_pooling == 'long_short':
                # print(time_lens.shape)
                # long_embs = snapshot_embs.sum(0) / time_lens
                long_embs = snapshot_embs.sum(0)
                short_embs = snapshot_embs[-1]
                # print(long_embs.shape, short_embs.shape)
                snapshot_readouts_list.append(torch.cat((long_embs, short_embs), dim=-1))

        # [batch, h]
        snapshot_readouts = torch.stack(snapshot_readouts_list, dim=1).mean(dim=1)  # layer mean pooling
        # print(snapshot_readouts.shape)
        if show_detail:
            return snapshot_readouts, other_outputs
        else:
            return snapshot_readouts


class SingleDHGCN(nn.Module):
    def __init__(self, in_feat, out_feat, time_length, ntypes, etypes, encoder_type, special_edges, node_attr_types,
                 mixed=True, time_encoder_layers=1, shared=False, updater='normal',
                 layer=0, activation=nn.LeakyReLU(inplace=True)):
        super(SingleDHGCN, self).__init__()
        print('>>>activation:', activation)
        self.time_length = time_length
        self.mixed = mixed
        self.shared = shared
        self.layer = layer
        if updater == 'simple':
            if self.layer == 0:
                print('!!!!!change the snapshot updater to simple version')
            SnapshotUpdater = SimpleSnapshotWeighter
        elif updater == 'basic':
            if self.layer == 0:
                print('!!!!!ablate the snapshot updater')
            SnapshotUpdater = BasicSnapshotWeighter
        else:
            SnapshotUpdater = SnapshotWeighter
        if self.shared:
            self.input_trans = dglnn.HeteroLinear({ntype: in_feat for ntype in ntypes}, out_feat)
            self.graph_encoder = CustomHeteroGCN(encoder_type, out_feat, out_feat,
                                                 ntypes, etypes, layer, activation,
                                                 special_edges=special_edges)
            self.snapshot_weighters = SnapshotUpdater(out_feat, 3, node_attr_types, activation=activation)
        else:
            if self.layer == 0:
                self.input_trans = nn.ModuleList([dglnn.HeteroLinear({ntype: in_feat for ntype in ntypes}, out_feat)
                                                  for t in range(time_length)])
            else:
                self.input_trans = None
            self.graph_encoder = nn.ModuleList([CustomHeteroGCN(encoder_type, out_feat, out_feat,
                                                                ntypes, etypes, layer, activation,
                                                                special_edges=special_edges)
                                                for t in range(time_length)])
            self.snapshot_weighters = nn.ModuleList(
                [SnapshotUpdater(out_feat, 3, node_attr_types, activation=activation)
                 for t in range(time_length)])
        if mixed:
            if layer == 0:
                print('change the mixed fc to multiple')
            # self.fc_mix = MLP(out_feat * 2, out_feat, activation=activation)
            self.fc_mix = nn.ModuleList([MLP(out_feat * 2, out_feat, activation=activation)
                                         for t in range(time_length)])

        self.time_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=out_feat, nhead=out_feat // 32, dim_feedforward=4 * out_feat,
                                       batch_first=True, activation='gelu')
            for i in range(time_encoder_layers)])
        # print(self.time_encoders)
        # self.time_pe = PositionalEncodingSC(out_feat, dropout=0, batch_first=True, max_len=time_length)
        # self.re_index = [t for t in range(time_length)]
        # self.re_index.reverse()

    def get_mixed_embs(self, papers, snapshots, t):
        return self.fc_mix[t](torch.cat((papers, snapshots), dim=-1))

    def forward(self, graphs, feats, cur_snapshot_types, snapshot_embs, index_dicts,
                snapshot_mask=None, show_detail=False):
        # [T, batch], [T, batch, h], [batch, T]
        # print(snapshot_embs.shape, cur_snapshot_types.shape)
        other_output = None
        # print('snapshot mask', snapshot_mask.shape)
        for t in range(self.time_length):
            if self.shared:
                if self.input_trans:
                    feats[t] = self.input_trans(feats[t])
                feats[t] = self.graph_encoder(graphs[t], feats[t])
                sum_out = self.snapshot_weighters(graphs[t], cur_snapshot_types[t], feats[t]['paper'],
                                                  snapshot_embs[t], index_dicts[t])
            else:
                if self.input_trans:
                    feats[t] = self.input_trans[t](feats[t])
                if show_detail and t == self.time_length - 1:
                    feats[t], other_output = self.graph_encoder[t](graphs[t], feats[t], show_detail=show_detail)
                else:
                    feats[t] = self.graph_encoder[t](graphs[t], feats[t])
                # print("after gcn:{:.2f}MB".format(torch.cuda.memory_allocated(0) / 1024 ** 2))
                sum_out = self.snapshot_weighters[t](graphs[t], cur_snapshot_types[t], feats[t]['paper'],
                                                     snapshot_embs[t], index_dicts[t])
            snapshot_embs[t] = snapshot_embs[t] + sum_out
            if self.mixed:
                index = index_dicts[t]['mixer_index']
                feats[t]['paper'] = self.get_mixed_embs(feats[t]['paper'], snapshot_embs[t][index, :], t)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            # if snapshot_embs.dtype != torch.float32:
            #     print(snapshot_embs.dtype)
            #     snapshot_embs = snapshot_embs.float()
            snapshot_embs = snapshot_embs.float()
            for time_encoder in self.time_encoders:
                snapshot_embs = time_encoder(snapshot_embs.transpose(0, 1),
                                             src_key_padding_mask=snapshot_mask).transpose(0, 1)
        time_mask = (~snapshot_mask).transpose(0, 1).unsqueeze(dim=-1)
        snapshot_embs = snapshot_embs * time_mask

        if other_output:
            return feats, snapshot_embs, other_output
        else:
            return feats, snapshot_embs


class SnapshotEncoder(nn.Module):
    def __init__(self, in_feat, out_feat, tgt_indicator, mode='mix', learned_time=False):
        super(SnapshotEncoder, self).__init__()
        self.fc_paper = nn.Linear(in_feat, out_feat)
        self.fc_time = nn.Linear(in_feat, out_feat)
        self.mode = mode
        self.tgt_indicator = tgt_indicator

        if self.mode == 'mix':
            self.fc_out = nn.Linear(out_feat * 2, out_feat)
        elif self.mode == 'gate':
            self.fc_out = GateModule(out_feat, out_feat, mode='trans')
        elif self.mode == 'mix_gate':
            self.fc_out = GateModule(out_feat, out_feat, mode='mix')

    def forward(self, graph, feat, time_emb):
        paper_index = torch.where(graph.nodes['paper'].data[self.tgt_indicator])
        # print(feat['paper'].device)
        papers = self.fc_paper(feat['paper'][paper_index].detach())
        # time_index = graph.in_edges(paper_index, etype=self.time_edge)
        # time_index = torch.where(graph.nodes['time'].data[dgl.NID] == cur_time)
        # times = feat['time'][time_index].detach()
        times = self.fc_time(torch.tile(time_emb.detach(), (papers.shape[0], 1)))

        if self.mode == 'mix':
            # print(papers.device, times.device)
            output = self.fc_out(torch.cat((papers, times), dim=-1))
        elif 'gate' in self.mode:
            output = self.fc_out(times, papers)
        return output


class BasicSnapshotWeighter(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BasicSnapshotWeighter, self).__init__()

    def forward(self, bg, cur_snapshot_types, papers, snapshots):
        h_src, h_dst = papers, snapshots

        index = []
        num_nodes = bg.batch_num_nodes('paper')
        for i in range(len(num_nodes)):
            index.extend([i] * num_nodes[i])
        index = torch.tensor(index, dtype=torch.long).to(h_src.device)

        with bg.local_scope():
            bg.nodes['paper'].data['h'] = h_src
            # print(a.shape, feat_src.shape)
            sum_out = dgl.readout_nodes(bg, 'h', ntype='paper', op='mean')
            # print(sum_out.shape)
        return sum_out, index


class SimpleSnapshotWeighter(nn.Module):
    def __init__(self, out_feats, snapshot_types, node_attr_types, activation=nn.LeakyReLU()):
        super(SimpleSnapshotWeighter, self).__init__()
        self.snapshot_type_emb = nn.Embedding(snapshot_types + 1, out_feats, padding_idx=-1)
        self.node_attr_type_emb = nn.ModuleDict({node_attr_type: nn.Embedding(2, out_feats, padding_idx=0)
                                                 for node_attr_type in node_attr_types})
        self.node_attr_types = node_attr_types
        self.leaky_relu = nn.LeakyReLU()
        self.fc_src = nn.Linear(out_feats, out_feats)
        self.fc_dst = nn.Linear(out_feats, out_feats)
        self.fc_src_t = nn.Linear(out_feats, out_feats)
        self.fc_dst_t = nn.Linear(out_feats, out_feats)
        # self.fc_dst = TFC(out_feats, out_feats)
        # self.attn = nn.Parameter(torch.FloatTensor(size=(1, out_feats)))
        self.out_feats = out_feats
        self.fc_out = nn.Linear(out_feats, out_feats)
        self.activation = activation
        # self.attn = nn.Parameter(torch.FloatTensor(size=(1, out_feats)))
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, self.out_feats)))
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self.out_feats)))
        self.bias = nn.Parameter(torch.FloatTensor(size=(1, self.out_feats)))
        self.bias_t = nn.Parameter(torch.FloatTensor(size=(1, self.out_feats)))
        # self._num_heads = num_heads
        # self._out_feats = out_feats
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.attn)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.attn_t)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_t, -bound, bound)

    def forward(self, bg, cur_snapshot_types, papers, snapshots):
        h_src, h_dst = papers, snapshots
        # print(h_src.shape, h_dst.shape, cur_snapshot_types.shape)
        feat_src = self.fc_src(h_src)
        # feat_dst = self.fc_dst(h_dst)

        # print(self.node_attr_types)
        src_type_h = torch.stack([self.node_attr_type_emb[node_attr_type](
            bg.nodes['paper'].data[node_attr_type]) for node_attr_type in self.node_attr_types], dim=0).sum(dim=0)
        dst_type_h = self.snapshot_type_emb(cur_snapshot_types)

        index = []
        num_nodes = bg.batch_num_nodes('paper')
        for i in range(len(num_nodes)):
            index.extend([i] * num_nodes[i])
        index = torch.tensor(index, dtype=torch.long).to(feat_src.device)
        # print(index)
        # tiled_embs = feat_dst[index]  # [N, h]

        feat_dst = self.fc_dst(h_dst[index])
        # feat_dst = self.fc_dst(h_dst).view(-1, self.n_heads, self.out_feats)[index]
        # tiled_embs = feat_dst + dst_type_h[index]

        # print(feat_src.shape, src_type_h.shape, tiled_embs.shape)
        # e = self.leaky_relu(feat_src + src_type_h + feat_dst + dst_type_h)
        # print(feat_src.shape, feat_dst.shape, self.bias.shape)
        e = self.leaky_relu(feat_src + feat_dst + self.bias)
        e = (e * self.attn).sum(dim=-1)  # [N]

        # print(index.shape)
        et = self.leaky_relu(self.fc_src_t(src_type_h) + self.fc_dst_t(dst_type_h[index]) + self.bias_t)
        et = (et * self.attn_t).sum(dim=-1)  # [N]

        e = (e + et).unsqueeze(dim=-1)

        mask_indicator = (torch.stack([bg.nodes['paper'].data[node_attr_type]
                                       for node_attr_type in self.node_attr_types], dim=-1).sum(dim=-1) == 0) * -1e8
        a = edge_softmax(e + mask_indicator.unsqueeze(dim=-1), index)
        # a = edge_softmax(e, index).unsqueeze(dim=-1)
        # print(a.sum(0))
        # print(a.shape)
        with bg.local_scope():
            bg.nodes['paper'].data['a'] = a
            # bg.nodes['paper'].data['h'] = feat_src
            bg.nodes['paper'].data['h'] = h_src
            # print(a.shape, feat_src.shape)
            sum_out = dgl.readout_nodes(bg, 'h', 'a', ntype='paper', op='sum')
            # print(sum_out.shape)
        sum_out = self.fc_out(sum_out)
        # print(sum_out.shape)
        return sum_out, index


class SnapshotWeighter(nn.Module):
    def __init__(self, out_feats, snapshot_types, node_attr_types, n_heads=4, activation=nn.LeakyReLU()):
        super(SnapshotWeighter, self).__init__()
        self.snapshot_type_emb = nn.Embedding(snapshot_types + 1, out_feats, padding_idx=-1)
        self.node_attr_type_emb = nn.ModuleDict({node_attr_type: nn.Embedding(2, out_feats, padding_idx=0)
                                                 for node_attr_type in node_attr_types})
        self.node_attr_types = node_attr_types
        self.leaky_relu = nn.LeakyReLU()
        self.fc_src = nn.Linear(out_feats, out_feats)
        self.fc_dst = nn.Linear(out_feats, out_feats)
        # self.fc_dst = TFC(out_feats, out_feats)
        self.out_feats = out_feats // n_heads
        self.n_heads = n_heads
        self.fc_out = nn.Linear(out_feats, out_feats)
        self.activation = activation
        # self.attn = nn.Parameter(torch.FloatTensor(size=(1, out_feats)))
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feats)))
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feats)))
        # self._num_heads = num_heads
        # self._out_feats = out_feats
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        # nn.init.xavier_normal_(self.dst_w, gain=gain)

    def forward(self, bg, cur_snapshot_types, papers, snapshots, index_dict):
        h_src, h_dst = papers, snapshots
        index = index_dict['index']
        valid_num_nodes = index_dict['valid_num_nodes']
        selected_indicator = index_dict['selected_indicator']
        # selected_indicator = (torch.stack([bg.nodes['paper'].data[node_attr_type]
        #                                    for node_attr_type in self.node_attr_types], dim=-1).sum(dim=-1) > 0)
        h_src = h_src[selected_indicator, :]
        # print(h_src.shape, h_dst.shape, cur_snapshot_types.shape)
        feat_src = self.fc_src(h_src).view(-1, self.n_heads, self.out_feats)
        # feat_dst = self.fc_dst(h_dst)

        src_type_h = (self.node_attr_type_emb['is_cite'](bg.nodes['paper'].data['is_cite'][selected_indicator]) +
                      self.node_attr_type_emb['is_ref'](bg.nodes['paper'].data['is_cite'][selected_indicator]) +
                      self.node_attr_type_emb['is_target'](bg.nodes['paper'].data['is_cite'][selected_indicator])
                      ).view(-1, self.n_heads, self.out_feats)
        dst_type_h = self.snapshot_type_emb(cur_snapshot_types).view(-1, self.n_heads, self.out_feats)

        feat_dst = self.fc_dst(h_dst.clone()).view(-1, self.n_heads, self.out_feats)
        feat_dst = torch.index_select(feat_dst, dim=0, index=index.to(feat_dst.device))

        e = self.leaky_relu(feat_src + feat_dst)
        e = (e * self.attn).sum(dim=-1)  # [N]

        # print(index.shape)
        et = self.leaky_relu(src_type_h + dst_type_h[index])
        et = (et * self.attn_t).sum(dim=-1)  # [N]

        e = e + et

        a = edge_softmax(e, index.to(e.device)).unsqueeze(dim=-1)
        x = feat_src * a
        # print(valid_num_nodes.shape, x.shape)
        sum_out = segment.segment_reduce(valid_num_nodes, x, reducer='sum')
        sum_out = self.fc_out(sum_out.view(-1, self.n_heads * self.out_feats))
        # print(sum_out.shape)
        return sum_out


def simple_node_drop(graph, prob, batched=False, reverse=False, focus=False):
    selected_index = (1 - graph.nodes['paper'].data['is_target']).to(torch.bool)
    if focus:
        focus_index = (graph.nodes['paper'].data['hop'] == 1) & (graph.nodes['paper'].data['is_cite'] == 1)
        # print(selected_index.sum())
        selected_index = focus_index & selected_index
        # print(selected_index.sum())
    count = torch.sum(selected_index).item()
    # print(count)
    if count > 0:
        if batched:
            num_nodes = graph.batch_num_nodes(ntype='paper')
            batch = []
            for i in range(len(num_nodes)):
                batch.extend([i] * num_nodes[i])
            batch = np.array(batch, dtype=np.int64)[selected_index]
            batch_count = Counter(batch.tolist())
            # print(batch_count)

            drop_rate = []
            for i in range(len(num_nodes)):
                drop_rate.extend([batch_count[i] * prob] * batch_count[i])
            drop_rate = torch.tensor(drop_rate)
            # print(drop_rate)

            batch = torch.from_numpy(batch).to(graph.device)
            # print(batch)
            cur_values = graph.nodes['paper'].data['citations'][selected_index]
            if not reverse:
                cur_values = -cur_values
            norm_p = edge_softmax(cur_values.float(), batch)
        else:
            drop_rate = count * prob
            cur_values = graph.nodes['paper'].data['citations'][selected_index]
            # print(cur_values)
            if not reverse:
                cur_values = -cur_values
            # print(cur_values)
            norm_p = torch.softmax(cur_values.float(), dim=-1)

        cur_p = norm_p * drop_rate
        drop_indicator = torch.rand(count) < cur_p
        drop_nodes = torch.where(selected_index)[0][drop_indicator]

        graph.remove_nodes(drop_nodes, ntype='paper')

    return graph


def simple_node_list_drop(detailed_dict, prob, cur_graph):
    selected_index = (1 - torch.tensor(detailed_dict['indicators']['is_target'])).bool()
    count = torch.sum(selected_index).item()
    # print(count)
    if count > 0:
        drop_rate = count * prob
        cur_nodes = torch.tensor(detailed_dict['nodes']['paper'])[selected_index]
        cur_citations = cur_graph.nodes['paper'].data['citations'][cur_nodes]
        cur_p = torch.softmax(cur_citations.float(), dim=-1).numpy() * drop_rate
        # print(cur_p)
        drop_indicator = np.random.rand(count) < cur_p
        # drop_nodes = torch.where(selected_index)[0][drop_indicator]
        drop_t = torch.tensor([False] * (count + 1))
        drop_t[selected_index] = torch.from_numpy(drop_indicator)
        # print(drop_t)
        drop_indexes = torch.where(drop_t)[0].numpy().tolist()
        # print(drop_indexes)
        for i in range(len(drop_indexes)):
            drop_index = drop_indexes[i]
            detailed_dict['nodes']['paper'].pop(drop_index - i)
            for indicator in detailed_dict['indicators']:
                detailed_dict['indicators'][indicator].pop(drop_index - i)
    return detailed_dict


if __name__ == '__main__':
    print('test')
    # time_embs = {'train': torch.randn(5, 384)}
    # module = AlignTopicModule(384, 128, time_embs, time_length=5, topic_num=10, sim_method='cos_sim')
    # snapshot_embs = torch.randn(8, 128)
    # rst = module(snapshot_embs, 'train')
    # print(rst[0])
