import copy
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import softmax as edge_softmax
from transformers import BertModel

from layers.GCN.CL_layers import ProjectionHead, simple_cl, triplet_cl
from layers.GCN.RGCN import CustomHeteroGCN
from layers.common_layers import MLP, RNNEncoder, GateModule, CapsuleNetwork
from utilis.scripts import SimLoss


class SimpleDisenModule(nn.Module):
    # def __init__(self, encoder, embed_dim, hidden_dim, time_length, device, time_embs=None, topic_num=10, disen_loss=None):
    def __init__(self, encoder, topic_module, pop_module, contri_module, device, mode='normal', disen_loss=None,
                 ablation_channel=(), sum_method='normal', loss_weights=None, use_constraint=False):
        super(SimpleDisenModule, self).__init__()
        # self.topic_module = AlignTopicModule(embed_dim, hidden_dim, time_embs, time_length, topic_num)
        # self.pop_module = CitationPopModule(hidden_dim)
        # self.contri_module = SimpleContriModule(hidden_dim)
        self.topic_module = topic_module
        self.pop_module = pop_module
        self.contri_module = contri_module
        self.encoder = encoder
        self.device = device
        # self.topic_weight = 0.5
        # self.pop_weight = 0.5
        # self.disen_weight = 0.5
        if loss_weights is None:
            self.loss_weights = {
                'topic': 0.5,
                'pop': 0.5,
                'disen': 0.5
            }
        else:
            self.loss_weights = loss_weights
        print('>>>loss_weights:', self.loss_weights)
        self.mode = mode
        self.ablation_channel = set(ablation_channel)
        self.sum_method = sum_method
        valid_channel = 3 - len(ablation_channel)
        if disen_loss and valid_channel > 1:
            self.disen_loss = SimLoss(valid_channel, method=disen_loss)
        else:
            self.disen_loss = None
        if self.mode == 'cl':
            self.p = 0.1
        elif self.mode == 'ecl':
            self.vice_encoder = copy.deepcopy(self.encoder).to(self.device)
            self.eta = 1.
        self.use_constraint = use_constraint
        self.constraint_weight = 10

    # def get_disen_loss(self, topic_embs, pop_embs, contri_embs):
    #     if self.disen_loss is not None:
    #         disen_loss = self.disen_loss(torch.stack((topic_embs, pop_embs, contri_embs), dim=0)).mean()
    #         # print(disen_loss)
    #         return disen_loss

    def get_constraint_loss(self, all_out):
        constraint_loss = 0
        zero_v = torch.zeros_like(all_out['contri'], device=all_out['contri'].device)
        for channel in all_out:
            # contri to be the biggest
            if channel != 'contri':
                cur_sub = all_out[channel] - all_out['contri']
                # print(cur_sub)
                cur_loss = torch.maximum(cur_sub, zero_v).mean()
                constraint_loss = constraint_loss + cur_loss * self.constraint_weight
                if self.use_constraint == 'aug':
                    # bigger than zero
                    aug_loss = torch.maximum(-all_out[channel], zero_v).mean()
                    constraint_loss = constraint_loss + aug_loss
                    # print('aug_loss:', aug_loss)
            elif self.use_constraint == 'aug':
                aug_loss = torch.maximum(-all_out[channel], zero_v).mean()
                constraint_loss = constraint_loss + aug_loss
                print('aug_loss:', aug_loss)
        print('constraint_loss:', constraint_loss)
        return constraint_loss

    def get_disen_loss(self, all_embs):
        if self.disen_loss is not None:
            input_embs = torch.stack([all_embs[channel]
                                      for channel in all_embs if channel not in self.ablation_channel], dim=0)
            disen_loss = self.disen_loss(input_embs).mean()
            # print(disen_loss)
            return disen_loss

    def get_aug_encoder(self):
        # print('aug here')
        for (adv_name, adv_param), (name, param) in zip(self.vice_encoder.named_parameters(),
                                                        self.encoder.named_parameters()):
            # print(name, param.data.std())
            # if param.data.std() < 0:
            #     print(name, param.data.std())
            adv_param.data = param.data.detach() + self.eta * torch.normal(0, torch.ones_like(
                param.data) * param.data.std()).detach().to(self.device)

    def topic_disen(self, graph_inputs, phase):
        if 'topic' not in self.ablation_channel:
            if self.mode == 'cl':
                g1_inputs, g2_inputs = graph_inputs
                cur_graph_inputs = g1_inputs
                g1_snapshots = self.encoder(g1_inputs)
                g2_snapshots = self.encoder(g2_inputs)
                snapshot_embs = (g1_snapshots + g2_snapshots) / 2
                topic_out, topic_embs, topic_loss = self.topic_module([g1_snapshots, g2_snapshots], phase)
            elif self.mode == 'ecl':
                self.get_aug_encoder()
                cur_graph_inputs = graph_inputs
                g1_snapshots = self.encoder(graph_inputs)
                g2_snapshots = self.vice_encoder(graph_inputs)
                snapshot_embs = g1_snapshots
                topic_out, topic_embs, topic_loss = self.topic_module([g1_snapshots, g2_snapshots], phase)
            elif self.mode == 'pcl':
                g1_inputs, g2_inputs = graph_inputs
                cur_graph_inputs = g1_inputs
                g1_snapshots = self.encoder(g1_inputs)
                g2_snapshots = self.encoder(g2_inputs)
                snapshot_embs = (g1_snapshots + g2_snapshots) / 2
                topic_out, topic_embs, topic_loss = self.topic_module(snapshot_embs, phase,
                                                                      graph_inputs=cur_graph_inputs)
                cur_graph_inputs = [g1_snapshots, g2_snapshots]
            elif self.mode == 'ocl':
                g1_inputs, g2_inputs = graph_inputs
                cur_graph_inputs = g1_inputs
                g1_snapshots = self.encoder(g1_inputs)
                g2_snapshots = self.encoder(g2_inputs)
                snapshot_embs = g1_snapshots
                topic_out, topic_embs, topic_loss = self.topic_module([g1_snapshots, g2_snapshots], phase)
            elif self.mode == 'scl':
                cur_graph_inputs, pos_inputs, neg_inputs = graph_inputs
                origin_snapshots = self.encoder(cur_graph_inputs)
                pos_snapshots = self.encoder(pos_inputs)
                neg_snapshots = self.encoder(neg_inputs)
                snapshot_embs = origin_snapshots
                topic_out, topic_embs, topic_loss = self.topic_module([origin_snapshots, pos_snapshots, neg_snapshots],
                                                                      phase)
            else:
                cur_graph_inputs = graph_inputs
                snapshot_embs = self.encoder(graph_inputs)
                topic_out, topic_embs, topic_loss = self.topic_module(snapshot_embs, phase,
                                                                      graph_inputs=cur_graph_inputs)
        else:
            if self.mode == 'pcl':
                g1_inputs, g2_inputs = graph_inputs
                g1_snapshots = self.encoder(g1_inputs)
                g2_snapshots = self.encoder(g2_inputs)
                snapshot_embs = (g1_snapshots + g2_snapshots) / 2
                cur_graph_inputs = [g1_snapshots, g2_snapshots]
            else:
                cur_graph_inputs = graph_inputs
                snapshot_embs = self.encoder(graph_inputs)
            topic_out, topic_embs, topic_loss = [None] * 3
        return snapshot_embs, topic_out, topic_embs, topic_loss, cur_graph_inputs

    def pop_disen(self, snapshot_embs, pop_indicators, cur_graph_inputs):
        if 'pop' not in self.ablation_channel:
            if self.mode == 'pcl':
                pop_out, pop_embs, pop_loss = self.pop_module(cur_graph_inputs, pop_indicators)
            else:
                pop_out, pop_embs, pop_loss = self.pop_module(snapshot_embs, pop_indicators,
                                                              graph_inputs=cur_graph_inputs)
        else:
            pop_out, pop_embs, pop_loss = [None] * 3
        return pop_out, pop_embs, pop_loss

    def show(self, graph_inputs, phase, show_detail=True):
        return self.predict(graph_inputs, phase, show_detail)

    def predict(self, graph_inputs, phase, show_detail=False):
        if show_detail:
            snapshot_readouts, other_outputs = self.encoder(graph_inputs, show_detail)
        else:
            snapshot_readouts = self.encoder(graph_inputs)
        topic_out, topic_embs = self.topic_module.predict(snapshot_readouts, phase)
        pop_out, pop_embs = self.pop_module.predict(snapshot_readouts)
        contri_out, contri_embs = self.contri_module.predict(snapshot_readouts)
        # if show_detail:
        #     return torch.stack([topic_out, pop_out, contri_out], dim=0)
        # else:
        #     return topic_out + pop_out + contri_out
        all_out = {'topic': topic_out, 'pop': pop_out, 'contri': contri_out}
        # print(all_out)
        all_out = torch.stack([all_out[channel]
                               for channel in all_out if channel not in self.ablation_channel], dim=0)
        # print(all_out.shape)
        if show_detail:
            # print(graph_inputs['graphs'])
            final_graph = graph_inputs['graphs'][-1]
            # print(torch.eq(final_graph.batch_num_edges(etype='cites'),
            #                final_graph.batch_num_edges(etype='is cited by')))
            # [layer, node, cites, ac]
            weight_list = []
            for i in range(len(other_outputs)):
                temp_list = []
                for etype in other_outputs[i]:
                    cur_weight = other_outputs[i][etype]
                    cur_weight[0] = cur_weight[0].mean(dim=1, keepdim=True)
                    cur_weight = torch.cat(cur_weight, dim=-1)
                    temp_list.append(cur_weight)
                weight_list.append(torch.cat(temp_list, dim=-2))
            other_outputs = torch.stack(weight_list, dim=0).cpu()
            split_nums = final_graph.batch_num_edges(etype='cites')
            # print(split_nums)
            other_outputs = torch.split(other_outputs, split_size_or_sections=split_nums.cpu().numpy().tolist(), dim=1)
            other_outputs = [temp.to(torch.float16).numpy() for temp in other_outputs]

            all_embs = torch.stack([topic_embs, pop_embs, contri_embs], dim=1)
            return [all_out, other_outputs, all_embs]
        else:
            if self.sum_method == 'exp':
                return torch.log(torch.exp(all_out).sum(dim=0))
            else:
                return all_out.sum(dim=0)

    def forward(self, graph_inputs, phase, pop_indicators):
        snapshot_embs, topic_out, topic_embs, topic_loss, cur_graph_inputs = self.topic_disen(graph_inputs, phase)
        pop_out, pop_embs, pop_loss = self.pop_disen(snapshot_embs, pop_indicators, cur_graph_inputs)
        contri_out, contri_embs = self.contri_module(snapshot_embs)
        # print('='*59)
        # print('graph encoder:', torch.isnan(snapshot_embs).sum())
        # print('topic:', torch.isnan(topic_out).sum(), torch.isnan(topic_embs).sum(), topic_loss)
        # print('pop:', torch.isnan(pop_out).sum(), torch.isnan(pop_embs).sum(), pop_loss)
        # print('contri:', torch.isnan(contri_out).sum(), torch.isnan(contri_embs).sum())

        # out
        all_out = {'topic': topic_out, 'pop': pop_out, 'contri': contri_out}
        if self.use_constraint in {'simple', 'aug'}:
            loss = self.get_constraint_loss(all_out)
        else:
            loss = 0
        all_out = torch.stack([all_out[channel]
                               for channel in all_out if channel not in self.ablation_channel], dim=0)
        # print(all_out)
        # out = topic_out + pop_out + contri_out
        # out = all_out.sum(dim=0)
        # print(self.ablation_channel)
        # print('all out', all_out.shape)
        if self.sum_method == 'exp':
            out = torch.log(torch.exp(all_out).sum(dim=0))
        else:
            out = all_out.sum(dim=0)

        # disen loss
        # loss = self.topic_weight * topic_loss + self.pop_weight * pop_loss
        if len(self.ablation_channel) < 2:
            all_loss = {'topic': topic_loss, 'pop': pop_loss}
            all_loss = torch.stack([all_loss[channel] * self.loss_weights[channel]
                                    for channel in all_loss if channel not in self.ablation_channel], dim=0)
            # print(all_loss.shape)
            loss = loss + all_loss.sum(dim=0)
        # print(topic_loss, pop_loss)
        all_embs = {'topic': topic_embs, 'pop': pop_embs, 'contri': contri_embs}
        if self.disen_loss:
            # loss = loss + self.disen_weight * self.get_disen_loss(topic_embs, pop_embs, contri_embs)
            loss = loss + self.loss_weights['disen'] * self.get_disen_loss(all_embs)
        # loss = 0
        # print('in forward:', self.loss_weights)
        return out, loss


class SimpleDisenModuleT(SimpleDisenModule):
    def __init__(self, *args, **kwargs):
        super(SimpleDisenModuleT, self).__init__(*args, **kwargs)
        assert self.mode != 'ecl'

    def topic_disen(self, graph_inputs, phase):
        if 'topic' not in self.ablation_channel:
            if self.mode == 'cl':
                cur_graph_inputs = None
                all_snapshots = self.encoder(graph_inputs)
                batch_len = all_snapshots.shape[0] // 2
                g1_snapshots = all_snapshots[:batch_len]
                g2_snapshots = all_snapshots[batch_len:]
                snapshot_embs = (g1_snapshots + g2_snapshots) / 2
                topic_out, topic_embs, topic_loss = self.topic_module([g1_snapshots, g2_snapshots], phase)
            elif self.mode == 'pcl':
                cur_graph_inputs = None
                all_snapshots = self.encoder(graph_inputs)
                batch_len = all_snapshots.shape[0] // 2
                g1_snapshots = all_snapshots[:batch_len]
                g2_snapshots = all_snapshots[batch_len:]
                snapshot_embs = (g1_snapshots + g2_snapshots) / 2
                topic_out, topic_embs, topic_loss = self.topic_module(snapshot_embs, phase,
                                                                      graph_inputs=cur_graph_inputs)
                cur_graph_inputs = [g1_snapshots, g2_snapshots]
            elif self.mode == 'ocl':
                cur_graph_inputs = None
                all_snapshots = self.encoder(graph_inputs)
                batch_len = all_snapshots.shape[0] // 2
                g1_snapshots = all_snapshots[:batch_len]
                g2_snapshots = all_snapshots[batch_len:]
                snapshot_embs = g1_snapshots
                topic_out, topic_embs, topic_loss = self.topic_module([g1_snapshots, g2_snapshots], phase)
            elif self.mode == 'scl':
                cur_graph_inputs = None
                all_snapshots = self.encoder(graph_inputs)
                batch_len = all_snapshots.shape[0] // 3
                origin_snapshots = all_snapshots[:batch_len]
                pos_snapshots = all_snapshots[batch_len:batch_len * 2]
                neg_snapshots = all_snapshots[batch_len * 2:]
                snapshot_embs = origin_snapshots
                topic_out, topic_embs, topic_loss = self.topic_module([origin_snapshots, pos_snapshots, neg_snapshots],
                                                                      phase)
            else:
                cur_graph_inputs = None
                snapshot_embs = self.encoder(graph_inputs)
                topic_out, topic_embs, topic_loss = self.topic_module(snapshot_embs, phase,
                                                                      graph_inputs=cur_graph_inputs)
        else:
            if self.mode == 'pcl':
                all_snapshots = self.encoder(graph_inputs)
                batch_len = all_snapshots.shape[0] // 2
                g1_snapshots = all_snapshots[:batch_len]
                g2_snapshots = all_snapshots[batch_len:]
                snapshot_embs = (g1_snapshots + g2_snapshots) / 2
                cur_graph_inputs = [g1_snapshots, g2_snapshots]
            else:
                cur_graph_inputs = None
                snapshot_embs = self.encoder(graph_inputs)
            topic_out, topic_embs, topic_loss = [None] * 3
        return snapshot_embs, topic_out, topic_embs, topic_loss, cur_graph_inputs


class PopDisenModule(SimpleDisenModule):
    def __init__(self, encoder, pop_module, contri_module, device, mode='normal', disen_loss=None):
        super(PopDisenModule, self).__init__(encoder, None, pop_module, contri_module, device, mode, disen_loss)
        if disen_loss:
            self.disen_loss = SimLoss(2, method=disen_loss)

    def predict(self, graph_inputs, phase, show_detail=False):
        snapshot_readouts = self.encoder(graph_inputs)
        # topic_out = self.topic_module.predict(snapshot_readouts, phase)
        pop_out, _ = self.pop_module.predict(snapshot_readouts)
        contri_out, _ = self.contri_module.predict(snapshot_readouts)
        if show_detail:
            return torch.stack([pop_out, contri_out], dim=0)
        else:
            return pop_out + contri_out

    def get_disen_loss(self, pop_embs, contri_embs):
        if self.disen_loss is not None:
            # print(torch.stack((pop_embs, contri_embs), dim=0).shape)
            disen_loss = self.disen_loss(torch.stack((pop_embs, contri_embs), dim=0)).mean()
            # print(disen_loss)
            return disen_loss

    def forward(self, graph_inputs, phase, pop_indicators):
        g1_inputs, g2_inputs = graph_inputs
        cur_graph_inputs = g1_inputs
        g1_snapshots = self.encoder(g1_inputs)
        g2_snapshots = self.encoder(g2_inputs)
        snapshot_embs = (g1_snapshots + g2_snapshots) / 2
        pop_out, pop_embs, pop_loss = self.pop_module([g1_snapshots, g2_snapshots], pop_indicators,
                                                      graph_inputs=cur_graph_inputs)
        contri_out, contri_embs = self.contri_module(snapshot_embs)

        out = pop_out + contri_out
        loss = self.pop_weight * pop_loss
        # print(topic_loss, pop_loss)
        if self.disen_loss:
            loss = loss + self.disen_weight * self.get_disen_loss(pop_embs, contri_embs)
        # loss = 0
        return out, loss


class SimpleContriModule(nn.Module):
    def __init__(self, hidden_dim, activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(SimpleContriModule, self).__init__()
        self.contri_encoder = MLP(hidden_dim, hidden_dim, activation=activation)  # encoder
        # self.fc_out = nn.Sequential(MLP(hidden_dim, 1, activation=activation),
        #                             nn.ReLU())  # out
        self.fc_out = MLP(hidden_dim, 1, activation=activation)  # out
        self.dropout = nn.Dropout(dropout)

    def predict(self, snapshot_readouts):
        contri_embs = self.contri_encoder(snapshot_readouts)
        return self.fc_out(contri_embs), contri_embs

    def forward(self, snapshot_readouts):
        contri_embs = self.contri_encoder(self.dropout(snapshot_readouts))
        out = self.fc_out(self.dropout(contri_embs))
        return out, contri_embs


class TopicModule(nn.Module):
    def __init__(self, hidden_dim, activation=nn.LeakyReLU(inplace=True), dropout=0):
        # prototype topic module
        super(TopicModule, self).__init__()
        self.topic_encoder = MLP(hidden_dim, hidden_dim, activation=activation)
        # self.fc_out = nn.Sequential(MLP(hidden_dim, 1, activation=activation),
        #                             nn.ReLU())
        self.fc_out = MLP(hidden_dim, 1, activation=activation)  # out
        self.dropout = nn.Dropout(dropout)

    def predict(self, snapshot_readouts, phase):
        topic_embs = self.topic_encoder(snapshot_readouts)
        return self.fc_out(topic_embs), topic_embs

    def get_topic_loss(self, topic_embs):
        return None

    def forward(self, snapshot_readouts, phase, **kwargs):
        return self.fc_out(self.topic_encoder(snapshot_readouts))


class AlignTopicModule(TopicModule):
    def __init__(self, embed_dim, hidden_dim, time_embs, time_length=5, topic_num=10, sim_method='cos_sim',
                 activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(AlignTopicModule, self).__init__(hidden_dim, activation, dropout)
        self.time_embs = time_embs
        self.time_encoder = nn.Sequential(MLP(embed_dim, hidden_dim, activation=activation),
                                          nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                                                                     dim_feedforward=2 * hidden_dim,
                                                                     batch_first=True))
        # self.topic_projector = MLP(hidden_dim, hidden_dim * topic_num, dim_inner=hidden_dim * topic_num // 2)
        self.time_topic_projector = MLP(hidden_dim * time_length, hidden_dim * topic_num,
                                        bias=False, activation=activation)
        # self.topic_w = nn.Parameter(torch.empty(topic_num, hidden_dim))
        self.topic_w = nn.Linear(hidden_dim, hidden_dim * topic_num, bias=False)
        self.topic_num = topic_num
        self.hidden_dim = hidden_dim
        self.sim_loss = SimLoss(topic_num, method=sim_method)
        self.sim_method = sim_method
        self.norm_p = nn.Parameter(torch.tensor([1 / topic_num] * topic_num).float().log().unsqueeze(dim=0),
                                   requires_grad=False)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def predict(self, snapshot_readouts, phase):
        snapshot_readouts = self.topic_encoder(snapshot_readouts)  # [B, h]
        cur_time_embs = self.time_encoder(self.time_embs[phase]).view(1, -1)  # [1, t * h]
        cur_time_topics = self.time_topic_projector(cur_time_embs).view(self.topic_num, self.hidden_dim)  # [tn, h]
        topic_a = torch.softmax(snapshot_readouts @ cur_time_topics.transpose(0, 1), dim=-1).unsqueeze(
            dim=-1)  # [B, tn, 1]
        # [B, h]
        topic_embs = (self.topic_w(snapshot_readouts).view(-1, self.topic_num, self.hidden_dim) * topic_a).sum(dim=1)
        out = self.fc_out(topic_embs)
        return out, topic_embs

    def get_topic_loss(self, topic_a, time_topics):
        # print(time_topics.shape)
        sim_loss = self.sim_loss(time_topics)
        if self.sim_method == 'cos_sim':
            sim_loss = torch.abs(sim_loss).mean()
        # elif self.sim_method == 'mmd_rbf':
        batch_size = topic_a.shape[0]
        norm_loss = self.kl_loss(torch.log_softmax(topic_a, dim=-1), self.norm_p.tile((batch_size, 1)))
        return sim_loss + norm_loss

    def forward(self, snapshot_readouts, phase, **kwargs):
        snapshot_readouts = self.topic_encoder(self.dropout(snapshot_readouts))  # [B, h]
        cur_time_embs = self.time_encoder(self.dropout(self.time_embs[phase])).view(1, -1)  # [1, t * h]
        cur_time_topics = self.time_topic_projector(cur_time_embs).view(self.topic_num, self.hidden_dim)  # [tn, h]
        # print(cur_time_topics.shape)
        topic_a = snapshot_readouts @ cur_time_topics.transpose(0, 1)  # [B, tn, 1]
        topic_a_sm = torch.softmax(topic_a, dim=-1).unsqueeze(dim=-1)
        print(torch.argmax(topic_a_sm.squeeze(dim=-1), dim=-1))
        # [B, h]
        topic_embs = (self.topic_w(snapshot_readouts).view(-1, self.topic_num, self.hidden_dim) * topic_a_sm).sum(dim=1)
        topic_loss = self.get_topic_loss(topic_a, cur_time_topics)
        # print(topic_loss)
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs, topic_loss


class PrototypeTopicModule(TopicModule):
    def __init__(self, hidden_dim, proto_num=10, sim_method='cos_sim',
                 activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(PrototypeTopicModule, self).__init__(hidden_dim, activation, dropout)
        # self.topic_projector = MLP(hidden_dim, hidden_dim * topic_num, dim_inner=hidden_dim * topic_num // 2)
        # self.topic_w = nn.Parameter(torch.empty(topic_num, hidden_dim))
        self.proto_w = nn.Linear(hidden_dim, hidden_dim * proto_num, bias=False)
        self.proto_num = proto_num
        self.protos = nn.Parameter(torch.randn(proto_num, hidden_dim))
        print('>>>proto shape:', self.protos.shape)
        self.hidden_dim = hidden_dim
        self.sim_loss = SimLoss(proto_num, method=sim_method)
        self.sim_method = sim_method
        # self.fc_cls = MLP(hidden_dim, proto_num, activation=activation)
        # self.criterion = nn.CrossEntropyLoss()
        self.norm_p = nn.Parameter(torch.tensor([1 / proto_num] * proto_num).float().log().unsqueeze(dim=0),
                                   requires_grad=False)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.sk_epsilon = 3e-2
        self.sk_iters = 100

    def predict(self, snapshot_readouts, phase):
        snapshot_readouts = self.topic_encoder(self.dropout(snapshot_readouts))  # [B, h]
        topic_a = torch.softmax(snapshot_readouts @ self.protos.transpose(0, 1), dim=-1).unsqueeze(
            dim=-1)  # [B, pn, 1]
        # [B, h]
        topic_embs = (self.proto_w(snapshot_readouts).view(-1, self.proto_num, self.hidden_dim) * topic_a).sum(dim=1)
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs

    def get_topic_loss(self, topic_embs, topic_a):
        # print(time_topics.shape)
        sim_loss = self.sim_loss(self.protos)
        if self.sim_method == 'cos_sim':
            sim_loss = torch.abs(sim_loss).mean()
        # labels = torch.argmax(topic_a.squeeze(dim=-1), dim=-1)
        # pred = self.fc_cls(topic_embs)
        # print(labels)
        # cls_loss = self.criterion(pred, labels)
        # print(topic_a.shape)
        batch_size = topic_a.shape[0]
        norm_loss = self.kl_loss(torch.log_softmax(topic_a, dim=-1), self.norm_p.tile((batch_size, 1)))

        return sim_loss + norm_loss

    def forward(self, snapshot_readouts, phase, **kwargs):
        snapshot_readouts = self.topic_encoder(self.dropout(snapshot_readouts))  # [B, h]
        # topic_a = torch.softmax(snapshot_readouts @ self.protos.transpose(0, 1), dim=-1).unsqueeze(
        #     dim=-1)  # [B, pn, 1]
        topic_a = snapshot_readouts @ self.protos.transpose(0, 1)  # [B, pn]
        topic_a_sm = torch.softmax(topic_a, dim=-1).unsqueeze(dim=-1)
        print(torch.argmax(topic_a_sm.squeeze(dim=-1), dim=-1))
        # [B, h]
        topic_embs = (self.proto_w(snapshot_readouts).view(-1, self.proto_num, self.hidden_dim) * topic_a_sm).sum(dim=1)
        topic_loss = self.get_topic_loss(topic_embs, topic_a)
        # print(topic_loss)
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs, topic_loss


class CapsuleTopicModule(TopicModule):
    def __init__(self, hidden_dim, num_caps=8, n_layers=3, sim_method='orthog', dropout=0):
        super(CapsuleTopicModule, self).__init__(hidden_dim, dropout=dropout)
        self.capsule_encoder = CapsuleNetwork(hidden_dim, hidden_dim, num_caps=num_caps, n_layers=n_layers,
                                              return_split=True)
        self.sim_loss = SimLoss(num_caps, method=sim_method)
        self.sim_method = sim_method

    def predict(self, snapshot_readouts, phase):
        snapshot_readouts = self.topic_encoder(self.dropout(snapshot_readouts))  # [B, h]
        topic_embs, _ = self.capsule_encoder(snapshot_readouts)
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs

    def get_topic_loss(self, topic_embs_split):
        # print(topic_embs_split.shape)
        sim_loss = self.sim_loss(topic_embs_split)
        if self.sim_method == 'cos_sim':
            sim_loss = torch.abs(sim_loss).mean()
        else:
            sim_loss = sim_loss.mean()
        return sim_loss

    def forward(self, snapshot_readouts, phase, **kwargs):
        snapshot_readouts = self.topic_encoder(self.dropout(snapshot_readouts))  # [B, h]
        topic_embs, topic_embs_split = self.capsule_encoder(snapshot_readouts)
        topic_loss = self.get_topic_loss(topic_embs_split.transpose(0, 1))
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs, topic_loss


class CLTopicModule(TopicModule):
    def __init__(self, hidden_dim, dropout=0):
        super(CLTopicModule, self).__init__(hidden_dim, dropout=dropout)
        self.proj_head = ProjectionHead(hidden_dim, hidden_dim)
        self.tau = 0.2

    def get_cl_inputs(self, snapshots, require_grad=True):
        if require_grad:
            snapshots = self.proj_head(snapshots)
        else:
            snapshots = self.proj_head(snapshots).detach()
        return snapshots

    def get_topic_loss(self, g1_snapshots, g2_snapshots):
        z1 = self.get_cl_inputs(g1_snapshots)
        z2 = self.get_cl_inputs(g2_snapshots)
        cl_loss = simple_cl(z1, z2, self.tau)
        return cl_loss

    def predict(self, snapshot_readouts, phase):
        topic_embs = self.topic_encoder(snapshot_readouts)
        return self.fc_out(topic_embs), topic_embs

    def forward(self, snapshot_readouts, phase):
        g1_snapshots, g2_snapshots = snapshot_readouts
        g1_snapshots = self.topic_encoder(self.dropout(g1_snapshots))
        g2_snapshots = self.topic_encoder(self.dropout(g2_snapshots))
        cl_loss = self.get_topic_loss(g1_snapshots, g2_snapshots)
        topic_embs = (g1_snapshots + g2_snapshots) / 2
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs, cl_loss


class OCLTopicModule(CLTopicModule):
    def __init__(self, hidden_dim, dropout=0):
        super(OCLTopicModule, self).__init__(hidden_dim, dropout)

    def forward(self, snapshot_readouts, phase):
        g1_snapshots, g2_snapshots = snapshot_readouts
        g1_snapshots = self.topic_encoder(self.dropout(g1_snapshots))
        g2_snapshots = self.topic_encoder(self.dropout(g2_snapshots))
        cl_loss = self.get_topic_loss(g1_snapshots, g2_snapshots)
        topic_embs = g1_snapshots
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs, cl_loss


class SCLTopicModule(CLTopicModule):
    def __init__(self, hidden_dim, dropout=0):
        super(SCLTopicModule, self).__init__(hidden_dim, dropout)

    def get_topic_loss(self, all_snapshots):
        all_snapshots = self.get_cl_inputs(all_snapshots)
        origin_snapshots, pos_snapshots, neg_snapshots = all_snapshots
        cl_loss = triplet_cl(origin_snapshots, pos_snapshots, neg_snapshots, self.tau)
        return cl_loss

    def forward(self, snapshot_readouts, phase):
        # origin_snapshots, pos_snapshots, neg_snapshots = snapshot_readouts
        all_snapshots = torch.stack(snapshot_readouts, dim=0)
        # print(all_snapshots.shape)
        all_snapshots = self.topic_encoder(self.dropout(all_snapshots))
        # origin_snapshots, pos_snapshots, neg_snapshots = all_snapshots
        cl_loss = self.get_topic_loss(all_snapshots)
        topic_embs = all_snapshots[0]
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs, cl_loss


class ECLTopicModule(CLTopicModule):
    def __init__(self, hidden_dim, dropout=0):
        super(ECLTopicModule, self).__init__(hidden_dim, dropout=dropout)

    def get_topic_loss(self, g1_snapshots, g2_snapshots):
        z1 = self.get_cl_inputs(g1_snapshots)
        z2 = self.get_cl_inputs(g2_snapshots, require_grad=False)
        cl_loss = simple_cl(z1, z2, self.tau)
        return cl_loss

    def forward(self, snapshot_readouts, phase):
        g1_snapshots, g2_snapshots = snapshot_readouts
        g1_snapshots = self.topic_encoder(self.dropout(g1_snapshots))
        g2_snapshots = self.topic_encoder(self.dropout(g2_snapshots))
        cl_loss = self.get_topic_loss(g1_snapshots, g2_snapshots)
        topic_embs = g1_snapshots
        out = self.fc_out(self.dropout(topic_embs))
        return out, topic_embs, cl_loss


class PopModule(nn.Module):
    def __init__(self, hidden_dim, activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(PopModule, self).__init__()
        self.pop_encoder = MLP(hidden_dim, hidden_dim, activation=activation)
        # self.fc_out = nn.Sequential(MLP(hidden_dim, 1, activation=activation),
        #                             nn.ReLU())
        self.fc_out = MLP(hidden_dim, 1, activation=activation)  # out
        self.dropout = nn.Dropout(dropout)

    def get_pop_loss(self, pop_embs, indicators):
        return None

    def predict(self, snapshot_readouts):
        pop_rep = self.pop_encoder(snapshot_readouts)
        return self.fc_out(pop_rep), pop_rep

    def forward(self, snapshot_readouts, indicators, **kwargs):
        pop_embs = self.pop_encoder(self.dropout(snapshot_readouts))
        pop_loss = self.get_pop_loss(pop_embs, indicators)
        out = self.fc_out(self.dropout(pop_embs))
        return out, pop_embs, pop_loss


class DistPopModule(PopModule):
    def __init__(self, hidden_dim, dist_classes=3, activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(DistPopModule, self).__init__(hidden_dim, activation, dropout=dropout)
        self.fc_dist = MLP(hidden_dim, dist_classes, activation=activation)
        self.criterion = nn.CrossEntropyLoss()

    def get_pop_loss(self, pop_embs, indicators):
        labels = self.fc_dist(pop_embs)
        pop_loss = self.criterion(labels, indicators)
        return pop_loss


class CitationPopModule(PopModule):
    def __init__(self, hidden_dim, box_count=5, activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(CitationPopModule, self).__init__(hidden_dim, activation, dropout=dropout)
        self.final_emb = nn.Embedding(box_count + 1, hidden_dim, padding_idx=-1)
        self.avg_emb = nn.Embedding(box_count + 1, hidden_dim, padding_idx=-1)
        self.seq_encoder = RNNEncoder('GRU', hidden_dim, hidden_dim, num_layers=2, dropout=0, bidirectional=False)
        self.mlp_trans = MLP(1, hidden_dim, dim_inner=hidden_dim // 2, activation=activation)
        self.pop_mix = MLP(3 * hidden_dim, hidden_dim, activation=activation)
        # self.sim_loss = SimLoss(2, method='cos_sim')

    def get_pop_loss(self, pop_embs, indicators):
        final_boxes, avg_boxes, seqs, masks = indicators
        final_embs = self.final_emb(final_boxes)
        avg_embs = self.avg_emb(avg_boxes)
        # print(final_embs.shape, avg_embs.shape)
        seq_embs = self.mlp_trans(seqs.unsqueeze(dim=-1))
        # print(seq_embs.shape, masks.shape)
        seq_embs, _ = self.seq_encoder(seq_embs, masks)
        # print(seq_embs.shape, masks.unsqueeze(dim=-1).shape)
        seq_embs = (seq_embs * masks.unsqueeze(dim=-1)).sum(dim=1) / masks.sum(dim=-1, keepdims=True)
        # print(seq_embs.shape)
        pop_mix_embs = self.pop_mix(self.dropout(torch.cat((final_embs, avg_embs, seq_embs), dim=-1)))
        # print(pop_mix_embs.shape)
        pop_sim_loss = (1 - torch.cosine_similarity(pop_embs, pop_mix_embs, dim=-1)).mean()
        return pop_sim_loss


class AccumCitationPopModule(PopModule):
    def __init__(self, hidden_dim, box_count=5, activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(AccumCitationPopModule, self).__init__(hidden_dim, activation, dropout=dropout)
        # self.sim_loss = SimLoss(2, method='cos_sim')
        self.fc_dist = MLP(hidden_dim, box_count, activation=activation)
        self.criterion = nn.CrossEntropyLoss()

    def get_pop_loss(self, pop_embs, indicators):
        final_boxes, avg_boxes, seqs, masks = indicators
        labels = self.fc_dist(pop_embs)
        pop_loss = self.criterion(labels, final_boxes)
        return pop_loss


class CLAccumCitationPopModule(AccumCitationPopModule):
    def __init__(self, hidden_dim, box_count=5, activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(CLAccumCitationPopModule, self).__init__(hidden_dim, box_count, activation, dropout)
        self.proj_head = ProjectionHead(hidden_dim, hidden_dim)
        self.tau = 0.2
        # self.fc_accum = MLP(hidden_dim, hidden_dim, activation=activation)
        # self.fc_cl = MLP(hidden_dim, hidden_dim, activation=activation)

    def get_cl_inputs(self, snapshots, require_grad=True):
        if require_grad:
            snapshots = self.proj_head(snapshots)
        else:
            snapshots = self.proj_head(snapshots).detach()
        return snapshots

    def get_pop_loss(self, pop_embs, indicators):
        # g1_snapshots, g2_snapshots, pop_embs = pop_embs
        # z1 = self.get_cl_inputs(self.fc_cl(g1_snapshots))
        # z2 = self.get_cl_inputs(self.fc_cl(g2_snapshots))
        # z1, z2 = self.get_cl_inputs(self.fc_cl(pop_embs))
        z1, z2 = self.get_cl_inputs(pop_embs)
        cl_loss = simple_cl(z1, z2, self.tau)
        # pop_embs = (g1_snapshots + g2_snapshots) / 2
        final_boxes, avg_boxes, seqs, masks = indicators
        # labels = self.fc_dist(self.fc_accum(pop_embs.mean(dim=0)))
        labels = self.fc_dist(pop_embs.mean(dim=0))
        pop_loss = self.criterion(labels, final_boxes)
        return cl_loss + pop_loss

    def forward(self, snapshot_readouts, indicators, **kwargs):
        g1_snapshots, g2_snapshots = snapshot_readouts
        g1_snapshots = self.pop_encoder(self.dropout(g1_snapshots))
        g2_snapshots = self.pop_encoder(self.dropout(g2_snapshots))
        # pop_embs = (g1_snapshots + g2_snapshots) / 2
        # pop_loss = self.get_pop_loss([g1_snapshots, g2_snapshots, pop_embs], indicators)
        pop_embs = torch.stack([g1_snapshots, g2_snapshots], dim=0)
        pop_loss = self.get_pop_loss(pop_embs, indicators)
        pop_embs = pop_embs.mean(dim=0)
        out = self.fc_out(self.dropout(pop_embs))
        # print(pop_embs.shape, pop_loss.shape)
        return out, pop_embs, pop_loss


class AuthorAccumCitationPopModule(PopModule):
    def __init__(self, embed_dim, hidden_dim, encode_method='accum_weighted_sum',
                 box_count=5, activation=nn.LeakyReLU(inplace=True), dropout=0):
        super(AuthorAccumCitationPopModule, self).__init__(hidden_dim, activation, dropout=dropout)
        # self.sim_loss = SimLoss(2, method='cos_sim')
        # self.fc_dist = MLP(hidden_dim, box_count, activation=activation)
        # self.criterion = nn.CrossEntropyLoss()
        self.accum_emb = nn.Embedding(box_count + 1, embed_dim, padding_idx=-1)
        self.encode_method = encode_method
        self.pop_mix = MLP(2 * embed_dim, hidden_dim, activation=activation)
        # if self.encode_method == 'accum_emb_mean':
        #     self.author_accume_emb = nn.Embedding(box_count + 1, hidden_dim, padding_idx=-1)

    def get_pop_loss(self, pop_embs, pop_mix_embs):
        pop_sim_loss = (1 - torch.cosine_similarity(pop_embs, pop_mix_embs, dim=-1)).mean()
        return pop_sim_loss

    def get_mix_embs(self, graphs, indicators):
        final_boxes, avg_boxes, seqs, masks = indicators
        last_graph = graphs[-1]
        with last_graph.local_scope():
            target_papers = torch.where(last_graph.nodes['paper'].data['is_target'])
            if self.encode_method == 'accum_weighted_sum':
                last_graph['writes'].apply_edges(fn.copy_u('accum_citations', 'c'))
                last_graph['writes'].edata['c'] = dglnn.edge_softmax(last_graph['writes'],
                                                                     last_graph['writes'].edata['c'])
                last_graph['writes'].prop_nodes(target_papers, fn.src_mul_edge('h', 'c', 'm'),
                                                fn.sum('m', 'h'), etype='writes')
                author_embs = last_graph.nodes['paper'].data['h'][target_papers].detach()
        paper_embs = self.accum_emb(final_boxes)
        pop_mix_embs = self.pop_mix(torch.cat((author_embs, paper_embs), dim=-1))
        return pop_mix_embs

    def forward(self, snapshot_readouts, indicators, graph_inputs=None):
        graphs = graph_inputs['graphs']
        pop_embs = self.pop_encoder(self.dropout(snapshot_readouts))
        pop_mix_embs = self.get_mix_embs(graphs, indicators)
        pop_loss = self.get_pop_loss(pop_embs, pop_mix_embs)
        out = self.fc_out(self.dropout(pop_embs))
        return out, pop_embs, pop_loss


if __name__ == '__main__':
    batch = joblib.load('../test/test_batch')
    # print(batch.subgraphs)
    content = batch.x

    graph_inputs = {'graphs': batch.subgraphs[:5]}
    snapshot_readouts = torch.randn(32, 128)
    # rst = module(snapshot_readouts, 'train', graph_inputs=graph_inputs)
    # print(rst[0])
    # print(rst[-1])
    # rst[-1].backward()

    # time_embs = {'train': torch.randn(5, 384)}
    # module = AlignTopicModule(384, 128, time_embs, time_length=5, topic_num=10, sim_method='cos_sim')
    # snapshot_embs = torch.randn(8, 128)
    # rst = module(snapshot_embs, 'train')
    # print(rst[0])
    # print(rst[-1])
