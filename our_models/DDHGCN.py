import copy
import json
import logging
import os
import pandas as pd
from tqdm import tqdm

from models.base_model import BaseModel
import torch
import torch.nn as nn
from layers.GCN.DDHGCN_layers import *
from layers.disen_layers import *
from utilis.log_bar import TqdmToLogger
from utilis.scripts import SampleAllNeighbors, add_new_elements, DealtKhopSubgraph, box_cut, mm_norm, IndexDict
import dgl.function as fn
import jsonlines as jl


class DDHGCN(BaseModel):
    def __init__(self, num_classes, embed_dim, hidden_dim, time_length, ntypes, etypes, start_times,
                 time_embs_path=None, model_type=None, snapshot_mode='gate', learned_time=False, time_pooling='sum',
                 encoder_type='DGINP', n_layers=3, time_layers=1, tree=False, dropout=0, residual=False,
                 graph_dropout=0, ablation_channel=(), graph_norm=False, mixed=True, updater='normal',
                 lambda_weight=0.5,
                 tau=0.2,
                 # shared_encoder=False,
                 **kwargs):
        super(DDHGCN, self).__init__(0, 0, num_classes, 0, **kwargs)
        self.model_name = 'DDHGCN_{}_{}_{}_n{}'.format(encoder_type, snapshot_mode, time_pooling, n_layers)
        self.model_name = self.get_model_name()
        # if time_layers != 1:
        #     self.model_name += '_t{}'.format(time_layers)
        self.time_length = time_length
        phases = ['train', 'val', 'test']
        self.start_times = {phases[i]: start_times[i] - self.time_length + 1 for i in range(len(start_times))}
        self.cut_distance = 5
        self.tree = tree
        if self.tree:
            self.model_name += '_tree'
        self.ntypes = ntypes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.graph_dropout = graph_dropout
        self.dropout = dropout
        self.save_graph = False
        self.residual = residual
        self.graph_norm = graph_norm
        self.updater = updater
        self.lambda_weight = lambda_weight
        if self.lambda_weight != 0.5:
            self.model_name += '_lw{}'.format(self.lambda_weight)
        if self.residual:
            self.model_name += '_res'
        if self.graph_norm:
            self.model_name += '_{}'.format(self.graph_norm)
        if not mixed:
            self.model_name += '_unmixed'
        if self.updater != 'normal':
            self.model_name += '_' + self.updater
        # self.time_le = None
        if learned_time:
            self.model_name += '_lt'
        if time_layers != 1:
            self.model_name += '_t{}'.format(time_layers)
        if hidden_dim != 128:
            self.model_name += '_h{}'.format(hidden_dim)
        if tau != 0.2:
            self.model_name += '_tau{}'.format(tau)
        if os.path.exists(time_embs_path):
            time_embs = joblib.load(time_embs_path)  # {phase: [time_length, h]}
            for phase in time_embs:
                time_embs[phase] = time_embs[phase].to(self.device)
        else:
            time_embs = None
        self.time_embs = time_embs
        self.encoder_type = encoder_type
        if encoder_type == 'DGINP':
            special_edges = {'DGINP': ['cites']}
        elif encoder_type == 'CCGIN':
            special_edges = {'CCGIN': ['cites']}
        elif encoder_type == 'GATO':
            special_edges = {'GATO': {
                'cites': {
                    'residual': self.residual,
                    'dropout': self.graph_dropout,
                    'graph_norm': self.graph_norm
                },
                'is cited by': {
                    'residual': self.residual,
                    'dropout': self.graph_dropout,
                    'graph_norm': self.graph_norm
                }}}
        elif encoder_type in {'CompGAT', 'CompGATM'}:
            special_edges = {'CompGAT': {
                'cites': {
                    'residual': self.residual,
                    'dropout': self.graph_dropout,
                    'graph_norm': self.graph_norm
                },
                'is cited by': {
                    'residual': self.residual,
                    'dropout': self.graph_dropout,
                    'graph_norm': self.graph_norm
                }}}
        elif encoder_type in {'CCompGAT', 'CCompGATS', 'CCompGATSM'}:
            special_edges = {encoder_type: {'cites': {'indicator_edge_attr': 'r_sim',
                                                      'norm_node_attr': 'refs',
                                                      'residual': self.residual,
                                                      'dropout': self.graph_dropout,
                                                      'graph_norm': self.graph_norm,
                                                      'lambda_weight': self.lambda_weight,
                                                      'reverse': True},
                                            'is cited by': {'indicator_edge_attr': 'r_sim',
                                                            'norm_node_attr': 'citations',
                                                            'residual': self.residual,
                                                            'dropout': self.graph_dropout,
                                                            'graph_norm': self.graph_norm,
                                                            'lambda_weight': self.lambda_weight
                                                            }}}
        elif encoder_type in {'CCompGATN', 'CCompGATSN', 'CCompGATSMN'}:
            cur_encoder_type = encoder_type[:-1]
            special_edges = {cur_encoder_type: {'cites': {'indicator_edge_attr': 'r_sim',
                                                          'norm_node_attr': 'refs',
                                                          'residual': self.residual,
                                                          'dropout': self.graph_dropout,
                                                          'graph_norm': self.graph_norm,
                                                          'lambda_weight': self.lambda_weight,
                                                          'reverse': True},
                                                'is cited by': {'indicator_edge_attr': 'r_sim',
                                                                'residual': self.residual,
                                                                'dropout': self.graph_dropout,
                                                                'graph_norm': self.graph_norm,
                                                                'lambda_weight': self.lambda_weight,
                                                                'norm_node_attr': 'citations'}},
                             'NGIN': [etype for etype in etypes if etype not in {'cites', 'is cited by'}]}
            print(special_edges)

        self.node_attr_types = ['is_cite', 'is_ref', 'is_target']
        self.disen_loss = False
        self.shared = False
        self.sum_method = 'normal'
        self.get_model_type(model_type)

        self.encoder = DHGCNEncoder(embed_dim, hidden_dim, time_length, ntypes, etypes, start_times, n_layers,
                                    encoder_type, special_edges, self.node_attr_types, self.time_embs,
                                    shared=self.shared,
                                    time_layers=time_layers, time_pooling=time_pooling,
                                    snapshot_mode=snapshot_mode, learned_time=learned_time, mixed=mixed,
                                    updater=updater,
                                    device=self.device)

        if time_pooling == 'long_short':
            self.hidden_dim = self.hidden_dim * 2  # long short concat
        self.topic_module = ECLTopicModule(self.hidden_dim)
        self.pop_module = CitationPopModule(self.hidden_dim)
        self.contri_module = SimpleContriModule(self.hidden_dim)
        self.ablation_channel = sorted(ablation_channel)
        if len(self.ablation_channel) > 0:
            self.model_name += '_wo'
            for channel in self.ablation_channel:
                self.model_name += '+{}'.format(channel)
        self.disen_module = SimpleDisenModule(self.encoder, self.topic_module, self.pop_module, self.contri_module,
                                              self.device, mode='ecl', disen_loss=self.disen_loss,
                                              sum_method=self.sum_method)

    def get_model_type(self, model_type):
        if model_type:
            if 'shared' in model_type:
                self.shared = True
                self.model_name += '_shared'
                model_type = model_type.replace('shared_', '')
            if model_type in {'orthog', 'simple_orthog', 'dcor', 'mmd_rbf', 'mmd_ms'}:
                self.model_name += '_' + model_type
                self.disen_loss = model_type
            if 'exp' in model_type:
                self.disen_loss = 'orthog'
                self.model_name += '_' + self.disen_loss + '_' + model_type
                self.sum_method = 'exp'

    def get_batch_input(self, batch, graph):
        '''
        :param batch:
        :param graph:
        :return:
        '''
        x = batch.x
        values = batch.values
        lengths = batch.lengths
        masks = batch.masks
        ids = batch.ids
        times = batch.times
        blocks = batch.subgraphs
        if type(x) == torch.Tensor:
            x = x.to(self.device)
        else:
            # x = [content.to(self.device) for content in x]
            for i in range(len(x)):
                if type(x[i]) == torch.Tensor:
                    x[i] = x[i].to(self.device)
        # print('io-time:', time.time()-start_time)
        values = values.to(self.device)
        lengths = lengths.to(self.device)
        masks = masks.to(self.device)
        if blocks:
            if type(blocks) == list:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(self.device)
            elif type(blocks) == tuple:
                all_valid_num_nodes = blocks[-1].to(self.device)
                # print(blocks[0])
                blocks = list(blocks)
                blocks[0] = list(blocks[0])
                for i in range(len(blocks[0])):
                    blocks[0][i] = blocks[0][i].to(self.device)
                    blocks[1][i]['valid_num_nodes'] = all_valid_num_nodes[i]
            else:
                blocks = blocks.to(self.device)
        if graph and type(ids) != torch.Tensor:
            ids = torch.tensor(list(map(lambda x: self.node_trans['paper'][x], ids)))
        # print('to input:', time.time() - start_time)
        # print('>>>>ccgl x:', x.device)
        return x, values, lengths, masks, ids, times, blocks

    def load_graphs(self, graphs):
        for pub_time in graphs['all_graphs']:
            cur_graph = graphs['all_graphs'][pub_time]
            cur_graph.nodes['paper'].data['citations'] = cur_graph.in_degrees(etype='cites').to(torch.float32)
            cur_graph.nodes['paper'].data['refs'] = cur_graph.in_degrees(etype='is cited by').to(torch.float32)
            cur_graph.update_all(fn.copy_u('citations', 'm'), fn.sum('m', 'accum_citations'), etype='is writen by')

        graphs['all_trans_index'] = json.load(open(graphs['graphs_path'] + '_trans.json', 'r'))
        graphs['all_masks'] = joblib.load(graphs['graphs_path'] + '_masks.job')
        # ego_graphs
        graphs['all_ego_graphs'] = {}
        for pub_time in tqdm(graphs['all_graphs']):
            # graphs['all_ego_graphs'][pub_time] = joblib.load(
            #     graphs['graphs_path'] + '_{}.job'.format(pub_time))
            with jl.open(graphs['graphs_path'] + '_{}.jsonl'.format(pub_time)) as fr:
                graphs['all_ego_graphs'][pub_time] = [line for line in fr]

        for phase in graphs['data']:
            time_span = range(self.start_times[phase], self.start_times[phase] + self.time_length)
            cur_graphs = [graphs['all_graphs'][pub_time] for pub_time in time_span]
            index_lists = [graphs['all_ego_graphs'][pub_time] for pub_time in time_span]
            # cur_masks = torch.from_numpy(
            #     graphs['all_masks'][:, np.array(time_span) - self.start_times['train']].astype(np.long)).to(self.device)
            cur_masks = torch.from_numpy(
                graphs['all_masks'][:, np.array(time_span) - self.start_times['train']].astype(np.long))
            print(cur_graphs, cur_masks[:5])
            # print(index_lists[0][:5])
            graphs['data'][phase] = [cur_graphs, index_lists, graphs['all_trans_index'], cur_masks, None]

        return graphs

    def deal_graphs(self, graphs, data_path=None, path=None, log_path=None):
        tqdm_out = None
        if log_path:
            logging.basicConfig(level=logging.INFO,
                                filename=log_path,
                                filemode='w+',
                                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                force=True)
            logger = logging.getLogger()
            # LOG = logging.getLogger(__name__)
            tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.save_graph:
            path = path.replace('DDHGCN', 'DDHGCN_saved')

        phases = list(graphs['data'].keys())
        for pub_time in graphs['all_graphs']:
            cur_graph = graphs['all_graphs'][pub_time]
            cur_graph.nodes['paper'].data['citations'] = cur_graph.in_degrees(etype='cites').to(torch.float32)
            cur_graph.nodes['paper'].data['refs'] = cur_graph.in_degrees(etype='is cited by').to(torch.float32)
            cur_graph.update_all(fn.copy_u('citations', 'm'), fn.sum('m', 'accum_citations'), etype='is writen by')
        dealt_graphs = {'data': {}, 'time': graphs['time'], 'node_trans': graphs['node_trans'],
                        'all_graphs': None}
        split_data = torch.load(data_path)
        all_selected_papers = [graphs['node_trans']['paper'][paper] for paper in split_data['test'][0]]
        all_selected_times = split_data['test'][-1]
        # paper_time_dict = dict(zip(all_selected_papers, all_selected_times))
        logging.info('all count: {}'.format(str(len(all_selected_papers))))
        # all_trans_index = dict(zip(all_selected_papers, range(len(all_selected_papers))))
        all_trans_index = dict(zip(split_data['test'][0], range(len(all_selected_papers))))
        pub_times = sorted(list(graphs['all_graphs'].keys()))
        # del graphs['all_graphs']
        del graphs['data']

        # saved frozen time embs
        time_embs = {}
        for phase in phases:
            cur_list = []
            for pub_time in range(self.start_times[phase], self.start_times[phase] + self.time_length):
                cur_graph = graphs['all_graphs'][pub_time]
                time_index = torch.where(cur_graph.nodes['time'].data[dgl.NID] == pub_time)
                cur_list.append(cur_graph.nodes['time'].data['h'][time_index])
            time_embs[phase] = torch.cat(cur_list, dim=0)
            print(time_embs[phase].shape)
        joblib.dump(time_embs, path + '_' + str(graphs['graph_type']) + '.job')
        #
        # all_dealt_graphs = set([os.path.split(path)[1] + '_' + str(pub_time) + '.job' for pub_time in pub_times])
        all_dealt_graphs = set([os.path.split(path)[1] + '_' + str(pub_time) + '.jsonl' for pub_time in pub_times])
        print(os.path.split(path))
        all_existed_files = set(os.listdir(os.path.split(path)[0]))
        print(all_dealt_graphs - all_existed_files)
        if all_dealt_graphs - all_existed_files:
            final_graph = graphs['all_graphs'][pub_times[-1]]
            final_graph = add_new_elements(dgl.node_type_subgraph(final_graph, ntypes=['paper']))
            # print(final_graph.ndata.keys())
            final_graph.ndata.pop('h')
            bigraph = dgl.add_reverse_edges(final_graph[('paper', 'is cited by', 'paper')])
            simple_sampler = SampleAllNeighbors(bigraph, 2, return_graph=True, origin_graph=final_graph)
            all_subgraphs = [simple_sampler.get_single_data(paper) for paper in tqdm(all_selected_papers,
                                                                                     total=len(all_selected_papers),
                                                                                     file=tqdm_out, mininterval=30
                                                                                     )]
            del final_graph, bigraph

            all_masks = []
            for paper_time in all_selected_times:
                unvalid_len = max(0, paper_time - self.start_times['train'])
                all_masks.append([0] * unvalid_len + [1] * (len(pub_times) - unvalid_len))
            all_masks = np.array(all_masks)
            # print(all_masks[:5])

            for pub_time in pub_times:
                cur_graph = graphs['all_graphs'][pub_time]
                # print(pub_time)
                for ntype in cur_graph.ntypes:
                    cur_graph.nodes[ntype].data.pop('h')
                index_dict = get_snapshot_types_dict({pub_time: cur_graph})
                cur_sampler = DealtKhopSubgraph(cur_graph, index_dict[pub_time], pub_time, return_graph=self.save_graph)
                input_data = zip(all_selected_times, all_subgraphs)
                cur_subgraphs = [cur_sampler.get_filter_subgraph(inputs) for inputs in tqdm(input_data,
                                                                                            total=len(
                                                                                                all_selected_papers),
                                                                                            file=tqdm_out,
                                                                                            mininterval=30
                                                                                            )]
                if path:
                    # joblib.dump(cur_subgraphs, path + '_' + str(pub_time) + '.job')
                    with jl.open(os.path.split(path)[0] + '/DDHGCN_graphs_{}.jsonl'.format(pub_time), 'w') as fw:
                        for line in cur_subgraphs:
                            fw.write(line)
            if path:
                json.dump(all_trans_index, open(path + '_trans.json', 'w+'))
                joblib.dump(all_masks, path + '_masks.job')

        del graphs['all_graphs']

        joblib.dump([], path)
        return dealt_graphs

    def get_snapshot_types(self, times):
        # [batch]
        # snapshot_times = torch.tensor(list(range(self.start_times[self.phase],
        #                                          self.start_times[self.phase] + self.time_length)),
        #                               dtype=torch.long).unsqueeze(dim=-1)
        snapshot_times = torch.arange(self.start_times[self.phase], self.start_times[self.phase] + self.time_length,
                                      dtype=torch.long, device=times.device).unsqueeze(dim=-1)
        # print(times)
        # sample_times = torch.tensor(times, dtype=torch.long).tile((self.time_length, 1))
        sample_times = times.tile((self.time_length, 1)).detach()
        # print(sample_times.shape)
        # print(snapshot_times.shape)
        pub_distances = snapshot_times - sample_times
        # print(sample_times.shape, snapshot_times.shape, pub_distances.shape)
        # print(sample_times.device, snapshot_times.device, pub_distances.device)

        # [T, batch]
        snapshot_types = torch.ones_like(pub_distances).to(self.device)
        new_papers = torch.where(pub_distances < self.cut_distance)
        old_papers = torch.where(pub_distances > 2 * self.cut_distance - 1)
        snapshot_types[new_papers] = 0
        snapshot_types[old_papers] = 2
        # print(snapshot_types)
        return snapshot_types

    def get_index_dicts(self, graphs):
        index_dicts = {}
        for t in range(self.time_length):
            cur_graph = graphs[t]
            selected_indicator = (torch.stack([cur_graph.nodes['paper'].data[node_attr_type]
                                               for node_attr_type in self.node_attr_types], dim=-1).sum(dim=-1) > 0)

            index = []
            mixer_index = []
            num_nodes = cur_graph.batch_num_nodes('paper')
            last_start = 0
            valid_num_nodes = []
            for i in range(len(num_nodes)):
                # index.extend([i] * num_nodes[i])
                cur_end = last_start + num_nodes[i].item()
                valid_num = selected_indicator[last_start: cur_end].sum()
                valid_num_nodes.append(valid_num)
                index.extend([i] * valid_num.item())
                mixer_index.extend([i] * num_nodes[i])
                last_start = cur_end
            valid_num_nodes = torch.tensor(valid_num_nodes, device=cur_graph.device)
            index = torch.tensor(index, dtype=torch.long).to(cur_graph.device)
            index_dicts[t] = {'selected_indicator': selected_indicator, 'index': index,
                              'valid_num_nodes': valid_num_nodes, 'mixer_index': mixer_index}
        return index_dicts

    def get_inputs(self, content, lengths, masks, ids, graph, times, phase, time_masks, **kwargs):
        # masks [batch, T]
        # start_time = time.time()
        graphs = graph
        # print(graphs)
        for t in range(self.time_length):
            cur_graph = graphs[t]
            if self.shared:
                for ntype in self.ntypes:
                    cur_graph.nodes[ntype].data['snapshot_idx'] = torch.tensor([t] * cur_graph.num_nodes(ntype=ntype),
                                                                               dtype=torch.long).to(self.device)

        cur_snapshot_types = self.get_snapshot_types(times)
        time_masks = ~time_masks.bool()
        index_dicts = self.get_index_dicts(graphs)
        return {'graphs': graphs,
                'cur_snapshot_types': cur_snapshot_types,
                'time_masks': time_masks,
                'index_dicts': index_dicts,
                'phase': phase}

    def predict(self, content, lengths, masks, ids, graph, times, **kwargs):
        # print(len(content))
        # print([temp.shape for temp in content])
        text, time_masks, _ = content
        graph_inputs = self.get_inputs(content, lengths, masks, ids, graph, times, self.phase, time_masks, **kwargs)
        out = self.disen_module.predict(graph_inputs, self.phase)
        return out, None, None

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        text, time_masks, pop_indicators = content
        # print(time_masks.shape, pop_indicators)
        graph_inputs = self.get_inputs(text, lengths, masks, ids, graph, times, self.phase, time_masks, **kwargs)
        # start_time = time.time()
        out, loss = self.disen_module(graph_inputs, self.phase, pop_indicators)
        return out, None, loss


class DDHGCNS(DDHGCN):
    def __init__(self, num_classes, embed_dim, hidden_dim, time_length, ntypes, etypes, start_times,
                 time_embs_path=None, model_type=None, snapshot_mode='mix', learned_time=False, time_pooling='sum',
                 topic_module='align', pop_module='citation', n_topics=10,
                 encoder_type='DGINP', n_layers=3, time_layers=1, tree=False, dropout=0.3, residual=True,
                 graph_dropout=0, ablation_channel=(), graph_norm=False, mixed=True, updater='normal',
                 adaptive_lr=False, loss_weights=None, lambda_weight=0.5,
                 use_constraint=False,
                 tau=0.2,
                 **kwargs):
        super(DDHGCNS, self).__init__(num_classes, embed_dim, hidden_dim, time_length, ntypes, etypes, start_times,
                                      time_embs_path, None, snapshot_mode, learned_time, time_pooling,
                                      encoder_type, n_layers, time_layers, tree, dropout, residual, graph_dropout,
                                      ablation_channel, graph_norm, mixed, updater, lambda_weight, tau,
                                      **kwargs)
        self.model_name = self.model_name.replace('DDHGCN', 'DDHGCNS')
        self.disen_loss = False
        self.shared = False
        self.get_model_type(model_type)
        self.topic_module_type = topic_module
        self.pop_module_type = pop_module
        self.adaptive_lr = adaptive_lr
        self.loss_weights = loss_weights
        self.use_constraint = use_constraint
        if self.use_constraint in {'simple', 'aug'}:
            self.model_name += '_uc{}'.format(self.use_constraint[0])
        if self.adaptive_lr:
            self.model_name += '_al'
        if dropout:
            self.model_name += '_dn'
        if graph_dropout:
            self.model_name = self.model_name.replace('_dn', '_sdn')
        self.n_topics = n_topics
        # if topic_module not in {'align', 'cl', 'ecl'}:
        if topic_module not in {'align', 'ecl'}:
            self.model_name += '_' + topic_module
            if n_topics != 10:
                self.model_name += '_nt{}'.format(n_topics)
        if pop_module not in {'citation'}:
            self.model_name += '_' + pop_module
        self.get_disen_modules(topic_module, pop_module)
        for channel in self.loss_weights:
            if self.loss_weights[channel] != 0.5:
                self.model_name += '_{}w{}'.format(channel[0], self.loss_weights[channel])

        model_config = kwargs.get('config', None)
        if model_config:
            if model_config['lr'] != 1e-4:
                self.model_name += '_lr{:.0e}'.format(model_config['lr'])
            if model_config['optimizer'] != 'ADAM':
                self.model_name += '_{}'.format(model_config['optimizer'])
            if model_config['graph_type'] != 'sbert':
                self.model_name += '_{}'.format(model_config['graph_type'])
            if model_config['etc']:
                self.model_name += '_{}'.format(model_config['etc'])

    @staticmethod
    def get_group_parameters(model, lr=0):
        if model.adaptive_lr:
            te_params = []
            other_params = []
            for n, p in model.named_parameters():
                if 'time_encoders' in n:
                    te_params.append(p)
                else:
                    other_params.append(p)
            print(len(te_params), len(other_params))
            params = [
                {
                    'params': te_params, 'lr': lr / 10
                },
                {
                    'params': other_params, 'lr': lr
                }
            ]
        else:
            params = model.parameters()
        return params

    def get_optimizer(self, model, lr, optimizer, weight_decay=1e-3, autocast=False):
        eps = 1e-4 if autocast else 1e-8
        optimizer_type = self.get_selected_optimizer_type(optimizer)
        if 'cluster' in self.topic_module_type:
            print('!!!>>>cluster different optimizer')
            print(lr)
            gmm_params = [p for n, p in model.named_parameters() if 'vae_encoder.gmm' in n]
            gmm_params_ids = set(map(id, gmm_params))
            discriminator_params = [p for n, p in model.named_parameters() if 'vae_encoder.discriminator' in n]
            discriminator_params_ids = set(map(id, discriminator_params))
            vae_all_params = [p for n, p in model.named_parameters() if 'vae_encoder' in n]
            vae_params = list(filter(lambda p: id(p) not in (gmm_params_ids | discriminator_params_ids),
                                     vae_all_params))
            vae_params_ids = set(map(id, vae_params))
            print(discriminator_params_ids)
            other_params = list(
                filter(lambda p: id(p) not in (gmm_params_ids | vae_params_ids | discriminator_params_ids),
                       model.parameters()))
            other_params_ids = set(map(id, other_params))
            optimizer = optimizer_type([{'params': other_params, 'lr': lr}], weight_decay=weight_decay)
            optimizer_gmm = optimizer_type([{'params': gmm_params, 'lr': lr}],
                                           weight_decay=weight_decay, eps=eps)
            optimizer_vae = optimizer_type([{'params': vae_params, 'lr': lr}],
                                           weight_decay=weight_decay, eps=eps)
            optimizer_dis = optimizer_type([{'params': discriminator_params, 'lr': lr, 'betas': (0.5, 0.9)}],
                                           weight_decay=weight_decay, eps=eps)
            all_params_ids = set(map(id, model.parameters()))
            for name, param in model.named_parameters():
                if id(param) not in (gmm_params_ids | discriminator_params_ids | vae_params_ids | other_params_ids):
                    print(name)
            assert len(
                all_params_ids - (gmm_params_ids | discriminator_params_ids | vae_params_ids | other_params_ids)) == 0
            # self.optimizers = [self.optimizer, self.optimizer_gmm, self.optimizer_dis, self.optimizer_vae]
            # return self.optimizer, self.optimizer_gmm, self.optimizer_vae, self.optimizer_dis
            return [optimizer, optimizer_gmm, optimizer_dis, optimizer_vae]
        elif 'vae' in self.topic_module_type:
            print('!!!>>>vae different optimizer')
            # print(lr)
            discriminator_params = [p for n, p in model.named_parameters() if 'vae_encoder.discriminator' in n]
            discriminator_params_ids = set(map(id, discriminator_params))
            vae_all_params = [p for n, p in model.named_parameters() if 'vae_encoder' in n]
            vae_params = list(filter(lambda p: id(p) not in discriminator_params_ids,
                                     vae_all_params))
            vae_params_ids = set(map(id, vae_params))
            # print(vae_params_ids, vae_params_ids)
            other_params = list(filter(lambda p: id(p) not in (vae_params_ids | discriminator_params_ids),
                                       model.parameters()))
            other_params_ids = set(map(id, other_params))
            # self.optimizer = optimizer_type([{'params': other_params, 'lr': lr}], weight_decay=weight_decay)
            # self.optimizer_vae = optimizer_type([{'params': vae_params, 'lr': lr * 5}],
            #                                       weight_decay=0)
            optimizer = optimizer_type(other_params, lr=lr, weight_decay=weight_decay, eps=eps)
            optimizer_vae = optimizer_type(vae_params, lr=lr * 5, weight_decay=weight_decay, eps=eps)
            # self.optimizer_gmm = torch.optim.SGD([{'params': gmm_params, 'lr': lr}],
            #                                       weight_decay=0)
            # self.optimizer_vae = torch.optim.SGD([{'params': vae_params, 'lr': lr}],
            #                                       weight_decay=0)
            optimizer_dis = optimizer_type(discriminator_params, lr=lr * 5, betas=(0.5, 0.9),
                                           weight_decay=weight_decay, eps=eps)
            all_params_ids = set(map(id, model.parameters()))
            assert len(all_params_ids - (vae_params_ids | discriminator_params_ids | other_params_ids)) == 0
            # self.optimizers = [self.optimizer, self.optimizer_dis, self.optimizer_vae]
            # return self.optimizer, self.optimizer_vae, self.optimizer_dis
            return [optimizer, optimizer_dis, optimizer_vae]
        else:
            return super().get_optimizer(model, lr, optimizer, weight_decay, autocast)

    def get_disen_modules(self, topic_module, pop_module, disen_module='normal'):
        if topic_module == 'cl':
            self.topic_module = CLTopicModule(self.hidden_dim, dropout=self.dropout)
        elif topic_module == 'ocl':
            print('>>>here we get the ocl')
            self.topic_module = OCLTopicModule(self.hidden_dim, dropout=self.dropout)
        elif topic_module == 'scl':
            print('>>>here we get the scl')
            self.topic_module = SCLTopicModule(self.hidden_dim, dropout=self.dropout)
        elif topic_module == 'ecl':
            self.topic_module = ECLTopicModule(self.hidden_dim, dropout=self.dropout)

        if pop_module == 'accum':
            self.pop_module = AccumCitationPopModule(self.hidden_dim, dropout=self.dropout)
        else:
            self.pop_module = CitationPopModule(self.hidden_dim, dropout=self.dropout)
        self.contri_module = SimpleContriModule(self.hidden_dim, dropout=self.dropout)

        self.disen_module = SimpleDisenModule(self.encoder, self.topic_module, self.pop_module, self.contri_module,
                                              self.device, mode=disen_module, disen_loss=self.disen_loss,
                                              sum_method=self.sum_method, ablation_channel=self.ablation_channel,
                                              loss_weights=self.loss_weights, use_constraint=self.use_constraint)

    def load_graphs(self, graphs):
        for pub_time in graphs['all_graphs']:
            cur_graph = graphs['all_graphs'][pub_time]
            cur_graph.nodes['paper'].data['citations'] = cur_graph.in_degrees(etype='cites').to(torch.float32)
            cur_graph.nodes['paper'].data['refs'] = cur_graph.in_degrees(etype='is cited by').to(torch.float32)
            if self.pop_module_type in {'aa'}:
                cur_graph.update_all(fn.copy_u('citations', 'm'), fn.sum('m', 'accum_citations'), etype='is writen by')

        graphs['all_trans_index'] = json.load(open(graphs['graphs_path'] + '_trans.json', 'r'))
        graphs['all_masks'] = joblib.load(graphs['graphs_path'] + '_masks.job')
        # ego_graphs
        graphs['all_ego_graphs'] = {}
        for pub_time in tqdm(graphs['all_graphs']):
            # graphs['all_ego_graphs'][pub_time] = joblib.load(
            #     graphs['graphs_path'] + '_{}.job'.format(pub_time))
            with jl.open(graphs['graphs_path'] + '_{}.jsonl'.format(pub_time)) as fr:
                graphs['all_ego_graphs'][pub_time] = [line for line in fr]

        for phase in graphs['data']:
            graphs['data'][phase] = self.get_phase_graph_data(phase, graphs)

        return graphs

    def get_phase_graph_data(self, phase, graphs, filter_nodes=True):
        selected_papers = graphs['selected_papers'][phase]
        trans_index = dict(zip(selected_papers, range(len(selected_papers))))
        cur_indexes = [graphs['all_trans_index'][paper] for paper in selected_papers]
        graphs['selected_papers'][phase] = selected_papers
        time_span = range(self.start_times[phase], self.start_times[phase] + self.time_length)
        print(len(trans_index))
        print(len(cur_indexes))
        # cur_masks = torch.from_numpy(
        #     graphs['all_masks'][:, np.array(time_span) - self.start_times['train']].astype(np.long))[cur_indexes, :].share_memory_()
        cur_masks = torch.from_numpy(
            graphs['all_masks'][:, np.array(time_span) - self.start_times['train']].astype(np.long))[cur_indexes, :]
        print(cur_masks.shape)
        # cur_graphs = [graphs['all_graphs'][pub_time] for pub_time in time_span]
        # index_lists = [graphs['all_ego_graphs'][pub_time] for pub_time in time_span]
        if filter_nodes:
            if phase == 'test':
                cur_graphs = []
                index_lists = []
                for pub_time in time_span:
                    cur_index_lists = []
                    cur_graphs.append(graphs['all_graphs'][pub_time])
                    for paper_index in tqdm(cur_indexes):
                        cur_info = graphs['all_ego_graphs'][pub_time][paper_index]
                        cur_index_lists.append(cur_info)
                    index_lists.append(cur_index_lists)
            else:
                cur_graphs = []
                index_lists = []
                for pub_time in time_span:
                    cur_index_lists = []
                    cur_node_index = {ntype: IndexDict() for ntype in self.ntypes}
                    for paper_index in tqdm(cur_indexes):
                        cur_info = copy.deepcopy(graphs['all_ego_graphs'][pub_time][paper_index])
                        new_nodes = {}
                        for ntype in cur_info['nodes']:
                            new_nodes[ntype] = torch.tensor(
                                [cur_node_index[ntype][node] for node in cur_info['nodes'][ntype]],
                                dtype=torch.long)
                        # print(new_nodes)
                        for indicator in cur_info['indicators']:
                            cur_info['indicators'][indicator] = torch.tensor(cur_info['indicators'][indicator],
                                                                             dtype=torch.long)
                        cur_info['nodes'] = new_nodes
                        cur_index_lists.append(cur_info)
                    subgraph_nodes = {ntype: list(cur_node_index[ntype].keys()) for ntype in self.ntypes}
                    index_lists.append(cur_index_lists)
                    print('=' * 59)
                    # print('index len:', len(cur_index_lists))
                    print(graphs['all_graphs'][pub_time])
                    cur_graph = dgl.node_subgraph(graphs['all_graphs'][pub_time], subgraph_nodes, store_ids=False)
                    print(cur_graph)
                    print('=' * 59)
                    # cur_graph = get_shared_graph(cur_graph, '{}_{}'.format(phase, pub_time))
                    cur_graphs.append(cur_graph)
        else:
            cur_graphs = [graphs['all_graphs'][pub_time] for pub_time in time_span]
            index_lists = None

        if phase == 'train':
            pop_indicators = self.get_pop_indicators(graphs)
        else:
            pop_indicators = None
        return [cur_graphs, index_lists, trans_index, cur_masks, pop_indicators]

    def get_pop_indicators(self, graphs):
        data_path = graphs['data_path']
        paper_ids = torch.load(data_path + 'split_data')['train'][0]
        citation_accum = json.load(open(data_path + 'sample_citation_accum.json'))
        time_index = list(range(graphs['time']['test'] - 3 * self.time_length + 1,
                                graphs['time']['test'] + self.time_length + 1))
        # print(len(time_index))
        index_trans = dict(zip(time_index, range(len(time_index))))
        final_citation = []
        all_citations = []
        train_time = graphs['time']['train']
        for paper in paper_ids:
            final_citation.append(citation_accum[paper][index_trans[train_time]])
            cur_list = []
            cur_citations = citation_accum[paper]
            for cur_time in range(train_time - self.time_length + 1, train_time + 1):
                new_value = cur_citations[index_trans[cur_time]]
                old_value = cur_citations[index_trans[cur_time - 1]]
                if new_value == -1:
                    cur_list.append(-1)
                elif old_value == -1:
                    cur_list.append(new_value)
                else:
                    cur_list.append(new_value - old_value)
            all_citations.append(cur_list)
        all_citations = np.array(all_citations)

        del citation_accum

        df = pd.DataFrame(data={'final': final_citation})
        for i in range(self.time_length):
            df[str(i)] = all_citations[:, i]
        mask = np.where(all_citations == -1, 0, 1)
        avg_values = (all_citations * mask).sum(1) / mask.sum(1)
        df['avg'] = avg_values

        df['final_box'] = box_cut(np.log(df['final'] + 1))
        df['avg_box'] = box_cut(np.log(df['avg'] + 1))
        final_boxes = torch.from_numpy(df['final_box'].to_numpy())
        avg_boxes = torch.from_numpy(df['avg_box'].to_numpy())

        norm_values = []
        for i in range(self.time_length):
            norm_values.append(mm_norm(df[str(i)].to_numpy(), log=True))
        norm_values = np.array(norm_values).T
        print('norm_values', norm_values.shape)

        seq_masks = []
        all_norm_seqs = []
        valid_indexes = norm_values > -1
        for i in range(len(paper_ids)):
            # print(norm_values[i], valid_indexes[i])
            cur_list = norm_values[i][valid_indexes[i]].tolist()
            valid_len = len(cur_list)
            cur_list += [-1] * (self.time_length - valid_len)
            seq_mask = [1] * valid_len + [0] * (self.time_length - valid_len)
            all_norm_seqs.append(cur_list)
            seq_masks.append(seq_mask)
        all_norm_seqs = torch.tensor(all_norm_seqs, dtype=torch.float32)
        seq_masks = torch.tensor(seq_masks, dtype=torch.bool)
        pop_index_trans = dict(zip(paper_ids, range(len(paper_ids))))

        # return {'final': final_boxes.share_memory_(), 'avg': avg_boxes.share_memory_(),
        #         'norm_seqs': all_norm_seqs.share_memory_(), 'seq_masks': seq_masks.share_memory_(),
        #         'pop_index_trans': pop_index_trans}
        return {'final': final_boxes, 'avg': avg_boxes,
                'norm_seqs': all_norm_seqs, 'seq_masks': seq_masks,
                'pop_index_trans': pop_index_trans}

    def show(self, content, lengths, masks, ids, graph, times, **kwargs):
        text, time_masks, _ = content
        graph_inputs = self.get_inputs(content, lengths, masks, ids, graph, times, self.phase, time_masks, **kwargs)
        out = self.disen_module.show(graph_inputs, self.phase)
        return out

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        text = content[0]
        time_masks = content[1]
        pop_indicators = content[2:]
        # print(time_masks.shape, pop_indicators)
        graph_inputs = self.get_inputs(text, lengths, masks, ids, graph, times, self.phase, time_masks, **kwargs)
        # start_time = time.time()
        out, loss = self.disen_module(graph_inputs, self.phase, pop_indicators)
        # print('disen time:', time.time() - start_time)
        # print(loss)
        # print(out)
        # self.other_loss = self.other_loss + loss
        return out, None, loss


class DDHGCNSCL(DDHGCNS):
    def __init__(self, num_classes, embed_dim, hidden_dim, time_length, ntypes, etypes, start_times,
                 time_embs_path=None, model_type=None, snapshot_mode='mix', learned_time=False, time_pooling='sum',
                 topic_module='cl', pop_module='citation', n_topics=10, aug_rate=0.1, focus=False,
                 encoder_type='DGINP', n_layers=3, time_layers=1, tree=False, dropout=0.3, residual=True,
                 graph_dropout=0, ablation_channel=(), graph_norm=False, mixed=True, updater='normal',
                 adaptive_lr=False, loss_weights=None, lambda_weight=0.5, use_constraint=False,
                 tau=0.2,
                 **kwargs):
        super(DDHGCNSCL, self).__init__(num_classes, embed_dim, hidden_dim, time_length, ntypes, etypes, start_times,
                                        time_embs_path, None, snapshot_mode, learned_time, time_pooling,
                                        topic_module, pop_module, n_topics,
                                        encoder_type, n_layers, time_layers, tree, dropout, residual, graph_dropout,
                                        ablation_channel, graph_norm, mixed, updater, adaptive_lr, loss_weights,
                                        lambda_weight, use_constraint, tau, **kwargs)
        self.model_name = self.model_name.replace('DDHGCNS', 'DDHGCNSCL')
        self.disen_loss = False
        self.shared = False
        self.get_model_type(model_type)
        if focus:
            self.model_name += '_focus'
        self.aug_rate = aug_rate
        self.focus = focus
        if aug_rate != 0.1:
            self.model_name += '_c{}'.format(aug_rate)
        disen_module = 'cl'
        if topic_module != 'cl':
            if pop_module == 'pcl':
                disen_module = 'pcl'
            elif topic_module == 'ocl':
                disen_module = 'ocl'
            elif topic_module == 'scl':
                disen_module = 'scl'
            else:
                raise Exception
        self.disen_module_type = disen_module
        self.get_disen_modules(topic_module, pop_module, disen_module)


class SingleSubgraph:
    def __init__(self, graph):
        self.graph = graph

    def get_subgraph(self, info_dict):
        cur_subgraph = dgl.node_subgraph(self.graph, nodes=info_dict['nodes'])
        for indicator in info_dict['indicators']:
            cur_subgraph.nodes['paper'].data[indicator] = torch.tensor(info_dict['indicators'][indicator],
                                                                       dtype=torch.long)
        return cur_subgraph


class DDHGCNSCLT(DDHGCNSCL):
    def __init__(self, *args, **kwargs):
        super(DDHGCNSCLT, self).__init__(*args, **kwargs)
        # self.model_name = self.model_name.replace('DDHGCNSCL', 'DDHGCNSCLT')
        self.model_name = self.model_name.replace('DDHGCNSCL', 'DPPDCC')
        self.disen_module = SimpleDisenModuleT(self.encoder, self.topic_module, self.pop_module, self.contri_module,
                                               self.device, mode=self.disen_module_type, disen_loss=self.disen_loss,
                                               ablation_channel=self.ablation_channel,
                                               sum_method=self.sum_method, loss_weights=self.loss_weights,
                                               use_constraint=self.use_constraint)

    def get_phase_graph_data(self, phase, graphs, filter_nodes=True):
        cur_graphs, index_lists, trans_index, cur_masks, pop_indicators = (
            super().get_phase_graph_data(phase, graphs, filter_nodes=filter_nodes))
        return [cur_graphs, index_lists, trans_index, cur_masks, pop_indicators]

    def get_train_info(self, data_inputs, record_path):
        cur_graphs, index_lists, trans_index, cur_masks, pop_indicators = data_inputs
        all_list = []
        for index in tqdm(range(len(trans_index))):
            for t in range(self.time_length):
                cur_dict = {'index': index, 't': t}
                cur_subgraph = index_lists[t][index]
                for ntype in cur_subgraph.ntypes:
                    cur_dict[ntype] = cur_subgraph.num_nodes(ntype=ntype)
                paper_subgraph = cur_subgraph['paper', 'cites', 'paper']
                cur_dict['paper_edges'] = paper_subgraph.num_edges()
                all_list.append(cur_dict)
        df = pd.DataFrame(all_list)
        df.to_csv(record_path + '{}_train_info.csv'.format(self.model_name))

    def get_inputs(self, content, lengths, masks, ids, graph, times, phase, time_masks, **kwargs):
        # masks [batch, T]
        # last_time = time.time()
        graphs, index_dicts, _ = graph
        # print(len(graphs))
        # for t in range(self.time_length):
        #     cur_graph = graphs[t]
        #     if not self.tree:
        #         bool_value = cur_graph.nodes['paper'].data['hop'] <= 1
        #         cur_graph.nodes['paper'].data['is_ref'] = cur_graph.nodes['paper'].data['is_ref'] * bool_value
        #         cur_graph.nodes['paper'].data['is_cite'] = cur_graph.nodes['paper'].data['is_cite'] * bool_value
        # cur_time = time.time()
        # print('>>hop time:', cur_time - last_time)
        # last_time = cur_time

        cur_snapshot_types = self.get_snapshot_types(times)
        # cur_time = time.time()
        # print('>>snap type time:', cur_time - last_time)
        # last_time = cur_time
        # [T, B]
        time_masks = ~time_masks.bool()
        # [B, T]
        if self.phase == 'train':
            if 'topic' not in self.ablation_channel:
                input_times = 3 if self.disen_module_type == 'scl' else 2
                cur_snapshot_types = cur_snapshot_types.tile(1, input_times)
                time_masks = time_masks.tile(input_times, 1)
        return {'graphs': graphs,
                'cur_snapshot_types': cur_snapshot_types,
                'time_masks': time_masks,
                'index_dicts': index_dicts,
                'phase': phase}


class DDHGCNSECL(DDHGCNS):
    def __init__(self, num_classes, embed_dim, hidden_dim, time_length, ntypes, etypes, start_times,
                 time_embs_path=None, model_type=None, snapshot_mode='mix', learned_time=False, time_pooling='sum',
                 topic_module='align', pop_module='citation', n_topics=10,
                 encoder_type='DGINP', n_layers=3, time_layers=1, tree=False, dropout=0.3, residual=False,
                 graph_dropout=0, ablation_channel=(), graph_norm=False, mixed=True, updater='normal',
                 adaptive_lr=False, loss_weights=None, lambda_weight=0.5, use_constraint=False, **kwargs):
        super(DDHGCNSECL, self).__init__(num_classes, embed_dim, hidden_dim, time_length, ntypes, etypes, start_times,
                                         time_embs_path, None, snapshot_mode, learned_time, time_pooling,
                                         topic_module, pop_module, n_topics,
                                         encoder_type, n_layers, time_layers, tree, dropout, residual, graph_dropout,
                                         ablation_channel, graph_norm, mixed, updater, adaptive_lr, loss_weights,
                                         lambda_weight, use_constraint, **kwargs)
        self.model_name = self.model_name.replace('DDHGCNS', 'DDHGCNSECL')
        self.disen_loss = False
        self.shared = False
        self.get_model_type(model_type)
        # self.disen_module = SimpleCLDisenModule(self.encoder, embed_dim, hidden_dim, time_length, self.device,
        #                                         disen_loss=disen_loss)
        if topic_module != 'ecl':
            raise Exception
        self.get_disen_modules(topic_module, pop_module, 'ecl')


class SparseSelecter:
    def __init__(self, matrix, graph=None):
        self.sparse = matrix
        self.graph = graph

        row, col = matrix.coalesce().indices()
        indices = list(zip(*map(lambda x: x.numpy().tolist(), (row, col))))
        print(len(indices), indices[0])
        self.indices = set(indices)

    def get_single_data(self, indexes):
        # print(indexes)
        if indexes in self.indices:
            return self.sparse[indexes]
        else:
            return torch.tensor(0)

    def get_single_index_data(self, node):
        linked_nodes = self.graph.out_edges(node)[1]
        cur_line = self.sparse.index_select(dim=0, index=torch.tensor([node])).to_dense().squeeze(dim=0)
        values = cur_line[linked_nodes]
        # print(values.shape)
        return values


def get_snapshot_types_dict(all_graphs):
    index_dict = {pub_time: {} for pub_time in all_graphs}
    for pub_time in all_graphs:
        for ntype in all_graphs[pub_time].ntypes:
            oids = all_graphs[pub_time].nodes[ntype].data['oid'].numpy().tolist()
            nids = all_graphs[pub_time].nodes(ntype).numpy().tolist()
            index_dict[pub_time][ntype] = dict(zip(oids, nids))
    return index_dict


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    batch = joblib.load('../test/test_batch')
    # x, values, lengths, masks, ids, times, blocks = batch

    ntypes = ["paper", "author", "journal", "time"]
    etypes = [
        "cites", "is cited by", "writes", "is writen by", "publishes", "is published by",
        "shows", "is shown in"
    ]
    start_times = [2013, 2015, 2017]
    time_embs_path = '../checkpoints/geography/DDHGCN_graphs_sbert.job'
    # model = DDHGCN(3, 384, 32, 5, ntypes, etypes, start_times, time_embs_path).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DDHGCNSCLT(3, 384, 128, 5, ntypes, etypes, start_times, time_embs_path, encoder_type='CCompGATSMN',
                       topic_module='scl', pop_module='accum', ablation_channel=(), device=device,
                       loss_weights={'topic': 0.5, 'pop': 0.5, 'disen': 0.5},
                       updater='dot').cuda()
    print(model.model_name)
    batch = model.get_batch_input(batch, None)
    x, values, lengths, masks, ids, times, blocks = batch
    # print(blocks[-1].device)
    # module = model.disen_module.pop_module
    # snapshot_embs = torch.randn(32, 128).cuda()
    # print(module(snapshot_embs, x[2:]))
    model.train()
    model.phase = 'train'
    rst = model(x, lengths, masks, ids, blocks, times)
    print(rst[0])
    print("1:{:.2f}MB".format(torch.cuda.memory_allocated() / 1024 ** 2))
    # model.other_loss.backward()
    rst[-1].backward()
