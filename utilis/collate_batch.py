import argparse
import copy
import datetime
import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import dgl
# import joblib
import numpy as np
import pandas as pd
import scipy as sp
import torch
from torch_geometric.data import Batch, Data
from dgl.nn.functional import edge_softmax

from layers.GCN.DDHGCN_layers import simple_node_drop


class saved_collate_batch:
    def __init__(self, dataProcessor, phase, sort=False):
        self.phase = phase
        self.dataProcessor = dataProcessor
        self.cur_graph, self.index_list, self.index_trans = self.dataProcessor.graph_dict['data'][phase]
        # print(self.index_trans)
        # self.index_list = self.dataProcessor.graph_dict['index_list'][phase]
        # self.index_trans = self.dataProcessor
        self.sort = sort

    def get_single_saved_subgraph(self, paper):
        subgraph = dgl.node_subgraph(self.cur_graph, self.index_list[self.index_trans[paper]])
        return subgraph

    def get_batched_graphs(self, paper_list):
        return dgl.batch([self.get_single_saved_subgraph(paper) for paper in paper_list])

    def __call__(self, batch, *args, **kwargs):
        # for those that need dealing with graph in each batch
        values_list, content_list = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)
        if self.sort:
            batch = sorted(batch, key=lambda x: int(x[self.sort]))

        for (_ids, _values, _contents, _time) in batch:
            processed_content, seq_len, mask = self.dataProcessor.tokenizer.encode(_contents)
            values_list.append(_values)
            content_list.append(processed_content)
            length_list.append(seq_len)
            mask_list.append(mask)
            ids_list.append(_ids.strip())
            time_list.append(_time)

        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        time_list = torch.tensor(time_list, dtype=torch.long)
        content_list = content_batch

        subgraphs = self.get_batched_graphs(ids_list)

        return SimpleCustomPinningBatch(content_list, values_list, length_list, mask_list, ids_list, time_list,
                                        subgraphs)


class saved_dynamic_homo_collate_batch:
    def __init__(self, dataProcessor, phase, sort=None):
        self.phase = phase
        self.dataProcessor = dataProcessor
        # self.cur_graphs, self.index_list, self.index_trans, self.node_count = self.dataProcessor.graph_dict['data'][
        #     phase]
        self.cur_graphs, self.index_list, self.index_trans, self.masks = self.dataProcessor.graph_dict['data'][phase]
        self.time_length = len(self.cur_graphs)
        # print(self.index_trans)
        # self.index_list = self.dataProcessor.graph_dict['index_list'][phase]
        # self.index_trans = self.dataProcessor
        self.sort = sort
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_single_saved_subgraph(self, paper):
        subgraph_list = [None] * self.time_length
        # node_masks = self.masks[self.index_trans[paper]]
        # print(len(self.node_count), self.index_trans[paper])
        # cur_node_count = self.node_count[self.index_trans[paper]]
        cur_node_dict = self.index_list[self.index_trans[paper]]
        for t in range(self.time_length):
            subgraph = dgl.node_subgraph(self.cur_graphs[t], nodes=cur_node_dict[t])
            # print(subgraph)
            # for ntype in cur_node_count:
            #     subgraph.add_nodes(ntype=ntype, num=cur_node_count[ntype] - subgraph.nodes(ntype).shape[0])
            # # print(subgraph.nodes['paper'].data)
            # # print(subgraph.edges['cites'].data)
            # if len(cur_node_dict[t]['paper']) == 0:
            #     subgraph.add_edges(etype='cites', u=0, v=0, data={'time': torch.zeros(1, 1, dtype=torch.int16)})
            # subgraph = dgl.to_homogeneous(subgraph, ndata=['h'], edata=['time'])
            # # print(subgraph.edata.keys())
            subgraph_list[t] = subgraph
        # print('here', subgraph_list)
        return subgraph_list

    def get_masks(self, paper_list):
        mask_list = [None] * len(paper_list)
        for i in range(len(paper_list)):
            mask_list[i] = self.masks[self.index_trans[paper_list[i]]]
        return torch.cat(mask_list, dim=1).unsqueeze(dim=-1).bool()

    def get_batched_graphs(self, paper_list):
        # start_time = time.time()
        # subgraph_list = [self.get_single_saved_subgraph(paper) for paper in paper_list]
        subgraph_list = [None] * len(paper_list)
        for i in range(len(paper_list)):
            subgraph_list[i] = self.get_single_saved_subgraph(paper_list[i])
        # print(subgraph_list)
        # print(list(zip(*subgraph_list)))
        # paper_list = [dgl.batch(subgraphs) for subgraphs in zip(*subgraph_list)]
        batch_list = [None] * self.time_length
        t = 0
        for subgraphs in zip(*subgraph_list):
            batch_list[t] = dgl.batch(subgraphs)
            t += 1
        # print(paper_list)
        # print('loader time:', time.time() - start_time)
        return batch_list

    def __call__(self, batch, *args, **kwargs):
        # for those that need dealing with graph in each batch
        # values_list, content_list = [], []
        # length_list = []
        # mask_list = []
        # ids_list = []
        # time_list = []
        batch_len = len(batch)
        values_list, content_list = [None] * batch_len, [None] * batch_len
        length_list = [None] * batch_len
        mask_list = [None] * batch_len
        ids_list = [None] * batch_len
        time_list = [None] * batch_len
        # print(batch)
        if self.sort is not None:
            batch = sorted(batch, key=lambda x: int(x[self.sort]))

        index = 0
        for (_ids, _values, _contents, _time) in batch:
            processed_content, seq_len, mask = self.dataProcessor.tokenizer.encode(_contents)
            values_list[index] = _values
            content_list[index] = processed_content
            length_list[index] = seq_len
            mask_list[index] = mask
            ids_list[index] = _ids.strip()
            time_list[index] = _time
            index += 1

        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        # mask_list = torch.tensor(mask_list, dtype=torch.int8)
        time_list = torch.tensor(time_list, dtype=torch.long)

        content_list = content_batch
        subgraphs = self.get_batched_graphs(ids_list)
        mask_list = self.get_masks(ids_list)
        # print(mask_list.shape)

        return SimpleCustomPinningBatch(content_list, values_list, length_list, mask_list, ids_list, time_list,
                                        subgraphs)



class ddhgcn_collate_batch:
    def __init__(self, dataProcessor, phase, sort=None, tree=False, **kwargs):
        self.phase = phase
        self.dataProcessor = dataProcessor
        self.cur_graphs, self.index_lists, self.index_trans, self.time_masks, self.pop_indicators = \
            self.dataProcessor.graph_dict['data'][phase]
        self.time_length = len(self.cur_graphs)
        # print(self.index_trans)
        # self.index_list = self.dataProcessor.graph_dict['index_list'][phase]
        # self.index_trans = self.dataProcessor
        self.sort = sort
        self.node_attr_types = ['is_cite', 'is_ref', 'is_target']
        self.tree = tree
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def get_single_saved_subgraph(self, paper):
        subgraph_list = [None] * self.time_length
        for t in range(self.time_length):
            # print(paper, self.index_trans[paper])
            info_dict = self.index_lists[t][self.index_trans[paper]]
            # print(info_dict)
            subgraph = dgl.node_subgraph(self.cur_graphs[t], nodes=info_dict['nodes'])
            # print(len(info_dict['nodes']['paper']))
            for indicator in info_dict['indicators']:
                cur_indicators = info_dict['indicators'][indicator]
                subgraph.nodes['paper'].data[indicator] = torch.tensor(cur_indicators, dtype=torch.long) \
                    if type(cur_indicators) != torch.Tensor else cur_indicators
            # print(subgraph)
            # subgraph_list.append(subgraph)
            subgraph_list[t] = subgraph
        # print('here', subgraph_list)
        return subgraph_list

    def get_batched_graphs(self, paper_list):
        # start_time = time.time()
        # subgraph_list = [self.get_single_saved_subgraph(paper) for paper in paper_list]
        # last_time = time.time()
        subgraph_list = [None] * len(paper_list)
        for i in range(len(paper_list)):
            subgraph_list[i] = self.get_single_saved_subgraph(paper_list[i])

        # cur_time = time.time()
        # print('subgraph time:', cur_time - last_time)
        # last_time = cur_time
        # print(subgraph_list)
        # print(list(zip(*subgraph_list)))
        # paper_list = [dgl.batch(subgraphs) for subgraphs in zip(*subgraph_list)]
        batch_list = [None] * self.time_length
        t = 0
        for subgraphs in zip(*subgraph_list):
            batch_list[t] = dgl.batch(subgraphs)
            t += 1

        # cur_time = time.time()
        # print('batch time:', cur_time - last_time)
        # last_time = cur_time
        # print(paper_list)
        # print('loader time:', time.time() - start_time)
        return batch_list

    def get_pop_indicators(self, values, papers):
        labels = torch.ones_like(values)
        tail = torch.where(values <= self.dataProcessor.cut_points[0])
        head = torch.where(values >= self.dataProcessor.cut_points[1])
        labels[tail] = 0
        labels[head] = 2
        return [labels]

    def get_time_masks(self, papers):
        # time_masks = torch.from_numpy(np.array([self.time_masks[self.index_trans[paper]]
        #                                         for paper in papers]).astype(np.long))
        # time_masks = torch.stack([self.time_masks[self.index_trans[paper]]
        #                           for paper in papers], dim=0)
        time_masks = [None] * len(papers)
        index = 0
        for paper in papers:
            time_masks[index] = self.time_masks[self.index_trans[paper]]
            index += 1
        time_masks = torch.stack(time_masks, dim=0)
        return time_masks

    def __call__(self, batch, *args, **kwargs):
        # for those that need dealing with graph in each batch
        # values_list, content_list = [], []
        # length_list = []
        # mask_list = []
        # ids_list = []
        # time_list = []
        batch_len = len(batch)
        values_list, content_list = [None] * batch_len, [None] * batch_len
        length_list = [None] * batch_len
        mask_list = [None] * batch_len
        ids_list = [None] * batch_len
        time_list = [None] * batch_len
        # print(batch)
        if self.sort is not None:
            batch = sorted(batch, key=lambda x: int(x[self.sort]))

        index = 0
        for (_ids, _values, _contents, _time) in batch:
            processed_content, seq_len, mask = self.dataProcessor.tokenizer.encode(_contents)
            # values_list.append(_values)
            # content_list.append(processed_content)
            # length_list.append(seq_len)
            # mask_list.append(mask)
            # ids_list.append(_ids.strip())
            # time_list.append(_time)
            values_list[index] = _values
            content_list[index] = processed_content
            length_list[index] = seq_len
            mask_list[index] = mask
            # content_list[index] = 0
            # length_list[index] = 0
            # mask_list[index] = 0
            ids_list[index] = _ids.strip()
            time_list[index] = _time
            index += 1

        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        time_list = torch.tensor(time_list, dtype=torch.long)

        time_masks = self.get_time_masks(ids_list)
        pop_indicators = self.get_pop_indicators(values_list, ids_list)
        content_list = [content_batch, time_masks] + pop_indicators
        # print(time_masks, pop_indicators)

        subgraphs = self.get_batched_graphs(ids_list)

        return SimpleCustomPinningBatch(content_list, values_list, length_list, mask_list, ids_list, time_list,
                                        subgraphs)


class ddhgcns_collate_batch(ddhgcn_collate_batch):
    def __init__(self, *args, **kwargs):
        super(ddhgcns_collate_batch, self).__init__(*args, **kwargs)

    def get_pop_indicators(self, values, papers):
        # print('here we get the pop indicators')
        if self.phase == 'train':
            pop_index_trans = self.pop_indicators['pop_index_trans']
            final_boxes = [None] * len(papers)
            avg_boxes = [None] * len(papers)
            norm_seqs = [None] * len(papers)
            seq_masks = [None] * len(papers)
            for i in range(len(papers)):
                final_boxes[i] = self.pop_indicators['final'][pop_index_trans[papers[i]]]
                avg_boxes[i] = self.pop_indicators['avg'][pop_index_trans[papers[i]]]
                norm_seqs[i] = self.pop_indicators['norm_seqs'][pop_index_trans[papers[i]]]
                seq_masks[i] = self.pop_indicators['seq_masks'][pop_index_trans[papers[i]]]
            all_list = [final_boxes, avg_boxes, norm_seqs, seq_masks]
            for i in range(len(all_list)):
                all_list[i] = torch.stack(all_list[i], dim=0)
                # print(all_list[i].shape)
                # print(all_list[i])
            return all_list
        else:
            return [torch.tensor([])]


class ddhgcns_cl_collate_batch(ddhgcns_collate_batch):
    def __init__(self, *args, **kwargs):
        super(ddhgcns_cl_collate_batch, self).__init__(*args, **kwargs)
        self.p = kwargs.get('aug_rate', 0.1)
        print('>>>aug_rate:', self.p)
        self.topic_module = kwargs.get('topic_module', 'cl')
        self.focus = kwargs.get('focus', False)
        print('>>>focus mode:', self.focus)


    def get_batched_graphs(self, paper_list):
        # last_time = time.time()
        subgraph_list = [None] * len(paper_list)
        for i in range(len(paper_list)):
            subgraph_list[i] = self.get_single_saved_subgraph(paper_list[i])

        # cur_time = time.time()
        # print('subgraph time:', cur_time - last_time)
        # last_time = cur_time
        batch_list = [None] * self.time_length

        t = 0
        for subgraphs in zip(*subgraph_list):
            batch_list[t] = dgl.batch(subgraphs)
            t += 1
        # print('loader time:', time.time() - start_time)
        # cur_time = time.time()
        # print('batch time:', cur_time - last_time)
        # last_time = cur_time

        if self.phase == 'train':
            new_list = [None] * self.time_length
            if self.topic_module == 'scl':
                neg_list = [None] * self.time_length
            for t in range(self.time_length):
                cur_graph = batch_list[t]
                new_list[t] = simple_node_drop(copy.deepcopy(cur_graph), self.p, batched=True, focus=self.focus)
                if self.topic_module not in {'ocl', 'scl'}:
                    batch_list[t] = simple_node_drop(cur_graph, self.p, batched=True, focus=self.focus)
                if self.topic_module == 'scl':
                    neg_list[t] = simple_node_drop(copy.deepcopy(cur_graph), self.p, batched=True, reverse=True,
                                                   focus=self.focus)
            batch_list += new_list
            if self.topic_module == 'scl':
                batch_list += neg_list

        # cur_time = time.time()
        # print('aug time:', cur_time - last_time)
        # last_time = cur_time
        return batch_list

class ddhgcns_cl_test_collate_batch(ddhgcns_cl_collate_batch):
    def __init__(self, *args, **kwargs):
        super(ddhgcns_cl_test_collate_batch, self).__init__(*args, **kwargs)

    def get_single_saved_subgraph(self, paper):
        subgraph_list = [None] * self.time_length
        for t in range(self.time_length):
            # if self.phase == 'train':
            #     subgraph = self.index_lists[t][self.index_trans[paper]]
            # else:
            #     info_dict = self.index_lists[t][self.index_trans[paper]]
            #     subgraph = dgl.node_subgraph(self.cur_graphs[t], nodes=info_dict['nodes'])
            #     for indicator in info_dict['indicators']:
            #         subgraph.nodes['paper'].data[indicator] = torch.tensor(info_dict['indicators'][indicator],
            #                                                                dtype=torch.long)
            info_dict = self.index_lists[t][self.index_trans[paper]]
            subgraph = dgl.node_subgraph(self.cur_graphs[t], nodes=info_dict['nodes'])
            for indicator in info_dict['indicators']:
                # subgraph.nodes['paper'].data[indicator] = torch.tensor(info_dict['indicators'][indicator],
                #                                                        dtype=torch.long)
                cur_indicators = info_dict['indicators'][indicator]
                subgraph.nodes['paper'].data[indicator] = torch.tensor(cur_indicators, dtype=torch.long) \
                    if type(cur_indicators) != torch.Tensor else cur_indicators
            # single graph complete
            cur_list = [subgraph]
            if self.phase == 'train':
                if self.topic_module == 'scl':
                    cur_list += [None] * 2
                else:
                    cur_list += [None] * 1

                cur_list[1] = simple_node_drop(copy.deepcopy(subgraph), self.p, batched=False, focus=self.focus)
                if self.topic_module not in {'ocl', 'scl'}:
                    cur_list[0] = simple_node_drop(subgraph, self.p, batched=False, focus=self.focus)
                if self.topic_module == 'scl':
                    cur_list[2] = simple_node_drop(copy.deepcopy(subgraph), self.p, batched=False, reverse=True,
                                                   focus=self.focus)
            subgraph_list[t] = cur_list
        return subgraph_list

class ddhgcns_cl_test_collate_batch(ddhgcns_cl_collate_batch):
    def __init__(self, *args, **kwargs):
        super(ddhgcns_cl_test_collate_batch, self).__init__(*args, **kwargs)

    def get_single_saved_subgraph(self, paper):
        subgraph_list = [None] * self.time_length
        for t in range(self.time_length):
            info_dict = self.index_lists[t][self.index_trans[paper]]
            subgraph = dgl.node_subgraph(self.cur_graphs[t], nodes=info_dict['nodes'])
            # cur_time = time.time()
            # print('>>subgraph time:', cur_time - last_time)
            # last_time = cur_time
            for indicator in info_dict['indicators']:
                # subgraph.nodes['paper'].data[indicator] = torch.tensor(info_dict['indicators'][indicator],
                #                                                        dtype=torch.long)
                cur_indicators = info_dict['indicators'][indicator]
                subgraph.nodes['paper'].data[indicator] = torch.tensor(cur_indicators, dtype=torch.long) \
                    if type(cur_indicators) != torch.Tensor else cur_indicators
            # single graph complete
            cur_list = [subgraph]
            if self.phase == 'train':
                if self.topic_module == 'scl':
                    cur_list += [None] * 2
                else:
                    cur_list += [None] * 1

                cur_list[1] = simple_node_drop(copy.deepcopy(subgraph), self.p, batched=False, focus=self.focus)
                if self.topic_module not in {'ocl', 'scl'}:
                    cur_list[0] = simple_node_drop(subgraph, self.p, batched=False, focus=self.focus)
                if self.topic_module == 'scl':
                    cur_list[2] = simple_node_drop(copy.deepcopy(subgraph), self.p, batched=False, reverse=True,
                                                   focus=self.focus)
            # cur_time = time.time()
            # print('>>augment time:', cur_time - last_time)
            # last_time = cur_time
            subgraph_list[t] = cur_list
        return subgraph_list

    @staticmethod
    def get_index_dict(cur_graph, tree, node_attr_types, filter_nodes=True):
        if (not tree) and filter_nodes:
            tgt_papers = torch.where(cur_graph.nodes['paper'].data['is_target'] == 1)[0].numpy().tolist()
            # print(tgt_papers)
            cites_paper = cur_graph.in_edges(tgt_papers, etype='cites')[0].numpy().tolist()
            cited_paper = cur_graph.in_edges(tgt_papers, etype='is cited by')[0].numpy().tolist()
            cur_graph.nodes['paper'].data['is_cite'] = torch.zeros(cur_graph.num_nodes('paper'), dtype=torch.long)
            cur_graph.nodes['paper'].data['is_ref'] = torch.zeros(cur_graph.num_nodes('paper'), dtype=torch.long)
            cur_graph.nodes['paper'].data['is_cite'][cites_paper + tgt_papers] = 1
            cur_graph.nodes['paper'].data['is_ref'][cited_paper + tgt_papers] = 1

        selected_indicator = (torch.stack([cur_graph.nodes['paper'].data[node_attr_type]
                                           for node_attr_type in node_attr_types], dim=-1).sum(dim=-1) > 0)

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
        valid_num_nodes = torch.tensor(valid_num_nodes)
        index = torch.tensor(index, dtype=torch.long)
        index_dict = {'selected_indicator': selected_indicator, 'index': index,
                      # 'valid_num_nodes': valid_num_nodes,
                      'mixer_index': mixer_index}
        return index_dict, valid_num_nodes

    def get_batched_graphs(self, paper_list):
        # last_time = time.time()
        subgraph_list = [None] * len(paper_list)
        for i in range(len(paper_list)):
            subgraph_list[i] = self.get_single_saved_subgraph(paper_list[i])
        batch_list = [None] * self.time_length
        index_list = [None] * self.time_length
        valid_num_list = [None] * self.time_length
        t = 0
        for subgraphs in zip(*subgraph_list):
            cur_list = reduce(lambda a, b: a + b, zip(*subgraphs))
            # print(cur_list)
            batch_list[t] = dgl.batch(cur_list)
            cur_index, cur_valid_num = self.get_index_dict(batch_list[t], self.tree, self.node_attr_types)
            index_list[t] = cur_index
            valid_num_list[t] = cur_valid_num
            t += 1
        valid_num_list = torch.stack(valid_num_list, dim=0)
        # print(valid_num_list.shape)
        return batch_list, index_list, valid_num_list

class dppdcc_collate_batch(ddhgcns_cl_test_collate_batch):
    def __init__(self, *args, **kwargs):
        super(dppdcc_collate_batch, self).__init__(*args, **kwargs)
        self.graph_path = './checkpoints/{}/DPPDCC/{}/'.format(self.dataProcessor.data_source, self.phase)

    def get_single_saved_subgraph(self, paper):
        origin_list = dgl.load_graphs(self.graph_path + '{}_subgraphs.dgl'.format(paper))[0]
        if self.phase == 'train' and 'cl' in self.topic_module:
            subgraph_list = [origin_list]
            pos_list = copy.deepcopy(origin_list)
            neg_list = copy.deepcopy(origin_list) if self.topic_module == 'scl' else None
            for t in range(self.time_length):
                pos_list[t] = simple_node_drop(pos_list[t], self.p,
                                               batched=False,
                                               focus=self.focus)
                if self.topic_module not in {'ocl', 'scl'}:
                    subgraph_list[0][t] = simple_node_drop(subgraph_list[0][t], self.p,
                                                           batched=False,
                                                           focus=self.focus)
                if self.topic_module == 'scl':
                    neg_list[t] = simple_node_drop(pos_list[t], self.p,
                                                   batched=False,
                                                   reverse=True, focus=self.focus)
            if neg_list is not None:
                subgraph_list = subgraph_list + [pos_list] + [neg_list]
            else:
                subgraph_list = subgraph_list + [pos_list]
            subgraph_list = list(zip(*subgraph_list))
        else:
            subgraph_list = [[subgraph] for subgraph in origin_list]
        return subgraph_list

    def get_batched_graphs(self, paper_list):
        # last_time = time.time()
        subgraph_list = [None] * len(paper_list)
        for i in range(len(paper_list)):
            subgraph_list[i] = self.get_single_saved_subgraph(paper_list[i])

        batch_list = [None] * self.time_length
        index_list = [None] * self.time_length
        valid_num_list = [None] * self.time_length
        t = 0
        for subgraphs in zip(*subgraph_list):
            cur_list = reduce(lambda a, b: a + b, zip(*subgraphs))
            # print(cur_list)
            cur_batch = dgl.batch(cur_list)
            batch_list[t] = cur_batch
            for ntype in cur_batch.ntypes:
                all_data_keys = self.cur_graphs[t].nodes[ntype].data.keys()
                cur_batch_keys = cur_batch.nodes[ntype].data.keys()
                for key in all_data_keys:
                    if key not in cur_batch_keys:
                        cur_batch.nodes[ntype].data[key] = (
                            self.cur_graphs[t].nodes[ntype].data)[key][cur_batch.nodes[ntype].data[dgl.NID]]

            cur_index, cur_valid_num = self.get_index_dict(batch_list[t], self.tree, self.node_attr_types,
                                                           filter_nodes=False)
            index_list[t] = cur_index
            valid_num_list[t] = cur_valid_num
            t += 1
        valid_num_list = torch.stack(valid_num_list, dim=0)
        # print(valid_num_list.shape)
        return batch_list, index_list, valid_num_list

def get_batched_graphs(self, paper_list):
    # last_time = time.time()
    subgraph_list = [None] * len(paper_list)
    for i in range(len(paper_list)):
        subgraph_list[i] = self.get_single_saved_subgraph(paper_list[i])
    # cur_time = time.time()
    # print('single time:', cur_time - last_time)
    # last_time = cur_time
    batch_list = [None] * self.time_length
    t = 0
    for subgraphs in zip(*subgraph_list):
        cur_list = reduce(lambda a, b: a + b, zip(*subgraphs))
        # print(cur_list)
        batch_list[t] = dgl.batch(cur_list)
        t += 1
    # cur_time = time.time()
    # print('batch time:', cur_time - last_time)
    # last_time = cur_time
    return batch_list


class saved_pyg_collate_batch:
    def __init__(self, dataProcessor=None, phase=None, is_graph=False):
        self.phase = phase
        self.dataProcessor = dataProcessor
        self.cur_graphs, self.index_list, self.index_trans = self.dataProcessor.graph_dict['data'][phase]
        # print(self.index_trans)
        # self.index_list = self.dataProcessor.graph_dict['index_list'][phase]
        # self.index_trans = self.dataProcessor
        self.is_graph = is_graph

    def get_single_saved_subgraph(self, paper):
        subgraph = self.index_list[self.index_trans[paper]]
        if not self.is_graph:
            if type(subgraph) != torch.Tensor:
                subgraph = torch.tensor(subgraph, dtype=torch.long)
            # subgraph = self.cur_graphs.subgraph(subgraph)
            # edges = pyg_subgraph(subgraph, edge_index=self.cur_graphs.edge_index)[0]
            # subgraph = Data(x=self.cur_graphs.x[subgraph, :], edge_index=edges)
            subgraph = dgl.node_subgraph(self.cur_graphs, subgraph)
            subgraph = Data(x=subgraph.ndata['h'], edge_index=torch.stack(subgraph.edges(), dim=0))
        return subgraph

    def get_batched_graphs(self, paper_list):
        # start_time = time.time()
        subgraphs = [None] * len(paper_list)
        for i in range(len(paper_list)):
            subgraphs[i] = self.get_single_saved_subgraph(paper_list[i])
        subgraph_list = Batch.from_data_list(subgraphs)
        return subgraph_list

    def __call__(self, batch, *args, **kwargs):
        # for those that need dealing with graph in each batch
        values_list, content_list = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)

        for (_ids, _values, _contents, _time) in batch:
            values_list.append(_values)
            # content_list.append(processed_content)
            length_list.append(0)
            mask_list.append(0)
            ids_list.append(_ids.strip())
            time_list.append(_time)

        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        time_list = torch.tensor(time_list, dtype=torch.long)

        subgraphs = self.get_batched_graphs(ids_list)

        content_list = content_batch
        # print(time_masks, pop_indicators)
        return SimpleCustomPinningBatch(content_list, values_list, length_list, mask_list, ids_list, time_list,
                                        subgraphs)


class saved_dynamic_collate_batch:
    def __init__(self, dataProcessor, phase, sort=None):
        self.phase = phase
        self.dataProcessor = dataProcessor
        self.cur_graphs, self.index_list, self.index_trans = self.dataProcessor.graph_dict['data'][phase]
        self.time_length = len(self.cur_graphs)
        # print(self.index_trans)
        # self.index_list = self.dataProcessor.graph_dict['index_list'][phase]
        # self.index_trans = self.dataProcessor

    def get_single_saved_subgraph(self, paper):
        selected_nodes = self.index_list[self.index_trans[paper]]
        sub_graphs = [None] * self.time_length
        lens = [None] * self.time_length
        max_len = len(selected_nodes[-1])
        for t in range(self.time_length):
            # print(selected_nodes[t])
            cur_nodes = torch.tensor(selected_nodes[t], dtype=torch.long)
            sub_graph = dgl.node_subgraph(self.cur_graphs[t], nodes=cur_nodes)
            sub_graph = dgl.add_nodes(sub_graph, max_len - sub_graph.num_nodes())
            # sub_graphs.append(sub_graph)
            sub_graphs[t] = sub_graph
            # lens.append(len(selected_nodes[t]))
            lens[t] = len(selected_nodes[t])
        # print(x_list.shape)
        # print(x_list.shape)
        mask_list = [None] * self.time_length
        last_len = lens[-1]
        for t in range(self.time_length):
            mask_list[t] = torch.tensor([1] * lens[t] + [0] * (last_len - lens[t]))
        mask_list = torch.stack(mask_list, dim=0).bool()
        return sub_graphs, mask_list

    def get_batched_graphs(self, paper_list):
        # start_time = time.time()
        single_out = [None] * len(paper_list)
        for i in range(len(paper_list)):
            single_out[i] = self.get_single_saved_subgraph(paper_list[i])
        subgraph_list, masks = zip(*single_out)
        # batches = []
        batches = [None] * self.time_length
        count = 0
        for subgraphs in zip(*subgraph_list):
            # print(len(subgraphs))
            cur_batched = dgl.batch(subgraphs)
            # batches.append(cur_batched)
            batches[count] = cur_batched
            count += 1

        masks = torch.cat(masks, dim=1).unsqueeze(dim=-1)
        # print(masks.shape)
        # print(masks)
        return batches, masks

    def __call__(self, batch, *args, **kwargs):
        # for those that need dealing with graph in each batch
        values_list, content_list = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)

        for (_ids, _values, _contents, _time) in batch:
            values_list.append(_values)
            # content_list.append(processed_content)
            length_list.append(0)
            mask_list.append(0)
            ids_list.append(_ids.strip())
            time_list.append(_time)

        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        # mask_list = torch.tensor(mask_list, dtype=torch.int8)
        time_list = torch.tensor(time_list, dtype=torch.long)

        subgraphs, mask_list = self.get_batched_graphs(ids_list)
        # print(subgraphs)

        content_list = content_batch
        # print(time_masks, pop_indicators)
        return SimpleCustomPinningBatch(content_list, values_list, length_list, mask_list, ids_list, time_list,
                                        subgraphs)


class saved_graph_collate_batch:
    def __init__(self, dataProcessor, phase):
        self.phase = phase
        self.dataProcessor = dataProcessor
        self.graph_list, self.index_trans = self.dataProcessor.graph_dict['data'][phase]
        # print(self.index_trans)
        # self.index_list = self.dataProcessor.graph_dict['index_list'][phase]
        # self.index_trans = self.dataProcessor

    def get_single_saved_subgraph(self, paper):
        subgraph = self.graph_list[self.index_trans[paper]]
        return subgraph

    def __call__(self, batch, *args, **kwargs):
        # for those that need dealing with graph in each batch
        values_list, content_list = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)
        for (_ids, _values, _contents, _time) in batch:
            processed_content, seq_len, mask = self.dataProcessor.tokenizer.encode(_contents)
            values_list.append(_values)
            content_list.append(processed_content)
            length_list.append(seq_len)
            mask_list.append(mask)
            ids_list.append(_ids.strip())
            time_list.append(_time)

        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        time_list = torch.tensor(time_list, dtype=torch.long)
        content_list = content_batch

        subgraphs = dgl.batch([self.get_single_saved_subgraph(paper) for paper in ids_list])

        return SimpleCustomPinningBatch(content_list, values_list, length_list, mask_list, ids_list, time_list,
                                        subgraphs)


class SimpleCustomPinningBatch:
    def __init__(self, x, values, lengths, masks, ids, times, subgraphs):
        # x, values, lengths, masks, ids, times, blocks
        self.x = x
        self.values = values
        self.lengths = lengths
        self.masks = masks
        self.ids = ids
        self.times = times
        self.subgraphs = subgraphs

    # custom memory pinning method on custom type
    def pin_memory(self):
        # print('here we pin!')
        # print(batch)
        # x, values, lengths, masks, ids, times, blocks = batch
        # print(x.is_pinned())
        # print(type(self.x))
        if type(self.x) == torch.Tensor:
            # print('here')
            self.x = self.x.pin_memory()
            # print(self.x.is_pinned())
        else:
            self.x = [temp.pin_memory() for temp in self.x]
        # print('io-time:', time.time()-start_time)
        self.values = self.values.pin_memory()
        self.lengths = self.lengths.pin_memory()
        self.masks = self.masks.pin_memory()
        if self.subgraphs:

            if type(self.subgraphs) == list:
                # print([temp.is_pinned() for temp in blocks])
                # print(type(self.subgraphs[0]))
                if 'dgl' in str(type(self.subgraphs[0])):
                    self.subgraphs = [temp.pin_memory_() for temp in self.subgraphs]
                else:
                    self.subgraphs = [temp.pin_memory() for temp in self.subgraphs]
            else:
                if 'dgl' in str(type(self.subgraphs[0])):
                    self.subgraphs = self.subgraphs.pin_memory_()
                else:
                    self.subgraphs = self.subgraphs.pin_memory()
        return self