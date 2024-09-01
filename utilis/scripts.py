import json
import logging
import math
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import dgl
import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, \
    precision_recall_curve, auc
import torch.nn.functional as F
from torch import nn
from torch._six import inf
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from tqdm import tqdm

from utilis.log_bar import TqdmToLogger
import torch.distributed as dist

reg_metrics = 'loss, acc, mae, r2, mse, rmse, Mrse, mrse, male, log_r2, msle, rmsle, smape, mape, ndcg'.split(', ')
aux_metrics = 'aux_loss, acc, roc_auc, log_loss_value, ' \
                       'micro_prec, micro_recall, micro_f1, ' \
                       'macro_prec, macro_recall, macro_f1'.split(', ')
aux_label_metrics = 'acc, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1'.split(', ')


def eval_result(all_true_values: np.array, all_predicted_values: np.array):
    # print(mean_absolute_error(all_true_values, all_predicted_values, multioutput='raw_values'))
    all_true_values = all_true_values.flatten(order='C')
    # print(all_true_values)
    all_predicted_values = all_predicted_values.flatten(order='C')
    all_predicted_values = np.where(all_predicted_values == np.inf, 1e20, all_predicted_values)
    # print(all_predicted_values)
    mae = mean_absolute_error(all_true_values, all_predicted_values)
    mse = mean_squared_error(all_true_values, all_predicted_values)
    rmse = math.sqrt(mse)
    r2 = r2_score(all_true_values, all_predicted_values)

    rse = (all_true_values - all_predicted_values) ** 2 / (all_true_values + 1)
    Mrse = np.mean(rse, axis=0)
    mrse = np.percentile(rse, 50)

    acc = np.mean(np.all(((0.5 * all_true_values) <= all_predicted_values,
                          all_predicted_values <= (1.5 * all_true_values)), axis=0), axis=0)
    # acc = np.mean(np.all(((0.5 * all_true_values) <= all_predicted_values,
    #                       all_predicted_values <= (1.5 * all_true_values)), axis=0), axis=0)

    # print(all_true_values)
    log_true_values = np.log(1 + all_true_values)
    log_predicted_values = np.log(1 + np.where(all_predicted_values >= 0, all_predicted_values, 0))

    # acc = np.mean(np.all(((0.5 * log_true_values) <= log_predicted_values,
    #                       log_predicted_values <= (1.5 * log_true_values)), axis=0), axis=0)

    male = mean_absolute_error(log_true_values, log_predicted_values)
    msle = mean_squared_error(log_true_values, log_predicted_values)
    rmsle = math.sqrt(msle)
    log_r2 = r2_score(log_true_values, log_predicted_values)
    smape = np.mean(np.abs(log_true_values - log_predicted_values) /
                    ((np.abs(log_true_values) + np.abs(log_predicted_values) + 0.1) / 2), axis=0)
    mape = np.mean(np.abs(log_true_values - log_predicted_values) / (np.abs(log_true_values) + 0.1), axis=0)
    ndcg = get_ranking_ndcg(log_predicted_values, log_true_values)
    # print('true', all_true_values[:10])
    # print('pred', all_predicted_values[:10])
    return [acc, mae, r2, mse, rmse, Mrse, mrse, male, log_r2, msle, rmsle, smape, mape, ndcg]


def eval_aux_results(all_true_label, all_predicted_result):
    all_predicted_label = np.argmax(all_predicted_result, axis=1)
    # all
    acc = accuracy_score(all_true_label, all_predicted_label)
    one_hot = np.eye(all_predicted_result.shape[-1])[all_true_label]
    try:
        roc_auc = roc_auc_score(one_hot, all_predicted_result)
        log_loss_value = log_loss(all_true_label, all_predicted_result)
    except Exception as e:
        print(e)
        roc_auc = 0
        log_loss_value = 0
    micro_prec = precision_score(all_true_label, all_predicted_label, average='micro')
    micro_recall = recall_score(all_true_label, all_predicted_label, average='micro')
    micro_f1 = f1_score(all_true_label, all_predicted_label, average='micro')
    macro_prec = precision_score(all_true_label, all_predicted_label, average='macro')
    macro_recall = recall_score(all_true_label, all_predicted_label, average='macro')
    macro_f1 = f1_score(all_true_label, all_predicted_label, average='macro')

    results = [acc, roc_auc, log_loss_value, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1]
    return results


def eval_aux_labels(all_true_label, all_predicted_label):
    acc = accuracy_score(all_true_label, all_predicted_label)
    micro_prec = precision_score(all_true_label, all_predicted_label, average='micro')
    micro_recall = recall_score(all_true_label, all_predicted_label, average='micro')
    micro_f1 = f1_score(all_true_label, all_predicted_label, average='micro')
    macro_prec = precision_score(all_true_label, all_predicted_label, average='macro')
    macro_recall = recall_score(all_true_label, all_predicted_label, average='macro')
    macro_f1 = f1_score(all_true_label, all_predicted_label, average='macro')

    results = [acc, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1]
    return results


def result_format(results):
    loss, acc, mae, r2, mse, rmse, Mrse, mrse, male, log_r2, msle, rmsle, smape, mape, ndcg = results
    format_str = '| loss {:8.4f} | acc {:9.4f} |\n' \
                 '| MAE {:9.4f} | R2 {:10.4f} |\n' \
                 '| MSE {:9.4f} | RMSE {:8.4f} |\n' \
                 '| MRSE {:8.4f} | mRSE {:8.4f} |\n' \
                 '| MALE {:8.4f} | LR2 {:9.4f} |\n' \
                 '| MSLE {:8.4f} | RMSLE {:7.4f} |\n' \
                 '| SMAPE {:7.4f} | MAPE {:8.4f} |\n' \
                 '| NDCG {:8.4F} |\n'.format(loss, acc,
                                                             mae, r2,
                                                             mse, rmse,
                                                             Mrse, mrse,
                                                             male, log_r2,
                                                             msle, rmsle,
                                                             smape, mape,
                                                             ndcg) + \
                 '-' * 59
    print(format_str)
    return format_str


def aux_result_format(results, phase):
    avg_loss, acc, roc_auc, log_loss_value, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1 = results

    format_str = '| aux loss {:8.3f} | {:9} {:7.3f} |\n' \
                 '| roc_auc {:9.3f} | log_loss {:8.3f} |\n' \
                 'micro:\n' \
                 '| precision {:7.3f} | recall {:10.3f} |\n' \
                 '| f1 {:14.3f} |\n' \
                 'macro:\n' \
                 '| precision {:7.3f} | recall {:10.3f} |\n' \
                 '| f1 {:14.3f} |\n'.format(avg_loss, phase + ' acc', acc, roc_auc, log_loss_value,
                                            micro_prec, micro_recall, micro_f1,
                                            macro_prec, macro_recall, macro_f1)
    print(format_str)
    return format_str


def get_configs(data_source, model_list, replace_dict=None):
    fr = open('./configs/{}.json'.format(data_source))
    configs = json.load(fr)
    full_configs = {'default': configs['default']}
    if replace_dict:
        print('>>>>>>>>>>replacing here<<<<<<<<<<<<<<<')
        print(replace_dict)
        for key in replace_dict:
            full_configs['default'][key] = replace_dict[key]
    for model in model_list:
        full_configs[model] = configs['default'].copy()
        if model in configs.keys():
            for key in configs[model].keys():
                full_configs[model][key] = configs[model][key]
    return full_configs


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        dgl.seed(seed)
    except Exception as E:
        print(E)
    print('add deterministic constraint')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ["OMP_NUM_THREADS"] = '1'


class IndexDict(dict):
    def __init__(self):
        super(IndexDict, self).__init__()
        self.count = 0

    def __getitem__(self, item):
        if item not in self.keys():
            super().__setitem__(item, self.count)
            self.count += 1
        return super().__getitem__(item)


def add_new_elements(graph, nodes={}, ndata={}, edges={}, edata={}):
    '''
    :param graph: Dgl
    :param nodes: {ntype: []}
    :param ndata: {ntype: {attr: []}
    :param edges: {(src_type, etype, dst_type): [(src, dst)]
    :param ndata: {etype: {attr: []}
    :return:
    '''
    # etypes = graph.canonical_etypes
    num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
    for ntype in nodes:
        num_nodes_dict[ntype] = nodes[ntype]
    # etypes.extend(list(edges.keys()))

    relations = {}
    for etype in graph.canonical_etypes:
        src, dst = graph.edges(etype=etype[1])
        relations[etype] = (src, dst)
    for etype in edges:
        relations[etype] = edges[etype]

    # print(relations)
    # print(num_nodes_dict)
    new_g = dgl.heterograph(relations, num_nodes_dict=num_nodes_dict)

    for ntype in graph.ntypes:
        for k, v in graph.nodes[ntype].data.items():
            new_g.nodes[ntype].data[k] = v.detach().clone()
    for etype in graph.etypes:
        for k, v in graph.edges[etype].data.items():
            new_g.edges[etype].data[k] = v.detach().clone()

    for ntype in ndata:
        for attr in ndata[ntype]:
            new_g.nodes[ntype].data[attr] = ndata[ntype][attr]

    for etype in edata:
        for attr in edata[etype]:
            # try:
            new_g.edges[etype].data[attr] = edata[etype][attr]
            # except Exception as e:
            #     print(e)
            #     print(etype, attr)
            #     print(new_g.edges(etype=etype))
            #     print(edata[etype][attr])
            #     raise Exception
    # print(new_g)
    return new_g


def get_norm_adj(sparse_adj):
    I = torch.diag(torch.ones(sparse_adj.shape[0])).to_sparse()
    sparse_adj = sparse_adj + I
    D = torch.diag(torch.sparse.sum(sparse_adj, dim=-1).pow(-1).to_dense())
    # print(I)
    # print(sparse_adj.to_dense())
    return torch.matmul(D, sparse_adj.to_dense()).to_sparse()


def custom_node_unbatch(g):
    # just unbatch the node attrs for cl, not including etypes for efficiency
    node_split = {ntype: g.batch_num_nodes(ntype) for ntype in g.ntypes}
    node_split = {k: dgl.backend.asnumpy(split).tolist() for k, split in node_split.items()}
    print(node_split)
    node_attr_dict = {ntype: {} for ntype in g.ntypes}
    for ntype in g.ntypes:
        for key, feat in g.nodes[ntype].data.items():
            subfeats = dgl.backend.split(feat, node_split[ntype], 0)
            # print(subfeats)
            node_attr_dict[ntype][key] = subfeats
    return node_attr_dict


def get_tsne(embs, file_name, indexes=None):
    perplexity_list = [5, 10, 20, 30, 50]
    embs = embs.astype(np.float32)
    embs = torch.from_numpy(embs)
    embs = F.normalize(embs).numpy()
    lr = max(embs.shape[0] / 12 / 4, 50)
    # lr = 1000
    print('add norm!')
    # print(lr)
    # tsne = TSNE(n_components=2, learning_rate=lr, init='random', perplexity=30)
    for perplexity in perplexity_list:
        tsne = TSNE(n_components=2, learning_rate=50, perplexity=perplexity, init='pca', verbose=1)
        result = tsne.fit_transform(embs)
        print(result.shape)
        # plt.scatter(x=result[:, 0], y=result[:, 1])
        # plt.savefig('./imgs/{}.pdf'.format(model_name), bbox_inches='tight')
        # joblib.dump(result, file_name.format(perplexity))
        torch.save([indexes, result], file_name.format(perplexity))


def get_label(citation):
    if citation < 10:
        return 0
    elif citation >= 100:
        return 2
    else:
        return 1


def get_sampled_index(data):
    print('wip')
    df = pd.DataFrame(data['test'][1])
    df['label'] = df[0].apply(get_label)
    sampled_count = 1000
    print(df[df['label'] == 0])
    cls_0 = df[df['label'] == 0].sample(sampled_count, random_state=123).index
    cls_1 = df[df['label'] == 1].sample(sampled_count, random_state=123).index
    print(cls_0[:5], cls_1[:5])
    cls_2 = df[df['label'] == 2].sample(sampled_count, random_state=123).index
    all_index = list(cls_0) + list(cls_1) + list(cls_2)
    return all_index


def get_selected_data(ids, dataProcessor):
    data = dataProcessor.get_data()


class ResultsContainer:
    def __init__(self, log_interval, batch_count, epoch, model_name, phase, record_path=None, record_name=None,
                 test_interval=0):
        self.start_time = time.time()
        self.all_predicted_values = []
        self.all_true_values = []
        self.all_loss = 0
        self.all_aux_loss = 0
        self.all_predicted_aux_values = []
        self.all_true_aux_values = []
        self.log_interval = log_interval
        self.batch_count = batch_count
        self.epoch = epoch
        self.model_name = model_name
        self.record_path = record_path
        self.record_name = record_name
        self.test_interval = test_interval
        self.phase = phase

    def get_interval_result(self, loss, idx):
        print_loss = loss.sum(dim=0).item()
        rmse = math.sqrt(max(print_loss, 0))

        if idx % self.log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| RMSE {:8.3f} | loss {:8.3f}'.format(self.epoch, idx, self.batch_count,
                                                         rmse, print_loss))
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches '
                         '| RMSE {:8.3f} | loss {:8.3f}'.format(self.epoch, idx, self.batch_count,
                                                                rmse, print_loss))

    def get_all_inputs(self, aux, all_times=None, all_ids=None):
        all_predicted_values = torch.cat(self.all_predicted_values, dim=0).numpy()
        all_true_values = torch.cat(self.all_true_values, dim=0).numpy()
        # print(all_predicted_values.shape)
        avg_loss = self.all_loss / self.batch_count

        if aux:
            avg_aux_loss = self.all_aux_loss / self.batch_count
            all_predicted_aux_values = torch.cat(self.all_predicted_aux_values, dim=0).numpy()
            all_true_aux_values = torch.cat(self.all_true_aux_values, dim=0).numpy()
        else:
            avg_aux_loss, all_true_aux_values, all_predicted_aux_values = [None] * 3

        if dist.is_initialized():
            all_data = {'avg_loss': avg_loss,
                        'all_true_values': all_true_values,
                        'all_predicted_values': all_predicted_values,
                        'avg_aux_loss': avg_aux_loss,
                        'all_true_aux_values': all_true_aux_values,
                        'all_predicted_aux_values': all_predicted_aux_values,
                        'all_times': all_times,
                        'all_ids': all_ids
                        }
            dist.barrier()
            gather_data = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_data, all_data)

            if dist.get_rank() == 0:
                gathered_dict = {key: [] for key in all_data}
                for i in range(dist.get_world_size()):
                    for key in all_data:
                        gathered_dict[key].append(gather_data[i][key])
                avg_loss = sum(gathered_dict['avg_loss']) / dist.get_world_size()
                all_true_values = np.concatenate(gathered_dict['all_true_values'], axis=0)
                all_predicted_values = np.concatenate(gathered_dict['all_predicted_values'], axis=0)
                if aux:
                    avg_aux_loss = sum(gathered_dict['avg_aux_loss']) / dist.get_world_size()
                    all_true_aux_values = np.concatenate(gathered_dict['all_true_aux_values'], axis=0)
                    all_predicted_aux_values = np.concatenate(gathered_dict['all_predicted_aux_values'], axis=0)
                else:
                    avg_aux_loss, all_true_aux_values, all_predicted_aux_values = [None] * 3
                if all_times is not None and all_ids is not None:
                    all_times = reduce(lambda a, b: a + b, gathered_dict['all_times'])
                    all_ids = reduce(lambda a, b: a + b, gathered_dict['all_ids'])

        return avg_loss, all_true_values, all_predicted_values, \
               avg_aux_loss, all_true_aux_values, all_predicted_aux_values, \
               all_times, all_ids

    def get_epoch_result(self, aux, all_times=None, all_ids=None):
        avg_loss, all_true_values, all_predicted_values, \
        avg_aux_loss, all_true_aux_values, all_predicted_aux_values, \
        all_times, all_ids = self.get_all_inputs(aux, all_times, all_ids)
        if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
            elapsed = time.time() - self.start_time
            if self.phase == 'train':
                print('-' * 59)
                print('| end of epoch {:3d} | time: {:5.2f}s |'.format(self.epoch, elapsed))
                logging.info('-' * 59)
                logging.info('| end of epoch {:3d} | time: {:5.2f}s |'.format(self.epoch, elapsed))
            else:
                print('-' * 59)
                print('| end of {} | time: {:5.2f}s |'.format(self.phase, elapsed))
                logging.info('-' * 59)
                logging.info('| end of {} | time: {:5.2f}s |'.format(self.phase, elapsed))
            '''
            calculate and print all results
            :param all_predicted_values:
            :param all_true_values:
            :param all_loss:
            =================================
            :param aux:
            :param all_aux_loss:
            :param all_predicted_aux_values:
            :param all_true_aux_values:
            =================================
            :param batch_count:
            :param phase:
            :param all_times:
            :param all_ids:
            :param record_name:
            :return:
            '''

            # print(len(dataloader))
            results = [avg_loss] + eval_result(all_true_values, all_predicted_values)
            format_str = result_format(results)

            for line in format_str.split('\n'):
                logging.info(line)

            if aux:
                print('=' * 89)
                logging.info('=' * 89)
                aux_results = [avg_aux_loss] + eval_aux_results(all_true_aux_values, all_predicted_aux_values)
                aux_format_str = aux_result_format(aux_results, phase='train')

                for line in aux_format_str.split('\n'):
                    logging.info(line)

                print('=' * 89)
                logging.info('=' * 89)
                results += aux_results

            if self.record_path and (self.phase == 'test'):
                # print(all_times.shape, all_true_values.shape, all_predicted_values.shape)
                result_data = {
                    'time': all_times,
                    'true': np.squeeze(all_true_values, axis=-1),
                    'pred': np.squeeze(all_predicted_values, axis=-1)
                }
                if aux:
                    result_data['aux_pred'] = np.argmax(all_predicted_aux_values, axis=-1)
                    result_data['aux_true'] = all_true_aux_values

                # print(len(all_ids), self.all_predicted_values.shape, len(all_times), self.all_true_values.shape)
                df = pd.DataFrame(index=all_ids, data=result_data)
                if self.test_interval == 1:
                    record_name = self.record_name.format(self.model_name) if self.record_name \
                        else '{}_results_{}.csv'.format(self.model_name, self.epoch)
                else:
                    record_name = self.record_name.format(self.model_name) if self.record_name \
                        else '{}_results.csv'.format(self.model_name)
                # df.to_csv(self.record_path + '{}_results.csv'.format(self.model_name))
                df.to_csv(self.record_path + record_name)
            return results
        else:
            return []

class SampleAllNeighbors:
    def __init__(self, graph, hop, return_graph=False, origin_graph=None):
        self.graph = graph
        self.hop = hop
        self.return_graph = return_graph
        self.origin_graph = origin_graph

    def get_single_data(self, paper):
        seed_nodes = [paper]
        all_nodes = [paper]
        for i in range(self.hop):
            frontier = dgl.sampling.sample_neighbors(self.graph, seed_nodes, -1)
            # print(frontier)
            seed_nodes = list(set(frontier.edges()[0].numpy().tolist()))
            all_nodes.extend(seed_nodes)
            # print(seed_nodes)
        all_nodes = set(all_nodes)
        # print(all_nodes)
        if self.return_graph:
            # print(paper, all_nodes)
            all_nodes.discard(paper)
            all_nodes = [paper] + list(all_nodes)
            if self.origin_graph:
                output = dgl.node_subgraph(self.origin_graph, all_nodes)
            else:
                output = dgl.node_subgraph(self.graph, all_nodes)
        else:
            output = list(all_nodes)
        return output


class SampleAllHeteroNeighbors:
    def __init__(self, graph, hop, in_edges, return_graph=True):
        self.graph = dgl.node_type_subgraph(graph, ntypes=['paper', 'journal', 'author'])
        self.paper_graph = dgl.edge_type_subgraph(graph, etypes=['cites'])
        self.paper_graph = dgl.add_reverse_edges(self.paper_graph)
        self.hop = hop
        self.return_graph = return_graph
        self.in_edges = in_edges

    def get_single_data(self, paper):
        seed_nodes = [paper]
        paper_nodes = [paper]
        all_nodes = {}
        for i in range(self.hop):
            frontier = dgl.sampling.sample_neighbors(self.paper_graph, seed_nodes, -1)
            # print(frontier)
            seed_nodes = list(set(frontier.edges()[0].numpy().tolist()))
            paper_nodes.extend(seed_nodes)
            # print(seed_nodes)
        # paper_graphs
        all_nodes['paper'] = list(set(paper_nodes))
        # print(all_nodes)
        for in_edge in self.in_edges:
            src_nodes, _ = self.graph.in_edges(all_nodes['paper'], etype=in_edge)
            all_nodes[in_edge[0]] = list(set(src_nodes.numpy().tolist()))
        if not self.return_graph:
            return all_nodes
        else:
            return dgl.node_subgraph(self.graph, nodes=all_nodes)


class SampleAllNeighborsPYG:
    def __init__(self, graph, hop, return_graph=True):
        self.graph = graph
        self.hop = hop
        self.return_graph = return_graph

    def get_single_data(self, paper):
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            paper, self.hop, self.graph.edge_index, relabel_nodes=True)
        # print(all_nodes)
        if self.return_graph:
            # print(subset.shape)
            return Data(id=subset, edge_index=edge_index)
        else:
            return subset


class SingleKhopSubgraph:
    def __init__(self, paper_subgraph, hop=2, detailed=False, max_nodes=(100, 20), return_graph=False):
        self.paper_subgraph = paper_subgraph
        # self.origin_graph = origin_graph
        # self.origin_graph = dgl.node_type_subgraph(origin_graph, ntypes=['paper', 'journal', 'author'])
        self.hop = hop
        # self.tree = tree
        # self.citation = citation
        self.max_nodes = max_nodes
        # self.oids = self.paper_subgraph.nodes['paper'].data[dgl.NID]
        if 'citations' not in self.paper_subgraph.nodes['paper'].data:
            self.paper_subgraph.nodes['paper'].data['citations'] = self.paper_subgraph.in_degrees(etype='cites')
        self.citations = self.paper_subgraph.nodes['paper'].data['citations']
        self.time = self.paper_subgraph.nodes['paper'].data['time']
        self.detailed = detailed
        self.return_graph = return_graph

    def get_node_k_hop_neighbors(self, node, max_count=None, graph=None, index_dict=None):
        ref_papers = graph.in_edges(node, etype='is cited by')[0].numpy().tolist()
        if index_dict:
            ref_papers_nids = [index_dict['paper'][paper]
                               for paper in graph.nodes['paper'].data['oid'][ref_papers].numpy().tolist()]
        else:
            ref_papers_nids = ref_papers
        # print(ref_papers, ref_papers_nids)
        if len(ref_papers) > max_count:
            temp_df = pd.DataFrame(
                data={
                    'id': ref_papers,
                    'citations': self.citations[ref_papers_nids].numpy(),
                    'time': self.time[ref_papers_nids].squeeze(dim=-1).numpy()})
            ref_papers = list(temp_df.sort_values(by=['time', 'citations'],
                                                  ascending=False).head(max_count)['id'].tolist())
        cite_papers = graph.in_edges(node, etype='cites')[0].numpy().tolist()
        if index_dict:
            cite_papers_nids = [index_dict['paper'][paper]
                                for paper in graph.nodes['paper'].data['oid'][cite_papers].numpy().tolist()]
        else:
            cite_papers_nids = cite_papers
        # print(cite_papers, cite_papers_nids)
        if len(cite_papers) > max_count:
            temp_df = pd.DataFrame(
                data={
                    'id': cite_papers,
                    'citations': self.citations[cite_papers_nids].numpy(),
                    'time': self.time[cite_papers_nids].squeeze(dim=-1).numpy()})
            cite_papers = list(temp_df.sort_values(by=['time', 'citations'],
                                                   ascending=False).head(max_count)['id'].tolist())
        neighbor_papers = list(set(ref_papers + cite_papers))
        return neighbor_papers, ref_papers, cite_papers

    def get_valid_subgraph(self, paper, graph, index_dict=None):
        seed_nodes = [paper]
        all_nodes = set(seed_nodes)
        # detailed_dict = {hop: [] for hop in range(self.hop)}
        detailed_dict = {'hops': defaultdict(list)}

        detailed_dict['hops'][-1] = seed_nodes.copy()
        seed_nodes, cur_refs, cur_cites = self.get_node_k_hop_neighbors(paper, self.max_nodes[0], graph=graph,
                                                                        index_dict=index_dict)
        detailed_dict['hops'][0] = seed_nodes.copy()
        detailed_dict['ref'] = set(cur_refs)
        detailed_dict['cite'] = set(cur_cites)

        for i in range(1, self.hop):
            cur_list = []
            cur_refs = []
            cur_cites = []
            for node in seed_nodes:
                if node not in all_nodes:
                    neighbors, _, _ = self.get_node_k_hop_neighbors(node, self.max_nodes[i], graph=graph,
                                                                    index_dict=index_dict)
                    cur_list.extend(neighbors)
                    all_nodes.add(node)
                    if node in detailed_dict['ref']:
                        cur_refs.extend(neighbors)
                    if node in detailed_dict['cite']:
                        cur_cites.extend(neighbors)
            seed_nodes = list(set([node for node in cur_list if node not in all_nodes]))
            detailed_dict['hops'][i] = seed_nodes.copy()
            # if self.tree:
            # print(detailed_dict['ref'], cur_refs)
            detailed_dict['ref'] = detailed_dict['ref'] | set(cur_refs)
            detailed_dict['cite'] = detailed_dict['cite'] | set(cur_cites)
            # detailed_dict['ref'] = set(list(detailed_dict['ref']) + cur_refs)
            # detailed_dict['cite'] = set(list(detailed_dict['cite']) + cur_cites)
        # all_nodes = list(all_nodes) + seed_nodes
        all_nodes = reduce(lambda x, y: x + y, detailed_dict['hops'].values())  # keeping the hop ordinal
        if len(all_nodes) != len(set(all_nodes)):
            raise Exception
        detailed_dict['all'] = all_nodes

        # some indicator
        hop_indicator = [0]
        for i in range(self.hop):
            hop_indicator.extend([i + 1] * len(detailed_dict['hops'][i]))
        tgt_indicator = [1] + [0] * (len(all_nodes) - 1)
        cite_indicator = tgt_indicator.copy()
        ref_indicator = tgt_indicator.copy()
        for idx in range(1, len(all_nodes)):
            node = all_nodes[idx]
            if node in detailed_dict['ref']:
                ref_indicator[idx] = 1
            if node in detailed_dict['cite']:
                cite_indicator[idx] = 1

        detailed_dict['indicators'] = {}
        detailed_dict['indicators']['hop'] = hop_indicator
        detailed_dict['indicators']['is_target'] = tgt_indicator
        detailed_dict['indicators']['is_cite'] = cite_indicator
        detailed_dict['indicators']['is_ref'] = ref_indicator
        del detailed_dict['ref'], detailed_dict['cite']
        return detailed_dict

    def get_single_subgraph(self, paper):
        indicators = ['hop', 'is_target', 'is_ref', 'is_cite']
        if paper is not None:
            detailed_dict = self.get_valid_subgraph(paper, self.paper_subgraph)
            all_nodes = detailed_dict['all']
            if not self.return_graph:
                output = all_nodes
                if self.detailed:
                    output = detailed_dict
            else:
                output = dgl.node_subgraph(self.paper_subgraph, nodes=all_nodes)
                for indicator in indicators:
                    output.ndata[indicator] = torch.tensor(detailed_dict['indicators'][indicator], dtype=torch.long)
            return output
        else:
            if not self.return_graph:
                output = []
                if self.detailed:
                    output = {}
            else:
                output = dgl.node_subgraph(self.paper_subgraph, nodes=[])
                for indicator in indicators:
                    output.ndata[indicator] = torch.zeros(0, dtype=torch.long)
            return output


class DealtKhopSubgraph(SingleKhopSubgraph):
    def __init__(self, origin_graph, index_dict, cur_time,
                 hop=2, max_nodes=(100, 20), return_graph=False):
        paper_subgraph = dgl.node_type_subgraph(origin_graph, ntypes=['paper'])
        super(DealtKhopSubgraph, self).__init__(paper_subgraph, hop, True, max_nodes, return_graph)
        self.origin_graph = origin_graph
        self.cur_time = cur_time
        self.max_nodes = max_nodes
        self.index_dict = index_dict
    def get_filter_subgraph(self, subgraph_info):
        pub_time, subgraph = subgraph_info
        if pub_time > self.cur_time:
            # output = None
            output = {'nodes': {'paper': []}, 'indicators': {}}
            indicators = ['hop', 'is_target', 'is_ref', 'is_cite']
            for indicator in indicators:
                output['indicators'][indicator] = []
            if self.return_graph:
                output = dgl.node_subgraph(self.origin_graph, nodes={'paper': []})
        else:
            # print('-'*59)
            start_time = time.time()
            # paper_subgraph = dgl.node_type_subgraph(subgraph, ntypes=['paper'])
            time_indicator = subgraph.nodes['paper'].data['time'].squeeze(dim=-1) <= self.cur_time
            valid_nodes = torch.where(time_indicator)[0]
            subgraph = dgl.node_subgraph(subgraph, nodes=valid_nodes)
            detailed_dict = self.get_valid_subgraph(0, graph=subgraph, index_dict=self.index_dict)
            cur_time = time.time()
            # print('time subgraph:', cur_time - start_time)

            # print(detailed_dict)
            all_oids = subgraph.ndata['oid'][detailed_dict['all']].numpy()
            all_papers = [self.index_dict['paper'][paper] for paper in all_oids]
            temp_graph = dgl.in_subgraph(self.origin_graph, nodes={'paper': all_papers}, relabel_nodes=True)
            # print('in subgraph:', time.time() - start_time)
            # journals = temp_graph.nodes('journal').numpy().tolist()
            # authors = temp_graph.nodes('author').numpy().tolist()
            journals = temp_graph.nodes['journal'].data[dgl.NID].numpy().tolist()
            authors = temp_graph.nodes['author'].data[dgl.NID].numpy().tolist()
            times = temp_graph.nodes['time'].data[dgl.NID].numpy().tolist()
            if self.return_graph:
                # indicators = ['hop', 'is_target', 'is_ref', 'is_cite']
                cur_graph = dgl.node_subgraph(self.origin_graph,
                                              nodes={'paper': all_papers, 'journal': journals,
                                                     'author': authors, 'time': times})
                for indicator in detailed_dict['indicators']:
                    cur_graph.nodes['paper'].data[indicator] = torch.tensor(detailed_dict['indicators'][indicator],
                                                                            dtype=torch.long)
                output = cur_graph
            else:
                detailed_dict['nodes'] = {'paper': all_papers, 'journal': journals, 'author': authors, 'time': times}
                output = detailed_dict
                del detailed_dict['hops']
                del detailed_dict['all']
        return output


def saved_pyg_load_graphs(graphs):
    for phase in graphs['data']:
        cur_graph = graphs['all_graphs'][graphs['time'][phase]]
        cur_graph = dgl.add_reverse_edges(cur_graph[('paper', 'cites', 'paper')])
        # edge_index = torch.stack(cur_graph.edges(), dim=0)
        # X = cur_graph.ndata['h']
        # cur_graph = Data(x=X, edge_index=edge_index)
        graphs['data'][phase] = [cur_graph] + graphs['data'][phase]
    del graphs['all_graphs']
    return graphs


def saved_pyg_deal_graphs(graphs, hop, data_path=None, path=None, log_path=None):
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
    data = torch.load(data_path)

    id_trans = {}
    for pub_time in graphs['all_graphs']:
        cur_graph = graphs['all_graphs'][pub_time]
        oids = cur_graph.nodes['paper'].data['oid'].numpy().tolist()
        nids = cur_graph.nodes('paper').numpy().tolist()
        id_trans[pub_time] = dict(zip(oids, nids))

    phases = list(graphs['data'].keys())
    dealt_graphs = {'data': {}, 'node_trans': graphs['node_trans']}
    for phase in phases:
        cur_graph = graphs['data'][phase][-1]

        selected_papers = [graphs['node_trans']['paper'][paper] for paper in data[phase][0]]
        all_trans_index = dict(zip(data[phase][0], range(len(selected_papers))))
        cur_graph = dgl.node_type_subgraph(cur_graph, ntypes=['paper'])
        cur_last_time = graphs['time'][phase]
        selected_papers = [id_trans[cur_last_time][paper] for paper in selected_papers]
        paper_subgraph = dgl.node_type_subgraph(cur_graph, ntypes=['paper'])
        sampler = SingleKhopSubgraph(paper_subgraph, hop=hop, detailed=False, return_graph=False)
        with ThreadPoolExecutor(max_workers=4) as executor:
            all_indexes = list(tqdm(executor.map(sampler.get_single_subgraph, selected_papers),
                                    total=len(selected_papers), file=tqdm_out, mininterval=30))
        print(all_indexes[:5])
        if path:
            # joblib.dump([cur_graph, all_indexes, all_trans_index], path + '_{}.job'.format(phase))
            joblib.dump([all_indexes, all_trans_index], path + '_{}.job'.format(phase))
    if path:
        # torch.save(dealt_graphs, path)
        # joblib.dump(dealt_graphs, path)
        joblib.dump([], path)
    return dealt_graphs


def saved_load_graphs(graphs):
    # for pub_time in graphs['all_graphs']:
    #     cur_graph = graphs['all_graphs'][pub_time]
    #     cur_graph = dgl.add_reverse_edges(cur_graph[('paper', 'cites', 'paper')])
    #     graphs['all_graphs'][pub_time] = cur_graph
    for phase in graphs['data']:
        cur_graph = graphs['all_graphs'][graphs['time'][phase]]
        cur_graph = dgl.add_reverse_edges(cur_graph[('paper', 'cites', 'paper')])
        graphs['data'][phase] = [cur_graph] + graphs['data'][phase]
    del graphs['all_graphs']
    return graphs


def saved_deal_graphs(graphs, hop, data_path=None, path=None, log_path=None):
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
    split_data = torch.load(data_path)

    phases = list(graphs['data'].keys())
    dealt_graphs = {'data': {}, 'node_trans': graphs['node_trans']}
    for phase in phases:
        cur_graph = graphs['data'][phase][-1]
        cur_graph = dgl.node_type_subgraph(cur_graph, ntypes=['paper'])
        # cur_graph = cur_graph['paper', 'cites', 'paper']
        # cur_graph = dgl.add_reverse_edges(cur_graph)
        phase_node_trans = dict(zip(cur_graph.ndata[dgl.NID].numpy().tolist(), cur_graph.nodes().numpy().tolist()))
        selected_papers = split_data[phase][0]
        all_trans_index = dict(zip(selected_papers, range(len(selected_papers))))
        selected_papers = [graphs['node_trans']['paper'][paper] for paper in split_data[phase][0]]
        selected_papers = [phase_node_trans[paper] for paper in selected_papers]
        selected_values = split_data[phase][1]
        print(selected_values[:10])
        sampler = SingleKhopSubgraph(cur_graph, hop=hop, detailed=False, return_graph=False)
        with ThreadPoolExecutor(max_workers=4) as executor:
            all_subgraphs = list(tqdm(executor.map(sampler.get_single_subgraph, selected_papers),
                                      total=len(selected_papers), file=tqdm_out, mininterval=30))
        print(all_subgraphs[:5])
        # cur_graph = dgl.add_reverse_edges(cur_graph[('paper', 'cites', 'paper')])
        if path:
            # joblib.dump([cur_graph, all_subgraphs, all_trans_index], path + '_{}.job'.format(phase))
            joblib.dump([all_subgraphs, all_trans_index], path + '_{}.job'.format(phase))
        # dealt_graphs['data'][phase] = [dealt_graph_list, phase_node_trans]
        # dealt_graphs['data'][phase] = [A_list, node_list, mask_list, phase_node_trans]

    if path:
        # torch.save(dealt_graphs, path)
        # joblib.dump(dealt_graphs, path)
        joblib.dump([], path)
    return dealt_graphs


def saved_dynamic_load_graphs(graphs, time_length):
    print(graphs['graphs_path'])
    for pub_time in tqdm(graphs['all_graphs']):
        cur_graph = graphs['all_graphs'][pub_time]
        cur_graph = dgl.add_reverse_edges(cur_graph[('paper', 'cites', 'paper')])
        graphs['all_graphs'][pub_time] = cur_graph

    for phase in graphs['data']:
        time_span = range(graphs['time'][phase] - time_length + 1, graphs['time'][phase] + 1)
        cur_graphs = [graphs['all_graphs'][pub_time] for pub_time in time_span]
        all_graphs, all_trans_index = graphs['data'][phase]
        graphs['data'][phase] = [cur_graphs, all_graphs, all_trans_index]
    return graphs


def saved_dynamic_deal_graphs(graphs, hop, time_length, data_path=None, path=None, log_path=None):
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
    # if hop != 2:
    #     path += '_h{}'.format(hop)

    phases = list(graphs['data'].keys())
    id_trans = {}
    data = torch.load(data_path)
    for pub_time in graphs['all_graphs']:
        cur_graph = graphs['all_graphs'][pub_time]
        oids = cur_graph.nodes['paper'].data['oid'].numpy().tolist()
        nids = cur_graph.nodes('paper').numpy().tolist()
        id_trans[pub_time] = dict(zip(oids, nids))

    for phase in phases:
        cur_last_time = graphs['time'][phase]
        last_graph = graphs['all_graphs'][cur_last_time]
        paper_subgraph = dgl.node_type_subgraph(last_graph, ntypes=['paper'])
        selected_papers = [graphs['node_trans']['paper'][paper] for paper in data[phase][0]]

        all_trans_index = dict(zip(data[phase][0], range(len(selected_papers))))

        selected_papers = [id_trans[cur_last_time][paper] for paper in selected_papers]
        sampler = SingleKhopSubgraph(paper_subgraph, hop=hop, detailed=True, return_graph=True)
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     # all_subgraphs = list(tqdm(executor.map(sampler.get_single_subgraph, selected_papers),
        #     #                           total=len(selected_papers), file=tqdm_out, mininterval=30))
        #     all_subgraphs = list(tqdm(executor.map(sampler.get_single_subgraph, selected_papers),
        #                               total=len(selected_papers)))
        all_subgraphs = [sampler.get_single_subgraph(paper)
                         for paper in tqdm(selected_papers, total=len(selected_papers))]

        all_graphs = []
        # for sampled_graph in tqdm(all_subgraphs, file=tqdm_out, mininterval=30):
        for sampled_graph in tqdm(all_subgraphs):
            # print(sampled_graph.ndata[dgl.NID])
            # print(sampled_graph.ndata['oid'])
            cur_nodes = sampled_graph.ndata['oid'].numpy().tolist()
            cur_times = sampled_graph.ndata['time'].squeeze(dim=-1).numpy().tolist()
            cur_nodes, cur_times = zip(*sorted(zip(cur_nodes, cur_times), key=lambda x: x[1]))
            # print(cur_times[:5])
            cur_nodes = np.array(cur_nodes)
            cur_times = np.array(cur_times)

            sampled_graphs = []
            for pub_time in range(graphs['time'][phase] - time_length + 1, graphs['time'][phase] + 1):
                selected_nodes = cur_nodes[cur_times <= pub_time]
                # print(selected_nodes[:5])
                selected_nodes = [id_trans[pub_time][node] for node in selected_nodes]
                sampled_graphs.append(selected_nodes)
            all_graphs.append(sampled_graphs)

        if path:
            joblib.dump([all_graphs, all_trans_index], path + '_{}.job'.format(phase))

    if path:
        # torch.save(dealt_graphs, path)
        # joblib.dump(dealt_graphs, path)
        joblib.dump([], path)
    return {'data': {}, 'time': graphs['time'], 'node_trans': graphs['node_trans'], 'all_graphs': None}


def get_pareto_dist(nums):
    df = pd.DataFrame({'num': nums}).sort_values(by='num', ascending=False)
    print(df.shape)
    df['percent'] = df['num'] / np.sum(df['num'])
    df['cum_sum'] = np.cumsum(df['percent'])
    cut_point = df[df['cum_sum'] >= 0.8]['num'].values[0]
    l = df[df['num'] >= cut_point]['percent'].sum()
    s = df[df['num'] > cut_point]['percent'].sum()
    print(l, s)
    if l - 0.8 > 0.8 - s:
        cut_point = df[df['num'] > cut_point]['num'].values[-1]

    print(cut_point, df.shape[0] - df[df['cum_sum'] > 0.8].shape[0])
    return cut_point


def box_cut(values, n=5):
    cut_points = []
    percents = [i * 1 / n for i in range(n + 1)]
    cut_points += [values.quantile(percents[0]) - 1]
    for percent in percents[1:]:
        cut_points.append(values.quantile(percent))
    if len(set(cut_points)) != len(cut_points):
        return box_cut(values, n - 1)
    else:
        print(cut_points)
        return pd.cut(values, cut_points, labels=[i for i in range(n)])


def mm_norm(values, invalid_value=-1, log=False):
    values = values.astype(np.float32)
    if invalid_value is not None:
        valid_index = (values != invalid_value)
    else:
        valid_index = range(values.shape[0])
    if log:
        values[valid_index] = np.log(values[valid_index] + 1)
    max_value = np.max(values[valid_index])
    min_value = np.min(values[valid_index])
    print(min_value, max_value)
    values[valid_index] = (values[valid_index] - min_value) / (max_value - min_value)
    return values


def excellent_label(values, invalid_value=-1, p=0.9, log=False):
    if invalid_value is not None:
        valid_index = (values != invalid_value)
    else:
        valid_index = range(values.shape[0])
    if log:
        values[valid_index] = np.log(values[valid_index] + 1)
    cut_value = np.percentile(values[valid_index], p * 100)
    print(cut_value)
    values[valid_index] = np.where(values[valid_index] >= cut_value, 1, 0)
    return values


class SimLoss(nn.Module):
    def __init__(self, seq_len=None, method='cos_sim'):
        super(SimLoss, self).__init__()
        if seq_len:
            self.indexes = self.get_indexes(seq_len)
        else:
            self.indexes = None
        self.method = method

    def get_indexes(self, seq_len):
        mask = torch.triu(1 - torch.eye(seq_len))
        return torch.where(mask)

    def forward(self, seq_t):
        # [L, *, h]
        if self.indexes:
            a, b = self.indexes
        else:
            a, b = self.get_indexes(seq_t.shape[0])
        if self.method == 'cos_sim':
            # print(seq_t.shape)
            norm = torch.linalg.vector_norm(seq_t, dim=-1)
            norm_l = (norm[a] * norm[b])
            # print(norm[a].shape)
            seq_l = (seq_t[a] * seq_t[b]).sum(dim=-1)
            # print(seq_l.shape, norm_l.shape)
            return seq_l / norm_l
        elif self.method == 'orthog':
            norm = torch.linalg.vector_norm(seq_t, dim=-1)
            norm_l = (norm[a] * norm[b])
            seq_l = (seq_t[a] * seq_t[b]).sum(dim=-1) ** 2
            # print(norm_l.shape, seq_l.shape)
            return seq_l / norm_l
        elif self.method == 'simple_orthog':
            return torch.linalg.vector_norm((seq_t[a] * seq_t[b]).sum(dim=-1) - 1, dim=-1)
        elif self.method == 'dcor':
            print(self.indexes)
            return _new_create_distance_correlation(seq_t, self.indexes)
        elif self.method == 'mmd_rbf':
            # print('here')
            return mmd_loss(seq_t[self.indexes[0]], seq_t[self.indexes[1]], kernel='rbf')
        elif self.method == 'mmd_ms':
            return mmd_loss(seq_t[self.indexes[0]], seq_t[self.indexes[1]], kernel='multiscale')


def _new_create_distance_correlation(X, indexes):
    # calculate the dCor
    # X [C, B, h]
    def _create_new_centered_distance(X):
        # [C, B, h]
        '''
            Used to calculate the distance matrix of N samples
        '''
        # calculate the pairwise distance of X
        # .... A with the size of [batch_size, embed_size/n_factors]
        # .... D with the size of [batch_size, batch_size]
        # X = tf.math.l2_normalize(XX, axis=1)
        r = torch.sum(torch.square(X), dim=-1, keepdims=True)
        w = torch.matmul(X.unsqueeze(-2), X.unsqueeze(-1)).squeeze(-1)
        D = torch.sqrt(torch.maximum(r - 2 * w + r.transpose(-1, -2), torch.tensor(0.0)) + 1e-8)
        # # calculate the centered distance of X
        # # .... D with the size of [batch_size, batch_size]
        D = D - torch.mean(D, dim=-2, keepdims=True) - torch.mean(D, dim=-1, keepdims=True) \
            + D.mean(-1, keepdims=True).mean(-2, keepdims=True)
        return D

    def _create_new_distance_covariance(D1, D2):
        # [C, B, B]
        # calculate distance covariance between D1 and D2
        n_samples = D1.shape[-2]
        # print((D1 * D2).shape)
        dcov = torch.sqrt(torch.maximum((D1 * D2).sum(-1).sum(-1) / (n_samples * n_samples), torch.tensor(0.0)) + 1e-8)
        # dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2)) / n_samples)
        return dcov

    # print(X)
    Ds = _create_new_centered_distance(X)  # [C, B, B]
    # pair-wised
    dcov_pair = _create_new_distance_covariance(Ds[indexes[0]], Ds[indexes[1]])
    # self
    eye_indexes = torch.where(torch.eye(X.shape[0]))
    dcov_self = _create_new_distance_covariance(Ds[eye_indexes[0]], Ds[eye_indexes[1]])

    # calculate the distance correlation
    # print(dcov_pair, dcov_self[indexes[0]], dcov_self[indexes[1]])
    # print(dcov_pair.shape, dcov_self.shape)
    dcor = dcov_pair / (torch.sqrt(torch.maximum(dcov_self[indexes[0]] * dcov_self[indexes[1]],
                                                 torch.tensor(0.0))) + 1e-10)

    # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
    return dcor


def mmd_loss(x, y, kernel='rbf'):
    xx, yy, zz = torch.matmul(x, x.transpose(-1, -2)), \
                 torch.matmul(y, y.transpose(-1, -2)), \
                 torch.matmul(x, y.transpose(-1, -2))
    rx = (torch.diagonal(xx, dim2=-2, dim1=-1).unsqueeze(-2).expand_as(xx))
    ry = (torch.diagonal(yy, dim2=-2, dim1=-1).unsqueeze(-2).expand_as(yy))
    # print(xx, yy, zz)

    dxx = rx.transpose(-1, -2) + rx - 2. * xx  # Used for A in (1)
    dyy = ry.transpose(-1, -2) + ry - 2. * yy  # Used for B in (1)
    dxy = rx.transpose(-1, -2) + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return (XX + YY - 2. * XY).mean(dim=-1).mean(dim=-1)


def custom_clip_grad_norm_(
        parameters, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        print(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, will set to zero before clipping .  ')

        for p in parameters:
            p_grad_ = p.grad.detach()
            nan_idxs = torch.isnan(p_grad_)
            inf_idxs = torch.isinf(p_grad_)
            p_grad_[nan_idxs] = 0
            p_grad_[inf_idxs] = 0
        return custom_clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite)

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm


def get_ranking_ndcg(before_ranking_array, gain_array):
    # before_ranking_list is the predicted values
    # gain_list is the true values
    # to show the ranking performance for discovering the breakout paper

    # dcg
    ranking_index = np.argsort(before_ranking_array)[::-1]
    pos = 1 / np.log2(np.array(range(before_ranking_array.shape[0])) + 2)
    dcg = np.sum(gain_array[ranking_index] * pos)
    # print(dcg)

    # idcg
    sorted_gain = np.sort(gain_array)[::-1]
    idcg = np.sum(sorted_gain * pos)
    # print(idcg)

    ndcg = dcg / idcg
    return ndcg


def get_shared_graph(graph, shared_name):
    ndata = defaultdict(dict)
    edata = defaultdict(dict)
    # get shared attr tensor
    for ntype in graph.ntypes:
        for node_attr in graph.nodes[ntype].data:
            ndata[ntype][node_attr] = graph.nodes[ntype].data[node_attr].share_memory_()
    for etype in graph.etypes:
        for edge_attr in graph.edges[etype].data:
            edata[etype][edge_attr] = graph.edges[etype].data[edge_attr].share_memory_()
    # get shared graph and attr
    graph = graph.shared_memory(shared_name)
    for ntype in graph.ntypes:
        for node_attr in ndata[ntype]:
            graph.nodes[ntype].data[node_attr] = ndata[ntype][node_attr]
    for etype in graph.etypes:
        for edge_attr in edata[etype]:
            graph.edges[etype].data[edge_attr] = edata[etype][edge_attr]
    del ndata, edata
    return graph


def get_model_config(configs, cur_model, args):
    model_config = configs[cur_model]
    # model_config['model_name'] = cur_model
    model_config['data_source'] = args.data_source
    if args.model_seed:
        model_config['model_seed'] = int(args.model_seed)
    # =============================================== args ===========================================
    model_config['graph_type'] = None
    model_config['aug_type'] = None
    if args.graph_type:
        model_config['graph_name'] = '_'.join(model_config['graph_name'].split('_')[:-1]
                                              + [args.graph_type.split('+')[0]])
        model_config['graph_type'] = args.graph_type
    if args.emb_type:
        model_config['emb_name'] = '_'.join(model_config['graph_name'].split('_')[:-1]
                                            + [args.emb_type.split('+')[0]])
        model_config['emb_type'] = args.emb_type
    if args.pred_type:
        model_config['pred_type'] = args.pred_type
    if args.cl_type:
        model_config['cl_type'] = args.cl_type
    if args.encoder_type:
        model_config['encoder_type'] = args.encoder_type
    if args.aug_type:
        model_config['aug_type'] = args.aug_type
    if args.aug_rate:
        model_config['aug_rate'] = float(args.aug_rate)
    if args.gcn_out:
        model_config['gcn_out'] = args.gcn_out
    if args.n_layers:
        model_config['n_layers'] = int(args.n_layers)
    if args.lr:
        model_config['lr'] = float(args.lr)
    if args.optimizer:
        model_config['optimizer'] = args.optimizer
    if args.edge_seq:
        model_config['edge_sequences'] = args.edge_seq
    if args.inter_mode:
        model_config['inter_mode'] = args.inter_mode
    if args.time_length:
        model_config['time_length'] = int(args.time_length)
    if args.hop:
        model_config['hop'] = int(args.hop)
    if args.tau:
        model_config['tau'] = float(args.tau)
    if args.cl_w:
        model_config['cl_weight'] = float(args.cl_w)
    if args.aux_w:
        model_config['aux_weight'] = float(args.aux_w)
    if args.type:
        model_config['model_type'] = args.type

    if args.graph_type in {'sbert', 'random'}:
        model_config['embed_dim'] = 384
    elif args.graph_type == 'specter2':
        model_config['embed_dim'] = 768
    if args.emb_type == 'sbert':
        model_config['bert_dim'] = 384
    else:
        model_config['bert_dim'] = 768

    if args.hidden_dim:
        model_config['hidden_dim'] = int(args.hidden_dim)
    if args.sem:
        model_config['snapshot_mode'] = args.sem
    if args.topic_module:
        model_config['topic_module'] = args.topic_module
    if args.pop_module:
        model_config['pop_module'] = args.pop_module
    if args.time_pooling:
        model_config['time_pooling'] = args.time_pooling
    if args.time_layers:
        model_config['time_layers'] = int(args.time_layers)
    if args.num_workers is not None:
        model_config['num_workers'] = int(args.num_workers)
    if args.prefetch_factor:
        model_config['prefetch_factor'] = int(args.prefetch_factor)
    if args.n_topics:
        model_config['n_topics'] = int(args.n_topics)
    if args.focus is not None:
        model_config['focus'] = bool(args.focus)
    if args.dn is not None:
        model_config['dropout'] = float(args.dn)
    if args.gdn is not None:
        model_config['graph_dropout'] = float(args.gdn)
    if args.batch_size:
        model_config['batch_size'] = int(args.batch_size)
    if args.epochs:
        model_config['epochs'] = int(args.epochs)
    if args.use_amp:
        args.use_amp = bool(args.use_amp)
    if args.ac:
        model_config['ablation_channel'] = args.ac.split('+')
    else:
        model_config['ablation_channel'] = ()
    model_config['tree'] = bool(args.tree)
    model_config['residual'] = bool(args.residual)
    model_config['graph_norm'] = args.gn
    model_config['mixed'] = not bool(args.unmixed)
    model_config['updater'] = str(args.updater)
    model_config['adaptive_lr'] = bool(args.adaptive_lr)
    model_config['use_constraint'] = args.use_constraint
    model_config['lambda_weight'] = float(args.lw)
    model_config['loss_weights'] = {
        'topic': float(args.tw),
        'pop': float(args.pw),
        'disen': float(args.dw)
    }
    model_config['etc'] = args.etc

    # special args
    print(args.oargs)
    if args.oargs:
        if 'lbg' in args.oargs:
            model_config['linear_before_gcn'] = True
        int_list = [arg.split('=') for arg in args.oargs.split(',') if '=' in arg]
        for key, value in int_list:
            model_config[key] = int(value)
        str_list = [arg.split(':') for arg in args.oargs.split(',') if ':' in arg]
        for key, value in str_list:
            model_config[key] = value

    # =============================================== args ===========================================
    return model_config