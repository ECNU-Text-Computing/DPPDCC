import argparse
import datetime
import json
import math
import os
import random
import logging
from functools import reduce

import dgl
import joblib
import numpy as np
import pandas as pd
import torch_geometric.data

from torch import nn
import time
import torch
import torch.nn.functional as F
import re

from torch.cuda.amp import GradScaler
from tqdm import tqdm

from models.base_model import BaseModel
from utilis.scripts import eval_result, result_format, eval_aux_results, aux_result_format, ResultsContainer, \
    custom_clip_grad_norm_, reg_metrics, aux_metrics
from trainers.model_factory import get_model
import torch.distributed as dist
from transformers.utils import ModelOutput

# self.local_rank = int(os.environ.get("self.local_rank", -1))
# print('》》》local rank:', self.local_rank)
# torch.cuda.set_device(self.local_rank)


class BaseTrainer:
    def __init__(self, model_name, config, vectors, use_amp=False, **kwargs):
        # seed, aux, interval
        self.local_rank = dist.get_rank() if dist.is_initialized() else -1
        if dist.is_initialized():
            device = torch.device(f"cuda:{self.local_rank}")
            model = get_model(model_name, config, vectors, device, **kwargs)
            self.model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[self.local_rank],
                                                                   output_device=self.local_rank,
                                                                   find_unused_parameters=True)
            # print(self.model)
            # print(self.model.module)
            self.model_name = self.model.module.model_name + '_ddp'
            # self.model = self.dist_model(model_name, config, vectors, **kwargs)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = get_model(model_name, config, vectors, device, **kwargs)
            self.model = model.to(device)
            self.model_name = self.model.model_name

        self.cur_model = self.model.module if dist.is_initialized() else self.model

        self.adaptive_lr = False
        self.warmup = False
        self.T = kwargs['T'] if 'T' in kwargs.keys() else 400
        self.seed = None
        self.epoch = 0
        self.relu_out = nn.ReLU()
        # self.relu_out = nn.ELU()
        # print('use elu instead of relu')
        self.test_interval = None
        if 'seed' in kwargs.keys():
            self.seed = kwargs['seed']
        if self.warmup:
            print('warm up T', self.T)
        self.aux_weight = kwargs.get('aux_weight', 1)
        self.aux_criterion = nn.CrossEntropyLoss()
        # self.model.aux_criterion = self.aux_criterion
        self.cut_threshold = kwargs.get('cut_threshold', [0, 10])
        print(self.cut_threshold)
        self.eval_interval = kwargs.get('eval_interval', 5)
        print('>>>eval interval:', self.eval_interval)
        if use_amp:
            self.scaler = GradScaler(init_scale=2.**16, enabled=True)
            # self.scaler = GradScaler(init_scale=2. ** 12, enabled=True, growth_interval=100)
            self.model_name += '_amp'
        else:
            self.scaler = None
        self.use_amp = use_amp

    def dist_model(self, model_name, config, vectors, **kwargs):
        # for shared model
        assert dist.is_initialized()
        self.local_rank = dist.get_rank()
        device = torch.device(f"cuda:{self.local_rank}")
        model = get_model(model_name, config, vectors, device, **kwargs)
        self.model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[self.local_rank],
                                                               output_device=self.local_rank,
                                                               find_unused_parameters=True)
        self.model_name += '_sddp'
        self.cur_model = self.model.module
        return self.model

    def deal_graphs(self, graphs, data_path=None, path=None, log_path=None):
        return self.cur_model.deal_graphs(graphs, data_path, path, log_path)

    def load_graphs(self, graphs):
        return self.cur_model.load_graphs(graphs)

    def get_batch_input(self, batch, graph):
        if dist.is_initialized():
            return self.model.module.get_batch_input(batch, graph)
        else:
            return self.model.get_batch_input(batch, graph)

    def optimizer_zero_grad(self):
        # self.optimizer.zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def optimizer_step(self):
        # self.optimizer.step()
        for optimizer in self.optimizers:
            optimizer.step()

    def batch_process(self, x, values, lengths, masks, ids, times, cur_graphs, dataloader, aux):
        # self.optimizer_zero_grad()
        if self.model.training:
            predicted_values, aux_output, other_loss = self.model(x, lengths, masks, ids, cur_graphs, times)
        else:
            predicted_values, aux_output, other_loss = self.cur_model.predict(x, lengths, masks, ids, cur_graphs, times)
        # aux
        loss = self.get_reg_loss_and_result(values, predicted_values, self.criterion,
                                            log=dataloader.log,
                                            mean=dataloader.mean, std=dataloader.std)
        if aux:
            aux_loss = self.get_aux_loss_and_result(values, aux_output)
            loss = loss + aux_loss

        if other_loss is not None:
            loss = loss + other_loss

        if self.model.training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer_step()
            self.optimizer_zero_grad()

        return loss

    def train_model(self, dataloader, epoch, criterion, optimizer, graph=None, aux=False):
        """
        training model for each epoch
        """
        self.model.train()
        self.phase = 'train'
        self.epoch = epoch
        self.cur_model.phase = 'train'
        self.cur_model.epoch = epoch
        log_interval = 100
        self.container = ResultsContainer(log_interval, len(dataloader), epoch, self.model_name, 'train',
                                          test_interval=self.test_interval)

        cur_graphs = None
        if graph:
            # load graphs_data
            self.cur_model.node_trans = graph['node_trans']
            cur_graphs = graph['data']['train']
            cur_graphs = self.cur_model.graph_to_device(cur_graphs)

        for idx, batch in enumerate(dataloader):
            x, values, lengths, masks, ids, times, blocks = self.get_batch_input(batch, graph)
            if blocks:
                cur_graphs = blocks
            loss = self.batch_process(x, values, lengths, masks, ids, times, cur_graphs, dataloader, aux)
            if self.local_rank in [0, -1]:
                self.container.get_interval_result(loss, idx)

        results = self.container.get_epoch_result(aux)

        return results

    def get_reg_loss_and_result(self, values, predicted_values, criterion, log, mean=None, std=None):
        '''
        reg loss and results
        :param values:
        :param predicted_values:
        :param criterion:
        :param log:
        :param mean:
        :param std:
        :return:
        '''
        true_values = values.unsqueeze(dim=-1)
        if self.relu_out:
            predicted_values = self.relu_out(predicted_values)
        if log:
            # loss = criterion(predicted_values, (true_values + 1).float().log())
            loss = criterion(predicted_values, (true_values + 1).log())
        else:
            loss = criterion(predicted_values, (true_values - mean) / std)

        print_loss = loss.sum(dim=0).item()
        self.container.all_loss += print_loss

        if log:
            # predicted_values = predicted_values.detach().float().cpu().exp() - 1
            predicted_values = predicted_values.detach().cpu().exp() - 1
        else:
            # predicted_values = predicted_values.detach().float().cpu() * std + mean
            predicted_values = predicted_values.detach().cpu() * std + mean
        # print(predicted_values.squeeze(dim=-1))
        self.container.all_predicted_values.append(predicted_values)
        self.container.all_true_values.append(true_values.cpu())

        return loss

    def get_aux_loss_and_result(self, values, aux_output, epoch=None):
        '''
        aug loss and results
        :param values:
        :param aux_output:
        :return:
        '''
        new_gt = self.get_new_gt(values)
        # predicted_values, aux_output = predicted_values
        aux_loss = self.cur_model.get_aux_loss(self.aux_criterion, self.aux_weight, aux_output, new_gt, epoch)
        # all_aux_loss = all_aux_loss + aux_loss.item()
        # self.cur_model.other_loss = self.cur_model.other_loss + aux_loss
        # self.other_loss = aux_loss
        predicted_aux_values = self.cur_model.get_aux_pred(aux_output)
        # self.container.all_predicted_aux_values.append(predicted_aux_values.detach().float().cpu())
        # self.container.all_true_aux_values.append(new_gt.detach().float().cpu())
        self.container.all_predicted_aux_values.append(predicted_aux_values.detach().cpu())
        self.container.all_true_aux_values.append(new_gt.detach().cpu())
        self.container.all_aux_loss = self.container.all_aux_loss + aux_loss.item()
        return aux_loss

    def get_new_gt(self, gt):
        new_gt = torch.zeros_like(gt, dtype=torch.long)
        mid_range = gt > self.cut_threshold[0]
        last_range = gt > self.cut_threshold[1]
        new_gt[mid_range ^ last_range] = 1
        new_gt[last_range] = 2
        return new_gt

    def get_optimizer(self, lr, optimizer, weight_decay=1e-3, use_amp=False):
        # optimizers = self.cur_model.get_optimizer(lr, optimizer, weight_decay, autocast)
        optimizers = self.cur_model.get_optimizer(self.model, lr, optimizer, weight_decay, use_amp)
        self.optimizers = optimizers
        return optimizers

    def get_criterion(self, criterion):
        if criterion == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif criterion == 'MSE':
            criterion = nn.MSELoss()
        self.criterion = criterion
        self.cur_model.criterion = criterion
        return criterion

    def get_metric_and_record(self, aux, record_path, phase='train'):
        if aux:
            self.model_name += '_aux'
            if self.aux_weight != 0.5:
                self.model_name += '_aw{}'.format(self.aux_weight)

        records_logger = logging.getLogger('records')
        # records_logger.setLevel(logging.DEBUG)
        if self.seed:
            print('{}_records_{}.csv'.format(self.model_name, self.seed))
            # fw = open(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), 'w')
            # fh = logging.FileHandler(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), mode='w+')
            record_name = '{}_records_{}.csv'.format(self.model_name, self.seed)

        else:
            # fw = open(record_path + '{}_records.csv'.format(self.model_name), 'w')
            # fh = logging.FileHandler(record_path + '{}_records.csv'.format(self.model_name), mode='w+')
            record_name = '{}_records.csv'.format(self.model_name)
        # if aux:
        #     record_name = record_name.replace('.csv', '_aux.csv')
        if phase == 'test':
            record_name = record_name.replace('.csv', '_test.csv')

        print('record name:', record_name)
        fh = logging.FileHandler(record_path + record_name, mode='w+')
        records_logger.handlers = []
        records_logger.handlers = []
        records_logger.addHandler(fh)
        # metrics = 'loss, acc, mae, r2, mse, rmse, Mrse, mrse, male, log_r2, msle, rmsle, smape, mape'.split(', ')
        # if aux:
        #     metrics += 'aux_loss, acc, roc_auc, log_loss_value, ' \
        #                'micro_prec, micro_recall, micro_f1, ' \
        #                'macro_prec, macro_recall, macro_f1'.split(', ')
        metrics = reg_metrics.copy()
        if aux:
            metrics += aux_metrics
        # print(metrics)
        raw_header = ',' + ','.join(['{}_' + metric for metric in metrics])
        if phase == 'train':
            records_logger.warning('epoch' + raw_header.replace('{}', 'train')
                                   + raw_header.replace('{}', 'val')
                                   + raw_header.replace('{}', 'test'))
        return metrics, records_logger

    def train_batch(self, dataloaders, epochs, lr=1e-3, weight_decay=1e-4, criterion='MSE', optimizer='ADAM',
                    scheduler=False, record_path=None, save_path=None, graph=None, test_interval=1, best_metric='male',
                    aux=False, sampler=None):
        """
        training process
        """
        final_results = []
        train_dataloader, val_dataloader, test_dataloader = dataloaders

        criterion = self.get_criterion(criterion)
        # encoder_optimizer, decoder_optimizer= self.get_optimizer(lr, optimizer)
        optimizer = self.get_optimizer(lr, optimizer, weight_decay=weight_decay, use_amp=self.use_amp)
        metrics, records_logger = self.get_metric_and_record(aux, record_path)

        interval_best = None
        all_interval_best = None
        eval_interval = test_interval if test_interval > 0 else self.eval_interval
        self.test_interval = test_interval
        self.record_path = record_path
        for epoch in range(1, epochs + 1):
            # self.epoch = epoch
            # self.model.epoch = epoch
            if dist.is_initialized():
                sampler.set_epoch(epoch)
            logging.basicConfig(level=logging.INFO,
                                filename=record_path + '{}_epoch_{}.log'.format(self.model_name, epoch),
                                filemode='w+',
                                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                force=True)
            # dealt_graphs = self.load_graphs(graph)
            train_results = self.train_model(train_dataloader, epoch, criterion, optimizer, graph, aux)
            if dist.is_initialized(): dist.barrier()  # for dist model
            if eval_interval > 0:
                val_results = self.evaluate(val_dataloader, graph=graph, aux=aux)
                # print(cur_value)

                if self.local_rank in [0, -1]:
                    cur_value = val_results[metrics.index(best_metric)]
                    interval_best, all_interval_best = self.get_result_and_model(interval_best, all_interval_best,
                                                                                 best_metric,
                                                                                 cur_value, epoch, eval_interval, save_path)

                if epoch % eval_interval == 0:
                    if test_interval > 0:
                        if dist.is_initialized(): dist.barrier()
                        test_results = self.test(test_dataloader, graph=graph, aux=aux)
                    else:
                        test_results = [0] * len(val_results)
                    interval_best = None
                else:
                    test_results = [0] * len(val_results)
            else:
                if self.local_rank in [0, -1]:
                    if save_path:
                        self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, epoch - 1))
                        cur_loss = train_results[0]
                        if interval_best:
                            if cur_loss <= interval_best:
                                interval_best = cur_loss
                                self.save_model(save_path + '{}.pkl'.format(self.model_name))
                        else:
                            interval_best = cur_loss
                            self.save_model(save_path + '{}.pkl'.format(self.model_name))
                val_results = [0] * len(train_results)
                test_results = [0] * len(train_results)

            all_results = train_results + val_results + test_results

            if self.local_rank in [0, -1]:
                records_logger.info(','.join([str(epoch)] + [str(round(x, 6)) for x in all_results]))

        # fw.close()
        return final_results

    def get_result_and_model(self, interval_best, all_interval_best, best_metric, cur_value, epoch,
                             eval_interval, save_path):
        if interval_best:
            if best_metric in ['acc', 'r2', 'log_r2']:
                if cur_value >= interval_best:
                    interval_best = cur_value
                    print('new checkpoint! epoch {}!'.format(epoch))
                    if save_path:
                        self.save_model(
                            save_path + '{}_{}.pkl'.format(self.model_name, (epoch - 1) // eval_interval))
            else:
                if cur_value <= interval_best:
                    interval_best = cur_value
                    print('new checkpoint! epoch {}!'.format(epoch))
                    if save_path:
                        self.save_model(
                            save_path + '{}_{}.pkl'.format(self.model_name, (epoch - 1) // eval_interval))
        else:
            interval_best = cur_value
            print('new checkpoint! epoch {}!'.format(epoch))
            if save_path:
                self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, (epoch - 1) // eval_interval))

        if all_interval_best is None:
            all_interval_best = cur_value
            if save_path:
                self.save_model(save_path + '{}.pkl'.format(self.model_name))
        else:
            if best_metric in ['acc', 'r2', 'log_r2']:
                if cur_value >= all_interval_best:
                    all_interval_best = cur_value
                    print('global new checkpoint! epoch {}!'.format(epoch))
                    if save_path:
                        self.save_model(save_path + '{}.pkl'.format(self.model_name))
            else:
                if cur_value <= all_interval_best:
                    all_interval_best = cur_value
                    print('global new checkpoint! epoch {}!'.format(epoch))
                    if save_path:
                        self.save_model(save_path + '{}.pkl'.format(self.model_name))
        return interval_best, all_interval_best

    def save_model(self, path):
        # torch.save(self, path)
        if self.local_rank == 0:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print('Save successfully!')

    def load_model(self, path):
        state_dict = torch.load(path)
        if dist.is_initialized():
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        print('Load successfully!')
        return self

    @torch.no_grad()
    def evaluate(self, dataloader, phase='val', record_name=None, graph=None, aux=False):
        self.model.eval()
        self.phase = phase
        self.cur_model.phase = phase
        all_times = []
        all_ids = []
        self.container = ResultsContainer(0, len(dataloader), 0, self.model_name, phase,
                                          record_path=self.record_path, record_name=record_name,
                                          test_interval=self.test_interval)

        cur_graphs = None
        if graph:
            self.cur_model.node_trans = graph['node_trans']
            cur_graphs = graph['data'][phase]
            cur_graphs = self.cur_model.graph_to_device(cur_graphs)

        for idx, batch in enumerate(dataloader):
            loss = 0
            x, values, lengths, masks, ids, times, blocks = self.get_batch_input(batch, graph)
            if blocks:
                cur_graphs = blocks

            loss = self.batch_process(x, values, lengths, masks, ids, times, cur_graphs, dataloader, aux)
            all_times.extend(times.numpy().tolist())
            all_ids.extend(ids)

        results = self.container.get_epoch_result(aux, all_times, all_ids)
        # print(results)

        return results

    def test(self, test_dataloader, phase='test', record_name=None, graph=None, aux=False):
        results = self.evaluate(test_dataloader, phase, record_name=record_name, graph=graph, aux=aux)
        return results

    @torch.no_grad()
    def show_results(self, dataloader, phase='val', record_name=None, graph=None, aux=False, return_graph=False,
                     return_weight=False):
        print('===start showing===')
        self.model.eval()
        self.phase = 'test'
        self.model.phase = phase
        all_times = []
        all_ids = []
        all_outputs = []
        start_time = time.time()

        cur_graphs = None
        if graph:
            # load graphs_data
            self.model.node_trans = graph['node_trans']
            # self.graph = graph['data'].to(self.device)
            # self.graphs_data = graph['data']
            # cur_graphs = self.graphs_data[phase]
            cur_graphs = graph['data'][phase]
            cur_graphs = self.model.graph_to_device(cur_graphs)

        # print('===', dataloader)
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # print(idx)
            loss = 0
            x, values, lengths, masks, ids, times, blocks = self.get_batch_input(batch, graph)
            if blocks:
                cur_graphs = blocks

            # if self.graphs_data:
            all_ids.extend(ids)
            all_times.extend(times.numpy().tolist())
            outputs = self.model.show(x, lengths, masks, ids, cur_graphs, times, return_graph=return_graph,
                                      return_weight=return_weight)
            # print(outputs)
            all_outputs.append(outputs)
        if 'DDHGCN' in self.model_name:
            # [normal outputs, edge_weights, pop_embs]
            all_outputs = list(zip(*all_outputs))
            normal_outputs = torch.cat(all_outputs[0], dim=1).cpu().numpy()
            edge_weights = reduce(lambda a, b: a + b, all_outputs[1])
            all_embs = torch.cat(all_outputs[2], dim=0).cpu().numpy()
            return [all_ids, all_times, normal_outputs, edge_weights, all_embs]
        elif 'H2CGL' in self.model_name:
            all_outputs = list(zip(*all_outputs))
            print('output len:', len(all_outputs))
            all_reps = np.concatenate(all_outputs[0], axis=0)
            print(all_reps.shape)
            if return_graph:
                all_subgraph = reduce(lambda a, b: a + b, all_outputs[1])
                return [all_ids, all_times, all_reps, all_subgraph]
            else:
                all_cgin = reduce(lambda a, b: a + b, all_outputs[1])
                all_rgat = np.concatenate(all_outputs[2], axis=0)
                print(len(all_cgin))
                print(all_rgat.shape)
                return [all_ids, all_times, all_reps, all_cgin, all_rgat]
        else:
            all_outputs = torch.cat(all_outputs, dim=1).cpu().numpy()
            return [all_ids, all_times, all_outputs]

    def get_test_results(self, dataloader, save_path, record_path, graph=None, best_metric='male', aux=False):
        metrics, records_logger = self.get_metric_and_record(aux, record_path, phase='test')
        raw_header = ',' + ','.join(['{}_' + metric for metric in metrics])
        global_best = self.model_name + '.pkl'
        ptr = re.compile('^' + self.model_name.replace('+', '\+') + '_\d.pkl$')
        ckpts = sorted([ckpt for ckpt in os.listdir(save_path) if ptr.match(ckpt)] + [global_best])
        if (len(ckpts) <= 2) | ('DGCBERT' in self.model_name):
            ckpts = [global_best]
        print(ckpts)
        # records_logger = logging.getLogger('records')
        self.record_path = record_path

        records_logger.warning('epoch'
                               + raw_header.replace('{}', 'test'))

        all_metric_values = []
        metric_index = metrics.index(best_metric)
        count = 0
        for ckpt in ckpts:
            try:
                self.load_model(save_path + ckpt)
                if dist.is_initialized(): dist.barrier()
                test_results = self.test(dataloader, graph=graph, record_name='{}_results_' + str(count) + '.csv',
                                         aux=aux)
                if self.local_rank in [0, -1]:
                    all_metric_values.append(test_results[metric_index])
                    records_logger.info(','.join([ckpt] + [str(round(x, 6)) for x in test_results]))
                    count += 1
            except Exception as e:
                print(e)
        if self.local_rank in [0, -1]:
            if best_metric in ['acc', 'r2', 'log_r2']:
                best_ckpt_index = np.argmax(all_metric_values)
            else:
                best_ckpt_index = np.argmin(all_metric_values)

            print(best_ckpt_index)

    def get_show_results(self, dataloader, save_path, record_path, graph=None, best_metric='male', aux=False,
                         phase='test',
                         show='weight'):
        if show == 'weight':
            return_weight = True
            return_graph = False
        else:
            return_weight = False
            return_graph = True
        if aux:
            if 'aux' not in self.model_name:
                self.model_name += '_aux'
                if self.aux_weight != 0.5:
                    self.model_name += '_aw{}'.format(self.aux_weight)
        global_best = self.model_name + '.pkl'
        # records_logger = logging.getLogger('records')
        self.record_path = record_path
        # records_logger.setLevel(logging.DEBUG)
        print(global_best)
        self.load_model(save_path + global_best)
        if dist.is_initialized(): dist.barrier()
        print('if return graph:', return_graph)
        if ('H2CGL' in self.model_name) and return_graph:
            # [all_ids, all_times, all_reps, all_subgraphs]
            all_show_results = self.show_results(dataloader, graph=graph, aux=aux, phase=phase, return_graph=return_graph,
                                                 return_weight=return_weight)
            joblib.dump(all_show_results[:2], save_path + self.model_name + '_info.job')
            joblib.dump(all_show_results[2], save_path + self.model_name + '_rep.job')
            # print(all_show_results[-1][0])
            dgl.save_graphs(save_path + self.model_name + '_graphs.dgl', all_show_results[-1])
        else:
            all_show_results = self.show_results(dataloader, graph=graph, aux=aux, phase=phase, return_graph=return_graph,
                                                 return_weight=return_weight)
            joblib.dump(all_show_results, save_path + self.model_name + '_detail.job')
