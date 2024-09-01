import torch

from our_models.DDHGCN import *


class DPPDCC(DDHGCNSCLT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = self.model_name.replace('DDHGCNSCLT', 'DPPDCC')
        if ('cl' in self.model_name) and ('topic' not in self.ablation_channel):
            self.disen_module = SimpleDisenModuleT(self.encoder, self.topic_module, self.pop_module, self.contri_module,
                                                   self.device, mode=self.disen_module_type, disen_loss=self.disen_loss,
                                                   ablation_channel=self.ablation_channel,
                                                   sum_method=self.sum_method, loss_weights=self.loss_weights,
                                                   use_constraint=self.use_constraint)
        else:
            self.disen_module = SimpleDisenModule(self.encoder, self.topic_module, self.pop_module, self.contri_module,
                                                  self.device, mode=self.disen_module_type, disen_loss=self.disen_loss,
                                                  ablation_channel=self.ablation_channel,
                                                  sum_method=self.sum_method, loss_weights=self.loss_weights,
                                                  use_constraint=self.use_constraint)

    def get_phase_graph_data(self, phase, graphs, filter_nodes=False):
        cur_graphs, index_lists, trans_index, cur_masks, pop_indicators = (
            super().get_phase_graph_data(phase, graphs, filter_nodes=filter_nodes))
        return [cur_graphs, index_lists, trans_index, cur_masks, pop_indicators]

    def load_graphs(self, graphs):
        for pub_time in graphs['all_graphs']:
            cur_graph = graphs['all_graphs'][pub_time]
            cur_graph.nodes['paper'].data['citations'] = cur_graph.in_degrees(etype='cites').to(torch.float32)
            cur_graph.nodes['paper'].data['refs'] = cur_graph.in_degrees(etype='is cited by').to(torch.float32)
            if self.pop_module_type in {'aa'}:
                cur_graph.update_all(fn.copy_u('citations', 'm'), fn.sum('m', 'accum_citations'), etype='is writen by')

        graphs['all_trans_index'] = json.load(
            open(graphs['graphs_path'].replace('DPPDCC', 'DDHGCN') + '_trans.json', 'r'))
        graphs['all_masks'] = joblib.load(graphs['graphs_path'].replace('DPPDCC', 'DDHGCN') + '_masks.job')
        # ego_graphs
        for phase in graphs['data']:
            graphs['data'][phase] = self.get_phase_graph_data(phase, graphs)

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

        phases = list(graphs['data'].keys())
        for pub_time in graphs['all_graphs']:
            cur_subgraph = graphs['all_graphs'][pub_time]
            for ntype in cur_subgraph.ntypes:
                cur_subgraph.nodes[ntype].data.pop('h')
            cur_subgraph.nodes['paper'].data['citations'] = cur_subgraph.in_degrees(etype='cites').to(torch.float32)
            # cur_subgraph.nodes['paper'].data['refs'] = cur_subgraph.in_degrees(etype='is cited by').to(torch.float32)

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

        # seeds = [123, 456, 789]
        phase_path = os.path.split(path)[0] + '/DPPDCC/{}/'
        scl_path = phase_path.format('train') + 'scl/'
        for phase in phases:
            if not os.path.exists(phase_path.format(phase)):
                os.makedirs(phase_path.format(phase))
        if not os.path.exists(scl_path): os.makedirs(scl_path)

        for phase in phases:
            time_span = list(range(self.start_times[phase], self.start_times[phase] + self.time_length))
            all_subgraph_infos = []
            for t in time_span:
                with jl.open(os.path.split(path)[0] + '/DDHGCN_graphs_{}.jsonl'.format(t), 'r') as fr:
                    all_subgraph_infos.append([line for line in fr])
            all_subgraph_infos = list(zip(*all_subgraph_infos))
            print(len(all_subgraph_infos))
            for sample in tqdm(split_data[phase][0]):
                cur_idx = all_trans_index[sample]
                cur_subgraph_info = all_subgraph_infos[cur_idx]

                origin_list = []
                for t in range(self.time_length):
                    info_dict = cur_subgraph_info[t]
                    # print(info_dict)
                    cur_subgraph = dgl.node_subgraph(graphs['all_graphs'][self.start_times[phase] + t],
                                                     nodes=info_dict['nodes'])
                    # print(len(info_dict['nodes']['paper']))
                    for indicator in info_dict['indicators']:
                        cur_indicators = info_dict['indicators'][indicator]
                        cur_subgraph.nodes['paper'].data[indicator] = torch.tensor(cur_indicators, dtype=torch.long) \
                            if type(cur_indicators) != torch.Tensor else cur_indicators
                    if cur_subgraph.nodes('paper').shape[0] > 0:
                        tgt_papers = torch.where(cur_subgraph.nodes['paper'].data['is_target'] == 1)[0].numpy().tolist()
                        # print(tgt_papers)
                        cites_paper = cur_subgraph.in_edges(tgt_papers, etype='cites')[0].numpy().tolist()
                        cited_paper = cur_subgraph.in_edges(tgt_papers, etype='is cited by')[0].numpy().tolist()
                        cur_subgraph.nodes['paper'].data['is_cite'] = torch.zeros(cur_subgraph.num_nodes('paper'),
                                                                                  dtype=torch.long)
                        cur_subgraph.nodes['paper'].data['is_ref'] = torch.zeros(cur_subgraph.num_nodes('paper'),
                                                                                 dtype=torch.long)
                        cur_subgraph.nodes['paper'].data['is_cite'][cites_paper + tgt_papers] = 1
                        cur_subgraph.nodes['paper'].data['is_ref'][cited_paper + tgt_papers] = 1
                        # cur_subgraph.nodes['paper'].data['is_cite'][cites_paper] = 1
                        # cur_subgraph.nodes['paper'].data['is_ref'][cited_paper] = 1
                    for ntype in cur_subgraph.ntypes:
                        # cur_subgraph.nodes[ntype].data['tid'] = cur_subgraph.nodes[ntype].data[dgl.NID]
                        cur_subgraph.nodes[ntype].data.pop('oid')
                        if ntype == 'paper':
                            cur_subgraph.nodes[ntype].data.pop('time')
                    origin_list.append(cur_subgraph)
                dgl.save_graphs(phase_path.format(phase) + '{}_subgraphs.dgl'.format(sample), origin_list)

        del graphs['all_graphs']

        joblib.dump([], path)
        return dealt_graphs
