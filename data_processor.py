import joblib
import jsonlines as jl
from sentence_transformers import SentenceTransformer
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer, BertModel
from torchtext.vocab import build_vocab_from_iterator, Vectors, vocab

from our_models.DDHGCN import SparseSelecter
from utilis.log_bar import TqdmToLogger
from utilis.scripts import get_configs, IndexDict, add_new_elements, SampleAllNeighbors, get_pareto_dist, \
    DealtKhopSubgraph
from utilis.bertwhitening_utils import input_to_vec, compute_kernel_bias, transform_and_normalize, inputs_to_vec
from utilis.collate_batch import *
import networkx as nx
from utilis.eval_sampler import DistributedEvalSampler


# tokenizer = get_tokenizer('basic_english')
UNK, PAD, SEP = '[UNK]', '[PAD]', '[SEP]'
SBERT_MODELS = ['sbert', 'sbw']
BERT_MODELS = ['bert', 'bw', 'specter2']


class DataProcessor:
    def __init__(self, data_source, max_len=256, seed=123, norm=False, log=False, time=None, model_config=None):
        print('Init...')
        self.data_root = './data/'
        self.data_source = data_source
        self.seed = int(seed)
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.mean = 0
        self.std = 1
        self.norm = norm
        self.log = log
        self.time = time
        self.config = model_config if model_config else dict()
        self.batch_graph = self.config.get('batch_graph', False)
        self.selected_ids = None
        self.train_sampler = None
        print(self.time)
        self.data_cat_path = self.data_root + self.data_source + '/'

    def split_data(self, rate=0.8, fixed_num=None, shuffle=True, by='normal', time=None, cut_time=2015, time_length=5):
        if self.time:
            time = self.time
        all_values = json.load(open(self.data_cat_path + 'sample_citation_accum.json'))
        print(len(all_values))
        all_ids = list(all_values.keys())

        info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json'))
        time_dict = dict(map(lambda x: (x[0], int(x[1]['year'])), info_dict.items()))
        title_dict = dict(map(lambda x: (x[0], x[1]['title']), info_dict.items()))
        del info_dict
        abs_dict = json.load(open(self.data_cat_path + 'sample_abstract_dict.json'))
        abs_dict = dict(map(lambda x: (x[0], x[1]['abstract']), abs_dict.items()))

        if by == 'time':
            # as by time, no rate and fixed_num, but use year
            # here only input year < time, but should know the difference
            time_index = list(range(cut_time - 3 * time_length + 1, cut_time + time_length + 1))
            print(len(time_index))
            index_trans = dict(zip(time_index, range(len(time_index))))

            train_time, val_time, test_time = time
            print('time:', time)
            print('cut time:', cut_time)
            test_ids = [key for key in all_values if int(time_dict[key]) <= test_time]
            print('test samples:', len(test_ids))
            # test_idx = test_time + time_length - cut_time - 1
            # print('test_idx', test_idx)
            # test_values = list(map(lambda x: all_values[x][1][test_idx] - all_values[x][0][test_idx], test_ids))
            test_idx = index_trans[test_time]
            print('test_idx', test_idx)
            test_values = list(map(lambda x: all_values[x][test_idx + time_length] - all_values[x][test_idx], test_ids))
            key_ids = set(map(lambda x: x[0],
                              filter(lambda x: x[1] >= self.config['cut_threshold'][-1], zip(test_ids, test_values))))
            # selected_ids = list(key_ids)
            print('key_ids', len(key_ids))
            # count = 0
            random.seed(self.seed)
            random.shuffle(test_ids)

            selected_ids = test_ids[:fixed_num]
            all_values = dict([(paper, all_values[paper]) for paper in selected_ids])

            train_ids = [key for key in all_values if int(time_dict[key]) <= train_time]
            print('train samples:', len(train_ids))
            val_ids = [key for key in all_values if int(time_dict[key]) <= val_time]
            print('val samples:', len(val_ids))
            test_ids = [key for key in all_values if int(time_dict[key]) <= test_time]
            print('test samples:', len(test_ids))

            # here choose the cur + 5 as predict value 2015?
            train_idx = index_trans[train_time]
            train_values = list(
                map(lambda x: all_values[x][train_idx + time_length] - all_values[x][train_idx], train_ids))
            val_idx = index_trans[val_time]
            val_values = list(map(lambda x: all_values[x][val_idx + time_length] - all_values[x][val_idx], val_ids))
            test_idx = index_trans[test_time]
            test_values = list(map(lambda x: all_values[x][test_idx + time_length] - all_values[x][test_idx], test_ids))
            print('time:', train_time, val_time, test_time)
            print('idx:', train_idx, val_idx, test_idx)

        else:
            if shuffle:
                random.seed(self.seed)
                print('data_processor seed', self.seed)
                random.shuffle(all_ids)

            total_count = len(all_ids)
            train_ids = all_ids[:int(total_count * rate)]
            val_ids = all_ids[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]
            test_ids = all_ids[int(total_count * ((1 - rate) / 2 + rate)):]

            train_values = list(map(lambda x: all_values[x], train_ids))
            val_values = list(map(lambda x: all_values[x], val_ids))
            test_values = list(map(lambda x: all_values[x], test_ids))

        train_contents = list(
            map(lambda x: re.sub('\s+', ' ', str(title_dict[x]) + '. ' + abs_dict[x]), train_ids))
        val_contents = list(
            map(lambda x: re.sub('\s+', ' ', str(title_dict[x]) + '. ' + abs_dict[x]), val_ids))
        test_contents = list(
            map(lambda x: re.sub('\s+', ' ', str(title_dict[x]) + '. ' + abs_dict[x]), test_ids))

        train_times = list(map(lambda x: int(time_dict[x]), train_ids))
        val_times = list(map(lambda x: int(time_dict[x]), val_ids))
        test_times = list(map(lambda x: int(time_dict[x]), test_ids))

        cut_data = {
            'train': [train_ids, train_values, train_contents, train_times],
            'val': [val_ids, val_values, val_contents, val_times],
            'test': [test_ids, test_values, test_contents, test_times],
        }

        torch.save(cut_data, self.data_cat_path + 'split_data')

    def show_graph_info(self, graph_name='graph_sample_feature_vector'):
        all_values = json.load(open(self.data_cat_path + 'sample_citation_accum.json'))
        print(len(all_values))
        all_ids = list(all_values.keys())
        trans_dict = json.load(open(self.data_cat_path + 'sample_node_trans.json'))
        all_papers = [trans_dict['paper'][paper] for paper in all_ids]
        graph = torch.load(self.data_cat_path + graph_name)
        all_refs = graph.in_degrees(torch.tensor(all_papers), etype='is cited by').numpy().tolist()
        all_cites = graph.in_degrees(torch.tensor(all_papers), etype='cites').numpy().tolist()
        all_times = graph.nodes['paper'].data['time'][all_papers, :].squeeze(dim=-1).numpy().tolist()
        df_count = pd.DataFrame(index=all_papers, data={'ref': all_refs, 'cite': all_cites, 'time': all_times})
        print(df_count.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))
        df_count = df_count[df_count['time'] <= 2011]
        print(df_count.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))

    def get_data(self, phases=None, selected_ids=None):
        cur_data = torch.load(self.data_cat_path + 'split_data')
        cur_phases = list(cur_data.keys())
        if phases is not None:
            for phase in cur_phases:
                if phase not in phases:
                    del cur_data[phase]
                # print(len(cur_data[phase]))
                # print(len(list(zip(*zip(*cur_data[phase])))))
                # print(len([*list(zip(cur_data[phase]))]))
        if selected_ids is not None:
            for phase in cur_data:
                temp_data = list(zip(*filter(lambda x: x[0] in selected_ids, zip(*cur_data[phase]))))
                cur_data[phase] = temp_data
                print('{}:'.format(phase), len(cur_data[phase][0]))
        return cur_data

    def get_selected_ids(self):
        df = pd.read_csv('./results/test/{}/hard_ones.csv'.format(self.data_source), index_col=0)
        self.selected_ids = set(list(df.index.astype(str)))
        return self.selected_ids

    def add_attrs(self):
        data = torch.load(self.data_cat_path + 'split_data')
        info = json.load(open(self.data_cat_path + 'sample_info_dict.json', 'r'))
        for phase in data:
            if len(data[phase]) > 3:
                continue
            if self.data_source == 'pubmed':
                pub_time = [int(info[paper]['pub_time']['year']) for paper in data[phase][0]]
            elif self.data_source in ['dblp', 'sdblp']:
                pub_time = [int(info[paper]['year']) for paper in data[phase][0]]
            # print(pub_time)
            data[phase].append(pub_time)
        torch.save(data, self.data_cat_path + 'split_data')

    def get_tokenizer(self, tokenizer_type='basic', tokenizer_path=None):
        if tokenizer_type == 'bert':
            self.tokenizer = CustomBertTokenizer(max_len=self.max_len, bert_path=tokenizer_path,
                                                 data_path=self.data_cat_path)
        elif tokenizer_type == 'glove':
            self.tokenizer = VectorTokenizer(max_len=self.max_len, vector_path=tokenizer_path,
                                             data_path=self.data_cat_path, name='glove')
        else:
            self.tokenizer = BasicTokenizer(max_len=self.max_len, data_path=self.data_cat_path)

        data = torch.load(self.data_cat_path + 'split_data')
        self.tokenizer.load_vocab(data['train'][2], seed=self.seed)
        return self.tokenizer

    def get_cl_dataloader(self, batch_size=32, num_workers=0, graph_dict=None):
        return self.get_dataloader(batch_size, num_workers, graph_dict)

    def get_dataloader(self, batch_size=32, num_workers=0, graph_dict=None, device=None, use_dist=False,
                       selected_phases=None):
        print('workers', num_workers)
        # data = torch.load(self.data_cat_path + 'split_data')
        # self.data = data

        phases = ['train', 'val', 'test']
        if graph_dict:
            phases = [phase for phase in phases if phase in graph_dict['data']]
        if selected_phases:
            phases = selected_phases
        data = self.get_data(phases, self.selected_ids)
        # all_last_values = map(lambda x: x[1][-1], data['train'][1])
        if self.norm:
            # all_train_values = np.sum(list(map(lambda x: [num for num in (x[0] + x[1]) if num >= 0], data['train'][1])))
            all_train_values = data['train'][1]
            self.mean = np.mean(all_train_values)
            self.std = np.std(all_train_values)
            print(self.mean)
            print(self.std)

        self.graph_dict = graph_dict
        collate_batch_method = self.collate_batch
        if self.batch_graph == 'saved':
            # collate_batch_method = {phase: self.graph_method_map(phase) for phase in phases}
            collate_batch_method = {phase: saved_collate_batch(self, phase) for phase in phases}
        elif self.batch_graph == 'saved_hetero':
            collate_batch_method = {phase: saved_collate_batch(self, phase) for phase in phases}
        elif self.batch_graph == 'saved_dynamic':
            collate_batch_method = {phase: saved_dynamic_collate_batch(self, phase) for phase in phases}
        elif self.batch_graph == 'saved_dynamic_homo':
            collate_batch_method = {phase: saved_dynamic_homo_collate_batch(self, phase, sort=-1) for phase in phases}
        elif self.batch_graph == 'ddhgcn':
            cut_point = 10
            if 'train' in data:
                all_train_values = data['train'][1]
                cut_point = get_pareto_dist(all_train_values)
            self.cut_points = [0, cut_point]
            collate_batch_method = {phase: ddhgcn_collate_batch(self, phase, sort=-1) for phase in phases}
        elif self.batch_graph == 'ddhgcns':
            cut_point = 10
            if 'train' in data:
                all_train_values = data['train'][1]
                cut_point = get_pareto_dist(all_train_values)
            self.cut_points = [0, cut_point]
            collate_batch_method = {phase: ddhgcns_collate_batch(self, phase, sort=-1) for phase in phases}
        elif self.batch_graph == 'ddhgcnscl':
            collate_batch_method = {phase: ddhgcns_cl_collate_batch(self, phase, sort=-1,
                                                                    topic_module=self.config['topic_module'],
                                                                    aug_rate=self.config['aug_rate'],
                                                                    focus=self.config['focus']
                                                                    ) for phase in phases}
        elif self.batch_graph == 'ddhgcnsclt':
            collate_batch_method = {phase: ddhgcns_cl_test_collate_batch(self, phase, sort=-1,
                                                                         topic_module=self.config['topic_module'],
                                                                         aug_rate=self.config['aug_rate'],
                                                                         focus=self.config['focus']
                                                                         ) for phase in phases}
        elif self.batch_graph == 'dppdcc':
            if 'topic' not in self.config['ablation_channel']:
                topic_module = self.config['topic_module']
            else:
                topic_module = 'none'
            collate_batch_method = {phase: dppdcc_collate_batch(self, phase, sort=-1,
                                                                topic_module=topic_module,
                                                                aug_rate=self.config['aug_rate'],
                                                                focus=self.config['focus']
                                                                ) for phase in phases}
        elif self.batch_graph == 'saved_graph':
            collate_batch_method = {phase: saved_graph_collate_batch(self, phase) for phase in phases}
        elif self.batch_graph == 'saved_pyg':
            collate_batch_method = {phase: saved_pyg_collate_batch(self, phase) for phase in phases}
        elif self.batch_graph == 'saved_pyg_graph':
            collate_batch_method = {phase: saved_pyg_collate_batch(self, phase, is_graph=True) for phase in phases}
        # else:
        #     print('no such batch graph')
        #     raise Exception

        if type(collate_batch_method) != dict:
            collate_batch_method = {phase: collate_batch_method for phase in phases}

        self.dataloaders = []
        for phase in phases:
            if self.batch_graph == 'torch_geometric':
                cur_graph, trans_dict = graph_dict['data'][phase]
                cur_dataloader = NeighborLoader(cur_graph, num_neighbors=[-1] * self.config['hop'],
                                                input_nodes=cur_graph.mask, batch_size=batch_size, disjoint=True)
                cur_dataloader.log = self.log
                cur_dataloader.mean = self.mean
                cur_dataloader.std = self.std
                self.dataloaders.append(cur_dataloader)
            else:
                cur_dataset = list(zip(*data[phase]))
                print('>>>data len:', len(cur_dataset), len(cur_dataset) % batch_size)
                shuffle = False
                cur_batch_size = batch_size if phase == 'train' else max(batch_size, max(4 * batch_size, 128))
                if phase == 'train':
                    drop_last = True if (len(cur_dataset) % cur_batch_size == 1) else False
                    if dist.is_initialized() | use_dist:
                        sampler = DistributedSampler(cur_dataset)
                        self.train_sampler = sampler
                    else:
                        sampler = RandomSampler(cur_dataset)
                else:
                    drop_last = False
                    if dist.is_initialized() | use_dist:
                        sampler = DistributedEvalSampler(cur_dataset, shuffle=False)
                    else:
                        sampler = None
                self.dataloaders.append(CustomDataLoader(dataset=cur_dataset, batch_size=cur_batch_size,
                                                         shuffle=shuffle, num_workers=num_workers,
                                                         collate_fn=collate_batch_method[phase],
                                                         prefetch_factor=self.config['prefetch_factor'],
                                                         # pin_memory=True, pin_memory_device=device,
                                                         # persistent_workers=True,
                                                         mean=self.mean, std=self.std, log=self.log, phase=phase,
                                                         drop_last=drop_last, sampler=sampler))

        while len(self.dataloaders) < 3:
            self.dataloaders.append(None)
        return self.dataloaders

    def collate_batch(self, batch):
        values_list, content_list = [], []
        # inputs_list, valid_lens = [], []
        length_list = []
        mask_list = []
        ids_list = []
        time_list = []
        # print(batch)
        for (_ids, _values, _contents, _time) in batch:
            # processed_content, seq_len, mask = self.text_pipeline(_content)
            # # print(_label)
            processed_content, seq_len, mask = self.tokenizer.encode(_contents)
            # values_list.append(_values)
            # inputs, valid_len = self.values_pipeline(_values)
            # values_list.append(self.label_pipeline(_values))
            values_list.append(_values)
            # inputs_list.append(inputs)
            # valid_lens.append(valid_len)
            content_list.append(processed_content)
            length_list.append(seq_len)
            mask_list.append(mask)
            # ids_list.append(int(_ids.strip()))
            ids_list.append(_ids.strip())
            time_list.append(_time)

        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        # inputs_list = torch.tensor(inputs_list, dtype=torch.float32)
        # valid_lens = torch.tensor(valid_lens, dtype=torch.int8)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        # ids_list = torch.tensor(ids_list, dtype=torch.int64)
        # print(len(label_list))
        # content_list = [content_batch, inputs_list, valid_lens]
        time_list = torch.tensor(time_list, dtype=torch.long)
        content_list = content_batch
        # return content_list, values_list, length_list, \
        #        mask_list, ids_list, time_list, None
        return SimpleCustomPinningBatch(content_list, values_list, length_list,
                                        mask_list, ids_list, time_list, None)


    def get_embs(self, tokenizer_type, tokenizer_path, name='graph_sample_feature', mode='vector',
                 split_abstract=False):
        print(name)
        print(tokenizer_path)
        print('graph mode:', mode)
        # graph = torch.load(self.data_cat_path + 'graph_sample')
        node_trans = json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r'))
        paper_trans = node_trans['paper']
        node_ids = list(paper_trans.values())
        node_ids.sort()
        print(len(node_ids))
        index_trans = dict(zip(paper_trans.values(), paper_trans.keys()))

        info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json'))
        title_dict = dict(map(lambda x: (x[0], x[1]['title']), info_dict.items()))
        del info_dict
        abs_dict = json.load(open(self.data_cat_path + 'sample_abstract_dict.json'))
        abs_dict = dict(map(lambda x: (x[0], x[1]['abstract']), abs_dict.items()))

        abstracts = list(
            map(lambda x: re.sub('\s+', ' ', str(title_dict[index_trans[x]]) + '. ' + SEP + ' '
                                 + abs_dict[index_trans[x]]), node_ids))

        feature_list = []

        self.tokenizer = self.get_tokenizer(tokenizer_type, tokenizer_path)
        self.tokenizer.load_vocab()
        # print('bert-whitening embeddings')
        # self.tokenizer = CustomBertTokenizer(max_len=self.max_len, bert_path=tokenizer_path)
        # self.tokenizer.load_vocab()
        print(tokenizer_path)
        print(self.tokenizer.tokenizer)

        if mode in BERT_MODELS:
            bert_model = BertModel.from_pretrained(tokenizer_path, return_dict=True, output_hidden_states=True).to(
                self.device)
        elif mode in SBERT_MODELS:
            bert_model = SentenceTransformer(tokenizer_path)

        # first get kernel and bias
        all_embs = []

        batch_size = 64
        if mode == 'bw':
            count = 0
            temp_tokens, temp_masks, temp_lens = [], [], []
            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                temp_tokens.append(processed_content)
                temp_masks.append(mask)
                temp_lens.append(seq_len)
                count += 1
                if ((count > 0) and (count % batch_size == 0)) | (count == len(abstracts)):
                    tokens = torch.tensor(temp_tokens, dtype=torch.long).to(self.device)
                    masks = torch.tensor(temp_masks, dtype=torch.long).to(self.device)
                    seq_lens = temp_lens
                    output = inputs_to_vec([tokens, masks], bert_model, 'first_last_avg', seq_lens)
                    # print(output.shape)
                    all_embs.append(output)
                    temp_tokens, temp_masks, temp_lens = [], [], []
        elif mode in BERT_MODELS:
            count = 0
            temp_tokens, temp_masks, temp_lens = [], [], []
            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                temp_tokens.append(processed_content)
                temp_masks.append(mask)
                temp_lens.append(seq_len)
                count += 1
                if ((count > 0) and (count % batch_size == 0)) | (count == len(abstracts)):
                    tokens = torch.tensor(temp_tokens, dtype=torch.long).to(self.device)
                    masks = torch.tensor(temp_masks, dtype=torch.long).to(self.device)
                    seq_lens = temp_lens
                    # print(tokens)
                    output = inputs_to_vec([tokens, masks], bert_model, 'last_avg', seq_lens)
                    # print(output.shape)
                    all_embs.append(output)
                    temp_tokens, temp_masks, temp_lens = [], [], []
        elif mode in SBERT_MODELS:
            count = 0
            all_embs = []
            temp_list = []
            for abstract in tqdm(abstracts):
                # print(abstract)
                temp_list.append(abstract)
                count += 1
                if ((count > 0) and (count % batch_size == 0)) | (count == len(abstracts)):
                    output = bert_model.encode(temp_list)
                    # print(output.shape)
                    all_embs.append(output)
                    temp_list = []

        all_embs = np.array(all_embs)
        all_embs = np.concatenate(all_embs, axis=0)

        print(all_embs.shape)
        joblib.dump(all_embs, self.data_cat_path + name + '_' + mode + '_embs')

    def get_feature_graph(self, tokenizer_type, tokenizer_path, name='graph_sample_feature', mode='vector',
                          split_abstract=False, time_list=(2001, 2015)):
        print(name)
        print(tokenizer_path)
        print('graph mode:', mode)
        graph = torch.load(self.data_cat_path + 'graph_sample')
        node_trans = json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r'))
        paper_trans = node_trans['paper']
        node_ids = list(paper_trans.values())
        node_ids.sort()
        print(len(node_ids))
        index_trans = dict(zip(paper_trans.values(), paper_trans.keys()))

        info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json'))
        title_dict = dict(map(lambda x: (x[0], x[1]['title']), info_dict.items()))
        del info_dict
        abs_dict = json.load(open(self.data_cat_path + 'sample_abstract_dict.json'))
        abs_dict = dict(map(lambda x: (x[0], x[1]['abstract']), abs_dict.items()))

        abstracts = list(
            map(lambda x: re.sub('\s+', ' ', str(title_dict[index_trans[x]]) + '. ' + SEP + ' '
                                 + abs_dict[index_trans[x]]), node_ids))
        del abs_dict

        feature_list = []

        self.tokenizer = self.get_tokenizer(tokenizer_type, tokenizer_path)
        self.tokenizer.load_vocab()

        if mode == 'vector':
            print(self.tokenizer.vectors.shape)

            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                node_embedding = self.tokenizer.vectors[processed_content][:seq_len].mean(dim=0, keepdim=True)
                if torch.isnan(node_embedding).sum().item() > 0:
                    print(abstract)
                    node_embedding = torch.zeros_like(node_embedding)
                feature_list.append(node_embedding)

        elif mode == 'bert':
            print(self.tokenizer.tokenizer)
            bert_model = BertModel.from_pretrained(tokenizer_path, return_dict=True, output_hidden_states=True).to(
                self.device)

            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                tokens = torch.tensor(processed_content).unsqueeze(dim=0).to(self.device)
                mask = torch.tensor(mask).unsqueeze(dim=0).to(self.device)
                output = bert_model(tokens, attention_mask=mask)
                node_embedding = output['hidden_states'][-2][0, :seq_len].mean(dim=0, keepdim=True).detach().cpu()
                # print(node_embedding.shape)
                feature_list.append(node_embedding)

        elif mode in BERT_MODELS + SBERT_MODELS:
            all_embs = joblib.load(self.data_cat_path + name + '_' + mode + '_embs')
            feature_list = torch.from_numpy(all_embs)

        elif mode == 'token':
            seq_lens = []
            masks = []

            for abstract in tqdm(abstracts):
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                processed_content = torch.tensor(processed_content).unsqueeze(dim=0)
                seq_len = torch.tensor(seq_len).unsqueeze(dim=0)
                mask = torch.tensor(mask).unsqueeze(dim=0)
                feature_list.append(processed_content)
                seq_lens.append(seq_len)
                masks.append(mask)

            graph.nodes['paper'].data['seq_len'] = torch.cat(seq_lens, dim=0)
            graph.nodes['paper'].data['mask'] = torch.cat(masks, dim=0)

        if mode.endswith('bw'):
            # graph.nodes['paper'].data['h'] = feature_list
            print('not here')
        elif mode in ['sbert', 'specter2']:
            graph.nodes['paper'].data['h'] = feature_list
        elif mode == 'random':
            paper_count = graph.nodes('paper').shape[0]
            graph.nodes['paper'].data['h'] = torch.randn(paper_count, 384, dtype=torch.float32)
        else:
            graph.nodes['paper'].data['h'] = torch.cat(feature_list, dim=0)
        print(graph.nodes['paper'].data['h'].shape)

        # all metadata embedding is averaged on the graph at current time point!!!
        author_src, author_dst = graph.edges(form='uv', etype='writes', order='eid')
        journal_src, journal_dst = graph.edges(form='uv', etype='publishes', order='eid')

        # torch.save(graph, self.data_cat_path + name + '_' + mode)
        joblib.dump(graph, self.data_cat_path + name + '_' + mode + '.job')

        for time_point in time_list:
            print('-' * 30 + str(time_point) + '-' * 30)
            sub_papers = graph.nodes('paper')[graph.nodes['paper'].data['time'].squeeze(dim=-1) <= time_point]
            print('paper', sub_papers.shape)
            selected_journal = list(
                set(journal_src[graph.edges['publishes'].data['time'].squeeze(dim=-1) <= time_point].numpy().tolist()))
            print('journal', len(selected_journal))
            selected_author = list(
                set(author_src[graph.edges['writes'].data['time'].squeeze(dim=-1) <= time_point].numpy().tolist()))
            print('author', len(selected_author))
            sub_nodes_dict = {
                'paper': sub_papers,
                'author': selected_author,
                'journal': selected_journal
            }
            sub_graph = dgl.node_subgraph(graph, sub_nodes_dict)

            sub_graph = dgl.remove_self_loop(sub_graph, 'is cited by')
            sub_graph = dgl.remove_self_loop(sub_graph, 'cites')

            if mode.endswith('bw'):
                cur_idx = sub_graph.nodes['paper'].data[dgl.NID]
                cur_embs = all_embs[cur_idx, :]
                print(cur_embs.shape)
                kernel, bias = compute_kernel_bias([cur_embs])
                kernel = kernel[:, :300]

                feature_list = torch.from_numpy(transform_and_normalize(cur_embs, kernel, bias).astype(np.float32))
                print(feature_list.shape)
                sub_graph.nodes['paper'].data['h'] = feature_list

            if mode != 'token':
                feature_array = sub_graph.nodes['paper'].data['h'].numpy()
                print(feature_array.shape)
                # emb_dim = sub_graph.nodes['paper'].data['h'].shape[1]
                emb_dim = feature_array.shape[-1]
                author_features = []
                cur_author_src, cur_author_dst = sub_graph.edges(form='uv', etype='writes', order='eid')
                cur_journal_src, cur_journal_dst = sub_graph.edges(form='uv', etype='publishes', order='eid')
                cur_author_src, cur_author_dst = cur_author_src.numpy(), cur_author_dst.numpy()
                cur_journal_src, cur_journal_dst = cur_journal_src.numpy(), cur_journal_dst.numpy()
                # print(oid_cid_trans)
                # print(cur_author_src, cur_author_dst)
                for author in tqdm(sub_graph.nodes('author').numpy()):
                    temp_papers = cur_author_dst[cur_author_src == author]
                    # temp_papers = cur_author_dst[np.argwhere(cur_author_src == author)]
                    # print(np.argwhere(cur_author_src == author))
                    # print(temp_papers)
                    # print(np.argwhere(cur_author_src == author))
                    if len(temp_papers) == 0:
                        raise Exception
                    # print(temp_papers.shape)
                    # print(feature_array[temp_papers].shape)
                    author_features.append(
                        np.mean(feature_array[temp_papers, :], axis=0))
                if len(author_features) == 0:
                    sub_graph.nodes['author'].data['h'] = torch.zeros((0, emb_dim), dtype=torch.float32)
                else:
                    # sub_graph.nodes['author'].data['h'] = torch.cat(author_features, dim=0)
                    sub_graph.nodes['author'].data['h'] = torch.from_numpy(np.stack(author_features, axis=0))
                print(sub_graph.nodes['author'].data['h'].shape)

                journal_features = []
                for journal in tqdm(sub_graph.nodes('journal').numpy()):
                    temp_papers = cur_journal_dst[cur_journal_src == journal]
                    # temp_papers = cur_journal_dst[np.argwhere(cur_journal_src == journal)]
                    if len(temp_papers) == 0:
                        raise Exception
                    # print(temp_papers)
                    journal_features.append(
                        np.mean(feature_array[temp_papers, :], axis=0))
                if len(journal_features) == 0:
                    sub_graph.nodes['journal'].data['h'] = torch.zeros((0, emb_dim), dtype=torch.float32)
                else:
                    # sub_graph.nodes['journal'].data['h'] = torch.cat(journal_features, dim=0)
                    print(np.stack(journal_features, axis=0).shape)
                    sub_graph.nodes['journal'].data['h'] = torch.from_numpy(np.stack(journal_features, axis=0))
                print(sub_graph.nodes['journal'].data['h'].shape)

                time_features = []
                cur_times = sorted(set(sub_graph.nodes['paper'].data['time'].squeeze(dim=-1).numpy().tolist()))
                for pub_time in tqdm(cur_times):
                    cur_index = np.argwhere(
                        sub_graph.nodes['paper'].data['time'].squeeze(dim=-1).numpy() == pub_time).squeeze(
                        axis=-1)
                    # print(feature_array[cur_index].mean(axis=0).shape)
                    time_features.append(feature_array[cur_index].mean(axis=0))
                time_features = torch.from_numpy(np.stack(time_features, axis=0))
                print(time_features.shape)

                papers = sub_graph.nodes('paper')
                trans_dict = dict(zip(cur_times,
                                      range(len(cur_times))))
                times = torch.tensor([trans_dict[t]
                                      for t in sub_graph.nodes['paper'].data['time'].squeeze(dim=-1).numpy().tolist()],
                                     dtype=torch.long)

                edges = {
                    ('paper', 'is shown in', 'time'): (papers, times),
                    ('time', 'shows', 'paper'): (times, papers)
                }

                sub_graph = add_new_elements(sub_graph, nodes={'time': len(cur_times)}, ndata={'time': {
                    'h': time_features,
                    '_ID': torch.tensor(cur_times, dtype=torch.long),
                    'oid': torch.tensor(cur_times, dtype=torch.long)
                }}, edges=edges)

            print(sub_graph)
            # joblib.dump(sub_graph, self.data_cat_path + name + '_' + mode + '_' + str(time_point) + '.job')
            dgl.save_graphs(self.data_cat_path + name + '_' + mode + '_' + str(time_point) + '.dgl', [sub_graph])


    def load_graphs(self, graph_name='graph_sample_feature', time_length=10, phases=('train', 'val', 'test')):
        print(graph_name)
        train_time, val_time, test_time = self.time
        print(self.time, time_length)
        phase_dict = {
            'train': list(range(train_time - (time_length - 1), train_time + 1)),
            'val': list(range(val_time - (time_length - 1), val_time + 1)),
            'test': list(range(test_time - (time_length - 1), test_time + 1))
        }
        phase_dict = dict(filter(lambda x: x[0] in phases, phase_dict.items()))
        print(phase_dict)
        # train_list = list(range(train_time - 9, train_time + 1))
        # val_list = list(range(val_time - 9, val_time + 1))
        # test_list = list(range(test_time - 9, test_time + 1))
        # all_time = set(phase_dict['train'] + phase_dict['val'] + phase_dict['test'])
        all_time = []
        for phase in phases:
            all_time += phase_dict[phase]
        all_time = sorted(list(set(all_time)))

        graphs = {}
        for time in all_time:
            # graphs[time] = torch.load(self.data_cat_path + graph_name + '_' + str(time))
            # graphs[time] = joblib.load(self.data_cat_path + graph_name + '_' + str(time) + '.job')
            print(graph_name + '_' + str(time))
            graphs[time] = dgl.load_graphs(self.data_cat_path + graph_name + '_' + str(time) + '.dgl')[0][0]

        data_dict = {phase: [graphs[time] for time in phase_dict[phase]] for phase in phase_dict}

        self.graph = {
            'data': data_dict,
            'node_trans': json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r')),
            'time': {'train': train_time, 'val': val_time, 'test': test_time},
            'all_graphs': graphs,
            'time_length': time_length,
            'graph_type': graph_name.split('_')[-1],
            'data_path': self.data_cat_path
        }
        return self.graph

    def get_time_dict(self):
        train_time, val_time, test_time = self.time
        return {'train': train_time, 'val': val_time, 'test': test_time}

    def values_pipeline(self, values):
        return values


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=None,
                 mean=0, std=1, log=False, phase=None, pin_memory=False, pin_memory_device='', **kwargs):
        super(CustomDataLoader, self).__init__(dataset=dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn,
                                               pin_memory=pin_memory, pin_memory_device=pin_memory_device, **kwargs)
        self.mean = mean
        self.std = std
        self.log = log
        self.phase = phase


class BasicTokenizer:
    def __init__(self, max_len=256, data_path=None):
        self.tokenizer = get_tokenizer('basic_english')
        self.max_len = max_len
        self.vocab = None
        self.data_path = data_path
        self.vectors = None

    def yield_tokens(self, data_iter):
        for content in data_iter:
            yield self.tokenizer(content)

    def build_vocab(self, text_list, seed):
        self.vocab = build_vocab_from_iterator(self.yield_tokens(text_list), specials=[UNK, PAD])
        self.vocab.set_default_index(self.vocab[UNK])
        torch.save(self.vocab, self.data_path + 'vocab_{}'.format(seed))

    def load_vocab(self, text_list, seed):
        try:
            self.vocab = torch.load(self.data_path + 'vocab_{}'.format(seed))
        except Exception as e:
            print(e)
            self.build_vocab(text_list, seed)

    def encode(self, text):
        tokens = self.tokenizer(text)
        seq_len = len(tokens)
        if seq_len <= self.max_len:
            tokens += (self.max_len - seq_len) * [PAD]
        else:
            tokens = tokens[:self.max_len]
            seq_len = self.max_len
        ids = self.vocab(tokens)
        masks = [1] * seq_len + [0] * (self.max_len - seq_len)
        return ids, seq_len, masks


class CustomBertTokenizer(BasicTokenizer):
    def __init__(self, max_len=256, bert_path=None, data_path=None):
        super(CustomBertTokenizer, self).__init__(max_len)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def build_vocab(self, text_list=None, seed=None):
        self.vocab = {
            PAD: self.tokenizer.convert_tokens_to_ids([PAD])[0],
            UNK: self.tokenizer.convert_tokens_to_ids([UNK])[0],
            SEP: self.tokenizer.convert_tokens_to_ids([SEP])[0]
        }
        print('bert already have vocab')

    def load_vocab(self, text_list=None, seed=None):
        self.vocab = {
            PAD: self.tokenizer.convert_tokens_to_ids([PAD])[0],
            UNK: self.tokenizer.convert_tokens_to_ids([UNK])[0],
            SEP: self.tokenizer.convert_tokens_to_ids([SEP])[0]
        }
        print('bert already have vocab')

    def encode(self, text):
        result = self.tokenizer(text)
        result = self.tokenizer.pad(result, padding='max_length', max_length=self.max_len)
        ids = result['input_ids']
        mask = result['attention_mask']
        seq_len = sum(mask)
        # SEP_IDX = self.tokenizer.convert_tokens_to_ids([SEP])
        SEP_IDX = self.tokenizer.vocab[SEP]
        if seq_len > self.max_len:
            ids = ids[:self.max_len - 1] + [SEP_IDX]
            mask = mask[:self.max_len]
            seq_len = self.max_len
        return ids, seq_len, mask


class VectorTokenizer(BasicTokenizer):
    def __init__(self, max_len=256, vector_path=None, data_path=None, name=None):
        super(VectorTokenizer, self).__init__(max_len, data_path)
        self.vector_path = vector_path
        self.vectors = None
        self.name = name

    def build_vocab(self, text_list=None, seed=None):
        vec = Vectors(self.vector_path)
        self.vocab = vocab(vec.stoi, min_freq=0)
        self.vocab.append_token(UNK)
        self.vocab.append_token(PAD)
        self.vocab.set_default_index(self.vocab[UNK])
        unk_vec = torch.mean(vec.vectors, dim=0).unsqueeze(0)
        pad_vec = torch.zeros(vec.vectors.shape[1]).unsqueeze(0)
        self.vectors = torch.cat([vec.vectors, unk_vec, pad_vec])
        if self.name:
            torch.save(self.vocab, self.data_path + '{}_vocab'.format(self.name))
            torch.save(self.vectors, self.data_path + self.name)
        else:
            torch.save(self.vocab, self.data_path + 'vector_vocab')
            torch.save(self.vectors, self.data_path + 'vectors')

    def load_vocab(self, text_list=None, seed=None):
        try:
            if self.name:
                self.vocab = torch.load(self.data_path + '{}_vocab'.format(self.name))
                self.vectors = torch.load(self.data_path + self.name)
            else:
                self.vocab = torch.load(self.data_path + 'vector_vocab')
                self.vectors = torch.load(self.data_path + 'vectors')
        except Exception as e:
            print(e)
            self.build_vocab()


def make_data(data_source, config, seed=True, graph=None):
    dataProcessor = DataProcessor(data_source, seed=int(seed), model_config=config)
    if os.path.exists(dataProcessor.data_cat_path + 'split_data') and (graph is not None):
        print('split data already exist')
    else:
        dataProcessor.split_data(by='time', fixed_num=config['fixed_num'], time=config['time'],
                                 cut_time=config['cut_time'])
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])
    if graph:
        dataProcessor.get_feature_graph(config['tokenizer_type'], config['tokenizer_path'],
                                        mode=graph, split_abstract=config['split_abstract'],
                                        time_list=range(config['time'][0] - config['time_length'] + 1,
                                                        config['cut_time'] + 1))


def add_graphs(data_source, config, seed=True, graph='vector', name='graph_sample_feature'):
    dataProcessor = DataProcessor(data_source, seed=int(seed), model_config=config)
    graph_name = name + '_' + graph + '_'
    files = [file for file in os.listdir(dataProcessor.data_cat_path)
             if file.startswith(graph_name) and file.endswith('.job')]
    existed_times = set([int(file.split(graph_name)[1].split('.')[0]) for file in files])
    new_times = [pub_time for pub_time in range(config['time'][0] - config['time_length'] + 1, config['cut_time'] + 1)
                 if pub_time not in existed_times]

    print(sorted(list(existed_times)), new_times)
    dataProcessor.get_feature_graph(config['tokenizer_type'], config['tokenizer_path'],
                                    name=name,
                                    mode=graph, split_abstract=config['split_abstract'],
                                    time_list=new_times)


def get_edge_dict(srcs, dsts, values=None):
    src_dst_dict = defaultdict(list)
    for i in range(len(srcs)):
        if values:
            src_dst_dict[srcs[i]].append((dsts[i], values[i]))
        else:
            src_dst_dict[srcs[i]].append(dsts[i])
    return src_dst_dict


def get_citation_cascade(graph, target, nodes):
    # print(nodes)
    # target_id = graph.nodes['paper'].data[dgl.NID][target].item()
    subgraph = dgl.node_subgraph(graph, {'paper': [target] + list(set(nodes))})
    src, dst = subgraph.edges(form='uv', etype='is cited by', order='srcdst')
    # print(src, dst)
    # special_index = np.intersect1d(np.argwhere(src.numpy() > 0)[:, 0], np.argwhere(dst.numpy() > 0)[:, 0]).tolist()
    # print(special_index)
    special_index = np.argwhere(src.numpy() > 0)[:, 0].tolist()
    src_s = src[special_index].numpy().tolist()
    dst_s = dst[special_index].numpy().tolist()
    nodes_counter = Counter(dst_s)
    oids = subgraph.nodes['paper'].data[dgl.NID].numpy().tolist()
    # print(nodes_counter)
    # target_time = subgraph.nodes['paper'].data['time'][0].item()

    dealt_src, dealt_dst = [], []
    for dst_node in dst.numpy().tolist():
        if (dst_node not in nodes_counter) and (dst_node != 0):
            dealt_src.append(0)
            dealt_dst.append(dst_node)
    add_nodes = [0] + dealt_dst.copy()

    for node, count in nodes_counter.items():
        if (count == 1) & (node != 0):
            dealt_dst.append(node)
            dealt_src.append(src_s[dst_s.index(node)])
        elif node != 0:
            # cur_time = subgraph.nodes['paper'].data['time'][node].item()
            cur_index = np.argwhere(np.array(dst_s) == node)[:, 0].tolist()
            src_nodes = np.array(src_s)[cur_index]
            time_value = subgraph.nodes['paper'].data['time'][src_nodes][:, 0].numpy()
            max_value = np.max(time_value)
            max_index = np.argwhere(time_value == max_value)[:, 0].tolist()
            final_nodes = set(src_nodes[max_index])
            # if cur_time == max_value:
            #     for src_node in final_nodes:
            #         dealt_dst.append(src_node)
            #         dealt_src.append(0)

            for src_node in final_nodes:
                dealt_dst.append(node)
                dealt_src.append(src_node)

    src_counter = Counter(src_s)
    edge_counter = Counter(list(zip(src_s, dst_s)) + list(zip(dst_s, src_s)))
    edge_counter = dict([(key, value) for key, value in edge_counter.items() if value > 1])
    for edge in edge_counter:
        add_nodes_set = set(add_nodes)
        if (edge[0] not in add_nodes_set) & (edge[1] not in add_nodes_set):
            if src_counter[edge[0]] == src_counter[edge[1]]:
                add_nodes.extend([edge[0], edge[1]])
                dealt_dst.append(edge[1])
                dealt_src.append(0)
                dealt_dst.append(edge[0])
                dealt_src.append(0)
            elif src_counter[edge[0]] < src_counter[edge[1]]:
                # add_nodes.append(edge[1])
                add_nodes.extend([edge[0], edge[1]])
                dealt_dst.append(edge[1])
                dealt_src.append(0)
            else:
                # add_nodes.append(edge[0])
                add_nodes.extend([edge[0], edge[1]])
                dealt_dst.append(edge[0])
                dealt_src.append(0)

    dealt_graph = nx.DiGraph()
    # all_nodes = subgraph.nodes('paper').numpy().tolist()
    all_nodes = [oids[node] for node in subgraph.nodes('paper').numpy().tolist()]

    dealt_graph.add_nodes_from(all_nodes)
    dealt_src = [oids[node] for node in dealt_src]
    dealt_dst = [oids[node] for node in dealt_dst]
    dealt_graph.add_edges_from(set(zip(dealt_src, dealt_dst)))
    # print(dealt_graph)
    shortest_paths = [[target]]
    target_nodes = [node for node in all_nodes if node != target]
    nc_nodes = [node for node in target_nodes if not nx.has_path(dealt_graph, source=target, target=node)]
    while len(nc_nodes) > 0:
        max_target_node = nc_nodes[np.argmax([src_counter[node] for node in nc_nodes])]
        dealt_graph.add_edge(target, max_target_node)
        nc_nodes = [node for node in nc_nodes if not nx.has_path(dealt_graph, source=target, target=node)]

    # for node in all_nodes[1:]:
    for node in target_nodes:
        try:
            shortest_paths.extend(
                [p for p in nx.all_shortest_paths(dealt_graph, source=target, target=node)])
        except Exception as e:
            print(e)
            print(target)
            print(target, nodes)
            print(subgraph.nodes['paper'].data[dgl.NID].numpy().tolist())
            print(src, dst)
            print(dealt_src, dealt_dst)
            raise Exception
    return shortest_paths


def get_simple_graph(filename, srcs, dsts):
    fw = open(filename, 'w+')
    cur_src = -1
    for src, dst in zip(srcs, dsts):
        if src != dst:
            if cur_src == -1:
                fw.write('{}:{}'.format(str(src), str(dst)))
                print('start writing {}'.format(filename))
                cur_src = src
            elif cur_src != src:
                fw.write('\n')
                fw.write('{}:{}'.format(str(src), str(dst)))
                cur_src = src
            else:
                fw.write(',' + str(dst))


def add_times(data_path, graphs, graph_name):
    # joblib.load(self.data_cat_path + graph_name + '_' + str(time) + '.job')
    for pub_time in tqdm(graphs['all_graphs']):
        cur_graph = graphs['all_graphs'][pub_time]
        papers = cur_graph.nodes('paper')
        trans_dict = dict(zip(cur_graph.nodes['time'].data[dgl.NID].numpy().tolist(),
                              cur_graph.nodes('time').numpy().tolist()))
        times = torch.tensor([trans_dict[t]
                              for t in cur_graph.nodes['paper'].data['time'].squeeze(dim=-1).numpy().tolist()],
                             dtype=torch.long)
        if 'oid' not in cur_graph.nodes['time'].data:
            cur_graph.nodes['time'].data['oid'] = cur_graph.nodes['time'].data[dgl.NID]

        if 'is shown in' not in cur_graph.etypes:
            edges = {
                ('paper', 'is shown in', 'time'): (papers, times),
                ('time', 'shows', 'paper'): (times, papers)
            }
            cur_graph = add_new_elements(cur_graph, edges=edges)
            print(cur_graph)
            joblib.dump(cur_graph, data_path + graph_name + '_' + str(pub_time) + '.job')


def add_graph_feats(data_source, graph_type=None):
    # add co-cited/citing strength to the graph
    data_path = './data/{}/'.format(data_source)
    graphs = [file for file in os.listdir(data_path) if file.startswith('graph_sample') and not file.endswith('embs')]
    if graph_type:
        graphs = [file for file in graphs if graph_type in file]
    graphs = sorted(graphs)[1:]
    logging.basicConfig(level=logging.INFO,
                        filename='./results/{}/add_feats_{}.log'.format(data_source, graph_type),
                        filemode='w+',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        force=True)
    logger = logging.getLogger()
    # LOG = logging.getLogger(__name__)
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for graph_name in graphs:
        print(graph_name)
        save_indicator = False
        if graph_name == 'graph_sample':
            cur_graph = torch.load(data_path + graph_name)
        else:
            cur_graph = joblib.load(data_path + graph_name)
        for edge in [('paper', 'cites', 'paper'), ('paper', 'is cited by', 'paper')]:
            cites_graph = cur_graph[edge]
            if 'r_sim' not in cites_graph.edata:
                save_indicator = True
                adj = cites_graph.adj()
                adj_T = torch.sparse_coo_tensor(indices=torch.flipud(adj.coalesce().indices()),
                                                values=adj.coalesce().values(),
                                                size=adj.size())
                r_sim = torch.mm(adj, adj_T) * adj
                print(r_sim.shape)
                edges = cites_graph.edges()
                edges = list(zip(*map(lambda x: x.numpy().tolist(), edges)))
                selector = SparseSelecter(r_sim)
                start_time = time.time()
                all_edata = [selector.get_single_data(edge) for edge in tqdm(edges, total=len(edges),
                                                                             file=tqdm_out, mininterval=30
                                                                             )]
                all_edata = torch.stack(all_edata, dim=0)

                print(time.time() - start_time)
                print(all_edata.shape)
                cites_graph.edata['r_sim'] = all_edata
        if save_indicator:
            if graph_name == 'graph_sample':
                torch.save(cur_graph, data_path + graph_name)
            else:
                joblib.dump(cur_graph, data_path + graph_name)


def job2dgl(data_source, graph_type):
    data_path = './data/{}/'.format(data_source)
    graph_list = sorted([graph_name for graph_name in os.listdir(data_path)
                         if graph_name.endswith('.job') and graph_type in graph_name])
    print(graph_list)
    for graph_name in tqdm(graph_list):
        print(graph_name)
        graph = joblib.load(data_path + graph_name)
        dgl.save_graphs(data_path + graph_name.split('.')[0] + '.dgl', [graph])


def job2jsonl(data_source):
    ckpt_path = './checkpoints/{}/'.format(data_source)
    data_list = sorted([data_name for data_name in os.listdir(ckpt_path)
                        if data_name.startswith('DDHGCN_graphs_20') and data_name.endswith('.job')])
    # time.sleep(10)
    dealt_list = [data_name.split('.')[0] for data_name in os.listdir(ckpt_path) if data_name.endswith('.jsonl')]
    for data_name in data_list:
        last_time = time.time()
        print(data_name)
        if data_name.split('.')[0] not in dealt_list:
            data = joblib.load(ckpt_path + data_name)

            cur_time = time.time()
            print('load ckpt:', cur_time - last_time)

            with jl.open(ckpt_path + data_name.split('.')[0] + '.jsonl', mode='w') as fw:
                for line in data:
                    fw.write(line)
            del data


def load_test(model_config, data_source):
    # if data_source == 'geography':
    #     selected_ids = {'2729020', '595522', '207565498', '7555152', '5242952', '130777312', '16207466', '6970474',
    #                     '43411435', '56474104'}
    # else:
    #     # selected_ids = None
    #     data = torch.load('./data/{}/split_data'.format(data_source))
    #     selected_ids = set(data['test'][0][:100])
    # model_config = configs[model]
    dataProcessor = DataProcessor(data_source, time=model_config['time'], log=True,
                                  model_config=model_config)
    phases = ['train']
    graph_dict = dataProcessor.load_graphs(model_config['graph_name'], time_length=model_config['time_length'],
                                           phases=phases)

    dataProcessor.get_tokenizer(model_config['tokenizer_type'], model_config['tokenizer_path'])
    dataloaders = dataProcessor.get_dataloader(2, model_config['num_workers'],
                                               graph_dict=graph_dict)
    dataloader = dataloaders[0]
    for i in range(int(model_config['epochs'])):
        for idx, batch in enumerate(dataloader):
            # if idx == 5:
            #     break
            print(idx, batch)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')

    parser.add_argument('--phase', default='job2jsonl', help='the function name.')
    parser.add_argument('--data_source', default='geography', help='the data source.')
    parser.add_argument('--seed', default=123, help='the data seed.')
    parser.add_argument('--graph', default='sbert', help='the graph embed.')
    parser.add_argument('--structure', default='group', help='the cascade structure.')

    args = parser.parse_args()
    print(args)
    data_source = args.data_source
    configs = get_configs(data_source, [])

    if args.phase == 'test':
        data_processor = DataProcessor(data_source='geography')
    elif args.phase == 'make_data':
        temp_config = configs['default']
        make_data(args.data_source, temp_config, seed=args.seed)
    elif args.phase == 'make_data_graph':
        temp_config = configs['default']
        if args.graph == 'vector':
            temp_config['tokenizer_type'] = 'glove'
            temp_config['tokenizer_path'] = './data/glove'
        elif args.graph == 'bert':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/scibert/'
        elif args.graph == 'bw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/scibert/'
        elif args.graph == 'sbert':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-MiniLM-L12-v2/'
        elif args.graph == 'sbw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-mpnet-base-v2/'
        elif args.graph == 'specter2':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/specter2/'
        else:
            temp_config['tokenizer_type'] = 'glove'
            temp_config['tokenizer_path'] = './data/glove'
        make_data(args.data_source, temp_config, seed=args.seed, graph=args.graph)
    elif args.phase == 'get_embs':
        temp_config = configs['default']
        if args.graph == 'bw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/scibert/'
        elif args.graph == 'sbw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-mpnet-base-v2/'
        elif args.graph == 'sbert':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-MiniLM-L12-v2/'
        elif args.graph == 'specter2':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/specter2/'
        dataProcessor = DataProcessor(args.data_source, norm=True)
        dataProcessor.get_embs(temp_config['tokenizer_type'], temp_config['tokenizer_path'],
                               mode=args.graph, split_abstract=temp_config['split_abstract'])
    elif args.phase == 'add_attrs':
        dataProcessor = DataProcessor(args.data_source, norm=True)
        dataProcessor.add_attrs()
    elif args.phase == 'show_graph_info':
        dataProcessor = DataProcessor(args.data_source, norm=True)
        dataProcessor.show_graph_info()
    elif args.phase == 'add_times':
        graph_name = configs['default']['graph_name']
        if args.graph:
            graph_name = '_'.join(configs['default']['graph_name'].split('_')[:-1]
                                  + [args.graph.split('+')[0]])
        dataProcessor = DataProcessor(args.data_source, norm=True, time=configs['default']['time'])
        graphs = dataProcessor.load_graphs(graph_name, time_length=configs['default']['time_length'])
        add_times(dataProcessor.data_cat_path, graphs, graph_name)
    elif args.phase == 'add_graphs':
        temp_config = configs['default']
        if args.graph == 'vector':
            temp_config['tokenizer_type'] = 'glove'
            temp_config['tokenizer_path'] = './data/glove'
        elif args.graph == 'bert':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/scibert/'
        elif args.graph == 'bw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/scibert/'
        elif args.graph == 'sbert':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-MiniLM-L12-v2/'
        elif args.graph == 'sbw':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/all-mpnet-base-v2/'
        elif args.graph == 'specter2':
            temp_config['tokenizer_type'] = 'bert'
            temp_config['tokenizer_path'] = './bert/specter2/'
        else:
            temp_config['tokenizer_type'] = 'glove'
            temp_config['tokenizer_path'] = './data/glove'
        add_graphs(args.data_source, temp_config, seed=args.seed, graph=args.graph)
    elif args.phase == 'graph_feats':
        add_graph_feats(args.data_source, args.graph)
    elif args.phase == 'job2dgl':
        job2dgl(args.data_source, args.graph)
    elif args.phase == 'job2jsonl':
        job2jsonl(args.data_source)
    elif args.phase == 'load_test':
        model_config = configs[args.model]
        if args.walk_method:
            model_config['walk_method'] = args.walk_method
        print(model_config)
        load_test(model_config, args.data_source)

    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
