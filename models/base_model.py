import argparse
import datetime

from torch import nn
import time
import torch



class BaseModel(nn.Module):
    def __init__(self, vocab_size=0, embed_dim=0, num_classes=0, pad_index=0, word2vec=None, dropout=0.5,
                 model_path=None, device=None, **kwargs):
        super(BaseModel, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if not model_path:
            if word2vec is not None:
                # print(word2vec)
                self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False)
            elif vocab_size > 0:
                self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False, padding_idx=pad_index)
            else:
                self.embedding = None
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.model_name = 'BaseModel'
        self.adaptive_lr = False
        self.warmup = False
        self.T = kwargs['T'] if 'T' in kwargs.keys() else 400
        self.seed = None
        self.graphs_data = None
        self.node_trans = None
        self.epoch = 0
        self.relu_out = nn.ReLU()
        self.test_interval = None
        if 'seed' in kwargs.keys():
            self.seed = kwargs['seed']
        if self.warmup:
            print('warm up T', self.T)
        self.aux_weight = kwargs.get('aux_weight', 0.5)
        self.aux_criterion = nn.CrossEntropyLoss()
        self.cut_threshold = kwargs.get('cut_threshold', [0, 10])
        print(self.cut_threshold)
        self.eval_interval = kwargs.get('eval_interval', 5)
        self.config = kwargs.get('config', {})

    def get_model_name(self):
        model_seed = self.config.get('model_seed', 123)
        if model_seed != 123:
            self.seed = model_seed
            self.model_name += '_sd{}'.format(model_seed)
        print('>>>final model name:', self.model_name)
        return self.model_name


    def init_weights(self):
        print('init')

    def forward(self, content, lengths, masks, ids, graph, times, **kwargs):
        # return predicted values and classes
        return 'forward'

    def predict(self, content, lengths, masks, ids, graph, times, **kwargs):
        return self(content, lengths, masks, ids, graph, times, **kwargs)

    # def get_group_parameters(self, lr=0):
    #     return self.parameters()
    @staticmethod
    def get_group_parameters(model, lr=0):
        print('---------------here we use the base parameters')
        return model.parameters()

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
                # print([temp.is_pinned() for temp in blocks])
                # blocks = [temp.to(self.device) for temp in blocks]
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(self.device)
            else:
                blocks = blocks.to(self.device)
        if graph and type(ids) != torch.Tensor:
            ids = torch.tensor(list(map(lambda x: self.node_trans['paper'][x], ids)))
        # print('to input:', time.time() - start_time)
        # print('>>>>ccgl x:', x.device)
        return x, values, lengths, masks, ids, times, blocks

    def deal_graphs(self, graphs, data_path=None, path=None, log_path=None):
        # for different model, the graphs_data are different
        return graphs

    def load_graphs(self, graphs):
        return graphs

    def graph_to_device(self, graphs):
        # only for complete graph2device
        return graphs

    # def get_other_loss

    def get_aux_loss(self, aux_criterion, aux_weight, pred, gt, epoch=None):
        # aux_loss = self.aux_criterion(pred, gt) * self.aux_weight
        aux_loss = aux_criterion(pred, gt) * aux_weight
        return aux_loss

    def get_aux_pred(self, pred):
        # return pred.argmax(dim=-1)
        aux_pred = torch.softmax(pred, dim=-1)
        return aux_pred

    @staticmethod
    def get_selected_optimizer_type(optimizer):
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD
        elif optimizer == 'ADAM':
            optimizer = torch.optim.Adam
        elif optimizer == 'ADAMW':
            optimizer = torch.optim.AdamW
        else:
            optimizer = torch.optim.Adam
        print('>>>cur optimizer:', optimizer)
        return optimizer

    def get_optimizer(self, model, lr, optimizer, weight_decay=1e-3, autocast=False):
        eps = 1e-4 if autocast else 1e-8
        optimizer_type = self.get_selected_optimizer_type(optimizer)
        if optimizer == 'SGD':
            optimizer = optimizer_type(self.get_group_parameters(model, lr), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'ADAM':
            optimizer = optimizer_type(self.get_group_parameters(model, lr), lr=lr, eps=eps, weight_decay=weight_decay)
        elif optimizer == 'ADAMW':
            optimizer = optimizer_type(self.get_group_parameters(model, lr), lr=lr, eps=eps)
        else:
            optimizer = optimizer_type(self.get_group_parameters(model, lr), lr=lr, weight_decay=weight_decay)
        # self.optimizers = [self.optimizer]
        # return self.optimizer
        return [optimizer]


class PretrainedModel(BaseModel):
    def __init__(self, **kwargs):
        super(PretrainedModel, self).__init__(0, 0, 1, 0, **kwargs)

    def get_pretrained_loss(self, output):
        return output


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done base_model!')
