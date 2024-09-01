from trainers.BaseTrainer import *

class PretrainedTrainer(BaseTrainer):
    def __init__(self, model_name, config, vectors, **kwargs):
        super(PretrainedTrainer, self).__init__(model_name, config, vectors, **kwargs)

    def train_model(self, dataloader, epoch, criterion, optimizer, graph=None, aux=False):
        self.model.train()
        # torch.autograd = True
        log_interval = 100
        batch_count = len(dataloader)
        all_loss = []
        start_time = time.time()

        cur_graphs = None
        if graph:
            # load graphs_data
            self.node_trans = graph['node_trans']
            cur_graphs = graph['data']['train']
            cur_graphs = self.model.graph_to_device(cur_graphs)

        for idx, batch in enumerate(dataloader):
            x, values, lengths, masks, ids, times, blocks = self.get_batch_input(batch, graph)
            if blocks:
                cur_graphs = blocks

            # optimizer.zero_grad()
            self.optimizer_zero_grad()
            output = self.model(x, lengths, masks, ids, cur_graphs, times)
            # aux
            loss = self.model.get_pretrained_loss(output)
            # if self.model.other_loss:
            #     # print('adding other loss')
            #     loss = loss + self.model.other_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            # optimizer.step()
            self.optimizer_step()

            print_loss = loss.sum(dim=0).item()
            all_loss.append(print_loss)
            # print_loss = loss.item()
            if idx % log_interval == 0 and idx > 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| loss {:8.3f}'.format(epoch, idx, batch_count,
                                              print_loss))
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches '
                             '| loss {:8.3f}'.format(epoch, idx, batch_count,
                                                     print_loss))

        elapsed = time.time() - start_time
        avg_loss = np.mean(all_loss)
        # print(all_loss)
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | loss {:8.4f} |'.format(epoch, elapsed, avg_loss))
        logging.info('-' * 59)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | loss {:8.4f} |'.format(epoch, elapsed, avg_loss))


        results = [avg_loss]

        return results

    def train_batch(self, dataloaders, epochs, lr=1e-3, weight_decay=1e-4, criterion='MSE', optimizer='ADAM',
                    scheduler=False, record_path=None, save_path=None, graph=None, test_interval=1, best_metric='male',
                    aux=False, sampler=None):
        final_results = []
        print(dataloaders)
        train_dataloader, _, _ = dataloaders

        # criterion = self.get_criterion(criterion)
        # encoder_optimizer, decoder_optimizer= self.get_optimizer(lr, optimizer)
        optimizer = self.get_optimizer(lr, optimizer, weight_decay=weight_decay)
        metrics, records_logger = self.get_metric_and_record(aux, record_path)

        interval_best = None
        all_interval_best = None
        eval_interval = test_interval if test_interval > 0 else self.eval_interval
        print(eval_interval, test_interval)
        self.test_interval = test_interval
        self.record_path = record_path
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            logging.basicConfig(level=logging.INFO,
                                filename=record_path + '{}_epoch_{}.log'.format(self.model.model_name, epoch),
                                filemode='w+',
                                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                force=True)
            train_results = self.train_model(train_dataloader, epoch, criterion, optimizer, graph, False)
            cur_value = train_results[0]
            interval_best, all_interval_best = self.get_result_and_model(interval_best, all_interval_best, best_metric,
                                                                         cur_value, epoch, eval_interval, save_path)

            all_results = train_results
            records_logger.info(','.join([str(epoch)] + [str(round(x, 6)) for x in all_results]))

        # fw.close()
        return final_results

    def get_metric_and_record(self, aux, record_path):
        records_logger = logging.getLogger('records')
        # records_logger.setLevel(logging.DEBUG)
        if self.seed:
            print('{}_records_{}.csv'.format(self.model.model_name, self.seed))
            # fw = open(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), 'w')
            # fh = logging.FileHandler(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), mode='w+')
            record_name = '{}_records_{}.csv'.format(self.model.model_name, self.seed)

        else:
            # fw = open(record_path + '{}_records.csv'.format(self.model_name), 'w')
            # fh = logging.FileHandler(record_path + '{}_records.csv'.format(self.model_name), mode='w+')
            record_name = '{}_records.csv'.format(self.model.model_name)
        # if aux:
        #     record_name = record_name.replace('.csv', '_aux.csv')

        fh = logging.FileHandler(record_path + record_name, mode='w+')
        records_logger.handlers = []
        records_logger.addHandler(fh)
        metrics = ['loss']
        raw_header = ',' + ','.join(['{}_' + metric for metric in metrics])
        records_logger.warning('epoch' + raw_header.replace('{}', 'train'))
        return metrics, records_logger

    def get_result_and_model(self, interval_best, all_interval_best, best_metric, cur_value, epoch,
                             eval_interval, save_path):
        if interval_best:
            if cur_value <= interval_best:
                interval_best = cur_value
                print('new checkpoint! epoch {}!'.format(epoch))
                if save_path:
                    self.save_model(save_path + '{}_{}.pkl'.format(self.model.model_name, (epoch - 1) // eval_interval))
        else:
            interval_best = cur_value
            print('new checkpoint! epoch {}!'.format(epoch))
            if save_path:
                self.save_model(save_path + '{}_{}.pkl'.format(self.model.model_name, (epoch - 1) // eval_interval))

        if all_interval_best is None:
            all_interval_best = cur_value
            if save_path:
                self.save_model(save_path + '{}.pkl'.format(self.model.model_name))
        else:
            if cur_value <= all_interval_best:
                all_interval_best = cur_value
                print('global new checkpoint! epoch {}!'.format(epoch))
                if save_path:
                    self.save_model(save_path + '{}.pkl'.format(self.model.model_name))
        return interval_best, all_interval_best
