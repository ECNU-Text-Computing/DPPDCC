from data_processor import *
from trainers.AdvTrainer import AdvTrainer
from trainers.BaseTrainer import BaseTrainer
from trainers.PretrainedTrainer import PretrainedTrainer
import torch.distributed as dist
from utilis.scripts import get_configs, setup_seed, ResultsContainer, get_model_config
import torch.multiprocessing as mp

LOAD_JOBLIB = True


def get_trainer(model_name, config, vectors=None, use_amp=False, **kwargs):
    print(config)
    trainer = BaseTrainer(model_name, config, vectors, use_amp=use_amp,
                          cut_threshold=config['cut_threshold'], aux_weight=config['aux_weight'],
                          eval_interval=config.get('eval_interval', 5),
                          **kwargs)
    print(">>>>>>>model name:" + trainer.model_name)
    return trainer


def load_data_and_model_wodl(data_source, model_name, config, seed=123, norm=False, log=False, log_tag=None,
                             phases=None,
                             cl=False, use_amp=False, use_dist=False,
                             **kwargs):
    record_path = './results/{}/'.format(data_source)
    save_path = './checkpoints/{}/'.format(data_source) if config['save_model'] else None
    if config['type'] == 'normal':
        dataProcessor = DataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm,
                                      time=config['time'],
                                      model_config=config, log=log)
    else:
        raise NotImplementedError

    print(config['tokenizer_type'])
    config['model_name'] = model_name
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])

    config['vocab_size'] = len(dataProcessor.tokenizer.vocab)
    config['pad_idx'] = dataProcessor.tokenizer.vocab[PAD]
    # trainer = get_model(model_name, config, dataProcessor.tokenizer.vectors, **kwargs)
    trainer = get_trainer(model_name, config, dataProcessor.tokenizer.vectors, use_amp=use_amp, **kwargs)
    # model = trainer.model
    trainer.get_criterion(config['criterion'])
    if log_tag:
        file_name = record_path + '{}_{}.log'.format(trainer.model_name, log_tag)
    else:
        file_name = record_path + '{}.log'.format(trainer.model_name)
    logging.basicConfig(level=logging.INFO,
                        filename=file_name,
                        filemode='w+',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        force=True)

    print('>>>here<<<', model_name)
    if config['use_graph']:
        if ('DPPDCC' in model_name) | ('saved' in str(config['batch_graph'])):
            graph_dict = dataProcessor.load_graphs(config['graph_name'], time_length=config['time_length'],
                                                   phases=phases)
            # print(graph_dict)
        else:
            train_time, val_time, test_time = dataProcessor.time
            graph_dict = {
                'data': {phase: None for phase in phases},
                'node_trans': json.load(open(dataProcessor.data_cat_path + 'sample_node_trans.json', 'r')),
                'time': {'train': train_time, 'val': val_time, 'test': test_time},
                'all_graphs': None,
                'time_length': config['time_length'],
                'data_path': dataProcessor.data_cat_path
            }
    else:
        graph_dict = {
            'data': {phase: None for phase in phases}}

    dealt_graph_dict = get_graphs(model_name, config, trainer, dataProcessor, graph_dict)
    # print(dealt_graph_dict.keys())

    emb_name = config.get('emb_name', None)
    if emb_name and dealt_graph_dict:
        dealt_graph_dict['emb_path'] = dataProcessor.data_cat_path + emb_name + '_embs'

    dealt_graph_dict = trainer.load_graphs(dealt_graph_dict)
    # batch_graphs = dealt_graph_dict if config['batch_graph'] else None
    batch_graphs = dealt_graph_dict if dealt_graph_dict else graph_dict
    return dataProcessor, dealt_graph_dict, batch_graphs, trainer, config, save_path, record_path


def load_data_and_model(data_source, model_name, config, seed=123, norm=False, log=False, log_tag=None, phases=None,
                        cl=False, use_amp=False, use_dist=False,
                        **kwargs):
    dataProcessor, dealt_graph_dict, batch_graphs, trainer, config, save_path, record_path = load_data_and_model_wodl(
        data_source, model_name, config, seed, norm, log, log_tag, phases, cl, use_amp, use_dist, **kwargs)

    dataProcessor.get_dataloader(config['batch_size'], num_workers=config['num_workers'], graph_dict=batch_graphs,
                                 device='cuda', use_dist=use_dist, selected_phases=phases)

    return dataProcessor, dealt_graph_dict, trainer, config, save_path, record_path


def train_single(data_source, model_name, config, seed=123, norm=False, log=False, aux=False, use_amp=False, **kwargs):
    phases = ['train', 'val', 'test']
    if config['test_interval'] == 0:
        phases.remove('test')
    if config.get('eval_interval', 5) == 0:
        phases.remove('val')
    dataProcessor, dealt_graph_dict, trainer, config, save_path, record_path = load_data_and_model(
        data_source, model_name, config, seed=seed, norm=norm, log=log, aux=aux, phases=phases, use_amp=use_amp,
        **kwargs)

    trainer.train_batch(dataloaders=dataProcessor.dataloaders, epochs=config['epochs'],
                        lr=config['lr'], weight_decay=config['weight_decay'], criterion=config['criterion'],
                        optimizer=config['optimizer'],
                        record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                        test_interval=config['test_interval'], best_metric=config['best_metric'], aux=aux,
                        sampler=dataProcessor.train_sampler)
    if save_path:
        if config['test_interval'] > 0:
            trainer.get_test_results(dataloader=dataProcessor.dataloaders[-1],
                                     record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                                     best_metric=config['best_metric'], aux=aux)
        else:
            del dataProcessor, dealt_graph_dict
            test_results(data_source, model_name, config, seed=seed, norm=norm, log=log, aux=aux, use_amp=use_amp,
                         **kwargs)


def test_results(data_source, model_name, config, seed=123, norm=False, log=False, aux=False, use_amp=False, **kwargs):
    phases = ['test']
    dataProcessor, dealt_graph_dict, trainer, config, save_path, record_path = load_data_and_model(
        data_source, model_name, config, seed=seed, norm=norm, log=log, log_tag='test', use_amp=use_amp,
        phases=phases, **kwargs)

    trainer.get_test_results(dataloader=dataProcessor.dataloaders[0],
                             record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                             best_metric=config['best_metric'], aux=aux)


def show_results(data_source, model_name, config, seed=123, norm=False, log=False, aux=False, use_amp=False,
                 show_phase='test',
                 show='weight', **kwargs):
    phases = [show_phase]
    dataProcessor, dealt_graph_dict, trainer, config, save_path, record_path = load_data_and_model(
        data_source, model_name, config, seed=seed, norm=norm, log=log, log_tag='show', phases=phases, use_amp=use_amp,
        **kwargs)
    trainer.get_show_results(dataloader=dataProcessor.dataloaders[0],
                             record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                             best_metric=config['best_metric'], aux=aux, phase=show_phase, show=show)


def selected_results(data_source, model_name, config, seed=123, norm=False, log=False, aux=False, use_amp=False,
                     show_phase='test',
                     **kwargs):
    phases = [show_phase]
    dataProcessor, dealt_graph_dict, trainer, config, save_path, record_path = load_data_and_model(
        data_source, model_name, config, seed=seed, norm=norm, log=log, log_tag='show', phases=phases, use_amp=use_amp,
        **kwargs)
    trainer.get_show_results(dataloader=dataProcessor.dataloaders[0],
                             record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                             best_metric=config['best_metric'], aux=aux, phase=show_phase, show='graph')



def get_graphs(model_name, config, trainer, dataProcessor, graph_dict, force=False, selected=False):
    if config['use_graph'] == 'one-time':
        # graph_dict = dataProcessor.load_graphs(config['graph_name'])
        dealt_graph_dict = trainer.deal_graphs(graph_dict, data_path=dataProcessor.data_cat_path + 'split_data',
                                               log_path='./results/{}/{}.log'.format(dataProcessor.data_source,
                                                                                     model_name))
    elif config['use_graph'] == 'repeatedly':
        # graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, model.model_name)
        graph_type = config.get('graph_type', config['graph_name'].split('_')[-1])
        graphs_path = './checkpoints/{}/{}_graphs'.format(dataProcessor.data_source, model_name)
        # if model_name.endswith('TSGCN'):
        #     graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, 'TSGCN')
        print(model_name)

        if 'DDHGCN' in model_name:
            graphs_path = graphs_path.replace(model_name, 'DDHGCN')
            model_name = 'DDHGCN'
        elif 'DPPDCC' in model_name:
            graphs_path = graphs_path.replace(model_name, 'DPPDCC')
            model_name = 'DPPDCC'

        if config['batch_graph'] in ['saved', 'saved_pyg', 'saved_dynamic']:
            # graphs_path = './checkpoints/{}/{}_graphs_{}'.format(dataProcessor.data_source,
            #                                                      config['batch_graph'], graph_type)
            graphs_path = './checkpoints/{}/{}_graphs'.format(dataProcessor.data_source,
                                                              config['batch_graph'])
            if config['hop'] != 2:
                graphs_path += '_h{}'.format(config['hop'])
                print('>>>here:', graphs_path)

        print('>>>graph_path:', graphs_path)

        if (os.path.exists(graphs_path)) & (not force):
            try:
                if model_name == 'DPPDCC':
                    # graphs_path = './checkpoints/{}/{}_graphs'.format(data_source, 'TSGCN')
                    # dealt_graph_dict = dataProcessor.load_graphs(config['graph_name'])
                    dealt_graph_dict = graph_dict
                    for phase in dealt_graph_dict['data']:
                        dealt_graph_dict['data'][phase] = None
                    split_data = torch.load(dataProcessor.data_cat_path + 'split_data')
                    # ids
                    dealt_graph_dict['selected_papers'] = {}
                    # print(dealt_graph_dict['data'])
                    for phase in dealt_graph_dict['data']:
                        dealt_graph_dict['selected_papers'][phase] = split_data[phase][0]
                    del split_data
                    dealt_graph_dict['graphs_path'] = graphs_path
                else:
                    dealt_graph_dict = graph_dict
                    dealt_graph_dict['graphs_path'] = graphs_path
                    phases = list(dealt_graph_dict['data'].keys())
                    for phase in phases:
                        dealt_graph_dict['data'][phase] = joblib.load(graphs_path + '_{}.job'.format(phase))
            except Exception as e:
                print(e)
                dealt_graph_dict = joblib.load(graphs_path)
            print('load graphs successfully!')
            if 'time' not in dealt_graph_dict:
                dealt_graph_dict['time'] = dataProcessor.get_time_dict()
            if 'all_graphs' in dealt_graph_dict:
                if dealt_graph_dict['all_graphs'] is None:
                    print('adding all graphs')
                    # graph_dict = dataProcessor.load_graphs(config['graph_name'])
                    dealt_graph_dict['all_graphs'] = graph_dict['all_graphs']
                    del graph_dict
        else:
            dealt_graph_dict = trainer.deal_graphs(graph_dict, data_path=dataProcessor.data_cat_path + 'split_data',
                                                   path=graphs_path,
                                                   log_path='./results/{}/{}_dealing.log'.format(
                                                       dataProcessor.data_source, model_name))
        dealt_graph_dict['data_source'] = dataProcessor.data_source
    else:
        dealt_graph_dict = None

    return dealt_graph_dict


def get_model_graph_data(data_source, model_name, config, seed=123, norm=False):
    if config['type'] == 'normal':
        dataProcessor = DataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm, time=config['time'],
                                      model_config=config)
    else:
        raise NotImplementedError
    print(config['tokenizer_type'])
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])

    config['vocab_size'] = len(dataProcessor.tokenizer.vocab)
    config['pad_idx'] = dataProcessor.tokenizer.vocab[PAD]
    # model = get_model(model_name, config, dataProcessor.tokenizer.vectors)
    trainer = get_trainer(model_name, config, dataProcessor.tokenizer.vectors)
    print('time_length:', config['time_length'])
    graph_dict = dataProcessor.load_graphs(config['graph_name'], time_length=config['time_length'])
    get_graphs(model_name, config, trainer, dataProcessor, graph_dict, force=True)


def get_best_values(data_source, metric, ascending=True):
    data_path = './results/{}/'.format(data_source)
    records = [file for file in os.listdir(data_path) if file.endswith('.csv')]
    temp_list = []
    for record in records:
        print(record)
        df = pd.read_csv((data_path + record))
        # print(df.columns)
        try:
            df = df.sort_values(by=metric, ascending=ascending).head(1)
            df['model'] = [record.split('_records')[0]]
            df = df[['model'] + list(df.columns[:-1])]
            # df.columns = ['model'] + list(df.columns[:-1])
            temp_list.append(df)
            print(df)
        except Exception as e:
            continue
    result_df = pd.DataFrame(columns=temp_list[0].columns)
    for record in temp_list:
        result_df = result_df.append(record)
    result_df.to_excel(data_path + 'all_{}_best.xlsx'.format(metric))


def get_detail_csv(data_source):
    ckpt_path = './checkpoints/{}/'.format(data_source)
    files = [file for file in os.listdir(ckpt_path) if file.endswith('detail.pkl')]
    for file in files:
        print(file)
        detail = joblib.load(ckpt_path + file)
        # ids = detail[0].numpy() if type(detail[0]) == torch.Tensor else detail[0]
        df = pd.DataFrame(data={'id': detail[0], 'time': detail[1],
                                'topic': detail[2][0].squeeze(axis=-1), 'pop': detail[2][1].squeeze(axis=-1),
                                'contri': detail[2][2].squeeze(axis=-1)})
        print(df.head())
        print(df.describe())
        df.to_csv('./results/{}/{}.csv'.format(data_source, file.split('.')[0]))


def ddp_setup(rank, world_size, port=12355):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '{}'.format(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def ddp_run(rank, world_size, cur_model, model_config, args):
    # rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp_setup(rank, world_size, args.port)
    print('before train:', int(dist.get_rank()))
    torch.cuda.set_device(dist.get_rank())
    train_single(args.data_source, cur_model, model_config, args.seed, args.norm, args.log, args.aux, args.use_amp,
                 oargs=args.oargs)
    dist.destroy_process_group()


def ddp_train(rank, world_size, dataProcessor, dealt_graph_dict, batch_graphs, trainer, config, save_path, record_path,
              args):
    ddp_setup(rank, world_size, args.port)
    print('before train:', int(dist.get_rank()))
    torch.cuda.set_device(dist.get_rank())
    dataProcessor.get_dataloader(config['batch_size'], num_workers=config['num_workers'], graph_dict=batch_graphs,
                                 device='cuda', use_dist=True)
    trainer.dist_model(cur_model, config, dataProcessor.tokenizer.vectors,
                       cut_threshold=config['cut_threshold'], aux_weight=config['aux_weight'])
    trainer.train_batch(dataloaders=dataProcessor.dataloaders, epochs=config['epochs'],
                        lr=config['lr'], weight_decay=config['weight_decay'], criterion=config['criterion'],
                        optimizer=config['optimizer'],
                        record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                        test_interval=config['test_interval'], best_metric=config['best_metric'], aux=args.aux,
                        sampler=dataProcessor.train_sampler)
    dist.destroy_process_group()


def ddp_test(rank, world_size, dataProcessor, dealt_graph_dict, batch_graphs, trainer, config, save_path, record_path,
             args):
    ddp_setup(rank, world_size, args.port)
    print('before test:', int(dist.get_rank()))
    torch.cuda.set_device(dist.get_rank())
    dataProcessor.get_dataloader(config['batch_size'], num_workers=config['num_workers'], graph_dict=batch_graphs,
                                 device='cuda', use_dist=True)
    trainer.dist_model(cur_model, config, dataProcessor.tokenizer.vectors,
                       cut_threshold=config['cut_threshold'], aux_weight=config['aux_weight'])
    trainer.get_test_results(dataloader=dataProcessor.dataloaders[0],
                             record_path=record_path, save_path=save_path, graph=dealt_graph_dict,
                             best_metric=config['best_metric'], aux=args.aux)
    dist.destroy_process_group()


def load_test(dataloader, trainer, phase='train'):
    cur_start = time.time()
    print('len:', len(dataloader))
    count_num = min(100, len(dataloader))
    sleep_time = 0.5
    forward_time = 0
    backward_time = 0
    cuda_time = 0
    max_gpu_memory = 0
    print('count_num:', count_num)

    model = trainer.model
    model.phase = phase
    trainer.epoch = 10
    model.epoch = 10
    trainer.container = ResultsContainer(100, len(dataloader), 10, model.model_name, 'train',
                                         test_interval=model.test_interval)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} parameters.')
    if phase == 'train':
        model = model.train()
    else:
        model = model.eval()

    for idx, batch in enumerate(dataloader):
        if idx == count_num:
            break
        else:
            if idx == 0:
                batch_size = len(batch.ids)
            print(idx)
            last_time = time.time()
            # save batch
            # joblib.dump(batch, './test/test_batch')
            x, values, lengths, masks, ids, times, blocks = model.get_batch_input(batch, None)
            cur_time = time.time()
            temp_time = cur_time - last_time
            print('cuda time:', temp_time)
            cuda_time += temp_time
            last_time = cur_time
            # print input here
            # print('here', blocks)
            # print(masks.shape)
            # with torch.no_grad():
            #     rst = model(x, lengths, masks, ids, blocks, times)
            with (torch.autocast(device_type='cuda', dtype=torch.float16, enabled=trainer.use_amp)):
                if phase == 'train':
                    predicted_values, aux_output, other_loss = model(x, lengths, masks, ids, blocks, times)
                else:
                    with torch.no_grad():
                        predicted_values, aux_output, other_loss = model.predict(x, lengths, masks, ids, blocks, times)
                loss = trainer.get_reg_loss_and_result(values, predicted_values, model.criterion,
                                                       log=dataloader.log,
                                                       mean=dataloader.mean, std=dataloader.std)
                if type(other_loss) == dict:
                    other_loss = other_loss['loss']
                if other_loss is not None:
                    loss = loss + other_loss
                cur_time = time.time()
                temp_time = cur_time - last_time
                print('forward time:', temp_time)
                forward_time += temp_time
                last_time = cur_time

                max_gpu_memory += torch.cuda.memory_allocated() / 1024 ** 2
                print("before backward:{:.2f}MB".format(torch.cuda.memory_allocated() / 1024 ** 2))

                if phase == 'train':
                    loss.backward()
                    cur_time = time.time()
                    temp_time = cur_time - last_time
                    print('backward time:', temp_time)
                    backward_time += temp_time
                    last_time = cur_time
                    # time.sleep(sleep_time)
                    # joblib.dump(batch, './test/test_batch')
                    print("after backward:{:.2f}MB".format(torch.cuda.memory_allocated() / 1024 ** 2))

    all_time = time.time() - cur_start
    print('=' * 59)
    print('per batch:', all_time / count_num)
    print('per data:', (all_time - cuda_time - forward_time - backward_time) / count_num)
    print('per cuda:', cuda_time / count_num)
    print('per forward:', forward_time / count_num)
    print('per backward:', backward_time / count_num)
    print('gpu time:', (forward_time + backward_time) / count_num)
    print('batch size:', batch_size)
    print('per sample:', (forward_time + backward_time) / count_num / batch_size)
    print('per max_gpu_memory:{:.2f}MB'.format(max_gpu_memory / count_num))
    print(f'The model has {num_params:,} parameters.')
    # print('per pure batch:', (time.time() - cur_start - sleep_time * count_num) / count_num)
    rst_dict = {}
    rst_dict['model_name'] = model.model_name
    rst_dict['batch_size'] = batch_size
    rst_dict['params'] = num_params
    rst_dict['batch'] = all_time / count_num
    rst_dict['data'] = (all_time - cuda_time - forward_time - backward_time) / count_num
    rst_dict['cuda'] = cuda_time / count_num
    rst_dict['forward'] = forward_time / count_num
    rst_dict['backward'] = backward_time / count_num
    rst_dict['gpu'] = (forward_time + backward_time) / count_num
    rst_dict['sample'] = (forward_time + backward_time) / count_num / batch_size
    rst_dict['gpu_memory'] = max_gpu_memory / count_num
    return rst_dict


if __name__ == '__main__':
    phase_start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')

    parser.add_argument('--phase', default='DPPDCC', help='the function name.')
    parser.add_argument('--ablation', default=None, help='the ablation modules.')
    parser.add_argument('--data_source', default='geography', help='the data source.')
    parser.add_argument('--norm', default=False, help='the data norm.')
    parser.add_argument('--log', default=True, help='apply log on the objective.')
    parser.add_argument('--mode', default=None, help='the model mode.')
    parser.add_argument('--type', default='orthog', help='the model type.')
    parser.add_argument('--seed', default=123, help='the data seed.')
    parser.add_argument('--model_seed', default=123, help='the model seed.')
    parser.add_argument('--model', default='DPPDCC', help='the selected model for other methods.')
    parser.add_argument('--model_path', default=None, help='the selected model for deep analysis.')
    parser.add_argument('--pred_type', default='snapshot', help='how to get the final emb.')
    parser.add_argument('--graph_type', default='sbert', help='the graph type to learn.')
    parser.add_argument('--emb_type', default=None, help='the graph type to learn.')
    parser.add_argument('--cl_type', default=None, help='the cl type to learn.')
    parser.add_argument('--aug_type', default='cg', help='the aug type to learn.')
    parser.add_argument('--aug_rate', default=0.1, help='the aug type to learn.')
    parser.add_argument('--encoder_type', default='CCompGATSM', help='the gcn encoder type to learn.')
    parser.add_argument('--gcn_out', default='mean', help='how to get the node emb.')
    parser.add_argument('--n_layers', default=4, help='the layers of gcn.')
    parser.add_argument('--time_layers', default=4, help='the layers of time transformer.')
    parser.add_argument('--edge_seq', default=None, help='the edge sequences.')
    parser.add_argument('--inter_mode', default='attn', help='the edge sequences.')
    parser.add_argument('--time_length', default=None, help='the edge sequences.')
    parser.add_argument('--lr', default=None, help='the learning rate.')
    parser.add_argument('--optimizer', default=None, help='the optimizer.')
    parser.add_argument('--aux', default=False, help='add the aux classification task.')
    parser.add_argument('--cl_w', default=None, help='cl weight.')
    parser.add_argument('--aux_w', default=None, help='aux weight.')
    parser.add_argument('--hop', default=None, help='the subgraph hop.')
    # cur modeling
    parser.add_argument('--tau', default=None, help='hard negative sampling method.')
    parser.add_argument('--oargs', default=None, help='other args to the model.')
    parser.add_argument('--show', default='weight', help='other args to the model.')
    parser.add_argument('--hidden_dim', default=None, help='other args to the model.')
    parser.add_argument('--sem', default='gate', help='snapshot encoder mode.')
    parser.add_argument('--topic_module', default='scl', help='topic encoder mode.')
    parser.add_argument('--pop_module', default='accum', help='pop encoder mode.')
    parser.add_argument('--time_pooling', default='sum', help='time pooling method.')
    parser.add_argument('--n_topics', default=10, help='prefetch factor of dataloader.')
    parser.add_argument('--focus', default=False, help='focus cl drop.')
    parser.add_argument('--use_constraint', default=False, help='use value constraint.')
    parser.add_argument('--tree', default=False, help='if use all the k-hop information of papers.')
    parser.add_argument('--residual', default=True, help='if use the skip-connect in CompGAT.')
    parser.add_argument('--dn', default=0.3, help='if use the dropout for disentanglement.')
    parser.add_argument('--gdn', default=0, help='if use the dropout for gcn.')
    parser.add_argument('--ac', default=None, help='if ablation.')
    parser.add_argument('--gn', default='ln', help='if add graph norm.')
    parser.add_argument('--unmixed', default=False, help='if add graph norm.')
    parser.add_argument('--updater', default='normal', help='the snapshot embs updater.')
    parser.add_argument('--adaptive_lr', default=False, help='if use adaptive lr.')
    parser.add_argument('--lw', default=0.5, help='the lambda weight of the CCompGATS.')
    parser.add_argument('--tw', default=0.5, help='the loss weight of topic.')
    parser.add_argument('--pw', default=0.5, help='the loss weight of pop.')
    parser.add_argument('--dw', default=0.5, help='the loss weight of disen.')
    parser.add_argument('--etc', default=None, help='something.')
    # training
    parser.add_argument('--num_workers', default=1, help='snapshot encoder mode.')
    parser.add_argument('--prefetch_factor', default=2, help='prefetch factor of dataloader.')
    parser.add_argument('--batch_size', default=None, help='the batch size.')
    parser.add_argument('--epochs', default=None, help='the epochs.')
    parser.add_argument('--port', default=12355, help='the local port.')
    parser.add_argument('--use_amp', default=False, help='if use amp.')

    args = parser.parse_args()
    print('args', args)
    print('data_seed', args.seed)
    # setup_seed(int(args.seed))
    MODEL_SEED = int(args.model_seed)
    setup_seed(MODEL_SEED)
    print('model_seed', MODEL_SEED)
    torch.set_num_threads(8)

    model_list = ['DDHGCNSCL', 'DPPDCC']
    data_source = args.data_source
    configs = get_configs(data_source, model_list)

    configs['DPPDCC'] = copy.deepcopy(configs['DDHGCNSCL'])
    configs['DPPDCC']['batch_graph'] = 'dppdcc'

    if args.phase in model_list:
        cur_model = args.phase
    elif args.model in model_list:
        cur_model = args.model
    elif args.phase in {'all_load_test', 'all_load_eval_test'}:
        print('well')
        cur_model = 'default'
    else:
        print('what are you doing')
        raise Exception
    print(cur_model)
    model_config = get_model_config(configs, cur_model, args)


    if args.phase == 'test':
        print('test')
    elif args.phase in model_list:
        train_single(args.data_source, args.phase, model_config, args.seed, args.norm, args.log, args.aux, args.use_amp,
                     oargs=args.oargs)
    elif args.phase == 'get_model_graph_data':
        get_model_graph_data(args.data_source, args.model, model_config, args.seed, args.norm)
    elif args.phase == 'test_results':
        test_results(args.data_source, args.model, model_config, args.seed, args.norm, args.log, args.aux, args.use_amp,
                     oargs=args.oargs)
    elif args.phase == 'show_results':
        show_results(args.data_source, args.model, model_config, args.seed, args.norm, args.log, args.aux, args.use_amp,
                     oargs=args.oargs, show=args.show)
    elif args.phase == 'selected_results':
        selected_results(args.data_source, args.model, model_config, args.seed, args.norm, args.log, args.aux,
                         args.use_amp,
                         oargs=args.oargs)
    elif args.phase == 'load_test':
        dataProcessor, dealt_graph_dict, trainer, config, save_path, record_path = \
            load_data_and_model(args.data_source, args.model, model_config, args.seed, args.norm, args.log,
                                use_amp=args.use_amp,
                                phases=['train'], oargs=args.oargs)
        # phases=['val'], oargs=args.oargs)
        print(dataProcessor.dataloaders)
        dataloader = dataProcessor.dataloaders[0]
        rst_dict = load_test(dataloader, trainer, phase='train')
        print(rst_dict)
        df = pd.DataFrame(index=[rst_dict['model_name']], data=rst_dict)
        df.to_csv('./results/{}/{}_load_test.csv'.format(data_source, rst_dict['model_name']))

    elif args.phase == 'load_eval_test':
        dataProcessor, dealt_graph_dict, trainer, config, save_path, record_path = \
            load_data_and_model(args.data_source, args.model, model_config, args.seed, args.norm, args.log,
                                use_amp=args.use_amp,
                                # phases=['train'], oargs=args.oargs)
                                phases=['test'], oargs=args.oargs)
        print(dataProcessor.dataloaders)
        dataloader = dataProcessor.dataloaders[0]
        rst_dict = load_test(dataloader, trainer, phase='test')
        print(rst_dict)
        df = pd.DataFrame(index=[rst_dict['model_name']], data=rst_dict)
        df.to_csv('./results/{}/{}_load_eval_test.csv'.format(data_source, rst_dict['model_name']))

    elif args.phase == 'detail_csv':
        get_detail_csv(args.data_source)
    elif args.phase == 'train_info':
        dataProcessor, dealt_graph_dict, trainer, config, save_path, record_path = \
            load_data_and_model(args.data_source, args.model, model_config, args.seed, args.norm, args.log,
                                phases=['train'], oargs=args.oargs)
        model = trainer.model
        model.get_train_info(dealt_graph_dict['data']['train'], record_path)
    elif args.phase == 'dist_test':
        world_size = torch.cuda.device_count()
        mp.spawn(ddp_run, args=(world_size, cur_model, model_config, args,), nprocs=world_size, join=True)

    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - phase_start_time).seconds))

    print('Done main!')
