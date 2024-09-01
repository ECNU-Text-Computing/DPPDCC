# from our_models.DDHGCN import DDHGCN, DDHGCNS, DDHGCNSCL, DDHGCNSECL, DDHGCNSP, DDHGCNSCLT
from our_models.DDHGCN import DDHGCNSCL
from our_models.DPPDCC import DPPDCC


def get_model(model_name, config, vectors=None, device=None, **kwargs):
    model = None
    # if model_name == 'DDHGCN':
    #     model = DDHGCN(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
    #                    hidden_dim=config['hidden_dim'], model_type=config['model_type'],
    #                    time_length=config['time_length'], ntypes=config['ntypes'], etypes=config['etypes'],
    #                    start_times=config['time'], encoder_type=config['encoder_type'], n_layers=config['n_layers'],
    #                    time_embs_path='./checkpoints/{}/DDHGCN_graphs_{}.job'.format(config['data_source'],
    #                                                                                  config['graph_type']),
    #                    snapshot_mode=config['snapshot_mode'], save_graph=False, device=device,
    #                    config=config,
    #                    **kwargs)
    # elif model_name == 'DDHGCNS':
    #     model = DDHGCNS(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
    #                     hidden_dim=config['hidden_dim'], model_type=config['model_type'],
    #                     time_length=config['time_length'], ntypes=config['ntypes'], etypes=config['etypes'],
    #                     start_times=config['time'], encoder_type=config['encoder_type'], n_layers=config['n_layers'],
    #                     time_embs_path='./checkpoints/{}/DDHGCN_graphs_{}.job'.format(config['data_source'],
    #                                                                                   config['graph_type']),
    #                     snapshot_mode=config['snapshot_mode'], time_pooling=config['time_pooling'],
    #                     topic_module=config['topic_module'], pop_module=config['pop_module'], tree=config['tree'],
    #                     time_layers=config['time_layers'], n_topics=config['n_topics'], device=device,
    #                     ablation_channel=config['ablation_channel'], config=config, graph_norm=config['graph_norm'],
    #                     dropout=config['dropout'], graph_dropout=config['graph_dropout'], residual=config['residual'],
    #                     mixed=config['mixed'], updater=config['updater'], adaptive_lr=config['adaptive_lr'],
    #                     lambda_weight=config['lambda_weight'], loss_weights=config['loss_weights'],
    #                     use_constraint=config['use_constraint'],
    #                     **kwargs)
    # elif model_name == 'DDHGCNSECL':
    #     model = DDHGCNSECL(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
    #                        hidden_dim=config['hidden_dim'], model_type=config['model_type'],
    #                        time_length=config['time_length'], ntypes=config['ntypes'], etypes=config['etypes'],
    #                        start_times=config['time'], encoder_type=config['encoder_type'], n_layers=config['n_layers'],
    #                        time_embs_path='./checkpoints/{}/DDHGCN_graphs_{}.job'.format(config['data_source'],
    #                                                                                      config['graph_type']),
    #                        snapshot_mode=config['snapshot_mode'], time_pooling=config['time_pooling'],
    #                        topic_module='ecl', pop_module=config['pop_module'], tree=config['tree'],
    #                        time_layers=config['time_layers'], n_topics=config['n_topics'], device=device,
    #                        ablation_channel=config['ablation_channel'], config=config, graph_norm=config['graph_norm'],
    #                        dropout=config['dropout'], graph_dropout=config['graph_dropout'],
    #                        residual=config['residual'],
    #                        mixed=config['mixed'], updater=config['updater'], adaptive_lr=config['adaptive_lr'],
    #                        lambda_weight=config['lambda_weight'], loss_weights=config['loss_weights'],
    #                        use_constraint=config['use_constraint'],
    #                        **kwargs)
    # elif model_name == 'DDHGCNSCL' or model_name == 'DDHGCNSCLT':
    # elif model_name == 'DDHGCNSCL':
    #     model = DDHGCNSCL(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
    #                       hidden_dim=config['hidden_dim'], model_type=config['model_type'],
    #                       time_length=config['time_length'], ntypes=config['ntypes'], etypes=config['etypes'],
    #                       start_times=config['time'], encoder_type=config['encoder_type'], n_layers=config['n_layers'],
    #                       time_embs_path='./checkpoints/{}/DDHGCN_graphs_{}.job'.format(config['data_source'],
    #                                                                                     config['graph_type']),
    #                       aug_rate=config['aug_rate'], focus=config['focus'],
    #                       snapshot_mode=config['snapshot_mode'], time_pooling=config['time_pooling'],
    #                       time_layers=config['time_layers'], n_topics=config['n_topics'], tree=config['tree'],
    #                       topic_module=config['topic_module'], pop_module=config['pop_module'], device=device,
    #                       ablation_channel=config['ablation_channel'], config=config, graph_norm=config['graph_norm'],
    #                       dropout=config['dropout'], graph_dropout=config['graph_dropout'], residual=config['residual'],
    #                       mixed=config['mixed'], updater=config['updater'], adaptive_lr=config['adaptive_lr'],
    #                       lambda_weight=config['lambda_weight'], loss_weights=config['loss_weights'],
    #                       use_constraint=config['use_constraint'],
    #                       **kwargs)
    # elif model_name == 'DDHGCNSCLT':
    #     model = DDHGCNSCLT(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
    #                        hidden_dim=config['hidden_dim'], model_type=config['model_type'],
    #                        time_length=config['time_length'], ntypes=config['ntypes'], etypes=config['etypes'],
    #                        start_times=config['time'], encoder_type=config['encoder_type'], n_layers=config['n_layers'],
    #                        time_embs_path='./checkpoints/{}/DDHGCN_graphs_{}.job'.format(config['data_source'],
    #                                                                                      config['graph_type']),
    #                        aug_rate=config['aug_rate'], focus=config['focus'],
    #                        snapshot_mode=config['snapshot_mode'], time_pooling=config['time_pooling'],
    #                        time_layers=config['time_layers'], n_topics=config['n_topics'], tree=config['tree'],
    #                        topic_module=config['topic_module'], pop_module=config['pop_module'], device=device,
    #                        ablation_channel=config['ablation_channel'], config=config, graph_norm=config['graph_norm'],
    #                        dropout=config['dropout'], graph_dropout=config['graph_dropout'],
    #                        residual=config['residual'],
    #                        mixed=config['mixed'], updater=config['updater'], adaptive_lr=config['adaptive_lr'],
    #                        lambda_weight=config['lambda_weight'], loss_weights=config['loss_weights'],
    #                        use_constraint=config['use_constraint'],
    #                        **kwargs)
    # elif model_name == 'DDHGCNSP':
    #     model = DDHGCNSP(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
    #                      hidden_dim=config['hidden_dim'], model_type=config['model_type'],
    #                      time_length=config['time_length'], ntypes=config['ntypes'], etypes=config['etypes'],
    #                      start_times=config['time'], encoder_type=config['encoder_type'], n_layers=config['n_layers'],
    #                      time_embs_path='./checkpoints/{}/DDHGCN_graphs_{}.job'.format(config['data_source'],
    #                                                                                    config['graph_type']),
    #                      snapshot_mode=config['snapshot_mode'], time_pooling=config['time_pooling'],
    #                      tree=config['tree'],
    #                      pop_module=config['pop_module'], time_layers=config['time_layers'], device=device,
    #                      ablation_channel=config['ablation_channel'], config=config, graph_norm=config['graph_norm'],
    #                      dropout=config['dropout'], graph_dropout=config['graph_dropout'], residual=config['residual'],
    #                      mixed=config['mixed'], updater=config['updater'], adaptive_lr=config['adaptive_lr'],
    #                      lambda_weight=config['lambda_weight'], loss_weights=config['loss_weights'],
    #                      use_constraint=config['use_constraint'],
    #                      **kwargs)

    if model_name == 'DPPDCC':
        model = DPPDCC(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
                       hidden_dim=config['hidden_dim'], model_type=config.get('model_type', None),
                       time_length=config['time_length'], ntypes=config['ntypes'], etypes=config['etypes'],
                       start_times=config['time'], encoder_type=config['encoder_type'], n_layers=config['n_layers'],
                       time_embs_path='./checkpoints/{}/DDHGCN_graphs_{}.job'.format(config['data_source'],
                                                                                     config['graph_type']),
                       aug_rate=config['aug_rate'], focus=config['focus'],
                       snapshot_mode=config['snapshot_mode'], time_pooling=config['time_pooling'],
                       time_layers=config['time_layers'], n_topics=config['n_topics'], tree=config['tree'],
                       topic_module=config['topic_module'], pop_module=config['pop_module'], device=device,
                       ablation_channel=config['ablation_channel'], config=config, graph_norm=config['graph_norm'],
                       dropout=config['dropout'], graph_dropout=config['graph_dropout'],
                       residual=config['residual'],
                       mixed=config['mixed'], updater=config['updater'], adaptive_lr=config['adaptive_lr'],
                       lambda_weight=config['lambda_weight'], loss_weights=config['loss_weights'],
                       use_constraint=config['use_constraint'],
                       tau=config.get('tau', 0.2),
                       **kwargs)
    elif model_name == 'DDHGCNSCL':
        model = DDHGCNSCL(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
                          hidden_dim=config['hidden_dim'], model_type=config['model_type'],
                          time_length=config['time_length'], ntypes=config['ntypes'], etypes=config['etypes'],
                          start_times=config['time'], encoder_type=config['encoder_type'], n_layers=config['n_layers'],
                          time_embs_path='./checkpoints/{}/DDHGCN_graphs_{}.job'.format(config['data_source'],
                                                                                        config['graph_type']),
                          aug_rate=config['aug_rate'], focus=config['focus'],
                          snapshot_mode=config['snapshot_mode'], time_pooling=config['time_pooling'],
                          time_layers=config['time_layers'], n_topics=config['n_topics'], tree=config['tree'],
                          topic_module=config['topic_module'], pop_module=config['pop_module'], device=device,
                          ablation_channel=config['ablation_channel'], config=config, graph_norm=config['graph_norm'],
                          dropout=config['dropout'], graph_dropout=config['graph_dropout'], residual=config['residual'],
                          mixed=config['mixed'], updater=config['updater'], adaptive_lr=config['adaptive_lr'],
                          lambda_weight=config['lambda_weight'], loss_weights=config['loss_weights'],
                          use_constraint=config['use_constraint'],
                          **kwargs)
    else:
        raise NotImplementedError

    return model
