import argparse
import datetime

import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from layers.common_layers import MLP, CapsuleNetwork
from layers.GCN.custom_gcn import CompGAT, CCompGAT, CCompGATS, GATO, CompGATM, CCompGATSM


def get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=None, **kwargs):
    # TODO [APPNP, GCN2, PNA, DGN]
    if encoder_type == 'GCN':
        return dglnn.GraphConv(in_feat, hidden_feat, norm='right', allow_zero_in_degree=True)
    elif encoder_type == 'GIN':
        return dglnn.GINConv(MLP(in_feat, hidden_feat, activation=activation))
    elif encoder_type == 'NGIN':
        apply_func = nn.Sequential(MLP(in_feat, hidden_feat, activation=None),
                                   nn.LayerNorm(hidden_feat))
        return dglnn.GINConv(apply_func, activation=activation)
    elif encoder_type == 'GAT':
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        return dglnn.GATv2Conv(in_feat, out_feat, num_heads=nheads, activation=activation, allow_zero_in_degree=True)
    elif encoder_type == 'GATO':
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        residual = kwargs.get('residual', False)
        dropout = kwargs.get('dropout', 0)
        use_norm = kwargs.get('graph_norm', False)
        return GATO(in_feat, out_feat, num_heads=nheads, activation=activation, residual=residual, feat_drop=dropout,
                    allow_zero_in_degree=True, use_norm=use_norm)
    elif encoder_type == 'APPNP':
        k = kwargs.get('k', 5)
        alpha = kwargs.get('alpha', 0.5)
        return dglnn.APPNPConv(k=k, alpha=alpha)
    elif encoder_type == 'GCN2':
        return dglnn.GCN2Conv(in_feat, layer, allow_zero_in_degree=True, activation=activation)
    elif encoder_type == 'SAGE':
        return dglnn.SAGEConv(in_feat, hidden_feat, 'mean', activation=activation)
    elif encoder_type == 'PNA':
        return dglnn.PNAConv(in_feat, hidden_feat, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)
    elif encoder_type == 'DGN':
        return dglnn.DGNConv(in_feat, hidden_feat, ['dir1-av', 'dir1-dx', 'sum'], ['identity', 'amplification'], 2.5)
    elif encoder_type == 'CompGAT':
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        residual = kwargs.get('residual', False)
        dropout = kwargs.get('dropout', 0)
        use_norm = kwargs.get('graph_norm', False)
        return CompGAT(in_feat, out_feat, nheads,
                       feat_drop=dropout, residual=residual,
                       activation=activation, allow_zero_in_degree=True, use_norm=use_norm)
    elif encoder_type == 'CompGATM':
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        residual = kwargs.get('residual', False)
        dropout = kwargs.get('dropout', 0)
        use_norm = kwargs.get('graph_norm', False)
        return CompGATM(in_feat, out_feat, nheads,
                        feat_drop=dropout, residual=residual,
                        activation=activation, allow_zero_in_degree=True, use_norm=use_norm)
    elif encoder_type == 'CCompGAT':
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        # out_feat = kwargs.get('out_feat', 32)
        # nheads = hidden_feat // out_feat
        residual = kwargs.get('residual', False)
        dropout = kwargs.get('dropout', 0)
        indicator_edge_attr = kwargs.get('indicator_edge_attr', None)
        norm_node_attr = kwargs.get('norm_node_attr', None)
        reverse = kwargs.get('reverse', False)
        use_norm = kwargs.get('graph_norm', False)
        return CCompGAT(indicator_edge_attr, norm_node_attr, in_feat, out_feat, nheads, reverse=reverse,
                        feat_drop=dropout, residual=residual,
                        activation=activation, allow_zero_in_degree=True, use_norm=use_norm)
    elif encoder_type == 'CCompGATS':
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        # out_feat = kwargs.get('out_feat', 32)
        # nheads = hidden_feat // out_feat
        residual = kwargs.get('residual', False)
        dropout = kwargs.get('dropout', 0)
        indicator_edge_attr = kwargs.get('indicator_edge_attr', None)
        norm_node_attr = kwargs.get('norm_node_attr', None)
        reverse = kwargs.get('reverse', False)
        use_norm = kwargs.get('graph_norm', False)
        lambda_weight = kwargs.get('lambda_weight', 0.5)
        return CCompGATS(indicator_edge_attr, norm_node_attr, in_feat, out_feat, nheads, reverse=reverse,
                         feat_drop=dropout, residual=residual, lambda_weight=lambda_weight,
                         activation=activation, allow_zero_in_degree=True, use_norm=use_norm)
    elif encoder_type == 'CCompGATSM':
        nheads = kwargs.get('nheads', 4)
        out_feat = hidden_feat // nheads
        # out_feat = kwargs.get('out_feat', 32)
        # nheads = hidden_feat // out_feat
        residual = kwargs.get('residual', False)
        dropout = kwargs.get('dropout', 0)
        indicator_edge_attr = kwargs.get('indicator_edge_attr', None)
        norm_node_attr = kwargs.get('norm_node_attr', None)
        reverse = kwargs.get('reverse', False)
        use_norm = kwargs.get('graph_norm', False)
        lambda_weight = kwargs.get('lambda_weight', 0.5)
        return CCompGATSM(indicator_edge_attr, norm_node_attr, in_feat, out_feat, nheads, reverse=reverse,
                          feat_drop=dropout, residual=residual, lambda_weight=lambda_weight,
                          activation=activation, allow_zero_in_degree=True, use_norm=use_norm)


class CustomGCN(nn.Module):
    def __init__(self, encoder_type, in_feat, hidden_feat, layer, activation=None, **kwargs):
        super(CustomGCN, self).__init__()
        self.encoder_type = encoder_type
        self.fc_out = None
        # self.conv = get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=activation)
        self.conv = get_gcn_layer(encoder_type, in_feat, hidden_feat, layer, activation=activation)

        if 'GAT' in self.encoder_type:
            self.fc_out = nn.Linear(hidden_feat, hidden_feat)
        elif self.encoder_type in ['APPNP', 'GCN2']:
            self.fc_out = nn.Linear(in_feat, hidden_feat)

    def forward(self, graph, feat, edge_weight=None, **kwargs):
        '''
        :param graph:
        :param feat: tuple of src and dst
        :param edge_weight:
        :param kwargs:
        :return: hidden_embs: {ntype: [N, out_feat]}
        '''
        # print('wip')
        # print(self.encoder_type)
        attn = None
        if self.encoder_type == 'GAT':
            feat = self.conv(graph, feat, **kwargs)
        else:
            feat = self.conv(graph, feat, edge_weight=edge_weight, **kwargs)
        if 'GAT' in self.encoder_type:
            if kwargs.get('get_attention', False):
                feat, attn = feat
            num_nodes = feat.shape[0]
            feat = feat.reshape(num_nodes, -1)
        if self.fc_out:
            feat = self.fc_out(feat)
        if attn is not None:
            return feat, attn
        else:
            return feat


class BasicHeteroGCN(nn.Module):
    def __init__(self, encoder_type, in_feat, out_feat, ntypes, etypes, layer, activation=None, **kwargs):
        super(BasicHeteroGCN, self).__init__()

        if encoder_type == 'HGT':
            nheads = kwargs.get('nheads', 4)
            self.conv = dglnn.HGTConv(in_feat, out_feat // nheads, nheads,
                                      num_ntypes=len(ntypes), num_etypes=len(etypes))
        else:
            self.conv = dglnn.HeteroGraphConv({etype: get_gcn_layer(encoder_type, in_feat, out_feat, layer,
                                                                    activation=activation) for etype in etypes})
        self.encoder_type = encoder_type
        print(self.encoder_type)
        self.fc_out = None
        if self.encoder_type == 'GAT':
            self.fc_out = dglnn.HeteroLinear({ntype: out_feat for ntype in ntypes}, out_feat)
        elif self.encoder_type in ['APPNP', 'GCN2']:
            self.fc_out = dglnn.HeteroLinear({ntype: in_feat for ntype in ntypes}, out_feat)

    def forward(self, graph, feat, edge_weight=None, **kwargs):
        '''
        :param graph:
        :param feat:
        :param edge_weight:
        :param kwargs:
        :return: hidden_embs: {ntype: [N, out_feat]}
        '''
        # print('wip')
        # print('start', [torch.isnan(feat[x]).sum() for x in feat])
        # print(self.encoder_type)
        # print(self.conv)
        if self.encoder_type == 'HGT':
            # print(feat.shape)
            feat = self.conv(graph, feat, graph.ndata[dgl.NTYPE], graph.edata[dgl.ETYPE], presorted=True)
        if self.encoder_type == 'GAT':
            feat = self.conv(graph, feat)
        else:
            # feat = self.conv(graph, feat, edge_weight=edge_weight)
            feat = self.conv(graph, feat, mod_kwargs={'edge_weight': edge_weight})
            # print('end', [torch.isnan(feat[x]).sum() for x in feat])
        if self.encoder_type == 'GAT':
            for ntype in feat:
                num_nodes = feat[ntype].shape[0]
                feat[ntype] = feat[ntype].reshape(num_nodes, -1)
        if self.fc_out:
            feat = self.fc_out(feat)
        return feat


class CustomHeteroGraphConv(dglnn.HeteroGraphConv):
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        other_outputs = {}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {})
                )
                if etype in mod_kwargs:
                    dstdata, a, c = dstdata
                    other_outputs[etype] = [a, c]
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {})
                )
                if etype in mod_kwargs:
                    # print(len(dstdata))
                    # dstdata = dstdata[0]
                    dstdata, a, c = dstdata
                    other_outputs[etype] = [a, c]
                    # print(len(other_outputs[etype]))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        if len(other_outputs) > 0:
            return rsts, other_outputs
        else:
            return rsts


class CustomHeteroGCN(nn.Module):
    def __init__(self, encoder_type, in_feat, out_feat, ntypes, etypes, layer, activation=None, **kwargs):
        super(CustomHeteroGCN, self).__init__()
        etype_dict = None

        if encoder_type == 'HGT':
            nheads = kwargs.get('nheads', 4)
            self.conv = dglnn.HGTConv(in_feat, out_feat // nheads, nheads,
                                      num_ntypes=len(ntypes), num_etypes=len(etypes))
        elif encoder_type in {'CompGAT', 'CCompGAT', 'CCompGATS', 'GATO', 'CCompGATSN', 'CompGATM', 'CCompGATSM',
                              'CCompGATSMN'}:
            etype_dict = {}
            if 'special_edges' in kwargs:
                for cur_encoder_type in kwargs['special_edges']:
                    for etype in kwargs['special_edges'][cur_encoder_type]:
                        if type(kwargs['special_edges'][cur_encoder_type]) == dict:
                            cur_kwargs = kwargs['special_edges'][cur_encoder_type][etype]
                            # print(cur_kwargs)
                            etype_dict[etype] = get_gcn_layer(cur_encoder_type, in_feat, out_feat, layer,
                                                              activation=activation,
                                                              **cur_kwargs)
                        else:
                            etype_dict[etype] = get_gcn_layer(cur_encoder_type, in_feat, out_feat, layer,
                                                              activation=activation)
            # print('>>>>here!!!!see here!!!!<<<<', etype_dict)
            for etype in etypes:
                if etype not in etype_dict:
                    etype_dict[etype] = get_gcn_layer('GIN', in_feat, out_feat, layer, activation=activation)
        elif 'encoder_dict' in kwargs:
            etype_dict = {}
            encoder_dict = kwargs['encoder_dict']
            for special_encoder_type in encoder_dict:
                for etype in encoder_dict[special_encoder_type]:
                    etype_dict[etype] = get_gcn_layer(special_encoder_type, in_feat, out_feat, layer,
                                                      activation=activation)
            print(etype_dict)
            for etype in etypes:
                if etype not in etype_dict:
                    etype_dict[etype] = get_gcn_layer(encoder_type, in_feat, out_feat, layer, activation=activation)
        else:
            etype_dict = {etype: get_gcn_layer(encoder_type, in_feat, out_feat, layer,
                                               activation=activation) for etype in etypes}
        if etype_dict:
            # self.conv = dglnn.HeteroGraphConv(etype_dict)
            self.conv = CustomHeteroGraphConv(etype_dict)
        self.encoder_type = encoder_type
        # print(self.encoder_type)
        self.fc_out = None
        if self.encoder_type == 'GAT':
            self.fc_out = dglnn.HeteroLinear({ntype: out_feat for ntype in ntypes}, out_feat)
        elif self.encoder_type in ['APPNP', 'GCN2']:
            self.fc_out = dglnn.HeteroLinear({ntype: in_feat for ntype in ntypes}, out_feat)

    def forward(self, graph, feat, edge_weight=None, **kwargs):
        '''
        :param graph:
        :param feat:
        :param edge_weight:
        :param kwargs:
        :return: hidden_embs: {ntype: [N, out_feat]}
        '''
        # print('wip')
        other_outputs = None
        if self.encoder_type == 'HGT':
            # print(feat.shape)
            feat = self.conv(graph, feat, graph.ndata[dgl.NTYPE], graph.edata[dgl.ETYPE], presorted=True)
        else:
            if kwargs.get('show_detail', False):
                feat, other_outputs = self.conv(graph, feat, mod_kwargs={
                    'cites': {'get_attention': True},
                    'is cited by': {'get_attention': True}
                })
            else:
                feat = self.conv(graph, feat)
        if self.encoder_type == 'GAT':
            for ntype in feat:
                num_nodes = feat[ntype].shape[0]
                feat[ntype] = feat[ntype].reshape(num_nodes, -1)
        if self.fc_out:
            feat = self.fc_out(feat)
        if other_outputs:
            return feat, other_outputs
        else:
            return feat


class StochasticKLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, ntypes, etypes, k=2, residual=False, encoder_type='GCN'):
        super().__init__()
        self.residual = residual
        self.convs = nn.ModuleList()
        print(ntypes)
        self.skip_fcs = nn.ModuleList()
        self.conv_start = BasicHeteroGCN(encoder_type, in_feat, hidden_feat, ntypes, etypes, layer=1,
                                         activation=nn.LeakyReLU())
        self.skip_fc_start = dglnn.HeteroLinear({key: in_feat for key in ntypes}, hidden_feat)
        self.conv_end = BasicHeteroGCN(encoder_type, hidden_feat, out_feat, ntypes, etypes, layer=k,
                                       activation=nn.LeakyReLU())
        self.skip_fc_end = dglnn.HeteroLinear({key: hidden_feat for key in ntypes}, out_feat)
        self.convs.append(self.conv_start)
        self.skip_fcs.append(self.skip_fc_start)
        for i in range(k - 2):
            self.convs.append(BasicHeteroGCN(encoder_type, hidden_feat, hidden_feat, ntypes, etypes, layer=i + 2,
                                             activation=nn.LeakyReLU()))
            self.skip_fcs.append(dglnn.HeteroLinear({key: hidden_feat for key in ntypes}, hidden_feat))
        self.convs.append(self.conv_end)
        self.skip_fcs.append(self.skip_fc_end)

    def forward(self, graph, x):
        # print(graph.device)
        # print(x.device)
        # count = 0
        for i in range(len(self.convs)):
            # print('-' * 30 + str(count) + '-' * 30)
            # print(x)
            if self.residual:
                # x = x + conv(graph, x)
                out = self.convs[i](graph, x)
                x = self.skip_fcs[i](x)
                for key in x:
                    x[key] = x[key] + out[key]
            else:
                x = self.convs[i](graph, x)
            # print([torch.isnan(x[e]).sum() for e in x])
            # print(x)
            # count += 1
        return x

    def forward_block(self, blocks, x):
        for i in range(len(self.convs)):
            x = self.convs[i](blocks[i], x)
        return x


class CustomRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, ntypes, etypes, k=2, residual=False, encoder_type='GCN',
                 return_list=False, **kwargs):
        super().__init__()
        self.residual = residual
        self.convs = nn.ModuleList()
        self.conv_start = CustomHeteroGCN(encoder_type, in_feat, hidden_feat, ntypes, etypes, layer=1,
                                          activation=nn.LeakyReLU(), **kwargs)
        self.conv_end = CustomHeteroGCN(encoder_type, hidden_feat, out_feat, ntypes, etypes, layer=k,
                                        activation=nn.LeakyReLU(), **kwargs)
        self.convs.append(self.conv_start)
        for i in range(k - 2):
            self.convs.append(CustomHeteroGCN(encoder_type, hidden_feat, hidden_feat, ntypes, etypes, layer=i + 2,
                                              activation=nn.LeakyReLU(), **kwargs))
        self.convs.append(self.conv_end)
        self.return_list = return_list

    def forward(self, graph, x):
        output_list = []
        for conv in self.convs:
            if self.residual:
                x = x + conv(graph, x)
            else:
                x = conv(graph, x)
            if self.return_list:
                output_list.append(x)
        return x, output_list


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

