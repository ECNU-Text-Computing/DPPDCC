import math

import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax

from layers.common_layers import MLP, PositionalEncoding

class GATO(dglnn.GATv2Conv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False,
                 use_norm=False):
        super(GATO, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual,
                                   activation, allow_zero_in_degree, bias, share_weights)
        self.fc_out = nn.Linear(num_heads * out_feats, num_heads * out_feats)
        self.final_norm = None
        if use_norm:
            self.final_norm = nn.LayerNorm(num_heads * out_feats)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
            graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)
            # message passing
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            rst = self.fc_out(rst.view(-1, self._out_feats * self._num_heads))
            if self.final_norm:
                # print(self.final_norm)
                rst = self.final_norm(rst)
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
class CompGAT(dglnn.GATv2Conv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False,
                 use_norm=False):
        super(CompGAT, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual,
                                   activation, allow_zero_in_degree, bias, share_weights)
        self.mix_w = nn.Parameter(torch.FloatTensor(size=(num_heads, out_feats * 2, out_feats)))
        self.mix_b = nn.Parameter(torch.FloatTensor(size=(num_heads, out_feats)))
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.fc_out = nn.Linear(num_heads * out_feats, num_heads * out_feats)
        self.init_weights()
        # if residual:
        #     print('residual start!')
        # if feat_drop:
        #     print('feat drop!')
        self.final_norm = None
        if use_norm:
            self.final_norm = nn.LayerNorm(num_heads * out_feats)


    def init_weights(self):
        # super(CompGAT, self).reset_parameters()
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.kaiming_uniform_(self.mix_w, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mix_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.mix_b, -bound, bound)

    def get_mix_feat(self, edges):
        feat = torch.cat((edges.src['feat_src'], edges.dst['feat_dst']), dim=-1)
        # print(feat.shape)
        # too slow for backward
        # feat = torch.einsum('bai,aio->bao', feat, self.mix_w) + self.mix_b
        # print(feat.shape)
        output = [None] * self._num_heads
        for i in range(self._num_heads):
            output[i] = torch.matmul(feat[:, i], self.mix_w[i])
        feat = torch.stack(output, dim=1) + self.mix_b
        return {'m': feat * edges.data['a']}

    def forward(self, graph, feat, get_attention=False, **kwargs):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        with graph.local_scope():
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'el': el, 'feat_src': feat_src})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': er, 'feat_dst': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

            # print(graph.edata['et'].shape)
            e = self.leaky_relu(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            # e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # (num_edge, num_heads)
            # message passing
            # graph.apply_edges(self.get_mix_feat)
            graph.update_all(self.get_mix_feat,
                             fn.sum('m', 'ft'))
            # graph.edata.pop('m')
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            # if self.activation:
            #     rst = self.activation(rst)

            rst = self.fc_out(rst.view(-1, self._out_feats * self._num_heads))
            if self.final_norm:
                rst = self.final_norm(rst)
            # activation 2
            if self.activation:
                rst = self.activation(rst)
            # print(rst.shape)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class CompGATM(CompGAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.mix_w
        del self.mix_b
        # simple yet effective version
        self.fc_mix = nn.Linear(2 * self._num_heads * self._out_feats, self._num_heads * self._out_feats)

    def get_mix_feat(self, edges):
        feat = torch.cat((edges.src['feat_src'], edges.dst['feat_dst']), dim=-1)
        feat = (self.fc_mix(feat.view(-1, 2 * self._num_heads * self._out_feats))
                .view(-1, self._num_heads, self._out_feats))
        return {'m': feat * edges.data['a']}

class CCompGAT(CompGAT):
    def __init__(self,
                 indicator_edge_attr,
                 norm_node_attr,
                 in_feats,
                 out_feats,
                 num_heads,
                 reverse=False,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False,
                 use_norm=False, **kwargs):
        super(CCompGAT, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual,
                                   activation, allow_zero_in_degree, bias, share_weights, use_norm)
        self.degree_trans = PositionalEncoding(out_feats * num_heads, max_len=100 + 1)
        self.indicator_edge_attr = indicator_edge_attr
        self.norm_node_attr = norm_node_attr
        self.reverse = reverse
        self.attn_c = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.init_attn_c()
        # if self.reverse:
        #     print('reversed:', norm_node_attr)
        # if residual:
        #     print('residual start!')
        # if feat_drop:
        #     print('feat drop!')

    def init_attn_c(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_c, gain=gain)

    def get_critical_weight(self, edges):
        # print(edges.data[self.indicator_edge_attr])
        # print(edges.dst[self.norm_node_attr])
        # print(self.indicator_edge_attr, self.norm_node_attr)
        norm_value = torch.clamp(edges.dst[self.norm_node_attr], min=1).detach()
        cw = torch.round((edges.data[self.indicator_edge_attr] / norm_value) * 100).long().detach()
        # print(torch.max(cw))
        if self.reverse:
            cw = 100 - cw
        # print(cw)
        ce = self.degree_trans(cw).view(-1, self._num_heads, self._out_feats)
        c = (ce * self.attn_c).sum(dim=-1).unsqueeze(-1)
        e = c + edges.data['e']
        return {'e': e}

    def forward(self, graph, feat, get_attention=False, **kwargs):
        with graph.local_scope():
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'el': el, 'feat_src': feat_src})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': er, 'feat_dst': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            # add critical embedding
            graph.apply_edges(self.get_critical_weight)

            # print(graph.edata['et'].shape)
            e = self.leaky_relu(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            # e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # (num_edge, num_heads)
            # message passing
            # graph.apply_edges(self.get_mix_feat)
            graph.update_all(self.get_mix_feat,
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            # if self.activation:
            #     rst = self.activation(rst)

            rst = self.fc_out(rst.view(-1, self._out_feats * self._num_heads))
            if self.final_norm:
                rst = self.final_norm(rst)
            if self.activation:
                rst = self.activation(rst)
            # print(rst.shape)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class CCompGATS(CCompGAT):
    def __init__(self, *args, **kwargs):
        super(CCompGATS, self).__init__(*args, **kwargs)
        self.attn_c = None
        self.degree_trans = None
        self.lambda_weight = kwargs.get('lambda_weight', 0.5)
        # print('>>>lambda weight:', self.lambda_weight)

    def get_critical_weight(self, edges):
        # print(edges.data[self.indicator_edge_attr])
        # print(edges.dst[self.norm_node_attr])
        norm_value = torch.clamp(edges.dst[self.norm_node_attr], min=1).detach()
        cw = (edges.data[self.indicator_edge_attr] / norm_value).detach().unsqueeze(dim=-1).unsqueeze(dim=-1)
        # print(torch.max(cw))
        if self.reverse:
            cw = 1 - cw
        # print(cw)
        return {'cw': cw}

    def forward(self, graph, feat, get_attention=False, **kwargs):
        with graph.local_scope():
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'el': el, 'feat_src': feat_src})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': er, 'feat_dst': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            # add critical embedding
            graph.apply_edges(self.get_critical_weight)

            # print(graph.edata['et'].shape)
            e = self.leaky_relu(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            # e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # compute softmax
            e = self.attn_drop(edge_softmax(graph, e))  # (num_edge, num_heads)
            c = graph.edata.pop('cw')
            c = edge_softmax(graph, c)
            # graph.edata['a'] = (c + e) / 2
            graph.edata['a'] = self.lambda_weight * c + (1 - self.lambda_weight) * e
            # print(e.shape, c.shape, graph.edata['a'].shape)
            # print(e.sum(0), c.sum(0), graph.edata['a'].sum(0))
            # message passing
            # graph.apply_edges(self.get_mix_feat)
            graph.update_all(self.get_mix_feat,
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            # if self.activation:
            #     rst = self.activation(rst)

            rst = self.fc_out(rst.view(-1, self._out_feats * self._num_heads))
            if self.final_norm:
                # print(self.final_norm)
                rst = self.final_norm(rst)
            if self.activation:
                rst = self.activation(rst)
            # print(rst.shape)

            if get_attention:
                return rst, graph.edata['a'], c
            else:
                return rst


class CCompGATSM(CCompGATS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.mix_w
        del self.mix_b
        # simple yet effective version
        self.fc_mix = nn.Linear(2 * self._num_heads * self._out_feats, self._num_heads * self._out_feats)

    def get_mix_feat(self, edges):
        feat = torch.cat((edges.src['feat_src'], edges.dst['feat_dst']), dim=-1)
        feat = (self.fc_mix(feat.view(-1, 2 * self._num_heads * self._out_feats))
                .view(-1, self._num_heads, self._out_feats))
        return {'m': feat * edges.data['a']}


if __name__ == '__main__':
    # dglnn.GATv2Conv()
    g = dgl.graph(([0, 1, 2, 3, 2, 5, 0, 4], [1, 2, 3, 4, 0, 3, 2, 2]))
    # feat = torch.ones(6, 10)
    feat = torch.randn(6, 10)
    time_ = torch.randint(2002, 2015, (6, 1)).squeeze(dim=-1)
    print(time_)
    print(g.edges())
    g.ndata['time'] = time_