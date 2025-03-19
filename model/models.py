import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric
import math
import numpy as np
from model import layers
from script import *
from scipy.sparse import csc_matrix


INF = 1e20
VERY_SMALL_NUMBER = 1e-12

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, epsilon=None, num_pers=16, metric_type='cosine'):
        super(GraphLearner, self).__init__()
        self.epsilon = epsilon
        self.metric_type = metric_type
        if metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        elif metric_type == 'cosine':
            pass
        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))
        print('use {}'.format(metric_type))

    def forward(self, context):
        if self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
            context_fc = context.unsqueeze(0)* expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0
        elif self.metric_type == 'cosine':
            context_norm = torch.nn.functional.normalize(context, p=2, dim=-1)
            attention = torch.bmm(context_norm, context_norm.transpose(-1, -2))
            markoff_value = 0
        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        return attention

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

def batch_diagflat(tensor):
    device = tensor.device
    tensor = tensor.unsqueeze(1)
    identity = torch.eye(tensor.size(-1)).to(device).unsqueeze(0)
    result = tensor * identity
    return result

def batch_trace(tensor):
    return tensor.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)



class InsiderClassifier(nn.Module):
    def __init__(self, num_features, cat_features, seq_features, cat_nums, cat_embedding_size, seq_embedding_size,
                 lstm_hidden_size, dropout, reduction=True, use_attention=False, LayerNorm=True, encoder_type='lstm',
                 num_lstm_layers=2, mlp_hidden_layers=[],
                 pooling_mode='origin', epsilon=0, num_pers=2, graph_metric_type='weighted_cosine',
                 topk=12, num_class=2, add_graph_regularization=False, gnn='GCN', embedding_hook=False):
        super(InsiderClassifier, self).__init__()

        self.tmp_conv1 = layers.TemporalConvLayer(3,1, 16, 170,'glu' ).to('cuda')
        self.align = layers.Align(16, 16).to('cuda')
        self.cheb_graph_conv = layers.ChebGraphConv(16, 16, 3,True ).to('cuda')
        self.tmp_conv2 = layers.TemporalConvLayer(3, 16, 64, 170,'glu').to('cuda')
        self.tc2_ln = nn.LayerNorm([170, 64], eps=1e-12).to('cuda')
        self.relu = nn.ReLU().to('cuda')
        self.dropout = nn.Dropout(p=0.5).to('cuda')

        self.tmp_conv1_1 = layers.TemporalConvLayer(3,64, 16, 170,'glu' ).to('cuda')
        self.align_1 = layers.Align(16, 16).to('cuda')
        self.cheb_graph_conv_1 = layers.ChebGraphConv(16, 16, 3,True ).to('cuda')
        self.tmp_conv2_1 = layers.TemporalConvLayer(3, 16, 64, 170,'glu').to('cuda')
        self.tc2_ln_1 = nn.LayerNorm([170, 64], eps=1e-12).to('cuda')
        self.relu_1 = nn.ReLU().to('cuda')
        self.dropout_1 = nn.Dropout(p=0.5).to('cuda')

        self.output = layers.OutputBlock(4,64, [128,128], 1, 170, 'glu', True,
                                     0.5).to('cuda')
        self.graph_learner = GraphLearner(12, lstm_hidden_size,
                                         epsilon=epsilon,
                                          num_pers=num_pers,
                                          metric_type=graph_metric_type).to("cuda")


    def forward(self, x):
        features = x['hist_activity']
        adj_batchsize = self.graph_learner(features.permute(0, 2, 1))
        adj_batchsize = adj_batchsize.detach()
        adj = torch.mean(adj_batchsize, dim=0)
        features=features.reshape(features.size(0), 1, 12,170 )
        adj=adj.cpu().numpy()
        adj = csc_matrix(adj)
        adj = adj.tocsc()
        gso = utility.calc_gso(adj, 'sym_norm_lap')
        gso = utility.calc_chebynet_gso(gso)
        gso = gso.toarray()
        gso = gso.astype(dtype=np.float32)
        gso = torch.from_numpy(gso).to("cuda")

        x = self.tmp_conv1(features)
        x_gc_in = self.align(x)
        x_gc = self.cheb_graph_conv(x_gc_in,gso)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)
        x = self.relu(x_gc_out)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        x = self.tmp_conv1_1(x)
        x_gc_in = self.align_1(x)
        x_gc = self.cheb_graph_conv_1(x_gc_in,gso)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)
        x = self.relu(x_gc_out)
        x = self.tmp_conv2_1(x)
        x = self.tc2_ln_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout_1(x)
        x = self.output(x)
        y_hat = x.view(32, -1)

        return y_hat




