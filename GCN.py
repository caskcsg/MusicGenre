# import torchvision.models as models
# from util import *
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, in_channel=100, out_channel=100,  adj_matrix=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, out_channel)
        self.relu = nn.LeakyReLU(0.2)
        self.A_data = torch.from_numpy(adj_matrix).float().cuda()

    def gen_adj(self, A):
        A += torch.eye(A.size(0)).cuda()
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def forward(self, inp):
        '''
        :param feature:
        :param inp:
        :return:
        '''
        data_adj = self.gen_adj(self.A_data)

        x = self.gc1(inp, data_adj)  # (22, 512)
        x = self.relu(x)
        x = self.gc2(x, data_adj)  # (22, 100)

        x = x.transpose(0, 1)
        if not self.training:
            pickle.dump(x, open("./label_embedding.pkl", 'wb'))

        return x

