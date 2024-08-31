import torch
import torch.nn as nn
from torch import einsum
from utils.entmax import EntmaxBisect


class MLP(nn.Module):

    def __init__(self, ninput, nlayer, nhid, dropout, noutput=1):
        super().__init__()
        layers = list()
        for i in range(nlayer):
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid
        if nlayer==0: nhid = ninput
        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)


class Linear(nn.Module):

    def __init__(self, nfeat):
        super().__init__()
        self.weight = nn.Embedding(nfeat, 1)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    linear transform of x
        """
        linear = self.weight(x['id']).squeeze(2) * x['value']   # B*F
        return torch.sum(linear, dim=1) + self.bias             # B


class Embedding(nn.Module):

    def __init__(self, nfield, nfeat, nemb):
        super().__init__()
        self.nemb, self.nfield = nemb, nfield
        self.embedding = nn.Embedding(nfeat, nemb)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    embeddings B*F*E
        """
        x_emb = self.embedding(x['id'])                 # B*F*E
        return x_emb * x['value'].unsqueeze(2)          # B*F*E


class CrossFeature(nn.Module):
    def __init__(self, nfield: int, nemb: int, ncross: int, alpha: float = 1.5):
        '''
        :param nemb:        Dimension of Feature Identification
        :param ncross:      Number of embedding cross
        '''
        super().__init__()
        self.scale = nemb ** -0.5
        # [Done]: [shared] vs separate
        self.bilinear_w = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(nemb, nemb)), gain=1.414)              # E*E
        # [Done]: check init - normal vs uniform gain=1-linear/[1.414-ReLU]
        self.query = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(ncross, nemb)), gain=1.414)
        self.value = nn.init.xavier_uniform_(
            nn.Parameter(torch.zeros(ncross, nfield)), gain=1.414)
        self.sparsemax = nn.Softmax(dim=-1) if alpha == 1. \
            else EntmaxBisect(alpha, dim=-1)

    def forward(self, x):
        """
        :param x:       [B, F, E], FloatTensor
        :return:        Cross Features [B, N, E], FloatTensor
        """
        key = einsum('bfe,de->bfd', x, self.bilinear_w)                     # B*F*E
        att_gate = einsum('bfe,ne->bnf', key, self.query) * self.scale      # B*N*F
        sparse_gate = self.sparsemax(att_gate)                              # B*N*F
        attn_weight = einsum('bof,of->bof', sparse_gate, self.value)        # B*N*F
        cross_feature = torch.exp(einsum('bfe,bnf->bne', x, attn_weight))   # B*N*E
        return cross_feature


class CrossEmbedding(nn.Module):
    def __init__(self, nfield: int, nfeat: int, nemb: int, ncross: int, alpha: float=1.5):
        '''
        :param nfeat:       Number of embedding features
        :param nemb:        Dimension of Feature Identification
        :param ncross:      Number of embedding cross
        '''
        super().__init__()
        self.nemb, self.nfield = nemb, ncross
        self.embedding = nn.Embedding(nfeat, nemb)
        # auto-cross features
        self.cross_feature = CrossFeature(nfield, nemb, ncross, alpha)
        # [Done]: BN for cross features
        self.bn = nn.BatchNorm1d(ncross)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    feature embeddings [B, F+N, E], FloatTensor
        """
        x_emb = self.embedding(x['id'])             # B*F*E
        x_emb = x_emb * x['value'].unsqueeze(2)     # B*F*E
        x_cross = self.cross_feature(x_emb)         # B*N*E
        x = self.bn(x_cross)                        # B*N*E
        return x


def get_embedding(nfield: int, nfeat: int, nemb: int, ncross: int) -> nn.Module:
    if ncross == 0:
        return Embedding(nfield, nfeat, nemb)
    else:
        return CrossEmbedding(nfield, nfeat, nemb, ncross)
