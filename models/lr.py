import torch
from models.layers import Linear,MLP

class LRModel(torch.nn.Module):
    """
    Model:  Logistic Regression
    """

    def __init__(self, nfeat):
        super().__init__()
        self.linear = Linear(nfeat)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        return self.linear(x)


class MLPModel(torch.nn.Module):
    """
    Model:  Logistic Regression
    """
    def __init__(self, nfeat):
        super().__init__()
        self.mlp = MLP(nfeat, nlayers, nhid, dropout, noutput=1)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        return self.mlp(x)

    