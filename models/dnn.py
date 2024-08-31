import torch
from models.layers import get_embedding, MLP


class DNNModel(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self, nfield: int, nfeat: int, nemb: int, mlp_nlayer: int, mlp_nhid: int, dropout: float,
                 emb_ncross: int, noutput: int = 1):
        super().__init__()
        self.embedding = get_embedding(nfield, nfeat, nemb, ncross=emb_ncross)
        self.mlp_ninput = self.embedding.nfield*self.embedding.nemb
        self.mlp = MLP(self.mlp_ninput, mlp_nlayer, mlp_nhid, dropout, noutput=noutput)

    def forward(self, x):
        """
        :param x:   {'id': [bsz, nfield], LongTensor, 'value': [bsz, nfield], FloatTensor}
        :return:    y: [bsz], FloatTensor of size B, for Regression or Classification
        """
        x_emb = self.embedding(x)                           # B*F*E
        y = self.mlp(x_emb.reshape(-1, self.mlp_ninput))    # B*1
        return y.squeeze(1)
