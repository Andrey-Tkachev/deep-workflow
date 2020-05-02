import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='f')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.hid_linears = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            #nn.Linear(hid_feats, out_feats),
           # nn.ReLU(),
        )
        self.msg_linears = nn.Sequential(
            nn.Linear(out_feats, hid_feats),
            nn.ReLU(),
            nn.Linear(hid_feats, out_feats),
            nn.ReLU(),
        )
        self.agg_linears = nn.Sequential(
            nn.Linear(out_feats, hid_feats),
            nn.ReLU(),
            nn.Linear(hid_feats, out_feats),
            nn.ReLU(),
        )

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            hid = self.hid_linears(feature)
            msg = self.msg_linears(hid)
            g.ndata['h'] = msg
            g.update_all(gcn_msg, gcn_reduce)
            return self.agg_linears(g.ndata['f']) + hid


class GCN(nn.Module):
    def __init__(
        self,
        real_dim=5,
        cat_dims=[16],
        emb_dim=5,
        hid_dim=32, 
        out_dim=16
    ):
        super(GCN, self).__init__()
        self.real_dim = real_dim
        self.cat_dim = len(cat_dims) * emb_dim
        self.input_dim = self.real_dim + self.cat_dim
        self.out_dim = out_dim
        self.layer1 = GCNLayer(self.input_dim, hid_dim, hid_dim * 2)
        self.layer2 = GCNLayer(hid_dim * 2, hid_dim, out_dim)
        self.embs = nn.ModuleList([
            nn.Embedding(dim, emb_dim) for dim in cat_dims
        ])

        self.n_cat_features = len(cat_dims)

    def forward(self, g, real_features, cat_features):
        real_features = real_features / (real_features.abs().max(0)[0] + 1e-12)
        cat_embs = torch.zeros((real_features.size(0), self.cat_dim), dtype=torch.float)
        for i in range(self.n_cat_features):
            st, end = i * self.cat_dim, (i + 1) * self.cat_dim
            cat_embs[:, st: end] += self.embs[i](cat_features[:, i])
        features = torch.cat((real_features, cat_embs), dim=-1)
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        return x
