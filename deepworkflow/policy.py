import logging

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from comet_ml import Experiment
from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical

from .gcn import GCN
from .memory import Memory


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class PolicyNet(nn.Module):
    def __init__(
        self,
        real_dim=5,
        cat_dims=[6],
        cat_emb_dim=16,
        hid_emb_dim=16,
        out_emb_dim=16,
        hid_pol_dim=32,
        global_memory: Memory = None
    ):
        super(PolicyNet, self).__init__()
        self.gcn = GCN(
            real_dim=real_dim,
            cat_dims=cat_dims,
            emb_dim=cat_emb_dim,
            hid_dim=hid_emb_dim,
            out_dim=out_emb_dim)
        self.fc1 = nn.Linear(out_emb_dim, hid_pol_dim)
        self.fc2 = nn.Linear(hid_pol_dim, hid_pol_dim)
        self.fc3 = nn.Linear(hid_pol_dim, 1)
        self.apply(init_weights)
        self.global_memory = global_memory

    def forward(self, g, real_features, cat_features, mask, return_embs=False):
        embs = self.gcn(g, real_features, cat_features)
        x = F.leaky_relu(self.fc1(embs))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x)[mask.flatten()], dim=0)
        if return_embs:
            return x.flatten(), embs
        return x.flatten()

    def act(self, g, real_features, cat_features, mask):
        real_features = torch.tensor(real_features, dtype=torch.float)
        cat_features = torch.tensor(cat_features, dtype=torch.int64)
        action_probs = self.forward(g, real_features, cat_features, mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        state = (
            g,
            real_features.detach(), 
            cat_features.detach(), 
            dist.log_prob(action).detach(),
            mask.clone())
        self.global_memory.register(action, state)
        return action.item()