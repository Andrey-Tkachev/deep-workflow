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


class PolicyNet(nn.Module):
    def __init__(
        self,
        real_dim=5,
        cat_dims=[6],
        emb_dim=5,
        out_dim=16,
        global_memory: Memory = None
    ):
        super(PolicyNet, self).__init__()
        self.gcn = GCN(
            real_dim=real_dim,
            cat_dims=cat_dims,
            emb_dim=emb_dim,
            out_dim=out_dim)
        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 1)
        self.value1 = nn.Linear(16, 24)
        self.value2 = nn.Linear(24, 1)
        self.global_memory = global_memory

    def forward(self, g, real_features, cat_features, mask, ret_value=False):
        graph_emb = self.gcn(g, real_features, cat_features)
        x = F.relu(self.fc1(graph_emb))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x)[mask.flatten()], dim=0)
        if not ret_value:
            return x.flatten()
        val = F.relu(self.value1(graph_emb.mean(dim=0).unsqueeze(0)))
        val = self.value2(val)
        return x.flatten(), val

    def act(self, g, real_features, cat_features, mask):
        real_features = torch.tensor(real_features, dtype=torch.float)
        cat_features = torch.tensor(cat_features, dtype=torch.int64)
        action_probs = self.forward(g, real_features, cat_features, mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        state = (g, real_features, cat_features, mask.clone())
        self.global_memory.register(action, state)
        return action.item()