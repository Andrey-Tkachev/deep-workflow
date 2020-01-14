import torch
import torch.nn as nn
import networkx as nx


class GCNNetwork(nn.Module):
    '''
        Network to encode nodes of directed graph
    '''

    def __init__(self, input_dim, hidden_dim, emb_dim):
        super(GCNNetwork, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.raw_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.LeakyReLU(),
        )
        self.node_network = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )
        self.summary_network = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )

    def forward(self, dag: nx.DiGraph, mask: torch.Tensor):
        '''Encodes directed graph
        Arguments"
            mask (torch.tensor): array of size (1, num_nodes) with values from {0, 1},
                where mask[i] == 0.0 means node_i is not actualy present in graph
            dag (nx.DiGraph): directed graph, each node has attribute 'features'
                    (dag.nodes[u]['freatures']) with numpy array of size (1, emb_dim)

        Returns:
            torch.Tensor: tensor of size (num_nodes, emb_dim) with grap's nodes embeddings
        '''
        output = torch.zeros((mask.size(0), self.emb_dim), dtype=torch.float)
        for node in reversed(list(nx.topological_sort(dag))):
            output[node] = self.raw_network(torch.Tensor(dag.nodes[node]['features']))
            descendants = list(nx.descendants(dag, node))
            if len(descendants) > 0:
                output[node] += self.summary_network(
                    self.node_network(output[descendants, :]).sum(dim=0)
                )
        return output
