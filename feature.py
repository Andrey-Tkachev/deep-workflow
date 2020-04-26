import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import dgl
from pysimgrid import simdag
from pysimgrid.simdag import Simulation
from pysimgrid import cscheduling, cplatform

from proxsim import FeatureExtractorBase


class FeatureExtractor(FeatureExtractorBase):
    REAL_FEATURES_NUM = 7
    CAT_FEATURES_NUM = 1
    CAT_DIMS = [
        max(map(lambda c: c.value, simdag.TaskState)),
    ]

    def __init__(self):
        assert self.CAT_FEATURES_NUM == len(self.CAT_DIMS)
        self.heft_rank = None
        self.critical_values = None
        self.task_ids = None
        self.real_features = None
        self.cat_features = None
        self.first_resquest = True


    def _first_request(self, simulation):
        self.graph = simulation.get_task_graph()
        platform_model = cscheduling.PlatformModel(simulation)
        ordered_tasks = cscheduling.heft_order(self.graph, platform_model)
        self.heft_rank = dict()
        for i, t in enumerate(ordered_tasks):
            self.heft_rank[t.name] = i
        self.critical_values = self._calculate_critical_value(self.graph)
        self.task_ids = dict()
        have_to_return_ids_map = True

        for ind, task in enumerate(self.graph):
            self.task_ids[task.name] = ind

        data_len = len(self.graph)
        self.real_features = np.zeros((data_len, self.REAL_FEATURES_NUM), dtype=np.float)
        self.cat_features = np.zeros((data_len, self.CAT_FEATURES_NUM), dtype=np.int64)

        # Init static features here
        for task in self.graph:
            ind = self.task_ids[task.name]
            self.real_features[ind][0] = task.amount
            self.real_features[ind][1] = self.heft_rank[task.name]
            self.real_features[ind][2] = self.critical_values[task.name]
            self.real_features[ind][3] = self.graph.in_degree(task)
            self.real_features[ind][4] = self.graph.out_degree(task)
            self.real_features[ind][5] = sum(data['weight']
                for _, _, data in self.graph.out_edges(task, data=True))
            self.real_features[ind][6] = max(data['weight']
                for _, _, data in self.graph.in_edges(task, data=True)) if task.name != 'root' else 0

        nxgraph = nx.DiGraph()
        for u, v, weight in self.graph.edges.data('weight'):
            nxgraph.add_edge(self.task_ids[u.name], self.task_ids[v.name], weight=weight)

        dglgraph = dgl.DGLGraph()
        dglgraph.from_networkx(nxgraph)
        return dglgraph, self.task_ids

    @staticmethod
    def _calculate_critical_value(graph: nx.DiGraph):
        '''Calcualte critical value for each node in graph

        Critical value of node u is max({critical_value[v]: v is child of u}) + weight[u]
        '''
        critical_values = dict()
        for node in reversed(list(nx.topological_sort(graph))):
            critical_values[node.name] = graph.nodes[node]['weight']
            descendants = nx.descendants(graph, node)
            if descendants:
                critical_values[node.name] += max(map(
                    lambda v: critical_values[v.name],
                    descendants
                ))
        return critical_values

    def get_task_graph(self, simulation: Simulation) -> nx.DiGraph:
        """
            Overriden.
        """

        if self.first_resquest:
            self.first_resquest = False
            return self._first_request(simulation)

        for task in self.graph:
            ind = self.task_ids[task.name]
            self.cat_features[ind] = np.array([
                task.state.value
            ])

        return self.real_features, self.cat_features


    def get_hosts_features(self, simulation: Simulation):
        """
            Overriden.
        """
        def host_features(host: cplatform.Host):
            return {
                'name': host.name,
                'speed': host.speed,
                'available_speed': host.available_speed,
            }

        return [host_features(host) for host in simulation.hosts]
