import torch
import torch.nn.functional as F
import networkx as nx
import dgl
from pysimgrid import simdag
from pysimgrid.simdag import Simulation
from pysimgrid import cscheduling, cplatform

from proxsim import FeatureExtractorBase


class FeatureExtractor(FeatureExtractorBase):
    REAL_FEATURES_NUM = 5
    CAT_FEATURES_NUM = 1
    CAT_DIMS = [
        max(map(lambda c: c.value, simdag.TaskState)),
    ]

    def __init__(self):
        assert self.CAT_FEATURES_NUM == len(self.CAT_DIMS)
        self.heft_rank = None
        self.critical_values = None
        self.task_ids = None

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

        graph = simulation.get_task_graph()
        if self.heft_rank is None:
            platform_model = cscheduling.PlatformModel(simulation)
            ordered_tasks = cscheduling.heft_order(graph, platform_model)
            self.heft_rank = dict()
            for i, t in enumerate(ordered_tasks):
                self.heft_rank[t.name] = i
        
        if self.critical_values is None:
            self.critical_values = self._calculate_critical_value(graph)

        have_to_return_ids_map = False
        if self.task_ids is None:
            self.task_ids = dict()
            have_to_return_ids_map = True

            for ind, task in enumerate(graph):
                self.task_ids[task.name] = ind

            nxgraph = nx.DiGraph()
        data_len = len(graph)
        real_features = torch.zeros((data_len, self.REAL_FEATURES_NUM), dtype=torch.float)
        cat_features = torch.zeros((data_len, self.CAT_FEATURES_NUM), dtype=torch.long)

        for task in graph:
            ind = self.task_ids[task.name]
            real_features[ind] = torch.tensor([
                task.amount,
                self.heft_rank[task.name],
                self.critical_values[task.name],
                graph.in_degree(task),
                graph.out_degree(task),
            ])
            cat_features[ind] = torch.tensor([
                task.state.value
            ])
            if have_to_return_ids_map:
                nxgraph.add_node(ind)

        if have_to_return_ids_map:
            for u, v, weight in graph.edges.data('weight'):
                nxgraph.add_edge(self.task_ids[u.name], self.task_ids[v.name], weight=weight)

        if have_to_return_ids_map:
            graph = dgl.DGLGraph()
            graph.from_networkx(nxgraph)
            return graph, self.task_ids
        return real_features.numpy(), cat_features.numpy()


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
