import abc
from typing import List, Any

import networkx as nx
from pysimgrid.simdag import Simulation
from pysimgrid import cplatform

class FeatureExtractorBase(object):

    @abc.abstractmethod
    def get_task_graph(self, simulation: Simulation) -> nx.DiGraph:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_hosts_features(self, simulation: Simulation) -> List[Any]:
        raise NotImplementedError()


class FeatureExtractorDummy(FeatureExtractorBase):

    def get_task_graph(self, simulation: Simulation) -> nx.DiGraph:
        """
            Overriden.
        """
        def task_features(task):
            return {
                'name': task.name,
                'amount': task.amount,
                'state': task.state.name,
            }

        graph = simulation.get_task_graph()
        picklable_graph = nx.DiGraph()
        for task in graph:
            picklable_graph.add_node(task.name, features=task_features(task))

        for u, v, weight in graph.edges.data('weight'):
            picklable_graph.add_edge(u.name, v.name, weight=weight)

        return picklable_graph


    def get_hosts_features(self, simulation: Simulation) -> List[Any]:
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
