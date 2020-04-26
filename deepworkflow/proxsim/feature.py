import abc
from typing import List, Any

import networkx as nx
from pysimgrid import simdag
from pysimgrid.simdag import Simulation
from pysimgrid import cplatform

class FeatureExtractorBase(object):

    @abc.abstractmethod
    def get_task_graph(self, simulation: Simulation) -> nx.DiGraph:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_hosts_features(self, simulation: Simulation) -> List[Any]:
        raise NotImplementedError()

    def get_eets(self, task, hosts) -> List[float]:
        result = []
        parent_connections = [p for p in task.parents if p.kind == simdag.TaskKind.TASK_KIND_COMM_E2E]
        for host_name in hosts:
            host = cplatform.host_by_name(host_name)
            #if (task_name, host_name) in self._estimate_cache:
            #    task_time = self._estimate_cache[(task, host)]
            #else:
            comm_times = [
                conn.get_ecomt(conn.parents[0].hosts[0], host) for conn in parent_connections if conn.parents[0].hosts]
            task_time = (max(comm_times) if comm_times else 0.) + task.get_eet(host)
            #    self._estimate_cache[(task_name, host_name)] = task_time
            result.append(task_time)
        return result


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
