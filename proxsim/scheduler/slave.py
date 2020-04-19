import logging
import typing
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

import networkx as nx

import pysimgrid
from pysimgrid import cplatform, simdag

from ..enums import ActionType, SimulationState
from ..simulation import ProximalSimulationSlave


class SlaveScheduler(simdag.DynamicScheduler):

    def __init__(self, simulation, connection: ProximalSimulationSlave):
        super(SlaveScheduler, self).__init__(simulation)
        self.connection = connection
        self.init_bindings()

    def action_unknown(self, simulation: simdag.Simulation, **params):
        return None

    @staticmethod
    def task(simulation, name):
        return simulation.tasks.by_prop('name', name)[0]

    @staticmethod
    def host(simulation, name):
        return simulation.hosts.by_prop('name', name)[0]

    def action_set_schedule(self, simulation: simdag.Simulation, **params) -> None:
        for task_name, host_name in params['schedule']:
            self.task(simulation, task_name).schedule(self.host(simulation, host_name))

    def action_get_eet(self, simulation: simdag.Simulation, **params) -> typing.List[float]:
        result = []
        task_name = params['task']
        for host_name in params['hosts']:
            result.append(self.task(simulation, task_name).get_eet(self.host(simulation, host_name)))
        return result

    def action_get_ecomt(self, simulation: simdag.Simulation, **params) -> typing.List[float]:
        result = []
        for task_name, (host1, host2) in zip(params['tasks'], params['hosts_pairs']):
            result.append(self.task(simulation, task_name), self.hosts(simulation, host1), self.hosts(simulation, host2))
        return result

    def action_get_graph(self, simulation: simdag.Simulation, **params) -> nx.DiGraph:
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

    def action_get_hosts(self, simulation: simdag.Simulation, **params) -> typing.List[typing.Dict]:
        def host_features(host: cplatform.Host):
            return {
                'name': host.name,
                'speed': host.speed,
                'available_speed': host.available_speed,
            }

        return [host_features(host) for host in simulation.hosts]

    def action_get_tasks(self, simulation: simdag.Simulation, **params) -> typing.List[typing.Dict]:
        if 'query' not in params:
            selector = simdag.TaskKind.TASK_KIND_COMM_E2E
        else:
            query = params['query']
            selector = {
                'state': simdag.TaskState,
                'kind':  simdag.TaskKind
            }[params.get('prop', 'state')][query]
        return [{
            'name': task.name,
            'hosts': [host.name for host in task.hosts]
        } for task in simulation.tasks[selector]]

    def init_bindings(self):
        self._action_binding = {
            ActionType.SetSchedule: self.action_set_schedule,
            ActionType.GetEet:   self.action_get_eet,
            ActionType.GetEcomt: self.action_get_ecomt,
            ActionType.GetGraph: self.action_get_graph,
            ActionType.GetHosts: self.action_get_hosts,
            ActionType.GetTasks: self.action_get_tasks,
        }

    def make_communications(self, simulation: simdag.Simulation, changed=None):
        for action in self.connection.iterate_actions():
            logging.info(f'Action {action.action.name} recived')
            handler = self._action_binding.get(action.action, self.action_unknown)
            action_result = handler(simulation, changed=changed, **action.params)
            logging.info(f'Action {action.action.name} processed: {action_result}')
            self.connection.send(action_result)

    def prepare(self, simulation: simdag.Simulation):
        self.connection.state(SimulationState.Prepare)
        self.make_communications(simulation)

    def schedule(self, simulation, changed):
        self.connection.state(SimulationState.Schedule)
        self.make_communications(simulation, changed)

    def _finally(self):
        self.connection.state(SimulationState.Finally)

    def run(self):
        super(SlaveScheduler, self).run()
        self._finally()
