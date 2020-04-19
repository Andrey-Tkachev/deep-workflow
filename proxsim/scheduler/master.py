import abc
import logging
import typing
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

import networkx as nx

import pysimgrid
from pysimgrid import cplatform, simdag

from ..action import Action
from ..enums import ActionType, SimulationState


class MasterSchedulerBase(object):

    def __init__(self, connection: Connection):
        self.connection = connection
        self.active = True
        self.init_bindings()
    
    def init_bindings(self):
        self.state_binding = {
            SimulationState.Initial: self.null,
            SimulationState.Prepare: self.prepare,
            SimulationState.Schedule: self.schedule,
            SimulationState.Finally: self.terminate,
            SimulationState.Terminal: self.terminate
        }

    def communicate(self, action: ActionType, params=dict()):
        logging.info(f'Sending action {action}')
        self.connection.send(Action(action, params))
        result = self.connection.recv()
        logging.info(f'Action result recived: {result}')
        return result

    def stop_communication(self):
        logging.info('Sending stop-communication flag')
        self.connection.send(Action(ActionType._End, dict()))

    def get_tasks(self, query, prop='state'):
        return self.communicate(ActionType.GetTasks, {'query': query, 'prop': prop})

    def get_eet(self, task, hosts):
        return self.communicate(ActionType.GetEet, {'task': task, 'hosts': hosts})

    def set_schedule(self, schedule):
        self.communicate(ActionType.SetSchedule, {'schedule': schedule})

    def terminate(self):
        self.active = False

    def null(self):
        pass

    @abc.abstractmethod
    def prepare(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def schedule(self):
        raise NotImplementedError()

    def run(self):
        while self.active:
            state = self.connection.recv()
            logging.info(f'Recived state: {state.name}')
            self.state_binding[state]()
            if state not in [SimulationState.Initial, SimulationState.Finally]:
                self.stop_communication()


class MasterSchedulerDummy(MasterSchedulerBase):
    
    def prepare(self):
        """
        Overridden.
        """
        self.hosts = self.communicate(ActionType.GetHosts)
        self.hosts_data = {
            host['name']: {'free': True} for host in self.hosts
        }

    def schedule(self):
        """
        Overridden.
        """
        for host_name in self.hosts_data:
            self.hosts_data[host_name]['free'] = True

        for task_type in [simdag.TaskState.TASK_STATE_RUNNING, simdag.TaskState.TASK_STATE_SCHEDULED]:
            for task in self.get_tasks(task_type.name):
                self.hosts_data[task['hosts'][0]]['free'] = False

        for task in self.get_tasks(simdag.TaskState.TASK_STATE_SCHEDULABLE.name):
            t = task['name']
            free_hosts = [host for host in self.hosts_data if self.hosts_data[host]['free']]
            eets = self.get_eet(t, free_hosts)
            free_hosts = sorted(list(zip(eets, free_hosts)), key=lambda t: t[0])
            if free_hosts:
                top_host = free_hosts[0][1]
                self.set_schedule([
                    (t, top_host)
                ])
                self.hosts_data[top_host]["free"] = False
            else:
                break
