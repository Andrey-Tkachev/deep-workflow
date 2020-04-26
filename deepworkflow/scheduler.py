import logging

import dgl
import numpy as np
import torch

from pysimgrid import simdag

from .proxsim import ActionType, Context, master_scheduling
from .proxsim.scheduler import MasterSchedulerBase


class MasterSchedulerRL(MasterSchedulerBase):

    def prepare(self):
        """
        Overridden.
        """

        # Custom behaviour for first call
        graph, task_ids = self.get_graph()
        self.scheduled = 0
        self.graph = graph
        self.task_ids = task_ids
        self.n_tasks = len(self.task_ids.items())
        self.hosts = self.get_hosts()
        self.hosts_data = {
            host['name']: {'free': True} for host in self.hosts
        }

    def _mask_from_tasks(self, tasks):
        mask = torch.zeros((self.n_tasks, 1), dtype=torch.bool)
        for task in tasks:
            mask[self.task_ids[task['name']]] = 1
        return torch.BoolTensor(mask)

    def _update_free_hosts(self):
        for host_name in self.hosts_data:
            self.hosts_data[host_name]['free'] = True

        for task_type in [simdag.TaskState.TASK_STATE_RUNNING, simdag.TaskState.TASK_STATE_SCHEDULED]:
            for task in self.get_tasks(task_type.name):
                self.hosts_data[task['hosts'][0]]['free'] = False

    def get_free_hosts(self):
        return [host for host in self.hosts_data if self.hosts_data[host]['free']]

    def get_top_host(self, task, hosts):
        eets = self.get_eet(task, hosts)
        return min(zip(eets, hosts), key=lambda eet_host: eet_host[0])[1]

    def schedule(self):
        """
        Overridden.
        """
        self._update_free_hosts()
        schedulable = sorted(self.get_tasks(simdag.TaskState.TASK_STATE_SCHEDULABLE.name), key=lambda t: self.task_ids[t['name']])
        schedulable_mask = self._mask_from_tasks(schedulable)
        logging.debug(f'Schedulable number: {len(schedulable)}')
        logging.debug(schedulable)
        while True:
            free_hosts = self.get_free_hosts()
            logging.debug(f'Number of free hosts: {len(free_hosts)}')
            if not free_hosts or not schedulable:
                break

            logging.debug('Request graph')
            real_features, cat_features = self.get_graph()
            logging.debug(f'Call model')
            action = self.context.model.act(self.graph, real_features, cat_features, schedulable_mask)
            logging.debug(f'Model prediction')

            task_to_schedule = schedulable.pop(action)['name']
            top_host = self.get_top_host(task_to_schedule, free_hosts)
            self.hosts_data[top_host]["free"] = False
            self.scheduled += 1
            self.set_schedule([
                (task_to_schedule, top_host)
            ])
            schedulable_mask[self.task_ids[task_to_schedule]] = 0
            logging.debug(f'Scheduled {self.scheduled} out of {self.n_tasks}')
