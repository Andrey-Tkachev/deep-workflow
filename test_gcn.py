import logging

import torch
import dgl
from agent import GCN
from feature import FeatureExtractor
from proxsim import ActionType, Context, master_scheduling
from proxsim.scheduler import MasterSchedulerBase


class MasterSchedulerRL(MasterSchedulerBase):

    def prepare(self):
        """
        Overridden.
        """
        graph, _, _ = self.get_graph()  # put graph in slave chache
        self.task_ids = {}
        for node in graph:
            self.task_ids[node['features']['name']] = node['features']['id']
        self.n_tasks = len(self.task_ids.items())
        self.hosts = self.get_hosts()
        self.hosts_data = {
            host['name']: {'free': True} for host in self.hosts
        }

    def _mask_from_tasks(self, tasks):
        mask = torch.zeros((self.n_tasks, 1))
        for task in tasks:
            mask[self.task_ids[task['name']]] = 1
        return mask

    def schedule(self):
        """
        Overridden.
        """
        for host_name in self.hosts_data:
            self.hosts_data[host_name]['free'] = True

        for task_type in [simdag.TaskState.TASK_STATE_RUNNING, simdag.TaskState.TASK_STATE_SCHEDULED]:
            for task in self.get_tasks(task_type.name):
                self.hosts_data[task['hosts'][0]]['free'] = False

        schedulable = sorted(self.get_tasks(simdag.TaskState.TASK_STATE_SCHEDULABLE.name), 
            key=lambda t: self.task_ids[t['name']])
        logging.info(f'Schedulable number: {len(schedulable)}')
        print(**schedulable, sep='\n')
        schedulable_mask = self._mask_from_tasks(schedulable)
        while True:
            free_hosts = [host for host in self.hosts_data if self.hosts_data[host]['free']]
            logging.info(f'Number of free hosts: {len(free_hosts)}')
            if not free_hosts:
                break

            graph, real_features, cat_features = self.get_graph()
            logging.info('Act')
            action = self.context.model.act(graph, real_features, cat_features, schedulable_mask)
            logging.info(f'Recived action {action}')
            t = schedulable.pop(action)['name']
            eets = self.get_eet(t, free_hosts)
            sorted_free_hosts = sorted(list(zip(eets, free_hosts)), key=lambda eet_host: eet_host[0])
            top_host = sorted_free_hosts[0][1]
            self.set_schedule([
                (t, top_host)
            ])
            self.hosts_data[top_host]["free"] = False
            schedulable_mask[action] = 0


if __name__ == '__main__':
    _LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    _LOG_FORMAT = "[%(process)d] [%(name)s] [%(levelname)5s] [%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.WARN, format=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)


    context = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='./data/workflows/dot/SIPHT.n.50.0.dot',
        master_callback=master_callback,
        feature=FeatureExtractor()
    )

    result = master_scheduling(context, MasterSchedulerRL)
