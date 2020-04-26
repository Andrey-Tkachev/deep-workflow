import logging
import multiprocessing
from itertools import count
from contextlib import contextmanager
from comet_ml import Experiment

import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli

from pysimgrid import simdag
import pysimgrid.simdag.algorithms as algorithms

from feature import FeatureExtractor
from proxsim import ActionType, Context, master_scheduling
from proxsim.scheduler import MasterSchedulerBase


from agent import GCN


class Memory:
    def __init__(self):
        self._reset()
    
    def _reset(self):
        self.actions = None
        self.states = None
        self.actions_batch = []
        self.states_batch = []
        self.rewards_batch = []
        self.active = True

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.actions_batch[:]
        del self.states_batch[:]
        del self.rewards_batch[:]
        self._reset()

    def register(self, action, state):
        if not self.active:
            return
        self.states.append(state)
        self.actions.append(action)

    def start_episode(self):
        self.actions = []
        self.states = []

    def end_episode(self, episode_reward, gamma=0.99):
        reward_pool = [0 for _ in range(len(self.actions))]
        reward_pool[-1] = episode_reward

        running_add = 0
        mid = len(reward_pool) // 2
        coefs = np.exp(-0.5 * ((np.arange(len(reward_pool)) - mid) / mid) ** 2.0)
        for i in reversed(range(len(reward_pool))):
            # running_add = running_add * gamma + reward_pool[i]
            # reward_pool[i] = running_add
            reward_pool[i] = coefs[i] * episode_reward

        self.rewards_batch.append(np.array(reward_pool))
        self.actions_batch.append(self.actions)
        self.states_batch.append(self.states)

    @contextmanager
    def no_memory(self):
        _active_prev = self.active
        self.active = False

        try:
            yield
        finally:
            self.active = _active_prev


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
        self.fc2 = nn.Linear(24, 1)
        self.global_memory = global_memory

    def forward(self, g, real_features, cat_features, mask):
        x = self.gcn(g, real_features, cat_features)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x)[mask.flatten()], dim=0)
        return x.flatten()

    def act(self, g, real_features, cat_features, mask):
        real_features = torch.tensor(real_features, dtype=torch.float)
        cat_features = torch.tensor(cat_features, dtype=torch.int64)
        action_probs = self.forward(g, real_features, cat_features, mask)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        state = (g, real_features, cat_features, mask.clone())
        self.global_memory.register(action, state)
        return action.item()


class MasterSchedulerRL(MasterSchedulerBase):

    def prepare(self):
        """
        Overridden.
        """
        graph, task_ids = self.get_graph()  # put some graph data in slave chache
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
        # print(*schedulable, sep='\n')
        while True:
            free_hosts = self.get_free_hosts()
            logging.debug(f'Number of free hosts: {len(free_hosts)}')
            if not free_hosts or not schedulable:
                break

            real_features, cat_features = self.get_graph()
            action = self.context.model.act(self.graph, real_features, cat_features, schedulable_mask)
            task_to_schedule = schedulable.pop(action)['name']
            top_host = self.get_top_host(task_to_schedule, free_hosts)
            self.hosts_data[top_host]["free"] = False
            self.scheduled += 1
            self.set_schedule([
                (task_to_schedule, top_host)
            ])
            schedulable_mask[self.task_ids[task_to_schedule]] = 0
            logging.debug(f'Scheduled {self.scheduled} out of {self.n_tasks}')


def create_experiment() -> Experiment:
    comet_key = None
    with open('comet_ml.key', 'r') as comet_file:
        comet_key = comet_file.read().rstrip()

    experiment = None
    if comet_key is not None:
        experiment = Experiment(
            api_key=comet_key,
            project_name="rl-manager",
            workspace="andrey-tkachev")
    return experiment



_SCHEDULERS = {
 "MinMin": algorithms.BatchMin,
# "MaxMin": algorithms.BatchMax,
 # "Sufferage": algorithms.BatchSufferage,
 # "DLS": algorithms.DLS,
 # "RandomSchedule": algorithms.RandomStatic,
 # "SimpleDynamic": SimpleDynamic,
 # "MCT": algorithms.MCT,
 # "OLB": algorithms.OLB,
 # "HCPT": algorithms.HCPT,
  "HEFT": algorithms.HEFT,
 # "Lookahead": algorithms.Lookahead,
 # "PEFT": algorithms.PEFT
}


def run_simulation(args):
    scheduler_name, env, task = args
    scheduler_class = _SCHEDULERS[scheduler_name]
    with simdag.Simulation(env, task) as simulation:
      scheduler = scheduler_class(simulation)
      scheduler.run()
    return simulation.clock, scheduler_name


def main():
    _LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    _LOG_FORMAT = "[%(process)d] [%(name)s] [%(levelname)5s] [%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    experiment = create_experiment()
    import datetime
    experiment.set_name(f'Test {datetime.datetime.now()}')

    # Parameters
    num_episode = 300
    batch_size = 5
    learning_rate = 0.02
    initial_factor = 1.1
    gamma = 0.99
    memory = Memory()
    feature_extractor = FeatureExtractor()

    policy_net = PolicyNet(
        real_dim=feature_extractor.REAL_FEATURES_NUM,
        cat_dims=feature_extractor.CAT_DIMS,
        emb_dim=10,
        global_memory=memory)

    context = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='./data/workflows/dot/LIGO.n.100.1.dot',
        # task_file='./data/workflows/dot/LIGO.n.50.2.dot',
        feature=feature_extractor,
    )


    ctx = multiprocessing.get_context("spawn")
    envs = [context.env_file] * len(_SCHEDULERS.keys())
    tasks = [context.task_file] * len(_SCHEDULERS.keys())
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        results = pool.map(run_simulation, zip(_SCHEDULERS.keys(), envs, tasks))

    heuristic_makespan = min(results)[0]
    logging.info(f'Heuristics result {results}')

    context.model = policy_net

    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)
    # Batch History
    rewards_batch = []
    steps = 0

    epoch = 0
    makespans = []
    makespans_history = [heuristic_makespan * initial_factor]

    for episode in range(num_episode):
        logging.info(f'Episode: {episode + 1}')
        memory.start_episode()
        with torch.no_grad():
            makespan = master_scheduling(context, MasterSchedulerRL)
            total_reward = -(makespan - np.mean(makespans_history))
        memory.end_episode(total_reward, gamma)
        makespans.append(makespan)
        logging.info(f'Reward is {total_reward}')
        logging.info(f'MakeSpan is {makespan}')
        if experiment is not None:
            experiment.log_metric('Reward', total_reward, step=episode, epoch=epoch)
            experiment.log_metric('MakeSpan', makespan, step=episode, epoch=epoch)

        logging.info(f'Batch size {len(memory.states_batch)}')
        # Update policy
        if len(memory.states_batch) == batch_size:
            epoch += 1
            rewards_batch = np.array(memory.rewards_batch)

            # Normalize reward
            reward_mean = np.mean(rewards_batch)
            reward_std = np.std(rewards_batch)
            rewards_batch = (rewards_batch - reward_mean) / reward_std

            makespans_mean = np.mean(makespans)
            if makespans_mean < np.mean(makespans_history):
                makespans_history.append(makespans_mean)

            if experiment is not None:
                experiment.log_metric('Avg epoch makespan', np.mean(makespans), step=epoch, epoch=0)
                experiment.log_metric('Avg epoch reward', reward_mean, step=epoch, epoch=0)
            makespans = []

            # Gradient Desent
            optimizer.zero_grad()

            logging.info('Update policy')
            for batch_id in range(batch_size):
                for state, action, reward in zip(memory.states_batch[batch_id], memory.actions_batch[batch_id], rewards_batch[batch_id]):
                    g, real_features, cat_features, mask = state

                    with memory.no_memory():
                        probs = policy_net(g, real_features, cat_features, mask)
                    dist = torch.distributions.Categorical(probs)
                    loss = -dist.log_prob(action) * reward  # Negtive score function x reward
                    loss.backward()

            optimizer.step()
            memory.clear_memory()


if __name__ == '__main__':
    main()
