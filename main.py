import datetime
import logging
import random
import glob
from itertools import count

import comet_ml
import configparser

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from comet_ml import Experiment
from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical


from deepworkflow.policy import PolicyNet
from deepworkflow.scheduler import MasterSchedulerRL
from deepworkflow.memory import Memory
from deepworkflow.proxsim import Context, master_scheduling
from deepworkflow.feature import FeatureExtractor
from deepworkflow import utils



TASK_TYPES = [
    'GENOME',
   # 'LIGO',
   # 'MONTAGE',
   # 'SIPHT'
]

TASK_SIZES_BY_TYPE = {
    'GENOME': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], #, 2000, 3000, 4000, 5000, 6000],
    'LIGO': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'MONTAGE': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SIPHT': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] #, 2000, 3000, 4000, 5000, 6000],
}


def size_probs(rng, curr_episode, num_episode):
    x = (torch.arange(0, rng) - rng * curr_episode / num_episode) / rng
    x = torch.exp(-5 * x ** 2.0)
    return F.softmax(x * 30, dim=0) 


def get_task(curr_episode, num_episode, task_type='GENOME'):
    sizes = TASK_SIZES_BY_TYPE[task_type]
    probs = size_probs(len(sizes), curr_episode, num_episode)
    size_id = Categorical(probs).sample().item()
    files = glob.glob(f'data/workflows/dot/{task_type}.n.{sizes[size_id]}.*')
    return random.choice(files), sizes[size_id]


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    utils.configure_logs(config['logs'])
    experiment = utils.create_experiment(config['comet'])

    # Parameters
    memory = Memory()
    num_episode = 1000
    batch_size = 5
    learning_rate = 0.01
    initial_factor = 1.2
    gamma = 0.99
    reward_mode = 'normal'

    feature_extractor = FeatureExtractor()
    policy_net = PolicyNet(
        real_dim=feature_extractor.REAL_FEATURES_NUM,
        cat_dims=feature_extractor.CAT_DIMS,
        emb_dim=10,
        global_memory=memory)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    # Better than average results here
    expected_makespan = {
        task_type: {size: [] for size in TASK_SIZES_BY_TYPE[task_type]}
        for task_type in TASK_TYPES
    }
    step_dict = {
        task_type: {size: 0 for size in TASK_SIZES_BY_TYPE[task_type]}
        for task_type in TASK_TYPES
    }
    makespans_batch = []
    epoch = 0

    context = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='',
        feature=feature_extractor,
    )
    context.model = policy_net
    for episode in range(num_episode):
        if episode % batch_size == 0:
            task_type = random.choice(TASK_TYPES)
            task_file, size = get_task(episode, num_episode, task_type)
            context.task_file = task_file
            logging.info(f'Current task: {task_file}')
            if not expected_makespan[task_type][size]:
                heuristic_makespan = utils.get_heuristics_estimation(context)
                expected_makespan[task_type][size].append(heuristic_makespan)
                logging.info(f'Heuristics makespan {task_type}.{size}: {heuristic_makespan}')
                if experiment is not None:
                    experiment.log_text(f'Heuristics makespan {task_type}.{size}: {heuristic_makespan}')
            expected_makespan_by_type = expected_makespan[task_type][size]
            step_dict[task_type][size] += 1
            step = step_dict[task_type][size]

        logging.info(f'Episode: {episode + 1}')
        with torch.no_grad(), memory.episode(gamma, reward_mode):
            makespan = master_scheduling(context, MasterSchedulerRL)
            total_reward = -(makespan - np.mean(expected_makespan_by_type))
            memory.set_reward(total_reward)
            makespans_batch.append(makespan)
            logging.info(f'Reward is {total_reward}')
            logging.info(f'MakeSpan is {makespan}')
        
        if experiment is not None:
            experiment.log_metric(f'Reward {task_type}.{size}', total_reward, step=step, epoch=0)
            experiment.log_metric(f'MakeSpan {task_type}.{size}', makespan, step=step, epoch=0)

        # Update policy
        logging.info(f'Batch size {len(memory.states_batch)}')
        if len(memory.states_batch) == batch_size:
            epoch += 1
            rewards_batch = np.array(memory.rewards_batch)

            # Normalize reward
            reward_mean = np.mean(rewards_batch)
            reward_std = np.std(rewards_batch)
            rewards_batch = (rewards_batch - reward_mean) / reward_std

            makespans_mean = np.mean(makespans_batch)
            expected_makespan_by_type.append(makespans_mean)

            if experiment is not None:
                experiment.log_metric('Avg epoch makespan', makespans_mean / np.std(makespans_batch), step=epoch, epoch=0)
                experiment.log_metric('Avg epoch reward', np.mean(rewards_batch), step=epoch, epoch=0)

            # Gradient Desent
            optimizer.zero_grad()

            logging.info('Update policy')
            for batch_id in range(batch_size):
                loss = None
                for state, action, reward in zip(memory.states_batch[batch_id], memory.actions_batch[batch_id], rewards_batch[batch_id]):
                    g, real_features, cat_features, mask = state

                    with memory.no_memory():
                        probs = policy_net(g, real_features, cat_features, mask)
                    dist = Categorical(probs)
                    curr_loss = -dist.log_prob(action) * reward
                    if loss is None:
                        loss = curr_loss
                    else:
                        loss += curr_loss
                loss /= batch_size
                loss.backward()

            makespans_batch = []
            memory.clear()
            optimizer.step()


if __name__ == '__main__':
    main()
