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
    x = torch.zeros((rng, )) - 100.0
    bound = int(rng * curr_episode / num_episode)
    x[:bound + 1] = 0.1 * (1.0 - curr_episode / num_episode)
    x[bound] = 1.0 + curr_episode / num_episode
    return F.softmax(x, dim=0)


def get_all_tasks_of_size(task_type='GENOME', size=50):
    files = glob.glob(f'data/workflows/dot/{task_type}.n.{size}.*')
    return files


def get_task(curr_episode, num_episode, task_type='GENOME'):
    sizes = TASK_SIZES_BY_TYPE[task_type][:4]
    probs = size_probs(len(sizes), curr_episode, num_episode)
    size_id = Categorical(probs).sample().item()
    files = get_all_tasks_of_size(task_type, sizes[size_id])
    return random.choice(files), sizes[size_id]


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    utils.configure_logs(config['logs'])
    experiment = None
    experiment = utils.create_experiment(config['comet'])
    # Parameters
    memory = Memory()
    num_episode = 600
    track_size = 10
    epochs_num = 4
    learning_rate = 0.001
    easy_factor = 1.2
    easy_factor_decay = 0.999
    gamma = 0.99
    reward_mode = 'gamma'

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
    tracks = 0

    context = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='',
        feature=feature_extractor,
    )
    file_statistics = {}
    context.model = policy_net
    for episode in range(num_episode):
        if episode % track_size == 0:
            task_type = random.choice(TASK_TYPES)
            task_file, size = get_task(episode, num_episode, task_type)
            logging.info(f'Current task: {task_file}')
            context.task_file = task_file
            if task_file not in file_statistics:
                heuristic_makespan = utils.get_heuristics_estimation(context)
                logging.info(f'Heuristics makespan {task_type}.{size}: {heuristic_makespan}')
                file_statistics[task_file] = {
                    'heuristic_makespan': heuristic_makespan,
                    'makespan_history': []
                }

        step_dict[task_type][size] += 1
        step = step_dict[task_type][size]

        logging.info(f'Episode: {episode + 1}')
        with torch.no_grad(), memory.episode(gamma, reward_mode):
            makespan = master_scheduling(context, MasterSchedulerRL)
            heuristics_makespan = file_statistics[task_file]["heuristic_makespan"]
            #total_reward = -makespan
            N = len(memory.states)
            total_reward = -(makespan - heuristics_makespan * easy_factor) / ((easy_factor - 1) * heuristics_makespan)
            # total_reward = heuristics_makespan / makespan # - N * easy_factor / heuristics_makespan)
            memory.set_reward(total_reward)
            makespans_batch.append(makespan)
            logging.info(f'Reward is {total_reward}')
            logging.info(f'MakeSpan is {makespan}; Heuristics is {heuristics_makespan}')
        
        #if experiment is not None:
            #experiment.log_metric(f'Reward {task_type}.{size}', total_reward, step=step, epoch=0)
            #experiment.log_metric(f'MakeSpan {task_type}.{size}', makespan, step=step, epoch=0)

        # Update policy
        logging.info(f'Batch size {len(memory.states_batch)}')
        if len(memory.states_batch) == track_size:
            easy_factor *= easy_factor_decay
            logging.info(f'Easiness factor: {easy_factor}')
            tracks += 1
            rewards_batch = np.array(memory.rewards_batch)

            # Normalize reward
            reward_mean = np.mean(rewards_batch)
            reward_std = np.std(rewards_batch)
            rewards_batch = (rewards_batch - reward_mean) / (reward_std + 1e-5)

            makespans_mean = np.mean(makespans_batch)
            file_statistics[context.task_file]['makespan_history'].append(makespans_mean)

            if experiment is not None:
                heuristics_makespan = file_statistics[task_file]["heuristic_makespan"]
                experiment.log_metric('Makespan ratio', makespans_mean / heuristics_makespan, step=tracks)
                #experiment.log_metric('Avg epoch makespan', makespans_mean / np.std(makespans_batch), step=epoch, epoch=0)
                #experiment.log_metric('Avg epoch reward', np.mean(rewards_batch), step=epoch, epoch=0)

            # Gradient Desent
            optimizer.zero_grad()
            logging.info('Update policy')

            for epoch in range(epochs_num):
                logging.info(f'Epoch {epoch + 1} out of {epochs_num}')
                loss = None
                actions_num = 0
                for batch_id in range(track_size):
                    batch_item = 0
                    for state, action in zip(memory.states_batch[batch_id], memory.actions_batch[batch_id]):
                        g, real_features, cat_features, mask = state

                        with memory.no_memory():
                            # probs, val = policy_net(g, real_features, cat_features, mask, ret_value=True)
                            probs = policy_net(g, real_features, cat_features, mask, ret_value=False)
                        dist = Categorical(probs)
                        reward = np.sum(rewards_batch[batch_id][batch_item:]) #- np.sum(rewards_batch[:, batch_item:], axis=0).mean()
                        #advantages = reward  - val.detach()
                        #curr_loss = -dist.log_prob(action) * advantages + (reward - val) ** 2.0 - 0.01 * dist.entropy()
                        curr_loss = -dist.log_prob(action) * reward #- 0.0001 * dist.entropy()
                        if loss is None:
                            loss = curr_loss
                        else:
                            loss += curr_loss
                        actions_num += 1
                        batch_item += 1

                loss /= actions_num
                loss.backward()

            makespans_batch = []
            memory.clear()
            optimizer.step()


    policy_net.eval()
    task_type = 'GNOME'
    for size in [50, 100, 200, 300]:
        files = get_all_tasks_of_size(size=size)
        heu = []
        model = []
        for task_file in files[:4]:
            context.task_file = task_file
            heuristic_makespan = utils.get_heuristics_estimation(context)
            for _ in range(5):
                with torch.no_grad(), memory.no_memory():
                    makespan = master_scheduling(context, MasterSchedulerRL)
                model.append(makespan)

            heu.append(heuristic_makespan)
        print(f'''
            {task_type}.n.{size}
                heuristics average makespan:    {np.mean(heu)}
                trained model average makespan: {np.mean(model)}
        ''')


if __name__ == '__main__':
    main()
