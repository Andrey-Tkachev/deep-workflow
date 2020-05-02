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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


TASK_TYPES = [
    'GENOME',
    #'LIGO',
    #'MONTAGE',
    #'SIPHT'
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
    files = glob.glob(f'data/workflows/dot/{task_type}.n.100.1.*')
    return random.choice(files), sizes[size_id]


def visualize_features(features, fig, ax):
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(features)

    alpha = 0.7
    ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        cmap="jet",
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    return 


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    utils.configure_logs(config['logs'])
    experiment = utils.create_experiment(config['comet'])

    # Parameters
    memory = Memory()
    epochs_num = 5
    track_size = 10
    num_episode = 1000
    learning_rate = 0.0005
    easiness_factor = 1.2
    easiness_decay = 0.999
    use_ppo = True
    eps_clip = 0.2
    entropy_loss = True
    entropy_coef = 0.0001
    reward_mode = 'gamma'
    gamma = 0.99
    weight_decay = 0.01

    if experiment is not None:
        experiment.log_parameters({
            'epochs_num': epochs_num,
            'num_episode': num_episode,
            'track_size': track_size,
            'learning_rate': learning_rate,
            'easiness_factor': easiness_factor,
            'easiness_decay': easiness_decay,
            'use_ppo': use_ppo,
            'eps_clip': eps_clip,
            'entropy_loss': entropy_loss,
            'reward_mode': reward_mode,
            'gamma': gamma,
            'weight_decay': weight_decay,
        })

    feature_extractor = FeatureExtractor()
    policy_net = PolicyNet(
        real_dim=feature_extractor.REAL_FEATURES_NUM,
        cat_dims=feature_extractor.CAT_DIMS,
        cat_emb_dim=12,
        hid_emb_dim=16,
        hid_pol_dim=16,
        global_memory=memory)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Better than average results here
    heuristic_chache = {}
    makespans_batch = []
    track = 0

    context = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='',
        feature=feature_extractor,
    )
    context.model = policy_net
    for episode in range(num_episode):
        if episode % track_size == 0:
            task_type = random.choice(TASK_TYPES)
            task_file, size = get_task(episode, num_episode, task_type)
            context.task_file = task_file
            logging.info(f'Current task: {task_file}')
            if task_file not in heuristic_chache:
                heuristic_makespan = utils.get_heuristics_estimation(context)
                heuristic_chache[task_file] = heuristic_makespan

        with torch.no_grad(), memory.episode(gamma, reward_mode):
            makespan = master_scheduling(context, MasterSchedulerRL)
            heu_makespan = heuristic_chache.get(context.task_file)
            total_reward = (heu_makespan * easiness_factor - makespan) / (0.5 * heu_makespan)
            # total_reward = 1.0 / makespan if makespan > heu_makespan * easiness_factor else 10.0 / makespan
            makespans_batch.append(makespan)
            memory.set_reward(total_reward)
        logging.info(f'Episode: {episode + 1}; Makespan: {makespan}; Heuristic: {heu_makespan}')


        #fig, ax = plt.subplots(figsize=(7, 7))
        # Update policy
        logging.info(f'Batch size {len(memory.states_batch)}')
        if len(memory.states_batch) == track_size:
            easiness_factor = max(easiness_factor * easiness_decay, 0.95)
            track += 1
            if experiment is not None:
                experiment.log_metric(
                    'Makespan ratio', np.mean(makespans_batch) / heuristic_chache.get(context.task_file), step=track
                )
                experiment.log_metric('Easiness factor', easiness_factor)
            rewards_batch = np.array(memory.rewards_batch)

            # Normalize reward
            reward_mean = np.mean(rewards_batch)
            reward_std = np.std(rewards_batch)
            rewards_batch = (rewards_batch - reward_mean) / reward_std

            # Gradient Desent
            logging.info('Update policy')
            for epoch in range(epochs_num):
                loss = None
                for track_id in range(track_size):
                    for item, (state, action) in enumerate(zip(memory.states_batch[track_id], memory.actions_batch[track_id])):
                        g, real_features, cat_features, old_logprob, mask = state

                        if mask.sum() == 1:
                            continue

                        with memory.no_memory():
                            probs, embs = policy_net(g, real_features, cat_features, mask, return_embs=True)
                        dist = Categorical(probs)
                        reward =  np.sum(rewards_batch[track_id][item:]) - np.mean(
                            np.sum(rewards_batch[:, item:], axis=-1)
                        )
                        log_prob = dist.log_prob(action)
                        ratios = torch.exp(log_prob - old_logprob)
                            
                        # Finding Surrogate Loss:
                        if use_ppo:
                            surr1 = ratios * reward
                            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * reward
                            curr_loss = -torch.min(surr1, surr2)
                        else:
                            curr_loss = -log_prob * reward

                        if entropy_loss:
                            curr_loss -= entropy_coef * dist.entropy()

                        if loss is None:
                            loss = curr_loss
                        else:
                            loss += curr_loss

                # if experiment is not None and epoch == epochs_num - 1:
                #     visualize_features(embs.detach().numpy(), fig, ax)
                #     experiment.log_figure(f'Embs {track}', figure=fig, step=track)
                #     ax.cla()

                loss /= track_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info(f'Loss {loss.item()}')

            makespans_batch = []
            memory.clear()


if __name__ == '__main__':
    main()
