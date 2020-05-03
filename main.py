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
    #'SIPHT',
    'RANDOM',
    #'RANDOM2',
    #'RANDOM3',
    #'RANDOM_FIXED_20',
    #'RANDOM_FIXED_30'
]

TASK_SIZES_BY_TYPE = {
    'RANDOM': [5, 10],
    'RANDOM2': [20, 30],
    'RANDOM3': [10, 20, 30, 40, 50],
    'RANDOM_FIXED_20': [20],
    'RANDOM_FIXED_30': [30],
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
    if not task_type.startswith('RANDOM'):
        pattern = f'data/workflows/dot/{task_type}.n.{size}.*'
    else:
        pattern =  f'data/workflows/{task_type.lower()}/daggen_{size}_*'

    files = glob.glob(pattern)
    return files


def get_task(curr_episode, num_episode, task_type='GENOME'):
    sizes = TASK_SIZES_BY_TYPE[task_type][:2]
    probs = size_probs(len(sizes), curr_episode, num_episode)
    size_id = Categorical(probs).sample().item()

    files = get_all_tasks_of_size(task_type, sizes[size_id])
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
    experiment = None
    experiment = utils.create_experiment(config['comet'])
    # Parameters
    memory = Memory()
    epochs_num = 8
    track_size = 8
    num_episode = 2000
    learning_rate = 0.0005
    easiness_factor = 1.2
    easiness_decay = 0.999
    use_ppo = True
    eps_clip = 0.2
    entropy_loss = False
    entropy_coef = 0.00001
    reward_mode = 'classic'
    substract_baseline_reward = True
    root_mode =  'cat' #'tahn_mul'
    gamma = 0.99
    weight_decay = 0.001

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
            'root_mode': root_mode,
            'substract_baseline_reward': substract_baseline_reward,
        })

    feature_extractor = FeatureExtractor()
    policy_net = PolicyNet(
        real_dim=feature_extractor.REAL_FEATURES_NUM,
        cat_dims=feature_extractor.CAT_DIMS,
        cat_emb_dim=12,
        hid_emb_dim=16,
        hid_pol_dim=16,
        root_features_mode=root_mode,
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
    file_statistics = {}
    context.model = policy_net
    for episode in range(num_episode):
        if episode % track_size == 0:
            task_type = random.choice(TASK_TYPES) + ['RANDOM'])
            task_file, size = get_task(episode, num_episode, task_type)
            context.task_file = task_file
            logging.info(f'Current task: {task_file}')
            if task_file not in heuristic_chache:
                heuristic_makespan = utils.get_heuristics_estimation(context)
                heuristic_chache[task_file] = heuristic_makespan

        with torch.no_grad(), memory.episode(gamma, reward_mode):
            makespan = master_scheduling(context, MasterSchedulerRL)
            heu_makespan = heuristic_chache.get(context.task_file)
            if reward_mode == 'classic':
                total_reward = -makespan
            else:
                total_reward = (heu_makespan * easiness_factor - makespan) / (0.1 * heu_makespan)
            # 1.0 / makespan if makespan > heu_makespan * easiness_factor else 10.0 / makespan
            memory.set_reward(total_reward)
            makespans_batch.append(makespan)
        logging.info(f'Episode: {episode + 1}; Makespan: {makespan}; Heuristic: {heu_makespan}')


        #fig, ax = plt.subplots(figsize=(7, 7))
        # Update policy
        logging.info(f'Batch size {len(memory.states_batch)}')
        if len(memory.states_batch) == track_size:
            track += 1
            policy_net.eval()
            eval_makespan = master_scheduling(context, MasterSchedulerRL)
            policy_net.train()
            if experiment is not None:
                experiment.log_metric(
                    'Makespan ratio', np.mean(makespans_batch) / heuristic_chache.get(context.task_file), step=track
                )
                experiment.log_metric(
                    'Eval makespan ratio', eval_makespan / heuristic_chache.get(context.task_file), step=track
                )
                experiment.log_metric('Easiness factor', easiness_factor)
            rewards_batch = np.array(memory.rewards_batch)
            # Normalize reward
            reward_mean = np.mean(rewards_batch)
            reward_std = np.std(rewards_batch)
            rewards_batch = (rewards_batch - reward_mean) / (reward_std + 1e-5)
            easiness_factor = max(easiness_factor * easiness_decay, 0.95)

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
                        reward =  np.sum(rewards_batch[track_id][item:])
                        if substract_baseline_reward:
                            reward -= np.mean(
                                np.sum(rewards_batch[:, item:], axis=-1)
                            )
                        log_prob = dist.log_prob(action)
                        if epoch == epochs_num - 1 and track_id == 0:
                            logging.debug(probs)
    
                        # Finding Surrogate Loss:
                        if use_ppo:
                            ratios = torch.exp(log_prob - old_logprob)
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
                loss /= track_size
                optimizer.zero_grad()
                loss.backward()
                for name, param in policy_net.named_parameters():
                    logging.debug(f'{name}: {(param.grad.data ** 2.0).sum().sqrt().item()}')
                optimizer.step()
                logging.info(f'Loss {loss.item()}')

            makespans_batch = []
            memory.clear()


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
