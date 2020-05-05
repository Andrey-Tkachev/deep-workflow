import datetime
import logging
import random
import glob
import argparse
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
from deepworkflow.parameters import Parameters
from deepworkflow.train_manager import TrainManager
from deepworkflow import utils

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


TASK_TYPES = [
   #'GENOME',
    #'LIGO',
    #'MONTAGE',
    #'SIPHT',
    #'RANDOM',
    #'RANDOM2',
    #'RANDOM3',
    #'GENOME_FIXED_100',
    #'GENOME_FIXED_200',
    'GENOME_FIXED_300',
    #'RANDOM_FIXED_20',
    'RANDOM_FIXED_30'
]

TASK_SIZES_BY_TYPE = {
    'RANDOM': [5, 10],
    'RANDOM2': [20, 30],
    'RANDOM3': [10, 20, 30, 40, 50],
    'RANDOM_FIXED_20': [20],
    'RANDOM_FIXED_30': [30],
    'GENOME': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], #, 2000, 3000, 4000, 5000, 6000],
    'GENOME_FIXED_100': [100],
    'GENOME_FIXED_200': [200],
    'GENOME_FIXED_300': [300],
    'LIGO': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'MONTAGE': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SIPHT': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] #, 2000, 3000, 4000, 5000, 6000],
}


def size_probs(rng, curr_episode, episodes_num):
    x = torch.zeros((rng, )) - 100.0
    bound = int(rng * curr_episode / episodes_num)
    x[:bound + 1] = 0.1 * (1.0 - curr_episode / episodes_num)
    x[bound] = 1.0 + curr_episode / episodes_num
    return F.softmax(x, dim=0)


def get_all_tasks_of_size(task_type='GENOME', size=50):
    if task_type.startswith('RANDOM'):
        pattern =  f'data/workflows/{task_type.lower()}/daggen_{size}_*'
    elif task_type.startswith('GENOME_FIXED_'):
        pattern = f'data/workflows/dot/GENOME.n.{size}.1.*'
    else:
        pattern = f'data/workflows/dot/{task_type}.n.{size}.*'

    files = glob.glob(pattern)
    return files


def get_task(curr_episode, episodes_num, task_type='GENOME'):
    sizes = TASK_SIZES_BY_TYPE[task_type][:2]
    probs = size_probs(len(sizes), curr_episode, episodes_num)
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


def train_policy(params: Parameters, policy_net: PolicyNet, feature_extractor: FeatureExtractor, manager: TrainManager):
    memory = policy_net.global_memory
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # Better than average results here
    heuristic_chache = {}
    makespans_batch = []
    track = 0
    easiness_factor = params.easiness_factor

    context = Context(
        env_file='./data/environment/exp1_systems/cluster_10_1-4_100_100_1.xml',
        task_file='',
        feature=feature_extractor,
    )
    context.model = policy_net
    for episode in manager.episode_range():
        if manager.have_to_update_file():
            task_type = random.choice(TASK_TYPES)
            task_file, size = get_task(episode, params.episodes_num, task_type)
            context.task_file = task_file
            manager.text(f'Current task is {task_file}')
            if task_file not in heuristic_chache:
                heuristic_makespan = utils.get_heuristics_estimation(context)
                heuristic_chache[task_file] = heuristic_makespan

        with torch.no_grad(), memory.episode(params.gamma, params.reward_mode):
            makespan = master_scheduling(context, MasterSchedulerRL)
            heu_makespan = heuristic_chache.get(context.task_file)
            if params.reward_mode == 'classic':
                total_reward = -makespan
            else:
                total_reward = (heu_makespan * easiness_factor - makespan) / (0.1 * heu_makespan)
            memory.set_reward(total_reward)
            makespans_batch.append(makespan)

        manager.text(f'Makespan {makespan}; Heuristic {heu_makespan}', per_episode=True)
        if not manager.have_to_update_policy():
            continue

        policy_net.eval()
        eval_makespan = master_scheduling(context, MasterSchedulerRL)
        policy_net.train()

        manager.metric('makespan_ratio', np.mean(makespans_batch) / heuristic_chache.get(context.task_file))
        manager.metric('eval_makespan_ratio', eval_makespan / heuristic_chache.get(context.task_file))
        manager.metric('easiness_factor', easiness_factor)
        rewards_batch = np.array(memory.rewards_batch)
        # Normalize reward
        # discounted_heuristic = np.array([[-heu_makespan * easiness_factor / params.track_size]] * params.track_size)
        #rewards_batch -= discounted_heuristic
        reward_mean = np.mean(rewards_batch)
        reward_std = np.std(rewards_batch)
        rewards_batch = (rewards_batch - reward_mean) / (reward_std + 1e-5)
        easiness_factor = max(easiness_factor * params.easiness_decay, 0.95)

        # Gradient Descent
        manager.text('Update policy', upload=False)
        for epoch in manager.epoch_range():
            loss = None
            for track_item_id in manager.track_item_range():
                for item, (state, action) in enumerate(zip(memory.states_batch[track_item_id], memory.actions_batch[track_item_id])):
                    g, real_features, cat_features, old_logprob, mask = state

                    # Skip actions without sense
                    if mask.sum() == 1:
                        continue

                    with memory.no_memory():
                        probs, embs = policy_net(g, real_features, cat_features, mask, return_embs=True)

                    dist = Categorical(probs)
                    reward =  np.sum(rewards_batch[track_item_id][item:])
                    if params.substract_baseline_reward:
                        reward -= np.mean(
                            np.sum(rewards_batch[:, item:], axis=-1)
                        )
                    log_prob = dist.log_prob(action)
                    if epoch == params.epochs_num - 1 and track_item_id == 0:
                        logging.debug(probs)

                    # Finding Surrogate Loss:
                    if params.use_ppo:
                        ratios = torch.exp(log_prob - old_logprob)
                        surr1 = ratios * reward
                        surr2 = torch.clamp(ratios, 1 - params.eps_clip, 1 + params.eps_clip) * reward
                        curr_loss = -torch.min(surr1, surr2)
                    else:
                        curr_loss = -log_prob * reward

                    if params.entropy_loss:
                        curr_loss -= params.entropy_coef * dist.entropy()

                    if loss is None:
                        loss = curr_loss
                    else:
                        loss += curr_loss

            loss /= params.track_size
            optimizer.zero_grad()
            loss.backward()
            for name, param in policy_net.named_parameters():
                logging.debug(f'{name}: {(param.grad.data ** 2.0).sum().sqrt().item()}')
            optimizer.step()
            manager.text(f'Loss {loss.item()}', upload=False)

        makespans_batch = []
        memory.clear()


def main(exp_config_section, exp_name):
    config = configparser.ConfigParser()
    config.read('config.ini')

    params_config = configparser.ConfigParser()
    params_config.read('parameters.ini')
    params = Parameters()
    params.from_config(params_config, exp_config_section)

    experiment = None
    experiment = utils.create_experiment(config['comet'], exp_name)
    utils.configure_logs(config['logs'])

    # Parameters
    memory = Memory()

    if experiment is not None:
        params_dict = params.to_dict()
        params_dict['task_types'] = TASK_TYPES
        experiment.log_parameters(params_dict)

    feature_extractor = FeatureExtractor()
    policy_net = PolicyNet(
        real_dim=feature_extractor.REAL_FEATURES_NUM,
        cat_dims=feature_extractor.CAT_DIMS,
        cat_emb_dim=12,
        hid_emb_dim=16,
        hid_pol_dim=16,
        root_features_mode=params.root_mode,
        global_memory=memory)
    manager = TrainManager(
        episodes_num=params.episodes_num,
        epochs_num=params.epochs_num,
        track_size=params.track_size,
        experiment=experiment)
    train_policy(params, policy_net, feature_extractor, manager)

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
    parser = argparse.ArgumentParser(description='Train deepRL scheduler')
    parser.add_argument('-n', '--name', action='store', default='')
    parser.add_argument('-p', '--parameters', action='store', default='DEFAULT')
    args = parser.parse_args()
    exp_name = f'{args.parameters}{"_" + args.name}'
    main(args.parameters, exp_name)
