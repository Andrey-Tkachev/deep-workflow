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
from sklearn.metrics import f1_score


TASK_TYPES = [
    'GENOME',
    'LIGO',
    'MONTAGE',
    'SIPHT',
    #'RANDOM',
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


def visualize_features(features, fig, ax, labels):
    X_embedded = TSNE(
            n_components=2,
            random_state=42,
            learning_rate=50,
            perplexity=10,
        ).fit_transform(features)
    print(X_embedded)
    alpha = 0.7
    ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=labels.detach().numpy(),
        cmap="jet",
        alpha=alpha,
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    return fig, ax 


def get_features(context):
    nxgraph, real_features, cat_features = utils.get_context_features(context)
    real_features = torch.tensor(real_features, dtype=torch.float)
    cat_features = torch.LongTensor(cat_features)
    graph = dgl.DGLGraph()
    graph.from_networkx(nxgraph)
    mask = torch.zeros((real_features.size(0), 1), dtype=torch.bool)
    mask[0] = True
    mask = torch.BoolTensor(mask)
    return graph, real_features, cat_features, mask


def pretrain_policy(policy_net, feature_extractor):
    context = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='',
        feature=feature_extractor,
    )
    context.model = policy_net
    loss_fn = nn.CrossEntropyLoss()

    #types = [random.choice(TASK_TYPES) for _ in range(10)]
    #valid = [
    #    (tp, get_task(random.choice([10, 90]), 100, task_type=tp)) for tp in types
    #]

    loss = 0
    ans = []
    predicted_ans = []
    for i in range(100):
        logging.info(f'Epoch {i}')
        batch_results = torch.zeros((10, 4))
        batch_embs = torch.zeros((10, 4))
        batch_ans = torch.zeros((10,), dtype=torch.long)
        optimizer.zero_grad()
        for b in range(10):
            logging.info(f'Batch item {b}')
            task_type = random.choice(TASK_TYPES)
            task_file, _ = get_task(i, 100, task_type=task_type)
            context.task_file = task_file
            graph, real_features, cat_features, mask = get_features(context)
            prediction = policy_net(graph, real_features, cat_features, mask)
            batch_results[b] += prediction
            logging.info(f'prediction: {prediction}; correct: {TASK_TYPES.index(task_type)}')
            batch_ans[b] = TASK_TYPES.index(task_type)
            predicted_ans.append(prediction.max(0)[1])
            ans.append(batch_ans[b])
        loss = loss_fn(batch_results, batch_ans)
        loss.backward()
        for name, parameter in policy_net.named_parameters():
            print(f'{name}: {(parameter.grad.data ** 2.0).sum().sqrt()}')
        optimizer.step()
        logging.info(f'Loss {loss.item()}')
        experiment.log_metric('F1-macro', f1_score(ans, predicted_ans, average='macro'), step=i)
        experiment.log_metric('Loss', loss.item(), step=i)
        if i % 5 == 0:
            batch_embs = torch.zeros((len(valid), 32))
            batch_real = torch.zeros((len(valid), feature_extractor.REAL_FEATURES_NUM))
            experiment.log_metric('Loss', loss.item())
            batch_ans = torch.zeros((len(valid),), dtype=torch.long)
            for b, (task_type, (task_file, _)) in enumerate(valid):
                logging.info(f'Process {task_file}')
                context.task_file = task_file
                graph, real_features, cat_features, mask = get_features(context)
                with torch.no_grad():
                    _, embs = policy_net(graph, real_features, cat_features, mask, True)
                batch_embs[b] += embs
                batch_real[b] += real_features.mean(0)
                batch_ans[b] = TASK_TYPES.index(task_type)
            print(batch_embs)
            fig, ax = plt.subplots(figsize=(7, 7))
            fig, ax = visualize_features(batch_embs, fig, ax, batch_ans)
            experiment.log_figure(f'TSNE embs: {i}', fig, step=i)

            fig, ax = plt.subplots(figsize=(7, 7))
            fig, ax = visualize_features(batch_real, fig, ax, batch_ans)
            experiment.log_figure(f'Real feats: {i}', fig, step=i)



def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    utils.configure_logs(config['logs'])
    experiment = None
    experiment = utils.create_experiment(config['comet'])
    # Parameters
    memory = Memory()
    learning_rate = 0.01
    weight_decay = 0.0001

    feature_extractor = FeatureExtractor()
    policy_net = PolicyNet(
        real_dim=feature_extractor.REAL_FEATURES_NUM,
        cat_dims=feature_extractor.CAT_DIMS,
        cat_emb_dim=12,
        hid_emb_dim=32,
        out_emb_dim=32,
        hid_pol_dim=32,
        root_features_mode=None,
        global_memory=memory)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    context = Context(
        env_file='./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml',
        task_file='',
        feature=feature_extractor,
    )
    context.model = policy_net
    loss_fn = nn.CrossEntropyLoss()

    types = [random.choice(TASK_TYPES) for _ in range(10)]
    valid = [
        (tp, get_task(random.choice([10, 90]), 100, task_type=tp)) for tp in types
    ]

    loss = 0
    ans = []
    predicted_ans = []
    for i in range(100):
        logging.info(f'Epoch {i}')
        batch_results = torch.zeros((10, 4))
        batch_embs = torch.zeros((10, 4))
        batch_ans = torch.zeros((10,), dtype=torch.long)
        optimizer.zero_grad()
        for b in range(10):
            logging.info(f'Batch item {b}')
            task_type = random.choice(TASK_TYPES)
            task_file, _ = get_task(i, 100, task_type=task_type)
            context.task_file = task_file
            graph, real_features, cat_features, mask = get_features(context)
            prediction = policy_net(graph, real_features, cat_features, mask)
            batch_results[b] += prediction
            logging.info(f'prediction: {prediction}; correct: {TASK_TYPES.index(task_type)}')
            batch_ans[b] = TASK_TYPES.index(task_type)
            predicted_ans.append(prediction.max(0)[1])
            ans.append(batch_ans[b])
        loss = loss_fn(batch_results, batch_ans)
        loss.backward()
        for name, parameter in policy_net.named_parameters():
            print(f'{name}: {(parameter.grad.data ** 2.0).sum().sqrt()}')
        optimizer.step()
        logging.info(f'Loss {loss.item()}')
        experiment.log_metric('F1-macro', f1_score(ans, predicted_ans, average='macro'), step=i)
        experiment.log_metric('Loss', loss.item(), step=i)
        if i % 5 == 0:
            batch_embs = torch.zeros((len(valid), 32))
            batch_real = torch.zeros((len(valid), feature_extractor.REAL_FEATURES_NUM))
            experiment.log_metric('Loss', loss.item())
            batch_ans = torch.zeros((len(valid),), dtype=torch.long)
            for b, (task_type, (task_file, _)) in enumerate(valid):
                logging.info(f'Process {task_file}')
                context.task_file = task_file
                graph, real_features, cat_features, mask = get_features(context)
                with torch.no_grad():
                    _, embs = policy_net(graph, real_features, cat_features, mask, True)
                batch_embs[b] += embs
                batch_real[b] += real_features.mean(0)
                batch_ans[b] = TASK_TYPES.index(task_type)
            print(batch_embs)
            fig, ax = plt.subplots(figsize=(7, 7))
            fig, ax = visualize_features(batch_embs, fig, ax, batch_ans)
            experiment.log_figure(f'TSNE embs: {i}', fig, step=i)

            fig, ax = plt.subplots(figsize=(7, 7))
            fig, ax = visualize_features(batch_real, fig, ax, batch_ans)
            experiment.log_figure(f'Real feats: {i}', fig, step=i)


if __name__ == '__main__':
    main()
