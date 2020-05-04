import datetime
import logging
import multiprocessing

from comet_ml import Experiment

import pysimgrid.simdag.algorithms as algorithms
from pysimgrid import simdag

from .proxsim import Context


def create_experiment(config: dict, display_name: str) -> Experiment:
    experiment = None
    key = config.get('api_key', None)
    name = config.get('project_name', None)
    workspace = config.get('workspace', None)
    if None not in (key, name, workspace):
        experiment = Experiment(
            api_key=key,
            project_name=name,
            workspace=workspace)
        experiment.set_name(f'{display_name}: {datetime.datetime.now().time()}')
    return experiment


_SCHEDULERS = {
    "MinMin": algorithms.BatchMin,
    "MaxMin": algorithms.BatchMax,
    "Sufferage": algorithms.BatchSufferage,
    "DLS": algorithms.DLS,
    "RandomSchedule": algorithms.RandomStatic,
    "MCT": algorithms.MCT,
    "OLB": algorithms.OLB,
    "HCPT": algorithms.HCPT,
    "HEFT": algorithms.HEFT,
    "Lookahead": algorithms.Lookahead,
    "PEFT": algorithms.PEFT
}


def run_plain_simulation(args):
    scheduler_name, env, task = args
    scheduler_class = _SCHEDULERS[scheduler_name]
    with simdag.Simulation(env, task) as simulation:
        scheduler = scheduler_class(simulation)
        scheduler.run()
    return simulation.clock, scheduler_name


def get_heuristics_estimation(context: Context, schedulers=['MinMin', 'HEFT']):
    ctx = multiprocessing.get_context("spawn")
    envs = [context.env_file] * len(schedulers)
    tasks = [context.task_file] * len(schedulers)
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        results = pool.map(run_plain_simulation, zip(schedulers, envs, tasks))
    return min(results)[0]


def run_feature_simulation(args):
    exractor, env, task = args
    exractor.return_dgl = False
    with simdag.Simulation(env, task) as simulation:
        graph, _ = exractor.get_task_graph(simulation) # Cache
        real_features, cat_features = exractor.get_task_graph(simulation)
        scheduler = algorithms.RandomStatic(simulation)
        scheduler.run()
    return (graph, real_features, cat_features)


def get_context_features(context: Context):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        nxgraph, real_features, cat_features = pool.map(run_feature_simulation, [(context.feature, context.env_file, context.task_file)])
    real_features = torch.tensor(real_features, dtype=torch.float)
    cat_features = torch.LongTensor(cat_features)
    graph = dgl.DGLGraph()
    graph.from_networkx(nxgraph)
    return graph, real_features, cat_features


def configure_logs(config):
    _LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    _LOG_FORMAT = "[%(process)d] [%(name)s] [%(levelname)5s] [%(asctime)s] %(message)s"
    level_name = config.get('level', 'INFO')
    logging.basicConfig(level=logging.getLevelName(level_name), format=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
