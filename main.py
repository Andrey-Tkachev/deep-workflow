"""
Temporary development test for quick-hacking.

Probably should be deleted on 'final release'.
It's pretty illustrative, however.

For bettter examples on C API wrappers look at test/test_capi.py.
"""

from __future__ import print_function

from multiprocessing import Pool

import random
import logging
import multiprocessing

import networkx

import pysimgrid
import pysimgrid.simdag.algorithms as algorithms
from pysimgrid import cscheduling

from pysimgrid import simdag



_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_FORMAT = "[%(name)s] [%(levelname)5s] [%(asctime)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

class RandomSchedule(simdag.StaticScheduler):
  def get_schedule(self, simulation):
    schedule = {host: [] for host in simulation.hosts}
    graph = simulation.get_task_graph()
    for task in networkx.topological_sort(graph):
      schedule[random.choice(simulation.hosts)].append(task)
    return schedule


class SimpleDynamic(simdag.DynamicScheduler):
  def prepare(self, simulation):
    for h in simulation.hosts:
      h.data = {}
    nxgraph = simulation.get_task_graph()
    platform_model = cscheduling.PlatformModel(simulation)
    ordered_tasks = cscheduling.heft_order(nxgraph, platform_model)
    for i, t in enumerate(ordered_tasks):
      t.data = i


  def schedule(self, simulation, changed):
    for h in simulation.hosts:
      h.data["free"] = True
    for task in simulation.tasks[simdag.TaskState.TASK_STATE_RUNNING, simdag.TaskState.TASK_STATE_SCHEDULED]:
      task.hosts[0].data["free"] = False
    #for t in sorted(simulation.tasks[simdag.TaskState.TASK_STATE_SCHEDULABLE], key=lambda t: t.data):
    for t in  simulation.tasks[simdag.TaskState.TASK_STATE_SCHEDULABLE]:
      free_hosts = simulation.hosts.by_data("free", True).sorted(lambda h: t.get_eet(h))
      if free_hosts:
        t.schedule(free_hosts[0])
        free_hosts[0].data["free"] = False
      else:
        break


_SCHEDULERS = {
 # "MinMin": algorithms.BatchMin,
  #"MaxMin": algorithms.BatchMax,
#  "Sufferage": algorithms.BatchSufferage,
 # "DLS": algorithms.DLS,
 # "RandomSchedule": algorithms.RandomStatic,
  "SimpleDynamic": SimpleDynamic,
 # "MCT": algorithms.MCT,
 # "OLB": algorithms.OLB,
 # "HCPT": algorithms.HCPT,
 # "HEFT": algorithms.HEFT,
 # "Lookahead": algorithms.Lookahead,
 # "PEFT": algorithms.PEFT
}


def run_simulation(args):
    scheduler_name, env, task = args
    scheduler_class = _SCHEDULERS[scheduler_name]
    with simdag.Simulation(env, task) as simulation:
    print(f'Start {scheduler_name} simulation')
    scheduler = scheduler_class(simulation)
    scheduler.run()
    print(f"""{scheduler_name}
        makespan: {simulation.clock}
        scheduling time: {scheduler.scheduler_time}
    """)
    return simulation.clock, scheduler_name


def main():
    results = []
    schedulers = list(_SCHEDULERS.keys())

    envs = ["./data/environment/exp1_systems/cluster_5_1-4_100_100_1.xml"] * len(schedulers)
    workflows = ["./data/workflows/dot/SIPHT.n.500.0.dot"] * len(schedulers)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        results = pool.map(run_simulation, zip(schedulers, envs, workflows))
    print(*sorted(results, key=lambda x: x[0]), sep='\n')
    return


if __name__ == '__main__':
    main()
