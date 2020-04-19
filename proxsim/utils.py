import typing
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from pysimgrid import simdag

from .context import Context
from .scheduler import MasterSchedulerBase, SlaveScheduler
from .simulation import ProximalSimulationMaster, ProximalSimulationSlave


def slave_scheduling(context: Context, connection: Connection):
    with simdag.Simulation(context.env_file, context.task_file) as simulation, ProximalSimulationSlave(connection) as proxy_slave:
        scheduler = SlaveScheduler(simulation, proxy_slave)
        scheduler.run()
        if context.slave_callback:
            proxy_slave.send(context.slave_callback(simulation))


def master_scheduling(context: Context, master_scheduler: MasterSchedulerBase):
    scheduling_result = None
    with ProximalSimulationMaster(context, slave_scheduling) as proxy_master:
        scheduler = master_scheduler(proxy_master.connection)
        scheduler.run()
        if context.master_callback:
            scheduling_result = context.master_callback(proxy_master.connection.recv())
    return scheduling_result
