import typing
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from pysimgrid import simdag

from .context import Context
from .scheduler import MasterSchedulerBase, SlaveScheduler
from .simulation import ProximalSimulationMaster, ProximalSimulationSlave


def slave_scheduling(context: Context, connection: Connection):
    with ProximalSimulationSlave(connection) as proxy_slave, simdag.Simulation(context.env_file, context.task_file) as simulation:
        with proxy_slave.scheduling_scope():
            scheduler = SlaveScheduler(simulation, proxy_slave, context.feature)
            scheduler.run()
        if context.slave_callback:
            proxy_slave.send(context.slave_callback(simulation))
        connection.close()


def master_scheduling(context: Context, master_scheduler: MasterSchedulerBase):
    scheduling_result = None
    with ProximalSimulationMaster(context, slave_scheduling) as proxy_master:
        scheduler = master_scheduler(proxy_master.connection, context)
        scheduler.run()
        if context.master_callback:
            scheduling_result = context.master_callback(proxy_master.connection.recv())
        proxy_master.connection.close()
    return scheduling_result
