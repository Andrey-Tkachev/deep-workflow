import logging
import typing
from multiprocessing import Pipe, Process

from ..context import Context
from ..enums import SimulationState


class ProximalSimulationMaster(object):

    def __init__(self, context, target, join_timeout=None):
        self.context = context
        self.target = target
        self.join_timeout = join_timeout

    def __enter__(self):
        master_connection, slave_connection = Pipe()
        self._connection = master_connection
        self._slave_process = Process(target=self.target, args=(self.context.light_copy(), slave_connection))
        self._slave_process.start()
        slave_connection.close()
        return self

    @property
    def connection(self):
        assert not self._connection.closed, 'Trying to accuire closed connection'
        return self._connection

    def __exit__(self, *args):
        if not self._connection.closed:
            self._connection.close()
        if args:
            self._slave_process.terminate()
        else:
            self._slave_process.join(self.join_timeout)
        return False
