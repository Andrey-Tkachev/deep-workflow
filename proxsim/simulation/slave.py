import logging
import typing
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from ..action import ActionsIterator
from ..context import Context
from ..enums import ActionType, SimulationState


class ProximalSimulationSlave(object):

    def __init__(self, connection: Connection):
        self.connection = connection

    def __enter__(self):
        logging.info('Enter simulation scope')
        return self

    def __exit__(self, *args):
        logging.info('Leave simulation scope')
        if args:
            exc_type, exc_value, traceback = args
            logging.error(exc_value, exc_info=True)
        return False

    @contextmanager    
    def scheduling_scope(self):
        self.state(SimulationState.Initial)
        try:
            yield
        except Exception as e:
            logging.error(e, exc_info=True)
        finally:
            self.state(SimulationState.Terminal)

    def iterate_actions(self) -> ActionsIterator:
        return ActionsIterator(self.connection)

    def state(self, state: SimulationState):
        logging.debug(f'Simulation state {state.name}')
        self.connection.send(state)

    def send(self, data):
        self.connection.send(data)

    def close(self):
        if self.state != SimulationState.Terminal:
            logging.warning('Close connection at not terminal state')
        self.connection.close()