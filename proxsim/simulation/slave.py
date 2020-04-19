import logging
import typing
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from ..action import ActionsIterator
from ..context import Context
from ..enums import ActionType, SimulationState


class ProximalSimulationSlave(object):

    def __init__(self, connection: Connection):
        self.connection = connection

    def __enter__(self):
        self.state(SimulationState.Initial)
        return self

    def __exit__(self, *args):
        self.state(SimulationState.Terminal)
        return False

    def iterate_actions(self) -> ActionsIterator:
        return ActionsIterator(self.connection)

    def state(self, state: SimulationState):
        logging.info(f'Simulation state {state.name}')
        self.connection.send(state)

    def send(self, data):
        self.connection.send(data)

    def close(self):
        if self.state != SimulationState.Terminal:
            logging.warning('Close connection at not terminal state')
        self.connection.close()