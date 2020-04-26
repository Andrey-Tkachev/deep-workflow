from .enums import ActionType
from multiprocessing.connection import Connection


class Action(object):
    def __init__(self, action: ActionType, params: dict):
        self.action = action
        self.params = params

    def __bool__(self):
        return self.action != ActionType._End


class ActionsIterator(object):

    def __init__(self, connection: Connection):
        self.connection = connection

    def __iter__(self):
        return self

    def __next__(self):
        action = self.connection.recv()
        if not action:
            raise StopIteration()
        return action
