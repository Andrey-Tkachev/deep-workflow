from enum import Enum

class SimulationState(Enum):
    Initial = 1
    Prepare = 2
    Schedule = 3
    Finally = 4
    Terminal = 5
    _End = 100


class ActionType(Enum):
    SetSchedule = 1
    GetEet = 2
    GetEcomt = 3
    GetGraph = 4
    GetHosts = 5
    GetTasks = 6
    _End = 100
