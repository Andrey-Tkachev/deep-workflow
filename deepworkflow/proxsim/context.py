from multiprocessing.connection import Connection
from typing import Type, Any, AnyStr, Callable, Optional, Union

from pysimgrid.simdag import Simulation

from .feature import FeatureExtractorBase, FeatureExtractorDummy


class Context(object):


    @staticmethod
    def default_slave_callback(sim: Simulation) -> float:
        return sim.clock

    @staticmethod
    def default_master_callback(result: float) -> float:
        return result

    def light_copy(self):
        return Context(
            self.env_file,
            self.task_file,
            self.slave_callback,
            self.master_callback,
            self.feature,
        )

    def __init__(self,
            env_file: AnyStr,
            task_file: AnyStr,
            slave_callback: Optional[Union[Callable[[Simulation], Any], str]] = 'default',
            master_callback: Optional[Union[Callable, str]] = 'default',
            feature: Optional[Union[Type[FeatureExtractorBase], str]] = 'dummy',
        ):
        assert (slave_callback is not None) == (master_callback is not None), 'Either specify both callbacks either dont specify any'
        if isinstance(slave_callback, str):
            assert slave_callback == 'default'
            slave_callback = self.default_slave_callback
        if isinstance(master_callback, str):
            assert master_callback == 'default'
            master_callback = self.default_master_callback
        if isinstance(feature, str):
            assert feature == 'dummy'
            feature = FeatureExtractorDummy()
    
        self.env_file = env_file
        self.task_file = task_file
        self.slave_callback = slave_callback
        self.master_callback = master_callback
        self.feature = feature
