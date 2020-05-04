import logging
from comet_ml import Experiment
from collections import defaultdict

class TrainManager(object):

    def __init__(
        self,
        episodes_num,
        epochs_num,
        track_size,
        experiment: Experiment = None
    ):
        self.experiment = experiment
        self.counters = {
            'episode': None,
            'epoch': None,
            'track_item': None,
        }
        self._track = 0
        self.episodes_num = episodes_num
        self.epochs_num = epochs_num
        self.track_size = track_size
        self.metrics = defaultdict(lambda : defaultdict(list))

    @property
    def episode(self):
        return self.counters['episode']

    @property
    def epoch(self):
        return self.counters['epoch']

    @property
    def track_item(self):
        return self.counters['track_item']

    @property
    def track(self):
        if self._track is None:
            return None
        if (self._track + 1) * self.track_size == self.episode:
            self._track += 1
        return self._track

    def _counter_range(self, counter_name, upper_bound):
        assert self.counters[counter_name] is None, f'Allready iterating over {counter_name}s'
        self.counters[counter_name] = 0
        while self.counters[counter_name] < upper_bound:
            yield self.counters[counter_name]
            self.counters[counter_name] += 1
        self.counters[counter_name] = None

    def episode_range(self):
        self._track = 0
        return self._counter_range('episode', self.episodes_num)

    def epoch_range(self):
        return self._counter_range('epoch', self.epochs_num)

    def track_item_range(self):
        return self._counter_range('track_item', self.track_size)

    def _episode_epoch_message(self):
        episode_message = f'episode {self.episode + 1};' if self.episode is not None else ''
        track_message = f' track {self.track + 1};' if self.track is not None else ''
        epoch_message = f' epoch {self.epoch + 1};' if self.epoch is not None else ''
        return f'{episode_message}{track_message}{epoch_message}'

    def have_to_update_file(self):
        return self.episode % self.track_size == 0
    
    def have_to_update_policy(self):
        return (self.episode + 1) % self.track_size == 0

    def metric(self, name, value, per_episode=False):
        self.metrics[name]['values'].append(value)
        self.metrics[name]['episode'].append(self.episode)
        self.metrics[name]['track'].append(self.track)
        if self.epoch is not None:
            self.metrics[name]['epoch'].append(self.epoch)

        if self.experiment is not None:
            self.experiment.log_metric(
                name, value, step=self.episode if per_episode else self.track
            )
        logging.info(f'{self._episode_epoch_message()} {name}: {value}')

    def text(self, text, per_episode=False, upload=False):
        message = f'{self._episode_epoch_message()} {text}'
        if self.experiment is not None and upload:
            self.experiment.log_text(message, step=self.episode if per_episode else self.track)
        logging.info(message)

    def figure(self, name, figure):
        if self.experiment is not None:
            self.experiment.log_figure(name, figure=figure, step=self.episode)
