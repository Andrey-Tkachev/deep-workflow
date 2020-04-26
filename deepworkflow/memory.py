from contextlib import contextmanager

import numpy as np


class Memory:
    def __init__(self):
        self._reset()
    
    def _reset(self):
        self.actions = None
        self.states = None
        self.actions_batch = []
        self.states_batch = []
        self.rewards_batch = []
        self.active = True
        self.episode_reward = 0

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.actions_batch[:]
        del self.states_batch[:]
        del self.rewards_batch[:]
        self._reset()

    def register(self, action, state):
        if not self.active:
            return
        self.states.append(state)
        self.actions.append(action)

    def set_reward(self, episode_reward):
        self.episode_reward = episode_reward

    def start_episode(self):
        self.actions = []
        self.states = []
        self.episode_reward = 0

    def end_episode(self, episode_reward, gamma=0.99, reward_mode='gamma'):
        reward_pool = np.zeros((len(self.actions,)))
        if reward_mode == 'gamma':
            running_add = 0
            reward_pool[-1] = episode_reward
            for i in reversed(range(len(reward_pool))):
                running_add = running_add * gamma + reward_pool[i]
                reward_pool[i] = running_add
        else:
            mid = len(reward_pool) // 2
            coefs = np.exp(-0.5 * ((np.arange(len(reward_pool)) - mid) / mid) ** 2.0)
            for i in reversed(range(len(reward_pool))):
                reward_pool[i] = coefs[i] * episode_reward

        self.rewards_batch.append(reward_pool)
        self.actions_batch.append(self.actions)
        self.states_batch.append(self.states)

    @contextmanager
    def episode(self, gamma=0.99, reward_mode='gamma'):
        assert reward_mode in ['gamma', 'normal']
        _active_prev = self.active
        self.active = True

        self.start_episode()
        try:
            yield
        finally:
            self.end_episode(self.episode_reward, gamma, reward_mode)
            self.active = _active_prev

    @contextmanager
    def no_memory(self):
        _active_prev = self.active
        self.active = False

        try:
            yield
        finally:
            self.active = _active_prev
