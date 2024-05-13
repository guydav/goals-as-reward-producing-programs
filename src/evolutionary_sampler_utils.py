from collections import defaultdict
import typing

import numpy as np

class Selector():
    def update(self, key, reward):
        raise NotImplementedError

    def select(self, keys, rng, n=1):
        raise NotImplementedError

    def _top_n_keys(self, keys: list, values: np.ndarray, n: int):
        """
        Returns the top n keys from the given list of keys, sorted by the given values
        """
        top_n_indices = np.argpartition(values, -n)[-n:]
        top_n_keys = [keys[i] for i in top_n_indices]
        return top_n_keys

DEFAULT_EXPLORATION_CONSTANT = np.sqrt(0.5)
DEFAULT_BUFFER_SIZE = 32

class UCBSelector(Selector):
    '''
    Implements the Upper Confidence Bound (UCB) algorithm for selecting from a provided
    set of keys (arms). Takes in an exploration constant 'c' and an optional buffer size
    '''
    def __init__(self,
                 c: float = DEFAULT_EXPLORATION_CONSTANT,
                 buffer_size: typing.Optional[int] = DEFAULT_BUFFER_SIZE,
                 **kwargs):


        self.c = c
        self.buffer_size = buffer_size

        self.reward_map = defaultdict(list)
        self.reward_sum = defaultdict(int)
        self.count_map = defaultdict(int)

        self.n_draws = 0

    def update(self, key, reward):
        '''
        Updates the buffer for the arm specified by the key with the provided reward. If the buffer
        is full, the oldest reward is removed
        '''
        self.reward_map[key].append(reward)
        self.reward_sum[key] += reward
        if self.buffer_size is not None and len(self.reward_map[key]) > self.buffer_size:
            out = self.reward_map[key].pop(0)
            self.reward_sum[key] -= out

    def select(self, keys: list, rng: np.random.Generator, n: int = 1):
        '''
        Given a list of keys, returns the key with the highest UCB score and updates the internal
        counter for the selected key (and overall count)
        '''
        reward_values = np.array([self.reward_sum[key] / self.count_map[key] if key in self.count_map else np.inf for key in keys])
        log_draws = np.log(self.n_draws) if self.n_draws > 0 else 0
        c_values = self.c * np.sqrt(log_draws / np.array([self.count_map[key] if key in self.count_map else 1 for key in keys]))
        ucb_values = reward_values + c_values

        self.n_draws += n
        if n == 1:
            max_index = np.argmax(ucb_values)
            max_key = keys[max_index]
            self.count_map[max_key] += 1
            return max_key

        else:
            max_keys = self._top_n_keys(keys, ucb_values, n)
            for max_key in max_keys:
                self.count_map[max_key] += 1

            return max_keys


class ThompsonSamplingSelector(Selector):
    '''
    Implements the Thompson Sampling algorithm for selecting from a provided set of keys (arms).
    Assumes that the rewards are Bernoulli distributed (i.e. 0 or 1)
    '''
    def __init__(self,
                 prior_alpha: int = 1,
                 prior_beta: int = 1,
                 buffer_size: typing.Optional[int] = DEFAULT_BUFFER_SIZE,
                 generation_size: int = 1,
                 **kwargs):

        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.buffer_size = buffer_size
        self.generation_size = generation_size

        self.reward_map = defaultdict(list)
        self.alpha = defaultdict(int)
        self.beta = defaultdict(int)

        self.pre_computed_sample_key_indices = None
        self.current_sample_key_index = 0

    def update(self, key, reward):
        self.reward_map[key].append(reward)
        if reward == 1:
            self.alpha[key] += 1
        else:
            self.beta[key] += 1

        if self.buffer_size is not None and len(self.reward_map[key]) > self.buffer_size:
            out = self.reward_map[key].pop(0)
            if out == 1:
                self.alpha[key] -= 1
            else:
                self.beta[key] -= 1

        self.current_sample_key_index = 0

    def select(self, keys, rng, n=1):
        '''
        Given a list of keys, returns the key with the highest sampled mean
        '''
        if self.generation_size == 1:
            alpha = np.array([self.alpha[key] for key in keys]) + self.prior_alpha
            beta = np.array([self.beta[key] for key in keys]) + self.prior_beta
            thompson_values = rng.beta(alpha, beta)

            if n == 1:
                return keys[np.argmax(thompson_values)]

            else:
                return self._top_n_keys(keys, thompson_values, n)

        else:
            if (self.current_sample_key_index) & self.generation_size == 0 or self.pre_computed_sample_key_indices is None:
                alpha = np.array([self.alpha[key] for key in keys]) + self.prior_alpha
                beta = np.array([self.beta[key] for key in keys]) + self.prior_beta
                thompson_values = rng.beta(alpha, beta, (self.generation_size, len(keys)))
                self.pre_computed_sample_key_indices = np.argmax(thompson_values, axis=1)

            if n == 1:
                max_key = keys[self.pre_computed_sample_key_indices[self.current_sample_key_index]]
                self.current_sample_key_index += 1
                return max_key

            else:
                max_keys = [keys[i] for i in self.pre_computed_sample_key_indices[self.current_sample_key_index:self.current_sample_key_index + n]]
                self.current_sample_key_index += n
                return max_keys
