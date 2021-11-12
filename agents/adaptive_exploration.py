import numpy as np


class PolyakAvgSuccessRate(object):
    def __init__(self, goal_num, tau=0.05):
        self.tau = tau
        self.success_rates = np.zeros(goal_num)

    def update_success_rates(self, new_tet_suc_rate):
        old_tet_suc_rate = self.success_rates.copy()
        self.success_rates = (1-self.tau)*old_tet_suc_rate + self.tau*new_tet_suc_rate

    def __call__(self, goal_ind):
        return self.success_rates[goal_ind].copy()


class AdaptiveEpsilonGaussian(object):
    def __init__(self, goal_num, action_dim, action_max, tau=0.05, epsilon=0.2, scale=1, mu=0, sigma=0.2, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.scale = scale
        self.action_dim = action_dim
        self.action_max = action_max
        self.goal_num = goal_num
        self.tau = tau
        self.success_rates = np.zeros(self.goal_num)

        self.base_epsilon = epsilon
        self.epsilon = np.ones(self.goal_num) * epsilon
        self.mu = mu
        self.base_sigma = sigma
        self.sigma = np.ones(self.goal_num) * sigma

    def update_success_rates(self, new_tet_suc_rate):
        old_tet_suc_rate = self.success_rates.copy()
        self.success_rates = (1-self.tau)*old_tet_suc_rate + self.tau*new_tet_suc_rate
        self.epsilon = self.base_epsilon*(1-self.success_rates)
        self.sigma = self.base_sigma*(1-self.success_rates)

    def __call__(self, action, goal_ind):
        # return a random action or a noisy action
        prob = self.rng.uniform(0, 1)
        if prob < self.epsilon[goal_ind]:
            return self.rng.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
        else:
            noise = self.scale*self.sigma[goal_ind]*self.rng.standard_normal(self.action_dim)
            return np.clip(action + noise, -self.action_max, self.action_max)


class EpsilonGaussian(object):
    # the one used in the HER paper: https://arxiv.org/abs/1707.01495
    def __init__(self, action_dim, action_max, epsilon=0.2, scale=1, mu=0, sigma=0.1, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.scale = scale
        self.action_dim = action_dim
        self.action_max = action_max
        self.epsilon = epsilon
        self.mu = mu
        self.sigma = sigma

    def __call__(self, action, _):
        prob = self.rng.uniform(0, 1)
        if prob < self.epsilon:
            return self.rng.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
        else:
            noise = self.scale*self.sigma*self.rng.standard_normal(self.action_dim)
            return np.clip(action + noise, -self.action_max, self.action_max)


class AdaptiveEpsilonGreedy(object):
    def __init__(self, goal_num, start=1, end=0.05, tau=0.05, rng=None):
        self.start = start
        self.end = end
        self.epsilon = np.array([start, start, start])
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.goal_num = goal_num
        self.tau = tau
        self.success_rates = np.zeros(self.goal_num)

    def update_success_rates(self, new_tet_suc_rate):
        old_tet_suc_rate = self.success_rates.copy()
        self.success_rates = (1-self.tau)*old_tet_suc_rate + self.tau*new_tet_suc_rate
        self.epsilon = self.end + (self.start - self.end) * (1 - self.success_rates)

    def __call__(self, goal_ind):
        prob = self.rng.uniform(0, 1)
        if prob < self.epsilon[goal_ind]:
            return True
        else:
            return False


class ExpDecayEpsilonGreedy(object):
    # e-greedy exploration with exponential decay
    def __init__(self, start=1, end=0.05, decay=50000, decay_start=None, rng=None):
        self.start = start
        self.end = end
        self.decay = decay
        self.decay_start = decay_start
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng

    def __call__(self, count):
        if self.decay_start is not None:
            count -= self.decay_start
            if count < 0:
                count = 0
        epsilon = self.end + (self.start - self.end) * np.exp(-1. * count / self.decay)
        prob = self.rng.uniform(0, 1)
        if prob < epsilon:
            return True
        else:
            return False
