from scipy.special import softmax
from scipy.signal import convolve2d
from .utils import *
import numpy as np


class SoftQAgent:

    def __init__(self,
                 env,                       # Environment;
                 q_table,                   # initialized Q-table;
                 kernel,                    # used to smooth the Q-table;
                 render_mode=None,
                 learning_rate=0.8,
                 temperature=0.1,
                 min_temperature=0.001,
                 max_temperature=1.0,
                 reduce_temperature=False,
                 decay_rate=0.001,
                 num_bins_per_obs=10
                 ):

        self.env = env
        self.q_table = q_table
        self.reward_log = []

        self.render_mode = render_mode
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.reduce_temp = reduce_temperature
        self.decay_rate = decay_rate
        self.n_bins = num_bins_per_obs

    def fit(self, epochs, bins):

        current_epoch = 0
        initial_state = self.env.reset()[0]
        discrete_state = discretize_observation(initial_state, bins)

        while current_epoch <= epochs:
            action = self.soft_q_action_selection()

            (next_state,
             reward,
             terminated,
             truncated,
             info) = self.env.step(action)

            # grab current q-value;
            old_q_value = self.q_table[discrete_state]

            # discretize the next state;
            next_state_discrete = discretize_observation(next_state, bins)

            # compute next q-value and update q-table;
            self.q_table = self.update_q_table(reward, next_state_discrete, old_q_value)

            # update state;
            discrete_state = next_state_discrete

            # by default keep temperature constant;
            self.reduce_temperature(current_epoch, reduce=self.reduce_temp)
            self.reward_log.append(reward)

            current_epoch += 1

        return self

    def soft_q_action_selection(self):

        # find q-values in the table and their corresponding action;
        q_values = self.q_table.flatten()
        n_actions = len(q_values)

        # calculate soft-Q values;
        softmax_values = softmax(q_values / self.temperature)

        # randomly select action with softmax probability;
        selected_action = np.random.choice(n_actions, p=softmax_values)
        action = np.array(np.unravel_index(selected_action, self.q_table.shape)) / self.n_bins

        return action

    def update_q_table(self, reward, discrete_state, old_q_value):

        # initialize q_table update;
        q_update = np.zeros(self.q_table.shape)
        q_update[discrete_state] = self.compute_next_q_value(old_q_value, reward)

        # update q_table;
        q_update = convolve2d(q_update, self.kernel, boundary='symm', mode='same')
        self.q_table += q_update
        return self.q_table

    def reduce_temperature(self, epoch, reduce=False):
        if reduce:
            self.temperature = self.min_temperature + (
                    self.max_temperature - self.min_temperature
            ) * np.exp(-self.decay_rate * epoch)

        return self.temperature

    def compute_next_q_value(self, old_q_value, reward):
        return self.learning_rate * (reward - old_q_value)
