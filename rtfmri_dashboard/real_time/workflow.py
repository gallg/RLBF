from .preprocessing import preprocess_volume, generate_hrf_regressor
from ..agents.utils import generate_gaussian_kernel, discretize_observation
from ..agents.soft_q_learner import SoftQAgent
from checkerboard_env.utils import load_checkerboard

import rtfmri_dashboard.config as config
import gymnasium as gym
import checkerboard_env
import numpy as np


def load_environment(render_mode=None):
    board, inverse, cross = load_checkerboard(
        "./checkerboard_env/assets/checkerboard.png",
        "./checkerboard_env/assets/cross.png"
    )

    env = gym.make(
        "checkerboard-v0",
        render_mode=render_mode,
        checkerboard=board,
        inverse=inverse,
        cross=cross
    )
    return env


class RealTimeEnv:
    def __init__(self, render_mode=None):
        # load environment;
        self.environment = load_environment(render_mode)

        # settings for real-time processing;
        self.block_size = config.block_size
        self.rest_size = config.rest_size
        self.stimulus_onset = config.hrf_stimulus_onset
        self.tr = config.repetition_time
        self.epoch_duration = None
        self.hrf = None

        # settings and parameters for Q-Learning;
        self.n_bins = config.num_bins_per_observation
        self.q_table_noise_mean = config.q_table_noise_mean
        self.q_table_noise_sigma = config.q_table_noise_sigma
        self.kernel_size = config.kernel_size
        self.kernel_sigma = config.kernel_sigma

        parameters = {
            "learning_rate": config.learning_rate,
            "temperature": config.temperature,
            "min_temperature": config.min_temperature,
            "max_temperature": config.max_temperature,
            "reduce_temperature": config.reduce_temperature,
            "decay_rate": config.decay_rate
        }

        # initialize environment;
        self.previous_state = None
        self.agent = None
        self.initialize_env(parameters)
        self.initialize_fmri()

    def initialize_env(self, parameters):
        observation, _ = self.environment.reset()

        # Create q_table;
        q_table_shape = (
            self.n_bins,
            self.n_bins
        )
        q_table = np.random.normal(
            self.q_table_noise_mean,
            scale=self.q_table_noise_sigma,
            size=q_table_shape
        )

        # Generate RBF kernel;
        kernel = generate_gaussian_kernel(
            self.kernel_size,
            self.kernel_sigma
        )

        # Load Soft-Q agent;
        self.previous_state = discretize_observation(observation, self.n_bins)
        self.agent = SoftQAgent(
            self.environment,
            q_table,
            kernel,
            **parameters
        )

    def initialize_fmri(self):
        self.epoch_duration = self.block_size + self.rest_size
        self.hrf = generate_hrf_regressor(
            time_length=self.epoch_duration + self.stimulus_onset,
            duration=self.block_size,
            onset=self.stimulus_onset,
            amplitude=1.0,
            tr=self.tr
        )

    def run_realtime(self):
        pass

    def stop_realtime(self):
        self.environment.close()

