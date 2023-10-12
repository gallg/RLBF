from ..agents.utils import generate_gaussian_kernel, discretize_observation
from ..agents.soft_q_learner import SoftQAgent
from checkerboard_env.utils import load_checkerboard
from .preprocessing import *

import rtfmri_dashboard.config as config
import gymnasium as gym
import checkerboard_env
import numpy as np
import json


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


def run_preprocessing(volume, template, affine, transformation=None):
    volume = reorient_volume(volume, affine, to_ants=True)

    if transformation is None:
        transformation = ants_registration(volume, template)

    volume = ants_transform(volume, template, transformation)
    return volume, transformation


class RealTimeEnv:
    def __init__(self, render_mode=None):
        # load environment;
        self.environment = load_environment(render_mode)

        # settings for real-time processing; #
        self.mask = None
        self.volume_counter = 0
        self.first_volume_has_arrived = False
        self.block_has_finished = False
        self.resting_state = True
        self.epoch_duration = None
        self.hrf = None

        # store real-time data;
        self.temporary_data = np.array([])
        self.real_time_data = np.array([])

        # initialize environment;
        self.current_epoch = 1
        self.previous_state = None
        self.agent = None
        self.initialize_env()
        self.initialize_hrf()

        # initialize log;
        self.log_realtime(
            [0, 0],
            0,
            init=True
        )

    def initialize_env(self):
        observation, _ = self.environment.reset()

        parameters = {
            "learning_rate": config.learning_rate,
            "temperature": config.temperature,
            "min_temperature": config.min_temperature,
            "max_temperature": config.max_temperature,
            "reduce_temperature": config.reduce_temperature,
            "decay_rate": config.decay_rate
        }

        # Create q_table;
        q_table_shape = (
            config.num_bins_per_observation,
            config.num_bins_per_observation
        )
        q_table = np.random.normal(
            config.q_table_noise_mean,
            scale=config.q_table_noise_sigma,
            size=q_table_shape
        )

        # Generate RBF kernel;
        kernel = generate_gaussian_kernel(
            config.kernel_size,
            config.kernel_sigma
        )

        # Load Soft-Q agent;
        self.previous_state = discretize_observation(
            observation,
            config.num_bins_per_observation
        )
        self.agent = SoftQAgent(
            self.environment,
            q_table,
            kernel,
            **parameters
        )

    def initialize_hrf(self):
        self.epoch_duration = config.block_size + config.rest_size
        self.hrf = generate_hrf_regressor(
            time_length=self.epoch_duration + config.hrf_stimulus_onset,
            duration=config.block_size,
            onset=config.hrf_stimulus_onset,
            amplitude=config.hrf_amplitude,
            tr=config.repetition_time
        )

    def update_rendering(self):
        # render checkerboard if resting state finished;
        if self.volume_counter > config.rest_size:
            self.resting_state = False

        _ = self.environment.step(
            [0, 0] if self.resting_state
            else self.previous_state
        )

    def get_mask_data(self, volume, mask):
        if not self.mask:
            self.mask = ants.image_read(mask)

        data = ants.utils.mask_image(volume, mask).numpy()
        return data[np.nonzero(data)]

    def calculate_reward(self):
        hrf_duration = self.epoch_duration + config.hrf_stimulus_onset \
            if config.hrf_stimulus_onset > 0 else self.epoch_duration

        mu = np.mean(self.temporary_data, axis=0)
        data_mean, data_std = np.mean(mu), np.std(mu)

        # save standardized data;
        standardized_data = (mu - data_mean) / data_std
        self.real_time_data = standardized_data if self.real_time_data.shape[0] == 0 \
            else np.hstack([self.real_time_data, standardized_data])

        reward = run_glm(
            self.real_time_data[-hrf_duration:].reshape(-1, 1),
            self.hrf.reshape(-1, 1)
        )
        return reward

    def run_realtime(self, volume, template, mask, affine, transformation=None):

        # render & update the environment;
        self.update_rendering()
        self.environment.render()

        if volume is not None:
            self.volume_counter += 1

            # preprocess volume;
            volume, transformation = run_preprocessing(
                volume,
                template,
                affine,
                transformation
            )

            # acquire data;
            data = self.get_mask_data(volume, mask)
            self.temporary_data = data if self.temporary_data.shape[0] == 0 \
                else np.vstack([self.temporary_data, data])

            # if block has finished, make environment step;
            if self.volume_counter == self.epoch_duration:
                action = self.agent.soft_q_action_selection()

                (next_state,
                 _,
                 terminated,
                 truncated,
                 info) = self.environment.step(action)

                # get old q-value;
                old_q_value = self.agent.q_table[self.previous_state]
                next_state = discretize_observation(next_state, self.agent.n_bins)
                reward = self.calculate_reward()

                # compute next q-value and update q-table;
                self.agent.q_table = self.agent.update_q_table(reward, next_state, old_q_value)
                self.previous_state = next_state

                # decide whether to reduce temperature or not;
                self.agent.reduce_temperature(
                    self.current_epoch,
                    reduce=config.reduce_temperature
                )

                # log current status and start a new epoch;
                self.log_realtime(
                    action,
                    reward,
                    init=False
                )
                self.reset_realtime()

    def log_realtime(self, action, reward, init=False):
        log = {
            "contrast": 0 if init else action[0],
            "frequency": 0 if init else action[1],
            "reward": 0 if init else reward,
            "resting_state": True if init else self.resting_state,
            "epoch": self.current_epoch
        }

        with open("../../data_out/log.json", "w") as json_file:
            json.dump(log, json_file)

    def reset_realtime(self):
        self.resting_state = True
        self.volume_counter = 0
        self.current_epoch += 1

    def stop_realtime(self):
        # save acquired data and close environment;
        np.save("../../data_out/data.npy", self.real_time_data)
        self.environment.close()
