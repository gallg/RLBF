from rtfmri_dashboard.real_time.utils import pad_array, inverse_roll, log_q_table, log_convergence
from rtfmri_dashboard.agents.utils import generate_gaussian_kernel, discretize_observation
from rtfmri_dashboard.agents.utils import convergence
from rtfmri_dashboard.agents.soft_q_learner import SoftQAgent, create_bins
from rtfmri_dashboard.envs.checkerboard import CheckerBoardEnv
from rtfmri_dashboard.real_time.preprocessing import *
from posixpath import join

import rtfmri_dashboard.config as config
import numpy as np
import threading
import json


class RealTimeEnv:
    def __init__(self):
        # load an instance of the environment for the agent;
        self.environment = CheckerBoardEnv()

        # settings for real-time processing; #
        self.mask = None
        self.volume_counter = 0
        self.collected_volumes = 0
        self.block_has_finished = False
        self.resting_state = True
        self.epoch_duration = 0
        self.hrf = None
        self.serializable_hrf = None
        self.output_dir = "../log"

        # store real-time data;
        self.noise = []
        self.temporary_data = []
        self.real_time_data = []
        self.real_time_noise = []

        # initialize environment;
        self.convergence_window_size = 0
        self.convergence = []
        self.action_log = []
        self.observation = None
        self.current_epoch = 1
        self.previous_state = None
        self.agent = None
        self.bins = None
        self.reward = []
        self.initialize_env()
        self.initialize_hrf()

        self.logger = None
        self.plot_volume = None

        # initialize log;
        self.log_realtime(
            [0, 0],
            None
        )

        self.serializable_hrf = []

    def initialize_env(self):
        self.observation = self.environment.reset()

        parameters = {
            "learning_rate": config.learning_rate,
            "temperature": config.temperature,
            "min_temperature": config.min_temperature,
            "max_temperature": config.max_temperature,
            "reduce_temperature": config.reduce_temperature,
            "decay_rate": config.decay_rate,
            "num_bins_per_obs": config.num_bins_per_observation
        }

        # generate bins;
        self.bins = create_bins(config.num_bins_per_observation)

        # Create q_table;
        q_table_shape = (
            config.num_bins_per_observation,
            config.num_bins_per_observation
        )

        # Start Q-table at 0.5 to allow agent punishment;
        q_table = np.ones(q_table_shape) * 0.5

        # Generate RBF kernel;
        kernel = generate_gaussian_kernel(
            config.kernel_size,
            config.kernel_sigma
        )

        # Load Soft-Q agent to get its functionalities;
        # The agent itself is not fitted;
        self.previous_state = discretize_observation(
            self.observation,
            self.bins
        )
        self.agent = SoftQAgent(
            self.environment,
            q_table,
            kernel,
            **parameters
        )

        # save initial state;
        state = np.array([
            self.resting_state,
            self.observation[0],
            self.observation[1]
        ])
        state.tofile(join(self.output_dir, "state.bin"))

        # convergence parameters;
        self.convergence_window_size = config.conv_window_size

        # log Q-table;
        log_q_table(
            self.agent.q_table,
            None,
            None,
            None,
            join(self.output_dir, "q_table.png")
        )

    def initialize_hrf(self):
        self.epoch_duration = config.block_size + config.rest_size
        self.hrf = generate_hrf_regressor(
            time_length=self.epoch_duration,
            duration=config.block_size,
            onset=config.hrf_stimulus_onset,
            amplitude=config.hrf_amplitude,
            tr=config.repetition_time
        )
        self.serializable_hrf = json.dumps(self.hrf.reshape(1, -1).tolist()[0])

    def update_rendering(self):
        # render checkerboard if resting state finished;
        self.resting_state = (True, False)[1 <= self.collected_volumes <= config.block_size]

        # save current state for rendering;
        state = np.array([
            self.resting_state,
            self.observation[0],
            self.observation[1]
        ])
        # np.save(join(self.output_dir, "state.npy"), state)
        state.tofile(join(self.output_dir, "state.bin"))

    def roll_over_data(self, data):
        overlap = (config.overlap_samples, 0)[self.current_epoch == 1]
        current_data = inverse_roll(
            data,
            overlap,
            self.epoch_duration
        )
        return np.array(current_data)

    def calculate_reward(self):
        self.real_time_data.extend(np.median(self.temporary_data, axis=1))

        # save standardized nuisance regressor;
        if len(self.noise) > 0:
            noise_mu, noise_mean, noise_std = standardize_signal(self.noise, axis=1)
            self.real_time_noise.extend((noise_mu - noise_mean) / noise_std)

        # roll over data arrays;
        current_data = self.roll_over_data(self.real_time_data)
        current_noise = self.roll_over_data(self.real_time_noise)

        reward = run_glm(
            current_data.reshape(-1, 1),
            self.hrf.reshape(-1, 1),
            current_noise.reshape(-1, 1)
        )
        return reward

    def run_realtime(self, volume, template, mask, affine, transformation=None, nuisance_mask=None):
        # render only with high contrast & frequency;
        if config.render_only and volume is not None:
            self.observation = np.array([1.0, 0.9])

        # update the environment rendering;
        self.update_rendering()

        if volume is not None:
            self.volume_counter += 1
            print("Volume: ", self.volume_counter)

            # preprocess volume;
            volume, transformation = run_preprocessing(
                volume,
                template,
                affine,
                transformation
            )

            # plot new, preprocessed, volume;
            if not (self.plot_volume and self.plot_volume.is_alive()):
                self.plot_volume = threading.Thread(
                    target=plot_image, args=(
                        volume,
                        mask,
                        False,
                        join(self.output_dir, "volume.png")
                    )
                )
                self.plot_volume.start()

            # acquire data, skip collecting the first resting block;
            if self.current_epoch > 1 or self.volume_counter > config.rest_size:
                self.collected_volumes += 1
                data, noise = get_mask_data(volume, mask, nuisance_mask=nuisance_mask)

                self.temporary_data.append(pad_array(data, self.temporary_data))
                if noise is not None:
                    self.noise.append(noise)

            # if block has finished, make environment step;
            if self.collected_volumes == self.epoch_duration:
                last_observation = self.observation

                if self.current_epoch > self.convergence_window_size:
                    self.convergence.append(
                        convergence(
                            self.action_log,
                            self.convergence_window_size
                        )
                    )

                # get old q-value;
                old_q_value = self.agent.q_table[self.previous_state]
                # next_state = discretize_observation(self.observation, self.bins)

                # get reward;
                reward = self.calculate_reward()
                self.reward.append(reward)

                # compute next q-value and update q-table;
                self.agent.q_table = self.agent.update_q_table(reward, self.previous_state, old_q_value)

                if not config.render_only:
                    self.observation = self.agent.soft_q_action_selection()
                    self.action_log.append(self.observation)

                next_state = discretize_observation(self.observation, self.bins)
                self.previous_state = next_state

                # decide whether to reduce temperature or not;
                self.agent.reduce_temperature(
                    self.current_epoch,
                    reduce=config.reduce_temperature
                )

                # log current status and start a new epoch;
                if not (self.logger and self.logger.is_alive()):
                    self.logger = threading.Thread(
                        target=self.log_realtime, args=(
                            last_observation,
                            self.observation
                        )
                    )
                    self.logger.start()

                # reset important variable and start new epoch;
                self.reset_realtime()

    def log_realtime(self, last_action, current_action):
        current_data = self.real_time_data

        # standardize signal for visualization purposes;
        if len(self.real_time_data) > 0:
            mu, mu_mean, mu_std = standardize_signal(self.temporary_data, axis=1)
            current_data = (mu - mu_mean) / mu_std

        # get correct data and noise samples;
        current_noise = self.roll_over_data(self.real_time_noise)
        current_data = self.roll_over_data(current_data)

        # ToDo: choose a good de-noising algorithm for data visualization;
        # ToDo: check action discretization and its visualization;
        # ToDo: add Q-table initialization as a parameter in the config file;
        # ToDo: check convergence calculation and output, add it to the dashboard;
        # ToDo: integrate new changes, with the dashboard and the program;
        # ToDo: add a motion correction algorithm compatible with RT processing;

        # serialize and log them in the json file;
        serializable_reward = json.dumps(self.reward)
        serializable_data = json.dumps(current_data.tolist())
        serializable_noise = json.dumps(current_noise.tolist())
        serializable_table = json.dumps(self.agent.q_table.tolist())

        log = {
            "contrast": last_action[0],
            "frequency": last_action[1],
            "reward": serializable_reward,
            "resting_state": self.resting_state,
            "epoch": self.current_epoch,
            "hrf": self.serializable_hrf,
            "fmri_data": serializable_data,
            "noise": serializable_noise,
            "q_table": serializable_table
        }

        try:
            with open(join(self.output_dir, "log.json"), "r") as json_file:
                json_data = json.load(json_file)
        except json.JSONDecodeError:
            json_data = []

        json_data.append(log)

        # write the updated data back to the log file;
        with open(join(self.output_dir, "log.json"), "w") as json_file:
            json_file.seek(0)
            json.dump(json_data, json_file, indent=4)

        # log Q-table;
        log_q_table(
            self.agent.q_table,
            last_action,
            current_action,
            config.num_bins_per_observation,
            join(self.output_dir, "q_table.png")
        )

        log_convergence(
            self.convergence,
            self.output_dir
        )

    def reset_realtime(self):
        self.temporary_data = []
        self.noise = []
        self.resting_state = True
        self.collected_volumes = 0
        self.volume_counter = 0
        self.current_epoch += 1

    def stop_realtime(self):
        # save acquired data and close environment;
        np.save(join(self.output_dir, "data.npy"), self.real_time_data)
        np.save(join(self.output_dir, "noise.npy"), self.real_time_noise)
