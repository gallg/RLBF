from rtfmri_dashboard.agents.utils import generate_gaussian_kernel, discretize_observation
from rtfmri_dashboard.agents.soft_q_learner import SoftQAgent, create_bins
from rtfmri_dashboard.real_time.utils import pad_array
from rtfmri_dashboard.envs.checkerboard import CheckerBoardEnv
from rtfmri_dashboard.real_time.preprocessing import *
from posixpath import join

import rtfmri_dashboard.config as config
import ants.core.ants_image
import numpy as np
import matplotlib
import threading
import json


class RealTimeEnv:
    def __init__(self):
        # load environment;
        self.environment = CheckerBoardEnv(
            board="../rtfmri_dashboard/envs/assets/checkerboard.png",
            cross="../rtfmri_dashboard/envs/assets/cross.png"
        )

        # settings for real-time processing; #
        self.mask = None
        self.volume_counter = 0
        self.block_has_finished = False
        self.resting_state = True
        self.epoch_duration = 0
        self.hrf = None
        self.serializable_hrf = None
        self.output_dir = "../log"

        # store real-time data;
        self.temporary_data = np.array([])
        self.real_time_data = np.array([])

        # initialize environment;
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
        matplotlib.use("Agg")
        self.log_realtime(
            [0, 0],
            self.epoch_duration
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
            "decay_rate": config.decay_rate
        }

        # generate bins;
        self.bins = create_bins(config.num_bins_per_observation)

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
        # self.log_q_table(q_table)

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

    def initialize_hrf(self):
        self.epoch_duration = config.block_size + config.rest_size
        self.hrf = generate_hrf_regressor(
            time_length=self.epoch_duration + config.hrf_stimulus_onset,
            duration=config.block_size,
            onset=config.hrf_stimulus_onset,
            amplitude=config.hrf_amplitude,
            tr=config.repetition_time
        )
        self.serializable_hrf = json.dumps(self.hrf.reshape(1, -1).tolist()[0])

    def update_rendering(self):
        # render checkerboard if resting state finished;
        if self.volume_counter > config.rest_size:
            self.resting_state = False

        self.environment.set(
            self.resting_state,
            self.observation[0],
            self.observation[1]
        )
        self.environment.step()

    def render_only(self):
        self.observation = (1.0, 0.9)
        self.volume_counter += 1
        self.update_rendering()

        print("Volume: ", self.volume_counter)
        if self.volume_counter == self.epoch_duration:
            self.reset_realtime()

    def get_mask_data(self, volume, mask):
        if not isinstance(mask, ants.core.ants_image.ANTsImage):
            self.mask = ants.image_read(mask)

        data = ants.utils.mask_image(volume, mask).numpy()
        return data[np.nonzero(data)]

    def calculate_reward(self):
        mu = np.mean(self.temporary_data, axis=1)
        data_mean, data_std = np.mean(mu), np.std(mu)

        # save standardized data;
        standardized_data = (mu - data_mean) / data_std
        self.real_time_data = standardized_data if self.real_time_data.shape[0] == 0 \
            else np.hstack([self.real_time_data, standardized_data])

        # overall function duration (allows overlaps between blocks);
        hrf_duration = self.epoch_duration + config.hrf_stimulus_onset

        # handle block overlaps;
        current_hrf = self.hrf.reshape(-1, 1) if self.current_epoch > 1 \
            else self.hrf[config.hrf_stimulus_onset:].reshape(-1, 1)

        reward = run_glm(
            self.real_time_data[-hrf_duration:].reshape(-1, 1),
            current_hrf
        )
        return reward, hrf_duration

    def run_realtime(self, volume, template, mask, affine, transformation=None):

        if config.render_only and volume is not None:
            # just render with high contrast & frequency;
            self.render_only()
            volume = None
        else:
            # render & update the environment;
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

            # acquire data;
            data = self.get_mask_data(volume, mask)

            self.temporary_data = data if self.temporary_data.shape[0] == 0 \
                else np.vstack([self.temporary_data, pad_array(data, self.temporary_data)])

            # if block has finished, make environment step;
            if self.volume_counter == self.epoch_duration:
                last_observation = self.observation
                self.observation = self.agent.soft_q_action_selection()

                # get old q-value;
                old_q_value = self.agent.q_table[self.previous_state]
                next_state = discretize_observation(self.observation, self.bins)

                # get reward;
                reward, hrf_duration = self.calculate_reward()
                self.reward.append(reward)

                # compute next q-value and update q-table;
                self.agent.q_table = self.agent.update_q_table(reward, next_state, old_q_value)
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
                            hrf_duration
                        )
                    )
                    self.logger.start()

                # log Q-table;
                # self.log_q_table(self.agent.q_table)

                # reset important variable and start new epoch;
                self.reset_realtime()

    # def log_q_table(self, q_table):
    #     plt.figure()
    #     plt.imshow(q_table)
    #     plt.savefig(join(self.output_dir, "q_table.png"))
    #     plt.close()

    def log_realtime(self, action, hrf_duration):
        serializable_reward = json.dumps(self.reward)
        serializable_data = json.dumps(self.real_time_data[-hrf_duration:].tolist())

        log = {
            "contrast": action[0],
            "frequency": action[1],
            "reward": serializable_reward,
            "resting_state": self.resting_state,
            "epoch": self.current_epoch,
            "hrf": self.serializable_hrf,
            "fmri_data": serializable_data
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

    def reset_realtime(self):
        self.resting_state = True
        self.volume_counter = 0
        self.current_epoch += 1

    def stop_realtime(self):
        # save acquired data and close environment;
        np.save(join(self.output_dir, "data.npy"), self.real_time_data)
        self.environment.close()
