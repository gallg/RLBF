import time

from rtfmri_dashboard.real_time.utils import StateManager
from rtfmri_dashboard import config
from posixpath import join
import numpy as np
import pyray as pr
import os


class CheckerBoardEnv:
    def __init__(self, scandir=None, board=None, cross=None, render_mode=None):
        self.screen_width = 1920
        self.screen_height = 1080
        self.board_scale = config.board_scale
        self.cross_scale = config.cross_scale
        self.center = (self.screen_width // 2, self.screen_height // 2)

        self.contrast = 0
        self.frequency = 0
        self.flickering_timer = 0
        self.resting_state = True
        self.render_mode = render_mode
        self.fps = config.fps

        # timing related variables;
        self.scandir = scandir
        self.t0 = len(os.listdir(scandir))
        self.n_vols = 0
        self.total_vols = 0
        self.t_start = 0
        self.timing = np.array([])
        self.last_state = True
        self.last_epoch = 0

        # agent actions management variables;
        self.state_manager = StateManager("/mnt/fmritemp/state.bin")
        self.log_scale = np.logspace(-0.7, np.log10(1.1), 10) - 0.1
        self.log_scale[0] = 0

        if self.render_mode == "human":
            pr.init_window(self.screen_width, self.screen_height, "Checkerboard Environment")
            pr.set_window_state(pr.ConfigFlags.FLAG_BORDERLESS_WINDOWED_MODE)
            pr.set_window_state(pr.ConfigFlags.FLAG_WINDOW_TOPMOST)
            pr.clear_background(pr.BLACK)
            pr.set_target_fps(self.fps)

            self.checkerboard_texture = pr.load_texture(str(board))
            self.cross_texture = pr.load_texture(str(cross))

            self.board_size = (self.checkerboard_texture.width, self.checkerboard_texture.height)
            self.cross_size = (self.cross_texture.width, self.cross_texture.height)

    def reset(self):
        self.resting_state = True
        observation = (np.random.rand(), np.random.rand())
        self.contrast, self.frequency = observation
        return observation

    def step(self):
        if len(os.listdir(self.scandir)) > self.t0:
            self.t0 += 1
            self.n_vols += 1
            self.total_vols += 1

            # debug time;
            # t_end = time.perf_counter() - self.t_start
            # self.debug_time(t_end)
            # np.save(join(output_dir, "timing.npy"), self.timing)

        if self.total_vols == 1:
            self.t_start = time.perf_counter()

        if self.n_vols > config.rest_size:
            state = self.state_manager.read_state()

            if state is not None and len(state) > 0:
                current_epoch = state[2]

                # check whether the state file has been updated,
                # then update rendering;
                if current_epoch > self.last_epoch or current_epoch == 0:
                    self.resting_state = False
                    self.contrast = self.log_scale[
                        int(round(state[0], 1) * config.num_bins_per_observation)
                    ]
                    self.frequency = state[1]
                    self.last_epoch = current_epoch

        if self.n_vols > config.rest_size + config.block_size:
            self.contrast, self.frequency = (0, 0)
            self.resting_state = True
            self.n_vols = 1

    def debug_time(self, t_end):
        if self.resting_state != self.last_state:
            self.last_state = self.resting_state
            self.timing = np.append(self.timing, t_end)

    def render(self):
        if self.render_mode != "human":
            return

        self.event_handler()
        pr.begin_drawing()

        cross_center = (
            self.center[0] - (self.cross_size[0] * self.cross_scale) / 2,
            self.center[1] - (self.cross_size[1] * self.cross_scale) / 2
        )

        board_center = (
            self.center[0] - (self.board_size[0] * self.board_scale) / 2,
            self.center[1] - (self.board_size[1] * self.board_scale) / 2
        )

        pr.draw_texture_ex(self.cross_texture, cross_center, 0, self.cross_scale, pr.GRAY)
        pr.end_drawing()

        if not self.resting_state:
            # adjust contrast
            brightness = int(self.contrast * 255)
            base_checkerboard = self.checkerboard_texture
            board_color = pr.Color(brightness, brightness, brightness, 255)

            # adjust frequency
            if self.frequency < 0.1:
                self.flickering_timer = 1.0
            else:
                adjusted_frequency = self.frequency * 30
                self.flickering_timer += adjusted_frequency / self.fps

            if self.flickering_timer >= 1.0:
                pr.begin_drawing()
                pr.draw_texture_ex(base_checkerboard, board_center, 0, self.board_scale, board_color)
                pr.end_drawing()
                self.flickering_timer = 0

        if self.frequency >= 0.1 or self.resting_state:
            pr.clear_background(pr.BLACK)

        pr.set_window_title(f"Contrast: {self.contrast}, Frequency: {self.frequency}")

    def event_handler(self):
        if pr.window_should_close():
            self.close()
        elif pr.is_key_pressed(pr.KEY_F11):
            pr.toggle_borderless_windowed()

    def close(self):
        self.state_manager.close()
        pr.close_window()
