from rtfmri_dashboard import config
from posixpath import join
import numpy as np
import pyray as rl
import os


class CheckerBoardEnv:
    def __init__(self, board=None, cross=None, render_mode=None):
        self.screen_width = 1280
        self.screen_height = 720
        self.board_scale = config.board_scale
        self.cross_scale = config.cross_scale
        self.center = (self.screen_width // 2, self.screen_height // 2)

        self.contrast = 0
        self.frequency = 0
        self.flickering_timer = 0
        self.resting_state = True
        self.render_mode = render_mode
        self.fps = config.fps

        if self.render_mode == "human":
            rl.init_window(self.screen_width, self.screen_height, "Checkerboard Environment")
            rl.clear_background(rl.BLACK)
            rl.set_target_fps(self.fps)

            self.checkerboard_texture = rl.load_texture(str(board))
            self.cross_texture = rl.load_texture(str(cross))

            self.board_size = (self.checkerboard_texture.width, self.checkerboard_texture.height)
            self.cross_size = (self.cross_texture.width, self.cross_texture.height)

    def reset(self):
        self.resting_state = True
        observation = (np.random.rand(), np.random.rand())
        self.contrast, self.frequency = observation
        return observation

    def step(self, output_dir):
        if os.path.isfile(join(output_dir, "state.bin")):
            state = np.fromfile(join(output_dir, "state.bin"))
            if not state.shape[0] == 0:
                (self.resting_state,
                    self.contrast,
                    self.frequency) = state
        else:
            (self.resting_state,
                self.contrast,
                self.frequency) = (True, 0, 0)

    def render(self):
        if self.render_mode != "human":
            return

        self.event_handler()
        rl.begin_drawing()

        cross_center = (
            self.center[0] - (self.cross_size[0] * self.cross_scale) / 2,
            self.center[1] - (self.cross_size[1] * self.cross_scale) / 2
        )

        board_center = (
            self.center[0] - (self.board_size[0] * self.board_scale) / 2,
            self.center[1] - (self.board_size[1] * self.board_scale) / 2
        )

        rl.draw_texture_ex(self.cross_texture, cross_center, 0, self.cross_scale, rl.GRAY)
        rl.end_drawing()

        if not self.resting_state:
            # adjust contrast
            brightness = int(self.contrast * 255)
            base_checkerboard = self.checkerboard_texture
            board_color = rl.Color(brightness, brightness, brightness, 255)

            # adjust frequency
            if self.frequency == 0:
                self.flickering_timer = 1.0
            else:
                adjusted_frequency = self.frequency * 30
                self.flickering_timer += adjusted_frequency / self.fps

                if self.flickering_timer >= 1.0:
                    rl.begin_drawing()
                    rl.draw_texture_ex(base_checkerboard, board_center, 0, self.board_scale, board_color)
                    rl.end_drawing()
                    self.flickering_timer = 0

        if self.frequency > 0:
            rl.clear_background(rl.BLACK)

        rl.set_window_title(f"Contrast: {self.contrast}, Frequency: {self.frequency}")

    def event_handler(self):
        if rl.window_should_close():
            self.close()
        elif rl.is_key_pressed(rl.KEY_F11):
            rl.toggle_fullscreen()

    @staticmethod
    def close():
        rl.close_window()
