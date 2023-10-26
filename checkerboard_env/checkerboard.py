from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pygame


class CheckerBoardEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode, checkerboard, cross):
        self.screen_width = 600
        self.screen_height = 600
        self.background = (0, 0, 0)
        self.board_size = 350
        self.cross_size = 30
        self.screen_size = (self.screen_width, self.screen_height)
        self.screen = None

        self.checkerboard = checkerboard
        self.cross = cross

        # initialize contrast and frequency spaces;
        self.contrast = 0
        self.frequency = 0

        self.contrast_low = 0
        self.contrast_high = 1.0
        self.frequency_low = 0
        self.frequency_high = 1.0

        self.action_space = spaces.Box(low=np.array([self.contrast_low, self.frequency_low]),
                                       high=np.array([self.contrast_high, self.frequency_high]),
                                       dtype=np.float64)

        self.observation_space = spaces.Box(low=np.array([self.contrast_low, self.frequency_low]),
                                            high=np.array([self.contrast_high, self.frequency_high]),
                                            dtype=np.float64)

        # render options;
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.fps_controller = pygame.time.Clock()
        self.board_render = None
        self.render_mode = render_mode
        self.invert = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.contrast = np.random.random()
        self.frequency = np.random.random()

        observation = np.array([self.contrast, self.frequency], dtype=np.float64)
        info = {}

        return observation, info

    def step(self, action):
        self.contrast, self.frequency = action
        observation = np.array([self.contrast, self.frequency], dtype=np.float64)

        # update contrast;
        if self.screen is not None:
            self.board_render = self.checkerboard.copy()
            brightness = self.contrast * 255
            self.board_render.fill((brightness, brightness, brightness, 128), special_flags=pygame.BLEND_RGBA_MULT)

        # reward is calculated externally;
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        # initialize screen if it has not been created yet;
        if not self.screen:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
            pygame.display.set_caption("CheckerBoardEnv")

            # load checkerboard assets;
            self.checkerboard = pygame.image.load(self.checkerboard).convert_alpha()
            self.cross = pygame.image.load(self.cross).convert_alpha()

            self.checkerboard = pygame.transform.scale(self.checkerboard, (self.board_size, self.board_size))
            self.cross = pygame.transform.scale(self.cross, (self.cross_size, self.cross_size))

            # create a copy that gets modified and rendered;
            self.board_render = self.checkerboard.copy()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # update screen size in case of resizing;
        board_x, board_y, cross_x, cross_y = self.resize()

        # clear the screen;
        self.screen.fill(self.background)

        # raw the images
        self.screen.blit(self.cross, (cross_x, cross_y))

        if self.invert:
            self.screen.blit(pygame.transform.flip(self.board_render, True, False), (board_x, board_y))
        else:
            self.screen.blit(self.board_render, (board_x, board_y))

        # update the display;
        pygame.display.update()

        # invert the flicker state;
        self.invert = not self.invert

        # Control the frame rate / frequency;
        self.fps_controller.tick(self.frequency * 30)

    def resize(self):
        board_x = (self.screen.get_size()[0] - self.checkerboard.get_size()[0]) / 2
        board_y = (self.screen.get_size()[1] - self.checkerboard.get_size()[1]) / 2
        cross_x = (self.screen.get_size()[0] - self.cross.get_size()[0]) / 2
        cross_y = (self.screen.get_size()[1] - self.cross.get_size()[1]) / 2
        return board_x, board_y, cross_x, cross_y

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
