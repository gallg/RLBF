from gymnasium import spaces
from gymnasium import Env
import numpy as np
import pygame


class CheckerBoardEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode, checkerboard, cross):
        self.screen_width = 600
        self.screen_height = 600
        self.background = (0, 0, 0)
        self.board_size = 350
        self.cross_size = 30
        self.screen_size = (self.screen_width, self.screen_height)
        self.screen = None

        self.board_render = None
        self.checkerboard = checkerboard
        self.cross = cross

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

        self.fps_controller = pygame.time.Clock()
        self.render_mode = render_mode
        self.invert = False

    def init_images(self, checkerboard, cross):
        self.checkerboard = pygame.image.load(checkerboard).convert_alpha()
        self.cross = pygame.image.load(cross).convert_alpha()
        self.checkerboard = pygame.transform.scale(self.checkerboard, (self.board_size, self.board_size))
        self.cross = pygame.transform.scale(self.cross, (self.cross_size, self.cross_size))
        self.board_render = self.checkerboard.copy()

    def render(self):
        if not self.screen:
            self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
            pygame.display.set_caption("CheckerBoardEnv")
            self.init_images(self.checkerboard, self.cross)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill(self.background)
        self.draw_images()
        pygame.display.update()

        self.invert = not self.invert
        self.fps_controller.tick(self.frequency * self.metadata["render_fps"])

    def draw_images(self):
        board_x, board_y, cross_x, cross_y = self.resize()

        self.screen.blit(self.cross, (cross_x, cross_y))
        render_board = pygame.transform.flip(self.board_render, True, False) if self.invert else self.board_render
        self.screen.blit(render_board, (board_x, board_y))

    def resize(self):
        screen_width, screen_height = self.screen_size
        board_x = (screen_width - self.board_size) / 2
        board_y = (screen_height - self.board_size) / 2
        cross_x = (screen_width - self.cross_size) / 2
        cross_y = (screen_height - self.cross_size) / 2
        return board_x, board_y, cross_x, cross_y

    def reset(self, seed=None, options=None):
        self.contrast = np.random.random()
        self.frequency = np.random.random()
        observation = np.array([self.contrast, self.frequency], dtype=np.float64)
        info = {}
        return observation, info

    def step(self, action):
        self.contrast, self.frequency = action
        observation = np.array([self.contrast, self.frequency], dtype=np.float64)

        if self.screen is not None:
            self.board_render = self.checkerboard.copy()
            brightness = self.contrast * 255
            self.board_render.fill((brightness, brightness, brightness, 128), special_flags=pygame.BLEND_RGBA_MULT)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
