from gymnasium import spaces
from PIL import ImageEnhance

import gymnasium as gym
import numpy as np
import pygame


class CheckerBoardEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, checkerboard=None, inverse=None, cross=None, snr=None):
        self.screen_width = 600
        self.screen_height = 600
        self.background = (93, 93, 93)
        self.checkerboard_size = 350
        self.cross_size = 30

        self.checkerboard = checkerboard
        self.inverse = inverse
        self.cross = cross
        self.screen = None

        self.contrast = 0
        self.frequency = 0

        self.contrast_low = 0
        self.contrast_high = 1.0
        self.frequency_low = 0
        self.frequency_high = 1.0

        self.activation = 0
        self.contrast_coeff = 1.0
        self.frequency_coeff = np.exp(-((np.arange(0.1, 10.1, 0.1) - 7) ** 2) / 20)  # Highest activation at 7hz;
        self.snr = snr

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.fps_controller = pygame.time.Clock()
        self.render_mode = render_mode
        self.render_steps = 0
        
        self.action_space = spaces.Box(low=np.array([self.contrast_low, self.frequency_low]),
                                       high=np.array([self.contrast_high, self.frequency_high]),
                                       dtype=np.float64)

        self.observation_space = spaces.Box(low=np.array([self.contrast_low, self.frequency_low]),
                                            high=np.array([self.contrast_high, self.frequency_high]),
                                            dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.contrast = np.random.random()
        self.frequency = np.random.random()

        observation = np.array([self.contrast, self.frequency], dtype=np.float64)
        info = {}

        return observation, info

    def set_action(self, contrast, frequency):
        self.contrast = contrast
        self.frequency = frequency

    def reward_handler(self, activation):
        noise = np.random.normal(0, 1/self.snr, size=1)
        reward = activation + noise
        return reward

    def step(self, action):
        
        self.contrast, self.frequency = action

        contrast_activation = self.contrast_coeff * self.contrast
        frequency_activation = self.frequency_coeff[int(self.frequency * 100) - 1]
        self.activation = contrast_activation * frequency_activation

        observation = np.array([self.contrast, self.frequency])
        reward = self.reward_handler(self.activation)[0]

        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_env()
    
    def _render_env(self):

        if self.screen is None:
            self.checkerboard = self.image_to_contrast(self.checkerboard, self.checkerboard_size)
            self.inverse = self.image_to_contrast(self.inverse, self.checkerboard_size)
            self.cross = self.image_to_contrast(self.cross, self.cross_size)

            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), vsync=True)
            pygame.display.set_caption("CheckerBoardEnv")
            
        if self.render_steps == 0:
            board, inverse, cross = self.update_image()

            self.sprites = pygame.sprite.Group()
            self.board_sprite = BoardComponents(board, self.screen.get_rect())
            self.cross_sprite = BoardComponents(cross, self.screen.get_rect())
            self.sprites.add(self.board_sprite)
            self.sprites.add(self.cross_sprite)
            self.render_steps += 1

        elif self.render_steps > 0:
            board, inverse, cross = self.update_image()
            self.board_sprite.image = board
            
        self.board_sprite.flicker(inverse, self.frequency)
        self.screen.fill(self.background)
        self.sprites.draw(self.screen)

        pygame.display.update()
        self.fps_controller.tick(self.metadata["render_fps"])

    def update_image(self):
        board = self.convert_image(self.checkerboard.enhance(self.contrast))
        inverse = self.convert_image(self.inverse.enhance(self.contrast))
        cross = self.convert_image(self.cross.enhance(1.0))
        return board, inverse, cross

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def image_to_contrast(image, size):
        image = image.resize((size, size))
        image = ImageEnhance.Contrast(image)
        return image

    @staticmethod
    def convert_image(image):
        return pygame.image.fromstring(image.tobytes(), image.size, image.mode)


class BoardComponents(pygame.sprite.Sprite):
    def __init__(self, image, screen_rect):
        super().__init__()

        self.image = image
        self.screen = screen_rect
        self.rect = self.image.get_rect(center=self.screen.center)
        self.flicker_timer = 0
        
    def flicker(self, inverse, frequency):
        self.flicker_timer += frequency
        if self.flicker_timer >= 1.0:
            self.image = inverse
            self.rect = self.image.get_rect(center=self.screen.center)
            self.flicker_timer = 0

    @staticmethod
    def invert_colors(image):
        inverted_image = image.copy()
        inverted_image.fill((255, 255, 255, 255), special_flags=pygame.BLEND_RGB_SUB)
        return inverted_image
    