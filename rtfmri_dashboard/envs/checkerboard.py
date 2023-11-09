from rtfmri_dashboard import config
import numpy as np
import pygame


class CheckerBoardEnv:
    def __init__(self, board, cross):
        self.screen_width = 600
        self.screen_height = 600
        self.board_size = config.board_size
        self.cross_size = config.cross_size

        self.contrast = 0
        self.frequency = 0
        self.flickering_timer = 0
        self.resting_state = True

        pygame.init()
        self.screen_size = (self.screen_width, self.screen_height)
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        self.background = (0, 0, 0)

        self.checkerboard_image = pygame.image.load(board).convert_alpha()
        self.cross_image = pygame.image.load(cross).convert_alpha()
        self.checkerboard = pygame.transform.scale(self.checkerboard_image, (self.board_size, self.board_size))
        self.cross = pygame.transform.scale(self.cross_image, (self.cross_size, self.cross_size))

    def reset(self):
        self.resting_state = True
        observation = (np.random.rand(), np.random.rand())
        self.contrast, self.frequency = observation
        return observation

    def set(self, resting_state, contrast, frequency):
        self.resting_state = resting_state
        self.contrast = contrast
        self.frequency = frequency

    def step(self):
        self.event_handler()
        self.screen.fill(self.background)

        if self.contrast <= 0 or self.resting_state:
            self.screen.blit(self.cross, ((self.screen.get_width() - self.cross.get_width()) / 2,
                                          (self.screen.get_height() - self.cross.get_height()) / 2))
            pygame.display.flip()
            return

        brightness = self.contrast * 255
        self.flickering_timer += self.frequency

        base_checkerboard = self.checkerboard.copy()
        base_checkerboard.fill((brightness, brightness, brightness, 128), special_flags=pygame.BLEND_RGBA_MULT)

        if self.flickering_timer >= 1.0:
            self.flickering_timer = 0
            base_checkerboard = pygame.transform.flip(base_checkerboard, True, False)

        self.screen_size = self.screen.get_size()
        checkerboard_center = (
            (self.screen_size[0] - self.board_size) / 2,
            (self.screen_size[1] - self.board_size) / 2
        )

        cross_center = (
            (self.screen_size[0] - self.cross.get_width()) / 2,
            (self.screen_size[1] - self.cross.get_height()) / 2
        )

        self.screen.blit(base_checkerboard, checkerboard_center)
        self.screen.blit(self.cross, cross_center)

        pygame.display.flip()
        pygame.time.Clock().tick(config.fps)

    def event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.VIDEORESIZE:
                new_size = event.dict['size']
                self.screen = pygame.display.set_mode(new_size, pygame.FULLSCREEN)

    @staticmethod
    def close():
        pygame.display.quit()
        pygame.quit()
