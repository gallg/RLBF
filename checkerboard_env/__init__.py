from gymnasium.envs.registration import register

register(
    id='checkerboard-v0',
    entry_point='checkerboard_env.checkerboard:CheckerBoardEnv',
)
