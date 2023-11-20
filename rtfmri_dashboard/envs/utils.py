from rtfmri_dashboard.envs.checkerboard import CheckerBoardEnv


def render_env(output_dir):
    environment = CheckerBoardEnv(
        board="../rtfmri_dashboard/envs/assets/checkerboard.png",
        cross="../rtfmri_dashboard/envs/assets/cross.png",
        render_mode="human"
    )

    while True:
        environment.step(output_dir)
        environment.render()
