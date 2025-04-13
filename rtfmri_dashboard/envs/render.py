from rtfmri_dashboard.envs.checkerboard import CheckerBoardEnv
import pyray as rl


def render_env():
    env = CheckerBoardEnv(
        "../rtfmri_dashboard/data_in/scandir",
        board="../rtfmri_dashboard/envs/assets/checkerboard.png",
        cross="../rtfmri_dashboard/envs/assets/cross.png",
        render_mode="human"
    )

    while not rl.window_should_close():
        env.step()
        env.render()

    env.close()


if __name__ == "__main__":
    render_env()
