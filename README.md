# Dashboard for Reinforcement Learning in Real-Time fMRI 

---
<b> ** Tested with Siemens Magnetom Vida 3T **</b>

### Features
The current version of the program consists in:

- A controller that manages incoming volumes from Siemens' real-time export function.
It handles preprocessing and reinforcement learning with a minimum TR of 1 sec.
- A custom RL environment made in [Raylib 5.0](https://www.raylib.com/) with a flickering
checkerboard that changes in contrast and frequency.
- A custom RL Soft-Q-Learning algorithm based on the work of 
[Haarnoja et al., 2017](https://proceedings.mlr.press/v70/haarnoja17a.html?ref=https://githubhelp.com).
- A Dashboard made in [Streamlit](https://streamlit.io/), to visualize the progress of real-time processing.


### Dependencies
The program has been tested in Ubuntu 20.04.

- Developed using Python 3.11.7.
- Preprocessing strongly depends on [ANTsPy](https://antspyx.readthedocs.io/en/latest/),
except for motion correction that is done using [FSL mcflirt](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MCFLIRT).
- The environment runs using the [python version](https://electronstudio.github.io/raylib-python-cffi/README.html#installation) 
of Raylib.
- The Dashboard runs on Streamlit 1.30.0.


### Run the program

Run the controller script to start the main program. The environment will spawn by itself
after the reference volume has been preprocessed.
```
python ./rtfmri_dashboard/controller.py
```

Run the Dashboard. The Dashboard is only for visualization purposes and doesn't need to be
run for the controller to work properly.
```
streamlit run ./rtfmri_dashboard/real_time/dashboard.py
```