import plotly.graph_objs as go
import plotly.express as px
import imageio.v2 as iio
import streamlit as st
import pandas as pd
import numpy as np
import ast
import os


def log_q_table(table, last_obs, cur_obs, n_bins):
    heatmap = px.imshow(table)

    # plot location of last action;
    if not (cur_obs is None):
        heatmap.add_trace(go.Scatter(
            x=np.clip([last_obs[1] * n_bins], a_min=0, a_max=n_bins - 1),
            y=np.clip([last_obs[0] * n_bins], a_min=0, a_max=n_bins - 1),
            mode='markers',
            marker=dict(color="red", size=15),
            showlegend=False,
            name="last action"
        ))

        heatmap.add_trace(go.Scatter(
            x=np.clip([cur_obs[1] * n_bins], a_min=0, a_max=n_bins - 1),
            y=np.clip([cur_obs[0] * n_bins], a_min=0, a_max=n_bins - 1),
            mode='markers',
            marker=dict(color="green", size=15),
            showlegend=False,
            name="current action"
        ))

    return heatmap


def check_empty_data(json_array):
    if isinstance(json_array, str):
        arr = np.array(ast.literal_eval(json_array)) if json_array else None
    else:
        arr = None
    return arr


def check_integrity(filesize, file_name):
    if os.path.isfile(file_name):
        current_size = os.stat(file_name).st_size
        if current_size == filesize:
            return True, filesize
        else:
            filesize = current_size
            return False, filesize
    else:
        return False, -1


st.set_page_config(
    page_title="Real-Time fMRI Dashboard",
    page_icon=":brain:",
    layout="wide"
)

# set-up main page;
st.markdown(
    "<h1 style='text-align: center; color: grey;'>Real-Time fMRI Dashboard</h1><br><br>",
    unsafe_allow_html=True
)

log_path = "./log/log.json"
placeholder = st.empty()
log_size = -1
ref_size = -1
vol_size = -1

# Dashboard MainLoop;
while True:
    with placeholder.container():
        try:  # check for integrity and load json log;
            integrity, log_size = check_integrity(log_size, log_path)
            if not integrity:
                st.markdown("Waiting for log file to be created...")
                continue
            else:
                data = pd.read_json(log_path)

            # load imaging and RL related data;
            contrast = data.loc[data.index.max(), "contrast"]
            frequency = data.loc[data.index.max(), "frequency"]
            resting_state = data.loc[data.index.max(), "resting_state"]
            epoch = data.loc[data.index.max(), "epoch"]
            q_table = data.loc[data.index.max(), "q_table"]
            fmridata = data.loc[data.index.max(), "fmri_data"]
            last_action = data.loc[data.index.max(), "last action"]
            current_action = data.loc[data.index.max(), "current action"]
            convergence = data.loc[data.index.max(), "convergence"]
            reward = data.loc[data.index.max(), "reward"]
            mc_ratio = data.loc[data.index.max(), "current_motion"]
            mc_max_ratio = data.loc[data.index.max(), "motion_max_ratio"]

            # Load motion parameters;
            rot_x = data.loc[data.index.max(), "rotation x"]
            rot_y = data.loc[data.index.max(), "rotation y"]
            rot_z = data.loc[data.index.max(), "rotation z"]
            trs_x = data.loc[data.index.max(), "translation x"]
            trs_y = data.loc[data.index.max(), "translation y"]
            trs_z = data.loc[data.index.max(), "translation z"]

            # the hypothesis function is found at the first index;
            hrf = data.loc[0, "hrf"]

        except (ValueError, KeyError) as e:
            continue

        # prepare data for visualization;
        hrf = check_empty_data(hrf)
        fmridata = check_empty_data(fmridata)
        q_table = check_empty_data(q_table)
        convergence = check_empty_data(convergence)
        reward = check_empty_data(reward)
        mc_ratio = check_empty_data(mc_ratio)
        last_action = check_empty_data(last_action)
        current_action = check_empty_data(current_action)

        rot_x = check_empty_data(rot_x)
        rot_y = check_empty_data(rot_y)
        rot_z = check_empty_data(rot_z)
        trs_x = check_empty_data(trs_x)
        trs_y = check_empty_data(trs_y)
        trs_z = check_empty_data(trs_z)

        # check for integrity and load reference volume image;
        ref_vol = "./log/reference.png"
        ref_integrity, ref_size = check_integrity(ref_size, ref_vol)
        if ref_integrity:
            ref_vol = iio.imread("./log/reference.png")
            ref_vol = np.asarray(ref_vol)
        else:
            ref_vol = None

        # check for integrity and load current volume image;
        cur_vol = "./log/volume.png"
        vol_integrity, vol_size = check_integrity(vol_size, cur_vol)
        if vol_integrity:
            cur_vol = iio.imread("./log/volume.png")
            cur_vol = np.asarray(cur_vol)
        else:
            cur_vol = None

        # set columns for the top of the dashboard;
        left_column, middle_left_column, middle_right_column, right_column = st.columns((2, 2, 3, 1))

        # show parameters;
        with right_column:
            st.markdown("##### Parameters")
            st.markdown(
                f"Contrast: {contrast} <br> Frequency: {frequency} <br> Resting State: {resting_state} <br> Epoch: {epoch}",
                unsafe_allow_html=True
            )

        # Show registration plots and q-table;
        with left_column:
            st.markdown("##### Reference Volume")
            if ref_vol is not None:
                try:
                    st.image(ref_vol, use_column_width=True)
                except (AttributeError, OSError) as e:
                    pass
            else:
                st.markdown("Waiting for reference volume.")

        with middle_left_column:
            st.markdown("##### Current Volume")
            if ref_vol is not None:
                try:
                    st.image(cur_vol, use_column_width=True)
                except (AttributeError, OSError) as e:
                    pass
            else:
                st.markdown("Waiting for reference volume.")

        with middle_right_column:
            st.markdown("##### Q-table")
            q_table = log_q_table(q_table, last_action, current_action, 10)
            st.plotly_chart(q_table, use_container_width=True)

        # Plot Real-Time Data;
        with st.container():
            st.markdown("##### Real-Time Data")
            if fmridata is None:
                fmridata = np.zeros(len(hrf))
            if epoch > 1:
                mridata = {"Hypothesis Function": hrf, "fMRI data": fmridata}
            else:
                mridata = {"Hypothesis Function": hrf}
            mridata = px.line(mridata)
            st.plotly_chart(mridata, use_container_width=True)
            st.markdown("----")

        # set columns for the bottom of the dashboard;
        bottom_left_column, bottom_right_column = st.columns((1, 1))

        # plot reward, convergence and motion;
        with bottom_left_column:
            st.markdown("##### Reward Chart")
            # create motion threshold line;
            plot_len = len(reward) if len(reward) > len(hrf) else len(hrf)
            max_ratio_thr = np.ones(plot_len) * mc_max_ratio

            # plot reward and current motion ratio;
            reward_update = {"Reward": reward, "motion ratio": mc_ratio}
            reward_chart = px.line(reward_update)
            reward_chart.add_trace(go.Scatter(y=max_ratio_thr, mode="lines", name="max motion ratio"))
            reward_chart.update_layout(showlegend=True)
            st.plotly_chart(reward_chart)

            st.markdown("##### Motion (rotations)")
            if isinstance(rot_x, np.ndarray):
                rotations = {"Rotation x": rot_x, "Rotation y": rot_y, "Rotation z": rot_z}
                motion_rot = px.line(rotations)
                st.plotly_chart(motion_rot)
            else:
                st.markdown("Waiting for motion correction output...")

        with bottom_right_column:
            st.markdown("##### Convergence")
            convergence = px.line(convergence)
            convergence.update_layout(showlegend=False)
            st.plotly_chart(convergence)

            st.markdown("##### Motion (translations)")
            if isinstance(trs_x, np.ndarray):
                translations = {"Translation x": trs_x, "Translation y": trs_y, "Translation z": trs_z}
                motion_trs = px.line(translations)
                st.plotly_chart(motion_trs)
            else:
                st.markdown("Waiting for motion correction output...")
