from real_time.utils import check_file_integrity, dcm_to_array, reset_log
from real_time.preprocessing import save_preprocessed_data, load_preprocessed_data
from real_time.preprocessing import get_image, select_preprocessing
from real_time.utils import scan_dicom_folder
from real_time.workflow import RealTimeEnv
from envs.utils import render_env
from posixpath import join

import threading
import time
import sys
import os


def initialize_realtime(environment, standard, roi_mask, nuisance_mask, scan_dir, out_dir):
    # save template and mask paths:
    path_to_template = standard
    path_to_mask = roi_mask

    # initialize variables for preprocessing;
    real_time_env = environment()
    standard, affine_matrix = get_image(standard, affine=True, to_ants=True)
    roi_mask, _ = get_image(roi_mask, affine=False, to_ants=True)
    transform_matrix = join(out_dir, "fwdtransforms.mat")
    preprocessed_data = join(out_dir, "preprocessed.pkl")
    custom_mask = join(out_dir, "custom_mask.nii.gz")

    if nuisance_mask is not None:
        nuisance_mask, _ = get_image(nuisance_mask, affine=False, to_ants=True)

    prompt = input("run preprocessing? [yes/no] ")
    if prompt == "yes":
        first_vol, _ = scan_dicom_folder(scan_dir)
        time.sleep(1)  # make sure the volume has been fully written on disk;

        print("preprocessing reference volume!")
        standard, roi_mask, affine_matrix, transform_matrix = select_preprocessing(
            first_vol,
            standard,
            roi_mask,
            affine_matrix,
            transform_matrix,
            scan_dir,
            out_dir
        )

        if standard is None:
            path_to_template = None
            path_to_mask = custom_mask

        save_preprocessed_data(
            (first_vol, path_to_template, path_to_mask, affine_matrix, transform_matrix),
            preprocessed_data
        )

    else:
        if not os.path.isfile(preprocessed_data):
            raise Exception("No preprocessed data, please run preprocessing")
        else:
            print("preprocessed data found, using old preprocessing data!")
            first_vol, standard, roi_mask, affine_matrix, transform_matrix = load_preprocessed_data(preprocessed_data)

    return real_time_env, first_vol, standard, roi_mask, nuisance_mask, affine_matrix, transform_matrix


def run_acquisition(
        scan_dir,
        first_vol_path,
        real_time_env,
        standard,
        roi_mask,
        nuisance_mask,
        affine_matrix,
        transformation_matrix
):
    print("**Starting rt-fmri acquisition**")
    processed = [first_vol_path]
    current_volume = None

    # initialize variables to check for file integrity;
    f_hash = ""
    f_size = -1

    while True:
        current_files = os.listdir(scan_dir)

        # update queue;
        f_queue = sorted([dcm for dcm in current_files
                          if dcm not in processed])

        if len(f_queue) > 0:
            dcm_file = join(scan_dir, f_queue[0])

            # check for file integrity;
            current_f_hash = check_file_integrity(dcm_file)
            current_f_size = os.stat(dcm_file).st_size

            if (current_f_hash != f_hash) or (current_f_size != f_size):
                f_hash = current_f_hash
                f_size = current_f_size
                time.sleep(0.010)
            else:
                current_volume = dcm_to_array(dcm_file)

        # REAL-TIME PROCESSING;
        real_time_env.run_realtime(
            current_volume,
            standard,
            roi_mask,
            affine_matrix,
            transformation_matrix,
            nuisance_mask=nuisance_mask
        )

        if current_volume is not None:
            # Mark volume as processed;
            processed.append(f_queue[0])

        # reset volume;
        current_volume = None


if __name__ == "__main__":
    output_dir = "/home/giuseppe/PNI/Bkup/Projects/rtfmri_dashboard/log"
    scanner_dir = "/home/giuseppe/PNI/Bkup/Projects/rtfmri_dashboard/data_in/scandir"
    reset_log(join(output_dir, "log.json"))

    env, path_to_first_vol, template, mask, noise_mask, affine, transformation = initialize_realtime(
        RealTimeEnv,
        "/home/giuseppe/PNI/Bkup/Projects/rtfMRI-controller/data_in/standard/MNI152_T1_2mm_brain.nii.gz",
        "/home/giuseppe/PNI/Bkup/Projects/rtfMRI-controller/data_in/standard/BA17_mask.nii.gz",
        "/home/giuseppe/PNI/Bkup/Projects/rtfMRI-controller/data_in/standard/CSF_mask_2mm.nii.gz",
        scanner_dir,
        output_dir
    )

    # ToDo: add clean way to close rendering thread;
    # load an instance of the environment for rendering;
    render = threading.Thread(
        target=render_env, args=(
            output_dir,
        )
    )
    render.start()

    try:
        run_acquisition(
            scanner_dir,
            path_to_first_vol,
            env,
            template,
            mask,
            noise_mask,
            affine,
            transformation
        )

    # save acquired data and decide whether to restart or not;
    except KeyboardInterrupt:
        env.stop_realtime()
        msg = input("Restart? [y/n]: ")
        if msg == "y":
            os.execv(sys.executable, ['python'] + [sys.argv[0]])
        elif msg == "n":
            sys.exit()
