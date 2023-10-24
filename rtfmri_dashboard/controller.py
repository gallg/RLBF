from real_time.utils import check_file_integrity, dcm_to_array
from real_time.utils import plot_image, scan_dicom_folder
from real_time.workflow import RealTimeEnv, run_preprocessing
from real_time.preprocessing import get_image
from shutil import copyfile
from posixpath import join
import json
import sys
import os


def reset_log(log_path):
    json_data = []
    with open(log_path, "w") as json_file:
        json_file.seek(0)
        json.dump(json_data, json_file)


def initialize_realtime(env, volume, template, mask, output_dir):
    real_time_env = env(render_mode="human")
    template, affine = get_image(template, affine=True, to_ants=True)
    mask, _ = get_image(mask, affine=False, to_ants=True)
    transform_matrix = join(output_dir, "fwdtransforms.mat")

    if not os.path.isfile(transform_matrix):
        print("No transformation matrix found, preprocessing reference volume!")
        volume, transformation = run_preprocessing(
            volume,
            template,
            affine,
        )
        copyfile(transformation, transform_matrix)

        # plot reference image;
        plot_image(
            volume,
            mask,
            reorient=False,
            filename=join(output_dir, "reference.png")
        )
        print("reference has been preprocessed!")

    return real_time_env, template, affine, mask, transform_matrix


def run_acquisition(scan_dir, template, mask, preprocessing):
    # When it becomes True we can pass to the second volume;
    first_volume_has_been_processed = False

    # Initialize variables for rt-scanning;
    first_vol, current_series = scan_dicom_folder(scan_dir)
    f_queue = [first_vol]
    processed = []

    # Initialize variables to check for file integrity;
    f_hash = ""
    f_size = -1

    output_dir = "../log/"
    real_time_env = None
    transformation = None
    affine = None
    volume = None

    # ToDo: check data & plots are correct;
    # ToDo: add stop function to the implementation;
    # ToDo: fix environment rendering;

    while True:
        if first_volume_has_been_processed:
            current_files = os.listdir(scan_dir)

            # Update queue;
            f_queue = sorted([dcm for dcm in current_files
                             if dcm not in processed])

        if len(f_queue) > 0:
            dcmfile = join(scan_dir, f_queue[0])

            # Check for file integrity;
            current_f_hash = check_file_integrity(dcmfile)
            current_f_size = os.stat(dcmfile).st_size

            if (current_f_hash != f_hash) or (current_f_size != f_size):
                f_hash = current_f_hash
                f_size = current_f_size
            else:
                volume = dcm_to_array(dcmfile)

        # REAL-TIME PROCESSING;
        if volume is not None:
            if not real_time_env:
                real_time_env, template, affine, mask, transformation = initialize_realtime(
                    RealTimeEnv,
                    volume,
                    template,
                    mask,
                    output_dir
                )

            # Mark volume as processed;
            processed.append(f_queue[0])
            first_volume_has_been_processed = True

        if first_volume_has_been_processed and not preprocessing:
            real_time_env.run_realtime(volume, template, mask, affine, transformation)

        # if preprocessing is true the first volume data is not collected;
        if preprocessing and volume is not None:
            preprocessing = False

        # reset volume;
        volume = None


if __name__ == "__main__":

    # reset log file;
    reset_log("../log/log.json")

    try:
        run_acquisition(
            "/home/giuseppe/PNI/Bkup/Projects/rtfmri_dashboard/data_in/scandir",
            "/home/giuseppe/PNI/Bkup/Projects/rtfMRI-controller/data_in/standard/MNI152_T1_2mm_brain.nii.gz",
            "/home/giuseppe/PNI/Bkup/Projects/rtfMRI-controller/data_in/standard/BA17_mask.nii.gz",
            preprocessing=False
        )

    except KeyboardInterrupt:
        msg = input("Restart? [y/n]: ")
        if msg == "y":
            os.execv(sys.executable, ['python'] + [sys.argv[0]])
        elif msg == "n":
            sys.exit()
