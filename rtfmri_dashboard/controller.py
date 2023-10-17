from real_time.utils import check_file_integrity, dcm_to_array, plot_image
from real_time.workflow import RealTimeEnv, run_preprocessing
from real_time.preprocessing import get_image
from shutil import copyfile
from posixpath import join

import sys
import os
import re

# Regex patterns to handle dicom filenames;
dcm_pattern = re.compile(r'\d{3}_\d{6}_\d{6}_[^_]+.dcm')
volume_number = re.compile(r'(?<=\d{3}_\d{6}_)[^_]+(?=_[^_]+\.dcm)')
series_number = re.compile('(?<=_)[^_]+')


def get_series_name(filenames):
    series = []
    all_vols = [f for f in filenames if dcm_pattern.match(f)]

    if len(all_vols) > 0:
        all_series = []
        for f in all_vols:
            all_series.append(series_number.search(f).group())
        series = set(all_series)

    return series


def scan_dicom_folder(folder_path):
    f_list = os.listdir(folder_path)
    new_file = None
    new_series = None

    if len(f_list) > 0:
        f_series = get_series_name(f_list)
        print("Available Series: {0}".format(f_series))
    else:
        f_series = []
        print("Waiting for new series to appear")

    while True:  # Search for new series;
        current_files = os.listdir(folder_path)
        current_series = get_series_name(current_files)

        if current_series != f_series:
            new_file = next(iter(set(current_files) - set(f_list)))
            new_series = next(iter(set(current_series) - set(f_series)))
            print("New series found: {0}, Starting acquisition.".format(new_series))
            break
        else:
            continue

    return new_file, new_series


def initialize_realtime(env, volume, template, mask, output_dir):
    real_time_env = env(render_mode="human")
    template, affine = get_image(template, affine=True, to_ants=True)
    mask, _ = get_image(mask, affine=False, to_ants=True)
    transform_matrix = join(output_dir, "fwdtransforms.mat")

    if not os.path.isfile(transform_matrix):
        volume, transformation = run_preprocessing(
            volume,
            template,
            affine,
        )
        copyfile(transformation, transform_matrix)

        # plot reference image;
        plot_image(volume, mask, join(output_dir, "reference.png"))

    return real_time_env, template, affine, mask, transform_matrix


def run_acquisition(scan_dir, template, mask):
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

    while True:
        if first_volume_has_been_processed:
            current_files = os.listdir(scan_dir)

            # Update queue;
            f_queue = [dcm for dcm in current_files
                       if current_series in dcm
                       and dcm not in processed]

        if len(f_queue) > 0:
            dcmfile = join(scan_dir, f_queue[0])  # ToDo: test the mainloop;

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

            # Mark volume as processed
            processed.append(f_queue[0])
            first_volume_has_been_processed = True

        if first_volume_has_been_processed:
            real_time_env.run_realtime(volume, template, mask, affine, transformation)

        # reset volume;
        volume = None


if __name__ == "__main__":
    try:
        run_acquisition(
            "scanner directory",
            "",
            ""
        )

    except KeyboardInterrupt:
        msg = input("Restart? [y/n]: ")
        if msg == "y":
            os.execv(sys.executable, ['python'] + [sys.argv[0]])
        elif msg == "n":
            sys.exit()
