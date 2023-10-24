from functools import partial
import pydicom
import hashlib
import ants
import re
import os


def check_file_integrity(filename):
    with open(filename, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


def dcm_to_array(filename):
    return pydicom.dcmread(filename, force=True).pixel_array


def get_series_name(filenames):
    # Regex patterns to handle dicom filenames;
    dcm_pattern = re.compile(r'\d{3}_\d{6}_\d{6}_[^_]+.dcm')
    series_number = re.compile('(?<=_)[^_]+')

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


def plot_image(image, mask, reorient=False, filename=None):
    ants.viz.plot_ortho_stack([image],
                              [mask],
                              filename=filename,
                              reorient=reorient)
