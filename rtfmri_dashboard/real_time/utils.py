from functools import partial
from ast import literal_eval
from posixpath import join
import numpy as np
import threading
import pydicom
import hashlib
import shutil
import time
import json
import re
import os


class StateManager:
    def __init__(self, filename):
        self.file = open(filename, "a+")
        self.lock = threading.Lock()

    def write_state(self, state):
        with self.lock:
            self.file.seek(0)
            self.file.truncate()
            self.file.write(state)
            self.file.flush()

    def read_state(self):
        with self.lock:
            self.file.seek(0)
            data = self.file.read()
            if data in ["", []]:
                return None
            else:
                return np.array(literal_eval(data), dtype=float)

    def close(self):
        if self.file.closed:
            self.file.close()


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
            print("New series found: {0}".format(new_series))
            break
        else:
            continue

    return new_file, new_series


def reset_log(log_path):
    json_data = []
    with open(log_path, "w") as json_file:
        json_file.seek(0)
        json.dump(json_data, json_file)


def pad_array(array, reference):
    if len(reference) > 0:
        padding_size = abs(len(reference[0]) - len(array))
        array = np.pad(array, (0, padding_size))
    return array


def backup_data(scan_dir, log_dir, out_dir):
    prompt = input("Save acquired data? [yes/no]")
    if prompt == "yes" or prompt == "y":
        current_time = time.strftime("%Y%m%d-%H%M%S")
        print("Wait while data is being saved...")
        shutil.make_archive(join(out_dir, f"run_{current_time}"), 'zip', scan_dir)
        shutil.make_archive(join(out_dir, f"log_{current_time}"), 'zip', log_dir)
        print("Data saved!")
    else:
        print("Data has not been saved!")


def clean_temporary_data():
    for f_name in os.listdir("/mnt/fmritemp"):
        if f_name == "reference.nii.gz":
            continue
        if f_name.endswith((".nii.gz", ".nii.gz.par")):
            os.remove("/mnt/fmritemp/" + f_name)
        else:
            continue
