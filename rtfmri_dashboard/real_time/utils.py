from functools import partial
import pydicom
import hashlib
import ants


def check_file_integrity(filename):
    with open(filename, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


def dcm_to_array(filename):
    return pydicom.dcmread(filename).pixel_array


def plot_image(image, mask, filename):
    ants.viz.plot_ortho_stack([image], [mask], filename)
