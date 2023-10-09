from functools import partial
import pydicom
import hashlib


def check_file_integrity(dcmfile):
    with open(dcmfile, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


def dcm_to_array(dcmfname):
    dcmfile = pydicom.dcmread(dcmfname)
    return dcmfile.pixel_array
