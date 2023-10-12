from nilearn.glm.first_level import compute_regressor
import statsmodels.api as sm
import nibabel as nib
import numpy as np
import ants


def get_template(template, to_ants=False):
    if isinstance(template, str):
        template = nib.load(template)

    affine = template.affine

    if to_ants:
        return ants.from_nibabel(template), affine
    else:
        return template, affine


def reorient_volume(volume, affine, to_ants=False):
    # swap dimensions;
    swap_dim = np.flip(volume, 0)
    swap_dim = np.transpose(swap_dim, (2, 1, 0))
    reoriented = np.flip(swap_dim)

    # save reoriented image;
    reoriented = nib.Nifti1Image(reoriented, affine)
    if to_ants:
        reoriented = ants.from_nibabel(reoriented)

    return reoriented


def ants_registration(moving, fixed, transform_type="SyNBold"):
    transformation = ants.registration(
        fixed,
        moving,
        transform_type=transform_type,
        reg_iterations=(1000, 2000),
        verbose=False
    )
    return transformation["fwdtransforms"][1]


def ants_transform(moving, fixed, transformation):
    registered_img = ants.apply_transforms(
        fixed,
        moving,
        transformation
    )
    return registered_img


def generate_hrf_regressor(time_length, duration, onset, amplitude, tr=1.0):
    # steps = int(time_length / tr)

    frame_times = np.linspace(0, time_length * tr, time_length)
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)

    # Generate boxcar stimulus;
    stim = np.zeros_like(frame_times)
    stim[(frame_times > onset) * (frame_times <= onset + duration)] = amplitude

    # Compute convolved HRF function;
    signal, _ = compute_regressor(
        exp_condition,
        "spm",
        frame_times,
        con_id="block"
    )
    return signal


def run_glm(y, x):
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit().params[-1]
