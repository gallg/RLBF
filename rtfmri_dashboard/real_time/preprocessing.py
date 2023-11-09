from rtfmri_dashboard.real_time.utils import dcm_to_array
from nilearn.glm.first_level import compute_regressor
from shutil import copyfile
from posixpath import join

import statsmodels.api as sm
import nibabel as nib
import numpy as np
import pickle
import pprint
import ants


def plot_image(image, mask, reorient=False, filename=None):
    ants.viz.plot_ortho_stack([image],
                              [mask],
                              filename=filename,
                              reorient=reorient)


def draw_roi(volume, output_dir, radius=10):
    affine = np.eye(4)

    volume = reorient_volume(volume, affine, to_ants=True)
    volume.to_filename(join(output_dir, "check_reference.nii.gz"))

    msg = input("x y z:  ")
    x = int(msg.split(" ")[0])
    y = int(msg.split(" ")[1])
    z = int(msg.split(" ")[2])

    img_shape = volume.shape
    mask = np.zeros(img_shape)

    for x1 in range(img_shape[0]):
        for y1 in range(img_shape[1]):
            for z1 in range(img_shape[2]):
                dist = np.sqrt((x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2)
                if dist <= radius:
                    mask[x1, y1, z1] = 1

    # convert mask to ants;
    mask = ants.from_nibabel(nib.Nifti1Image(mask, affine))
    mask.to_filename(join(output_dir, "custom_mask.nii.gz"))

    return volume, mask, affine


def get_image(image, affine=True, to_ants=False):
    if isinstance(image, str):
        image = nib.load(image)

    if affine:
        affine = image.affine

    if to_ants:
        return ants.from_nibabel(image), affine
    else:
        return image, affine


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


def run_preprocessing(volume, template, affine, transformation=None, transform_type="SyNBold", preprocessing=False):
    volume = reorient_volume(volume, affine, to_ants=True)

    # calculate transformation matrix if it is not available;
    if preprocessing and transformation is None:
        transformation = ants_registration(
            volume,
            template,
            transform_type=transform_type
        )

    if transformation is not None:
        volume = ants_transform(volume, template, transformation)

    return volume, transformation


def select_preprocessing(
        first_vol,
        template,
        mask,
        affine,
        transform_matrix,
        scan_dir,
        output_dir):

    while True:
        volume = dcm_to_array(join(scan_dir, first_vol))
        msg = input("write '1' for co-registration, '2' for manual ROI definition: ")

        if msg == "1":
            print("re-running the registration, please select the registration strategy:")
            prompt = input(
                "choose 'default' (SynBold) or another one from the documentation:"
                "https://antspyx.readthedocs.io/en/latest/registration.html "
            )

            # ToDo: add error handling for incorrect prompts;
            if prompt == "default":
                prompt = "SynBold"

            volume, transformation = run_preprocessing(
                volume,
                template,
                affine,
                transform_type=prompt,
                preprocessing=True
            )
            copyfile(transformation, transform_matrix)

        elif msg == "2":
            print("defining the ROI manually, please ROI center coordinate!")
            volume, mask, affine = draw_roi(volume, output_dir)
            transform_matrix = None
            template = None

        # plot reference image for visual inspection;
        plot_image(
            volume,
            mask,
            reorient=False,
            filename=join(output_dir, "reference.png")
        )

        # decide whether to end preprocessing or not;
        prompt = input("end preprocessing? [yes/no]")
        if prompt == "yes":
            break
        elif prompt == "no":
            continue
        else:
            print("invalid input!")
            continue

    return template, mask, affine, transform_matrix


def save_preprocessed_data(data, preprocessed_file):
    with open(preprocessed_file, "wb") as f:
        pickle.dump(data, f)


def load_preprocessed_data(preprocessed_file):
    with open(preprocessed_file, "rb") as f:
        labels = ["first_vol", "template", "mask", "affine", "transformation"]
        data = pickle.load(f)
        preprocessed = {}

        for idx, label in enumerate(labels):
            preprocessed[label] = data[idx]

        print("please check preprocessed data:")
        pprint.pprint(preprocessed, indent=4)
        input("press a key to continue or restart the program if the data is incorrect!")

        # load data;
        if preprocessed["template"] is not None:
            template = ants.image_read(preprocessed[labels[1]])
        else:
            template = preprocessed[labels[1]]

        first_vol = preprocessed[labels[0]]
        mask = ants.image_read(preprocessed[labels[2]])
        affine = preprocessed[labels[3]]
        transformation = preprocessed[labels[4]]

    return first_vol, template, mask, affine, transformation


def generate_hrf_regressor(time_length, duration, onset, amplitude, tr=1.0):

    frame_times = np.linspace(0, time_length * tr, time_length)
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)

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
