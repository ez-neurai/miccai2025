import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_nifti(file_path):
    """
    Load a NIfTI image and retrieve its data and affine transformation matrix.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        tuple: A tuple containing the image data (numpy.ndarray) and the affine matrix.
    """
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    return data, affine

def calculate_axis_aligned_box(mask_data, scale_factor=1.0):
    """
    Compute an axis-aligned bounding box from a binary mask and apply an optional scaling factor.

    The function determines the minimal and maximal voxel indices where the mask is non-zero,
    calculates the center and size of the bounding box, scales the box, and ensures that the
    coordinates remain within the volume bounds.

    Parameters:
        mask_data (numpy.ndarray): 3D binary mask array.
        scale_factor (float): Scaling factor for the bounding box dimensions.

    Returns:
        tuple: Two numpy arrays representing the minimum and maximum voxel coordinates of the scaled box.
    """
    non_zero_indices = np.argwhere(mask_data > 0)
    box_min = non_zero_indices.min(axis=0)
    box_max = non_zero_indices.max(axis=0)
    box_center = (box_min + box_max) / 2.0
    box_size = (box_max - box_min) * scale_factor
    scaled_min = box_center - box_size / 2.0
    scaled_max = box_center + box_size / 2.0
    scaled_min = scaled_min.astype(int)
    scaled_max = scaled_max.astype(int)
    vol_shape = np.array(mask_data.shape)
    scaled_min = np.clip(scaled_min, [0, 0, 0], vol_shape - 1)
    scaled_max = np.clip(scaled_max, [0, 0, 0], vol_shape - 1)
    return scaled_min, scaled_max

def extract_rotation_and_scaling(affine_matrix):
    """
    Extract the rotation matrix and scaling factors from the upper-left 3x3 submatrix of an affine transformation.

    This function uses singular value decomposition (SVD) to decouple the rotational component from the scaling factors.

    Parameters:
        affine_matrix (numpy.ndarray): A 4x4 affine transformation matrix.

    Returns:
        tuple: A tuple containing the rotation matrix and an array of scaling factors.
    """
    R_sub = affine_matrix[:3, :3]
    U, S, Vt = np.linalg.svd(R_sub)
    rotation_matrix = np.dot(U, Vt)
    scaling_factors = S
    return rotation_matrix, scaling_factors

def calculate_angles_from_rotation(rotation_matrix, order='xyz'):
    """
    Compute Euler angles from a given rotation matrix using the specified axis order.

    Parameters:
        rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
        order (str): Axis order for Euler angle extraction (default is 'xyz').

    Returns:
        numpy.ndarray: Array of Euler angles in degrees.
    """
    r = R.from_matrix(rotation_matrix)
    angles = r.as_euler(order, degrees=True)
    return angles

def display_segmentation_box(mask_data, corners_voxel, title):
    """
    Visualize orthogonal slices of a 3D segmentation volume with an overlaid rotated bounding box.

    This function extracts sagittal, coronal, and axial slices from the segmentation volume and projects
    the corresponding rotated bounding box (computed separately) onto each slice. The overlaid box facilitates
    the visual assessment of segmentation accuracy and spatial orientation.

    Parameters:
        segmentation_data (numpy.ndarray): 3D segmentation volume.
        rotated_corners_per_slice (dict): Dictionary with keys 'Sagittal', 'Coronal', and 'Axial'
                                          that map to rotated bounding box corner coordinates (numpy.ndarray).
        title (str): Title to be displayed on each subplot.
    """
    sagittal_slice = mask_data[int(mask_data.shape[0] // 2), :, :]
    coronal_slice = mask_data[:, int(mask_data.shape[1] // 2), :]
    axial_slice = mask_data[:, :, int(mask_data.shape[2] // 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slice_corners = {"Sagittal": [0, 1, 3, 2], "Coronal": [2, 3, 7, 6], "Axial": [0, 4, 6, 2]}
    for ax, slice_data, dims, slice_name in zip(
        axes, [sagittal_slice, coronal_slice, axial_slice], [(1, 2), (0, 2), (0, 1)], ["Sagittal", "Coronal", "Axial"]
    ):
        dim1, dim2 = dims
        ax.imshow(slice_data.T, cmap='gray', origin='lower')
        relevant_corners = slice_corners[slice_name]
        projected_corners = corners_voxel[relevant_corners][:, [dim1, dim2]]
        for i in range(len(projected_corners)):
            start = projected_corners[i]
            end = projected_corners[(i + 1) % len(projected_corners)]
            ax.plot([start[0], end[0]], [start[1], end[1]], color='r', lw=2)
        ax.set_title(f"{slice_name} Slice - {title}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define file paths for input data and JSON output
    base_path = "/niix_Data_fMRS_local_struct"
    json_path = "/output_matrix.json"
    output_json_path = "/obb_box_information.json"

    # Load affine transformation matrices from JSON
    with open(json_path, 'r') as f:
        affine_data = json.load(f)

    subject_ids = list(affine_data.keys())
    result_data = {}

    # Mapping for the indices of the bounding box corners for each slice view
    slice_corners = {
        "Sagittal": [0, 1, 3, 2],
        "Coronal": [2, 3, 7, 6],
        "Axial": [0, 4, 6, 2],
    }

    for subject_id in subject_ids:
        # Parse subject identifier
        subject_id_clean = subject_id.split("id_")[1]
        aligned_path = os.path.join(base_path, subject_id_clean, "T1_MPRAGE_ADNI_0002", "c1align.nii")
        original_path = os.path.join(base_path, subject_id_clean, "T1_MPRAGE_ADNI_0002", "origin.nii")
        print({subject_id_clean})

        # Validate the existence of required files
        if not os.path.exists(aligned_path) or not os.path.exists(original_path):
            print(f"Missing files for subject {subject_id_clean}, skipping...")
            continue

        # Load NIfTI data for aligned and original images
        aligned_data, aligned_affine = load_nifti(aligned_path)
        original_data, original_affine = load_nifti(original_path)

        # Calculate the axis-aligned bounding box for the aligned data
        box_min, box_max = calculate_axis_aligned_box(aligned_data)
        corners_voxel_aligned = np.array([
            [box_min[0], box_min[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_max[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_max[0], box_max[1], box_max[2]],
        ])

        # Extract rotation matrix and calculate Euler angles from the affine transformation data
        affine_matrix = np.array(affine_data[subject_id])
        rotation_matrix, _ = extract_rotation_and_scaling(affine_matrix)
        angles = calculate_angles_from_rotation(rotation_matrix)

        # Rotate bounding box corners for each slice view
        rotated_corners_per_slice = {}
        for slice_name, corners in slice_corners.items():
            slice_center = np.mean(corners_voxel_aligned[corners], axis=0)
            rotated_corners = np.dot(rotation_matrix, (corners_voxel_aligned - slice_center).T).T + slice_center
            rotated_corners_per_slice[slice_name] = rotated_corners

        # Aggregate computed parameters for each subject
        result_data[subject_id_clean] = {
            "center": ((box_min + box_max) / 2.0).tolist(),
            "size": (box_max - box_min).tolist(),
            "angles": angles.tolist()
        }

        # Visualize the original image with the rotated bounding box overlaid on three orthogonal slices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        slice_names = ["Sagittal", "Coronal", "Axial"]

        for ax, slice_name in zip(axes, slice_names):
            slice_data = original_data
            rotated_corners = rotated_corners_per_slice[slice_name]

            if slice_name == "Sagittal":
                slice_image = slice_data[int(slice_data.shape[0] // 2), :, :]
                dim1, dim2 = 1, 2  # Display YZ-plane
            elif slice_name == "Coronal":
                slice_image = slice_data[:, int(slice_data.shape[1] // 2), :]
                dim1, dim2 = 0, 2  # Display XZ-plane
            elif slice_name == "Axial":
                slice_image = slice_data[:, :, int(slice_data.shape[2] // 2)]
                dim1, dim2 = 0, 1  # Display XY-plane

            ax.set_facecolor('black')
            ax.imshow(slice_image.T, cmap="gray", origin="lower")
            relevant_corners = slice_corners[slice_name]
            projected_corners = rotated_corners[relevant_corners][:, [dim1, dim2]]
            for i in range(len(projected_corners)):
                start = projected_corners[i]
                end = projected_corners[(i + 1) % len(projected_corners)]
                ax.plot([start[0], end[0]], [start[1], end[1]], color='r', lw=2)
            ax.set_title(f"{slice_name} Slice")

        plt.tight_layout()
        plt.show()

        # display_segmentation_box(aligned_data, corners_voxel_aligned, "Aligned Data")

    # Write the computed subject parameters to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(result_data, json_file, indent=4)

    print(f"Results saved to {output_json_path}")


        

