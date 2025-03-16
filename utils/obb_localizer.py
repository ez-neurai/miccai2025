import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Path templates
localizer_base_path_template = "/32CH_LOCALIZER_0001/"
mprage_base_path_template = "/T1_MPRAGE_ADNI_0002/"
box_info_path = "/obb_box_information.json"

# Path for saving the updated JSON file
updated_box_info_path = "/obb_box_information_localizer.json"

# List of Subject IDs
subject_ids = [
    '4102_1', '4102_2', '4105_1', '4105_2', '4107', '4108_1', '4108_2',
    '4109', '4112', '4113', '4115', '4116', '4117', '4118', '4125',
    '4128', '4130', '4133', '4134', '4135', '4139', '4140',
    '4141', '4143', '4144', '4146', '4148', '4149', '4150', '4151',
    '4204', '4205', '4206', '4207', '4208', '4209', '4210', '4211_1', '4211_2',
    '4212', '4213'
]

# Load JSON file containing box information
with open(box_info_path, 'r') as f:
    box_information = json.load(f)

# Dictionary to store the final (transformed) center, size, and angles
updated_box_info = {}

def display_localizer_with_center(subject_id):
    # Define the paths for localizer and MPRAGE images using the subject ID
    localizer_folder_path = localizer_base_path_template.format(subject_id)
    mprage_folder_path = mprage_base_path_template.format(subject_id)

    # List localizer files (assumed to be in the order: sagittal, coronal, axial)
    localizer_files = sorted([
        os.path.join(localizer_folder_path, f)
        for f in os.listdir(localizer_folder_path)
        if f.endswith(".nii.gz")
    ])
    if len(localizer_files) < 3:
        print(f"Subject {subject_id}: Fewer than 3 localizer files available.")
        return

    # Use the 'origin.nii' file for MPRAGE
    mprage_file_path = os.path.join(mprage_folder_path, "origin.nii")
    if not os.path.exists(mprage_file_path):
        print(f"Subject {subject_id}: 'origin.nii' file is missing.")
        return

    # Retrieve box information (values in MPRAGE space)
    subject_info = box_information[subject_id]
    # The following values are obtained in MPRAGE space:
    #   - box_center_mprage: center coordinates (mm) for each view (3 values)
    #   - box_dimensions_physical: physical dimensions (mm)
    #   - box_angles: a 3-element vector [sagittal, coronal, axial] (degrees)
    box_center_mprage = np.array(subject_info["center"])
    box_dimensions_physical = np.array(subject_info["size"])
    box_angles = np.array(subject_info["angles"])

    # Load the MPRAGE image and compute its physical center
    mprage_nii = nib.load(mprage_file_path)
    mprage_center_voxel = np.array(mprage_nii.shape) / 2  
    mprage_center_physical = nib.affines.apply_affine(mprage_nii.affine, mprage_center_voxel)

    # For each localizer file, convert to canonical orientation and compute its physical center
    localizer_nii_list = []
    localizer_center_physical_list = []
    localizer_data_list = []
    for fp in localizer_files:
        nii = nib.as_closest_canonical(nib.load(fp))
        localizer_nii_list.append(nii)
        data = nii.get_fdata()
        data = np.squeeze(data)
        localizer_data_list.append(data)
        center_voxel = np.array(nii.shape) / 2
        center_phys = nib.affines.apply_affine(nii.affine, center_voxel)
        localizer_center_physical_list.append(center_phys)

    # Compute the offset and corrected center for each localizer file
    adjusted_centers = []
    for center_phys in localizer_center_physical_list:
        offset = (mprage_center_physical - center_phys)
        # box_offset: the deviation of the box center (in MPRAGE space) from [128, 160, 160] (mm),
        # scaled empirically by a factor of 0.7
        box_offset = (box_center_mprage - np.array([128, 160, 160])) * 0.7  
        adjusted_center = (box_offset + offset) / 0.48828125 + np.array([256, 256, 256])
        adjusted_centers.append(adjusted_center)

    # Determine a unified center that is common across all views.
    # For example, the unified center can be defined as:
    #  - x: x-coordinate from the coronal view (adjusted_centers[1][0])
    #  - y: y-coordinate from the sagittal view (adjusted_centers[0][1])
    #  - z: z-coordinate from the sagittal view (adjusted_centers[0][2]) (assumed to be equal to that from the coronal view)
    unified_center = np.array([adjusted_centers[1][0],
                               adjusted_centers[0][1],
                               adjusted_centers[0][2]])

    # Convert the box dimensions to localizer voxel units
    box_size_in_localizer_vox = box_dimensions_physical * 0.7 / 0.48828125

    # Output results to the console
    print(f"Subject: {subject_id}")
    print(f"  MPRAGE center (physical): {mprage_center_physical}")
    print("  Individual adjusted centers:")
    for i, adj_center in enumerate(adjusted_centers):
        print(f"    View {i}: {adj_center}")
    print(f"  Unified center (localizer voxel): {unified_center}")
    print(f"  Box size in localizer voxels: {box_size_in_localizer_vox}")
    print(f"  Box angles: {box_angles}")
    print("-" * 60)

    # Display the results for each view using the unified center
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    views = ['Sagittal', 'Coronal', 'Axial']
    half_dims = box_size_in_localizer_vox / 2

    # Extract 2D pivot coordinates for each view based on the unified center:
    #   - Sagittal: use the Y and Z components
    #   - Coronal: use the X and Z components
    #   - Axial: use the X and Y components
    pivot_sag = unified_center[1:3]
    pivot_cor = np.array([unified_center[0], unified_center[2]])
    pivot_axi = unified_center[:2]

    pivots = [pivot_sag, pivot_cor, pivot_axi]

    for i, (data, view, pivot) in enumerate(zip(localizer_data_list, views, pivots)):
        axes[i].imshow(data.T, cmap="gray", origin="lower")
        axes[i].axis("off")
        if view == 'Sagittal':
            base_corners = np.array([[-half_dims[1], -half_dims[2]],
                                     [-half_dims[1],  half_dims[2]],
                                     [ half_dims[1],  half_dims[2]],
                                     [ half_dims[1], -half_dims[2]]])
            theta = box_angles[0]
        elif view == 'Coronal':
            base_corners = np.array([[-half_dims[0], -half_dims[2]],
                                     [-half_dims[0],  half_dims[2]],
                                     [ half_dims[0],  half_dims[2]],
                                     [ half_dims[0], -half_dims[2]]])
            theta = box_angles[1]
        elif view == 'Axial':
            base_corners = np.array([[-half_dims[0], -half_dims[1]],
                                     [-half_dims[0],  half_dims[1]],
                                     [ half_dims[0],  half_dims[1]],
                                     [ half_dims[0], -half_dims[1]]])
            theta = box_angles[2]

        # Compute the rotation matrix for the given angle (in degrees)
        rotation_matrix = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                                    [np.sin(np.deg2rad(theta)),  np.cos(np.deg2rad(theta))]])
        # Rotate the base corners and add the pivot offset
        rotated_corners = np.dot(base_corners, rotation_matrix.T) + pivot

        # Draw the bounding box on the image
        for j in range(len(rotated_corners)):
            start = rotated_corners[j]
            end = rotated_corners[(j+1) % len(rotated_corners)]
            axes[i].plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=2)
        axes[i].set_title(f"{subject_id} - {view}")

    plt.tight_layout()
    plt.show()

    # Save the updated box information based on the unified center (a single 3D coordinate)
    updated_box_info[subject_id] = {
        "center": unified_center.tolist(),
        "size": box_size_in_localizer_vox.tolist(),
        "angles": box_angles.tolist()
    }

# Process each subject and display the localizer image with the computed center and bounding box
for sid in subject_ids:
    display_localizer_with_center(sid)

# Write the updated box information to a JSON file
with open(updated_box_info_path, 'w') as f:
    json.dump(updated_box_info, f, indent=4)

print(f"Transformed box information has been saved to '{updated_box_info_path}'.")
