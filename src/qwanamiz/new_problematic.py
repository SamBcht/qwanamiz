# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:13:37 2025

@author: sambo
"""

#### MAP CELLS WITH THE CORRESPONDING BOUNDARY REGION
def map_cell_to_region(boundary_regions, boundary_labeled, expanded_labels):
    # Step 1: Identify pixels that belong to boundary regions
    boundary_pixels = np.where(boundary_regions)  # Get indices of non-background pixels
    # Step 2: Get region labels from boundary_labeled
    region_labels = boundary_labeled[boundary_pixels]

    # Step 3: Get cell labels from expanded_labels
    cell_labels = expanded_labels[boundary_pixels]

    # Step 4: Create mapping of cell label → region label
    cell_to_region = {}
    for cell, region in zip(cell_labels, region_labels):
        if cell > 0 and region > 0:  # Ignore background
            cell_to_region[cell] = region  # Keep only the last associated region

#from collections import defaultdict

    region_to_cells = defaultdict(set)
    for cell, region in cell_to_region.items():
        region_to_cells[region].add(cell)
        
    return cell_to_region, region_to_cells


right_to_region, region_to_right = map_cell_to_region(rightcells_mask, final_boundary_corrected, expanded_labels)

###### FIND PROBLEMATIC BOUNDARY REGIONS
# Keep only cells whose label is in right_neighbor_labels
#rightcells_df = celldata[celldata["label"].isin(right_neighbor_labels)].copy()


# Step 1: Map lastcell labels to their corresponding boundary region
rightcells_df["boundary_region"] = rightcells_df["label"].map(right_to_region)

# Step 2: Count unique lastcell labels per (radial_file, boundary_region)
region_counts = rightcells_df.groupby(["radial_file", "boundary_region"])["label"].nunique()

# Step 3: Filter for regions with more than one lastcell in the same radial_file
problematic_regions = region_counts[region_counts > 1].reset_index()["boundary_region"].unique()

print("Regions with multiple lastcells in the same radial_file:", problematic_regions)

# Step 1: Create an empty mask
problematic_mask = np.zeros_like(final_boundary_corrected, dtype=bool)

# Step 2: Set pixels belonging to problematic regions to True
problematic_mask[np.isin(final_boundary_corrected, problematic_regions)] = True

viewer.add_image(problematic_mask, name="Problematic Regions", opacity=0.5, colormap="magenta", scale=[pix_to_um, pix_to_um])  # Highlighted problem regions


#import numpy as np
from skimage.morphology import erosion, disk
#from skimage.measure import label

# Step 1: Create a binary mask of region 405 from boundary_labeled
region_id = 502
region_mask_405 = right_labeled == region_id

# Optional: Visualize this original region
#viewer.add_image(region_mask_405.astype(np.uint8), name="Region 405", scale=[pix_to_um, pix_to_um])

# Step 2: Apply erosion to shrink it
# You can adjust the radius depending on how aggressive you want the erosion
eroded_mask = erosion(region_mask_405, disk(5))

# Step 3: Label the eroded mask to check for splitting
eroded_labeled = skimage.measure.label(eroded_mask)

# Step 4: Visualize
viewer.add_labels(eroded_labeled, name="Eroded Region 405", scale=[pix_to_um, pix_to_um])

map_to_new, _ = map_cell_to_region(eroded_mask, eroded_labeled, expanded_labels)

cells_in_eroded = celldata[celldata["label"].isin(map_to_new.keys())].copy()
cells_in_eroded["eroded_region"] = cells_in_eroded["label"].map(map_to_new)

# Step 2: Count unique lastcell labels per (radial_file, boundary_region)
region_counts = cells_in_eroded.groupby(["radial_file", "eroded_region"])["label"].nunique()

# Step 3: Filter for regions with more than one lastcell in the same radial_file
problematic_regions = region_counts[region_counts > 1].reset_index()["eroded_region"].unique()


from skimage.morphology import erosion, disk
from skimage.measure import label
from collections import defaultdict


# --- Setup ---
region_id = 405
region_mask_405 = right_labeled == region_id

max_radius = 10
final_eroded = None
final_cell_mapping = None
final_label = None

for radius in range(3, max_radius + 1):
    eroded_mask = erosion(region_mask_405, disk(radius))
    eroded_labeled = label(eroded_mask, connectivity=1)

    map_to_new, _ = map_cell_to_region(eroded_mask, eroded_labeled, expanded_labels)

    cells_in_eroded = celldata[celldata["label"].isin(map_to_new.keys())].copy()
    cells_in_eroded["eroded_region"] = cells_in_eroded["label"].map(map_to_new)

    # Step 2: Count unique lastcell labels per (radial_file, boundary_region)
    region_counts = cells_in_eroded.groupby(["radial_file", "eroded_region"])["label"].nunique()

    # Step 3: Filter for regions with more than one lastcell in the same radial_file
    problematic_regions = region_counts[region_counts > 1].reset_index()["eroded_region"].unique()

    print(f"Erosion radius {radius}: {len(problematic_regions)} problematic regions")

    if len(problematic_regions) == 0:
        print(f"✅ Region successfully split at erosion radius {radius}")
        final_eroded = eroded_mask
        final_cell_mapping = cell_mapping
        final_label = eroded_labeled
        break
else:
    print("⚠️ Reached max radius without resolving problem.")

# Optional: visualize result
if final_label is not None:
    viewer.add_labels(final_label, name=f"Region {region_id} split", scale=[pix_to_um, pix_to_um])



def split_problematic_regions(problematic_regions, labeled_mask, expanded_labels, celldata, max_radius=10, min_radius=3):
    """
    Split problematic boundary regions by iterative erosion and apply corrections directly
    to the original labeled_mask using cell regions in expanded_labels.

    Returns:
    - corrected_labeled_mask: np.ndarray, updated labeled mask with corrected regions.
    - updated_mappings: dict of {region_id: {cell_label → new region}}.
    - success_regions: list of successfully split region IDs.
    - failed_regions: list of region IDs that couldn't be split.
    """
    corrected_labeled_mask = labeled_mask.copy()
    updated_mappings = {}
    success_regions = []
    failed_regions = []

    current_max_label = corrected_labeled_mask.max() + 1  # for generating unique labels

    def map_cell_to_region(boundary_regions, boundary_labeled, expanded_labels):
        boundary_pixels = np.where(boundary_regions)
        region_labels = boundary_labeled[boundary_pixels]
        cell_labels = expanded_labels[boundary_pixels]
        cell_to_region = {}
        for cell, region in zip(cell_labels, region_labels):
            if cell > 0 and region > 0:
                cell_to_region[cell] = region
        return cell_to_region

    for region_id in problematic_regions:
        region_mask = labeled_mask == region_id
        for radius in range(min_radius, max_radius + 1):
            eroded_mask = erosion(region_mask, disk(radius))
            eroded_labeled = label(eroded_mask, connectivity=1)
            cell_to_eroded = map_cell_to_region(eroded_mask, eroded_labeled, expanded_labels)

            cells_in_eroded = celldata[celldata["label"].isin(cell_to_eroded.keys())].copy()
            cells_in_eroded["eroded_region"] = cells_in_eroded["label"].map(cell_to_eroded)

            region_counts = cells_in_eroded.groupby(["radial_file", "eroded_region"])["label"].nunique()
            problems = region_counts[region_counts > 1].reset_index()["eroded_region"].unique()

            if len(problems) == 0:
                # ✅ Success: update corrected_labeled_mask using expanded_labels
                for cell_label, new_subregion in cell_to_eroded.items():
                    # Assign a new global label for each new subregion
                    corrected_labeled_mask[expanded_labels == cell_label] = current_max_label + new_subregion
                updated_mappings[region_id] = {
                    k: current_max_label + v for k, v in cell_to_eroded.items()
                }
                current_max_label += eroded_labeled.max() + 1  # increment label ID base
                success_regions.append(region_id)
                break
        else:
            failed_regions.append(region_id)

    return corrected_labeled_mask, updated_mappings, success_regions, failed_regions

corrected_right_labeled, updated_mappings, success_regions, failed_regions = split_problematic_regions(
    problematic_regions=problematic_regions,
    labeled_mask=right_labeled,
    expanded_labels=expanded_labels,
    celldata=celldata[celldata["label"].isin(right_neighbor_labels)]
)

# Optional: visualize
viewer.add_labels(corrected_right_labeled, name="Corrected Right Labeled", scale=[pix_to_um, pix_to_um])

##############################################################################
# Solution by skeletonization
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

# Step 1: Skeletonize
skeleton = skeletonize(region_mask_405)

# Step 2: Distance transform
dist_transform = distance_transform_edt(region_mask_405)

# Step 3: Extract distance values on skeleton
thin_values = dist_transform[skeleton]

# Step 4: Threshold to find narrow parts (necks)
threshold = threshold_otsu(thin_values)
narrow_skeleton = (skeleton & (dist_transform < threshold))

# Step 5: Remove the narrow parts from the original region to split it
split_mask = region_mask_405.copy()
split_mask[narrow_skeleton] = False

# Step 6: Label the new connected components
split_labeled = skimage.measure.label(split_mask)

viewer.add_labels(split_labeled, name="Region 405 Split", scale=[pix_to_um, pix_to_um])