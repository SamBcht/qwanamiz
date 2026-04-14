# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:43:45 2026

@author: sambo
"""

import os
import datetime
import argparse
import pickle

# Application library imports
import numpy as np
import pandas as pd
import networkx as nx
from skimage.measure import regionprops_table
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
from skimage import measure
# scipy imports
from scipy import ndimage as ndi

from skimage.morphology import erosion, disk
from skimage.measure import label
from collections import defaultdict


import napari

import networkx as nx
import rings_functions as qrings
import os

sampleID = "L20_F33-1M2-Sc7"
base_folder = "C:/Users/sambo/Desktop/QWAnamiz_store/qwanamiz_dev"

output_folder = f"{base_folder}/{sampleID}_outputs"

images = np.load(f"{output_folder}/{sampleID}_imgs.npz")

celldata = pd.read_csv(f"{output_folder}/{sampleID}_cells.csv")

adjacency = pd.read_csv(f"{output_folder}/{sampleID}_adjacency.csv")
adjacency.set_index(['label1', 'label2'], inplace=True)

prediction = images['bw_img']

expanded_labels = images['explabs']

pix_to_um = 0.55

# Launch Napari viewer
viewer = napari.Viewer()

viewer.add_image(prediction, name='Original B&W', scale = [pix_to_um, pix_to_um])

# Add the expanded labeled image
viewer.add_labels(expanded_labels, 
                  name = 'Cells', 
                  scale = [pix_to_um, pix_to_um])

##############################################################################

celldata.set_index('label', drop = False, inplace = True)
celldata = qrings.morks_index(celldata)

celldata["tan_rad_ratio"] = np.where(
    celldata["diameter_rad"] > 0,
    celldata["diameter_tan"] / celldata["diameter_rad"],
    np.nan
)

tan_thresh = 6        # in µm (adapt to your data!)
ratio_thresh = 0.3    # very narrow tangentially

#tan_thresh = celldata["diameter_tan"].quantile(0.1)
#ratio_thresh = celldata["tan_rad_ratio"].quantile(0.1)

suspect_cells = celldata[
    (celldata["diameter_tan"] < tan_thresh) &
    (celldata["tan_rad_ratio"] < ratio_thresh)
].copy()

suspect_labels = suspect_cells["label"].values

suspect_mask = np.zeros_like(expanded_labels, dtype=bool)

suspect_mask[np.isin(expanded_labels, suspect_labels)] = True

viewer.add_image(
    suspect_mask,
    name="Suspect Narrow Cells",
    opacity=0.5,
    colormap="chartreuse",
    scale=[pix_to_um, pix_to_um]
)

lastcells, rightcells = qrings.get_lastcells(celldata, adjacency)

lastcells = lastcells - set(suspect_labels)
rightcells = rightcells - set(suspect_labels)


lastcells_df = celldata[celldata["label"].isin(lastcells)].copy()

# Create an empty mask with the same shape as expanded_labels
lastcells_mask = np.zeros_like(expanded_labels, dtype=bool)
rightcells_mask = np.zeros_like(expanded_labels, dtype=bool)
left_neighbors_mask = np.zeros_like(expanded_labels, dtype=bool)

# Get the labels of lastcells and their right_neighbors
lastcells_labels = lastcells_df["label"].values
rightcells_labels = lastcells_df["right_neighbor"].values
leftcells_labels = lastcells_df["left_neighbor"].values

# Retain labels that are both lastcell and a right neighbor
lastfirst_inter = set(lastcells_labels) & set(rightcells_labels)

leftlast_inter = set(lastcells_labels) & set(leftcells_labels)

leftright_inter = set(rightcells_labels) & set(leftcells_labels)

# Create a mask where pixels belong to lastcells or their right_neighbors
lastcells_mask[np.isin(expanded_labels, lastcells_labels)] = True
rightcells_mask[np.isin(expanded_labels, rightcells_labels)] = True
left_neighbors_mask[np.isin(expanded_labels, leftcells_labels)] = True

viewer.add_image(lastcells_mask, name="LastCells Mask", opacity=0.5, colormap="red", scale = [pix_to_um, pix_to_um])
viewer.add_image(rightcells_mask, name="Right-N Mask", opacity=0.5, colormap="orange", scale = [pix_to_um, pix_to_um])
viewer.add_image(left_neighbors_mask, name="Left-N Mask", opacity=0.5, colormap="yellow", scale = [pix_to_um, pix_to_um])

###############################################################################
# Now we can filter the cell and adjacency dataframes based on cell classification
# This allow us to filter the edges (adjacencies) and nodes (cells) involved in
# a ring transition
#lastcells_df = celldata[celldata["label"].isin(lastcells_labels)].copy()

# Filter using MultiIndex levels
adjacency_lastcells = adjacency[
    adjacency.index.get_level_values("label1").isin(lastcells_labels) &
    adjacency.index.get_level_values("label2").isin(lastcells_labels)
].copy()

# Visualize edges connecting 2 lastcells
lines = []

for index, row in adjacency_lastcells.iterrows():
    coords1 = qrings.parse_centroid(row['centroid1'])
    coords2 = qrings.parse_centroid(row['centroid2'])
    
    # Append line coordinates and color to respective lists
    lines.append([coords1, coords2])

viewer.add_shapes(lines, 
                  shape_type = 'line', 
                  edge_color = "orange", 
                  name = 'Lastcells Edges')

# Keep only cells whose label is in right_neighbor_labels
rightcells_df = celldata[celldata["label"].isin(rightcells_labels)].copy()
# Extract the lastcell labels as a set
rightcells_labels = set(rightcells_df["label"])

# Filter using MultiIndex levels
adjacency_rightcells = adjacency[
    adjacency.index.get_level_values("label1").isin(rightcells_labels) &
    adjacency.index.get_level_values("label2").isin(rightcells_labels)
].copy()

# Visualize edges connecting 2 rightcells
lines = []

for index, row in adjacency_rightcells.iterrows():
    coords1 = qrings.parse_centroid(row['centroid1'])
    coords2 = qrings.parse_centroid(row['centroid2'])
    
    # Append line coordinates and color to respective lists
    lines.append([coords1, coords2])

viewer.add_shapes(lines, 
                  shape_type = 'line', 
                  edge_color = "cyan", 
                  name = 'Rightcells Edges')


# Now we extract edges between a lastcell and its precise right neighbor
# Step 1: Create set of lastcell-right_neighbor pairs (ignoring NaNs)
pairs = {
    frozenset((row["label"], row["right_neighbor"]))
    for _, row in lastcells_df.iterrows()
    if pd.notna(row["right_neighbor"])
}

# Step 2: Filter adjacency DataFrame where the index (label1, label2) is in pairs
adjacency_neighbors = adjacency[
    adjacency.index.to_frame().apply(
        lambda row: frozenset((row["label1"], row["label2"])) in pairs,
        axis=1
    )
].copy()

# Step 3: Visualize the edges as lines
lines = []
for _, row in adjacency_neighbors.iterrows():
    coords1 = qrings.parse_centroid(row['centroid1'])
    coords2 = qrings.parse_centroid(row['centroid2'])
    lines.append([coords1, coords2])

viewer.add_shapes(
    lines,
    shape_type='line',
    edge_color='yellow',
    name='Neighbors Edges'
)

###############################################################################
# Now we extract edges between a lastcell and its precise left neighbor
# Step 1: Create set of lastcell-left_neighbor pairs (ignoring NaNs)
pairs = {
    frozenset((row["left_neighbor"], row["label"]))
    for _, row in lastcells_df.iterrows()
    if pd.notna(row["left_neighbor"])
}

# Step 2: Filter adjacency DataFrame where the index (label1, label2) is in pairs
adjacency_left = adjacency[
    adjacency.index.to_frame().apply(
        lambda row: frozenset((row["label1"], row["label2"])) in pairs,
        axis=1
    )
].copy()

# Step 3: Visualize the edges as lines
lines = []
for _, row in adjacency_left.iterrows():
    coords1 = qrings.parse_centroid(row['centroid1'])
    coords2 = qrings.parse_centroid(row['centroid2'])
    lines.append([coords1, coords2])

viewer.add_shapes(
    lines,
    shape_type='line',
    edge_color='red',
    name='Left Edges'
)

###############################################################################
pairs = {
    frozenset((row["label"], row["right_neighbor"]))
    for _, row in rightcells_df.iterrows()
    if pd.notna(row["right_neighbor"])
}

# Step 2: Filter adjacency DataFrame where the index (label1, label2) is in pairs
adjacency_to_right = adjacency[
    adjacency.index.to_frame().apply(
        lambda row: frozenset((row["label1"], row["label2"])) in pairs,
        axis=1
    )
].copy()

# Step 3: Visualize the edges as lines
lines = []
for _, row in adjacency_to_right.iterrows():
    coords1 = qrings.parse_centroid(row['centroid1'])
    coords2 = qrings.parse_centroid(row['centroid2'])
    lines.append([coords1, coords2])

viewer.add_shapes(
    lines,
    shape_type='line',
    edge_color='magenta',
    name='Right Edges'
)

###############################################################################
graph, boundaries, right_to_region, region_to_right, rightcells_df = qrings.find_boundaries(celldata,
                                                                                           adjacency,
                                                                                           lastcells,
                                                                                           rightcells,
                                                                                           expanded_labels)

viewer.add_labels(boundaries, name="First Boundary Segments", scale=[pix_to_um, pix_to_um])

problematic_regions = qrings.get_problematic_regions(rightcells_df)
print("Regions with multiple lastcells in the same radial_file:", problematic_regions)

# Step 1: Create an empty mask
problematic_mask = np.zeros_like(boundaries, dtype=bool)

# Step 2: Set pixels belonging to problematic regions to True
problematic_mask[np.isin(boundaries, problematic_regions)] = True

viewer.add_image(problematic_mask, name="Problematic Regions", opacity=0.5, colormap="magenta", scale=[pix_to_um, pix_to_um])  # Highlighted problem regions

# We update the boundary_labeled image image to keep only rightcells
rightcells_boundary = qrings.create_boundary_array(right_to_region, expanded_labels)


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
    labeled_mask=rightcells_boundary,
    expanded_labels=expanded_labels,
    celldata=rightcells_df
    #celldata[celldata["label"].isin(right_neighbor_labels)]
)

# Optional: visualize
viewer.add_labels(corrected_right_labeled, name="Corrected Right Labeled", scale=[pix_to_um, pix_to_um])

right_to_region, region_to_right = qrings.map_cell_to_region(rightcells_mask, corrected_right_labeled, expanded_labels)
rightcells_df["boundary_region"] = rightcells_df["label"].map(right_to_region)
rightcells_boundary = corrected_right_labeled.copy()
##############################################################################

def split_regions_to_target(
    problematic_regions,
    labeled_mask,
    expanded_labels,
    celldata,
    target_n_regions=2,
    max_radius=10,
    min_radius=3
):
    """
    Split regions until a desired number of subregions is reached.

    Returns:
    - corrected_labeled_mask
    - updated_mappings
    - success_regions
    - failed_regions
    """

    corrected_labeled_mask = labeled_mask.copy()
    updated_mappings = {}
    success_regions = []
    failed_regions = []

    current_max_label = corrected_labeled_mask.max() + 1

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

        success = False

        for radius in range(min_radius, max_radius + 1):

            eroded_mask = erosion(region_mask, disk(radius))
            eroded_labeled = label(eroded_mask, connectivity=1)

            cell_to_eroded = map_cell_to_region(
                eroded_mask,
                eroded_labeled,
                expanded_labels
            )

            # number of meaningful subregions
            unique_subregions = set(cell_to_eroded.values())
            n_subregions = len(unique_subregions)

            # stop when target is reached
            if n_subregions >= target_n_regions:

                for cell_label, subregion in cell_to_eroded.items():
                    corrected_labeled_mask[expanded_labels == cell_label] = (
                        current_max_label + subregion
                    )

                updated_mappings[region_id] = {
                    k: current_max_label + v for k, v in cell_to_eroded.items()
                }

                current_max_label += max(unique_subregions) + 1
                success_regions.append(region_id)
                success = True
                break

        if not success:
            failed_regions.append(region_id)

    return corrected_labeled_mask, updated_mappings, success_regions, failed_regions

regions_to_split = np.array([57, 141, 706, 382, 419, 313, 443])


corrected_right_labeled, updated_mappings, success_regions, failed_regions = split_regions_to_target(    
    regions_to_split,
    rightcells_boundary,
    expanded_labels,
    celldata,
    target_n_regions=2,
    max_radius=10,
    min_radius=3)

viewer.add_labels(corrected_right_labeled, name="Corrected Right Labeled", scale=[pix_to_um, pix_to_um])

right_to_region, region_to_right = qrings.map_cell_to_region(rightcells_mask, corrected_right_labeled, expanded_labels)
rightcells_df["boundary_region"] = rightcells_df["label"].map(right_to_region)
rightcells_boundary = corrected_right_labeled.copy()

###############################################################################
# VERTICAL MERGING
#### Now we find the most up- and downward cells in each ring boundary segments
up_extremities, down_extremities = qrings.get_extremities(region_to_right, rightcells_df)

common_neighbors, up_down_pairs, remaining_labels, upward_neighbors, downward_neighbors = qrings.get_extremity_neighbors(up_extremities, down_extremities, celldata)

# Create the mask for the common neighbors
common_neighbors_mask = np.isin(expanded_labels, list(common_neighbors))

# Add this mask to Napari as an image layer
viewer.add_image(common_neighbors_mask.astype(int), name="Common Neighbors", opacity=0.5, colormap="magenta", scale=[pix_to_um, pix_to_um])

# Extract unique labels from the pairs
paired_labels = set(label for pair in up_down_pairs for label in pair)

# Create the mask for these labels
up_down_pairs_mask = np.isin(expanded_labels, list(paired_labels)).astype(np.float32)

# Add this mask to Napari
viewer.add_image(up_down_pairs_mask, name="Up-Down Pairs", opacity=0.5, colormap="cyan", scale=[pix_to_um, pix_to_um])

common_to_reject = set([])
common_neighbors = common_neighbors - common_to_reject

# We start by integrating common neighbors and merge ring segments accordingly
updated_boundaries = qrings.integrate_commons(upward_neighbors, 
                                                       downward_neighbors, 
                                                       common_neighbors, 
                                                       rightcells_boundary, 
                                                   expanded_labels)

# We then integrate up and down pairs and also merge regions accordingly
# An update of the cell_to_region mapping is done internally
final_boundaries = qrings.integrate_updown(upward_neighbors, 
                                                    downward_neighbors, 
                                                    up_down_pairs, 
                                                    updated_boundaries, 
                                                    expanded_labels)

# We update the mapping of cells to their boundary region
cell_to_region, region_to_cells = qrings.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)

# We find in the remaining cells adjacent to extremities the ones that show
# characteristics of ring transition
labels_to_integrate = qrings.get_candidate_cells(celldata, remaining_labels, lastcells, diameter_factor = 1.8)

integration_mask = np.zeros_like(expanded_labels, dtype=bool)
integration_mask[np.isin(expanded_labels, list(labels_to_integrate))] = True

viewer.add_image(integration_mask, name="Integrated cells", opacity=0.5, colormap="orange", scale = [pix_to_um, pix_to_um])

candidate_to_reject = set([12661])
labels_to_integrate = labels_to_integrate - candidate_to_reject
# Cells retained for integration are the ones with their direct left neighbor
# showing a X times lower diameter
# or a transition between earlywood and latewood
boundaries = qrings.integrate_candidates(final_boundaries, 
                                                  expanded_labels, 
                                                  labels_to_integrate, 
                                                  cell_to_region, 
                                                  upward_neighbors, 
                                                  downward_neighbors)

viewer.add_labels(boundaries, name="New Boundary Segments", scale=[pix_to_um, pix_to_um])

# UPDATE THE RIGHTCELLS DATAFRAME WITH NEWLY INTEGRATED CELLS
# Step 1: Gather all labels already accounted for
integrated_labels = set(common_neighbors)  # Labels in both up and down neighbor sets
integrated_labels.update(label for pair in up_down_pairs for label in pair)  # Labels in up-down pairs
# Keep only cells whose label is in right_neighbor_labels
integrated_labels.update(labels_to_integrate)

# Step 1: Filter celldata for the integrated labels
new_rows = celldata[celldata["label"].isin(integrated_labels)].copy()

# Step 2: Append to rightcells_df (without duplicate labels)
rightcells_df = pd.concat([rightcells_df, new_rows]).drop_duplicates(subset="label")

cell_to_region, region_to_cells = qrings.map_cell_to_region(boundaries > 0, boundaries, expanded_labels)
rightcells_df["boundary_region"] = rightcells_df["label"].map(cell_to_region)

# Find the extrmities of the new ring segments
up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)


########################
# The function return a list of tuples with labels of the CONNECTED CELLS
connected_regions = qrings.get_segment_adjacency(adjacency, cell_to_region, up_extremities, down_extremities)

final_boundaries, new_cell_to_region = qrings.merge_by_cells(connected_regions, cell_to_region, boundaries, expanded_labels)

cell_to_region, region_to_cells = qrings.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)

viewer.add_labels(final_boundaries, name="New Boundary Segments", scale=[pix_to_um, pix_to_um])

rightcells_df["boundary_region"] = rightcells_df["label"].map(cell_to_region)
problematic_regions = qrings.get_problematic_regions(rightcells_df)
print("Regions with multiple lastcells in the same radial_file:", problematic_regions)

# Step 1: Create an empty mask
problematic_mask = np.zeros_like(final_boundaries, dtype=bool)

# Step 2: Set pixels belonging to problematic regions to True
problematic_mask[np.isin(final_boundaries, problematic_regions)] = True

viewer.add_image(problematic_mask, name="Problematic Regions", opacity=0.5, colormap="magenta", scale=[pix_to_um, pix_to_um])  # Highlighted problem regions

###############################################################################
#### 2nd round
up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)

common_neighbors, up_down_pairs, remaining_labels, upward_neighbors, downward_neighbors = qrings.get_extremity_neighbors(up_extremities, down_extremities, celldata)

#common_to_reject = set([7181])
common_neighbors = common_neighbors - common_to_reject


# Create the mask for the common neighbors
common_neighbors_mask = np.isin(expanded_labels, list(common_neighbors))

# Add this mask to Napari as an image layer
viewer.add_image(common_neighbors_mask.astype(int), name="Common Neighbors", opacity=0.5, colormap="magenta", scale=[pix_to_um, pix_to_um])

# Extract unique labels from the pairs
paired_labels = set(label for pair in up_down_pairs for label in pair)

# Create the mask for these labels
up_down_pairs_mask = np.isin(expanded_labels, list(paired_labels)).astype(np.float32)

# Add this mask to Napari
viewer.add_image(up_down_pairs_mask, name="Up-Down Pairs", opacity=0.5, colormap="cyan", scale=[pix_to_um, pix_to_um])

# We start by integrating common neighbors and merge ring segments accordingly
updated_boundaries = qrings.integrate_commons(upward_neighbors, 
                                                       downward_neighbors, 
                                                       common_neighbors, 
                                                       final_boundaries, 
                                                   expanded_labels)

# We then integrate up and down pairs and also merge regions accordingly
# An update of the cell_to_region mapping is done internally
final_boundaries = qrings.integrate_updown(upward_neighbors, 
                                                    downward_neighbors, 
                                                    up_down_pairs, 
                                                    updated_boundaries, 
                                                    expanded_labels)

# We update the mapping of cells to their boundary region
cell_to_region, region_to_cells = qrings.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)

# We find in the remaining cells adjacent to extremities the ones that show
# characteristics of ring transition
labels_to_integrate = qrings.get_candidate_cells(celldata, remaining_labels, lastcells, diameter_factor = 1.8)

integration_mask = np.zeros_like(expanded_labels, dtype=bool)
integration_mask[np.isin(expanded_labels, list(labels_to_integrate))] = True

viewer.add_image(integration_mask, name="Integrated cells", opacity=0.5, colormap="orange", scale = [pix_to_um, pix_to_um])




# Cells retained for integration are the ones with their direct left neighbor
# showing a X times lower diameter
# or a transition between earlywood and latewood
if len(labels_to_integrate) > 0 :
    boundaries = qrings.integrate_candidates(final_boundaries, 
                                                  expanded_labels, 
                                                  labels_to_integrate, 
                                                  cell_to_region, 
                                                  upward_neighbors, 
                                                  downward_neighbors)
else:
    boundaries = final_boundaries

#viewer.add_labels(boundaries, name="New Boundary Segments", scale=[pix_to_um, pix_to_um])

# UPDATE THE RIGHTCELLS DATAFRAME WITH NEWLY INTEGRATED CELLS
# Step 1: Gather all labels already accounted for
integrated_labels = set(common_neighbors)  # Labels in both up and down neighbor sets
integrated_labels.update(label for pair in up_down_pairs for label in pair)  # Labels in up-down pairs
# Keep only cells whose label is in right_neighbor_labels
integrated_labels.update(labels_to_integrate)

# Step 1: Filter celldata for the integrated labels
new_rows = celldata[celldata["label"].isin(integrated_labels)].copy()

# Step 2: Append to rightcells_df (without duplicate labels)
rightcells_df = pd.concat([rightcells_df, new_rows]).drop_duplicates(subset="label")

cell_to_region, region_to_cells = qrings.map_cell_to_region(boundaries > 0, boundaries, expanded_labels)


# Find the extrmities of the new ring segments
up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)

# The function return a list of tuples with labels of the CONNECTED CELLS
connected_regions = qrings.get_segment_adjacency(adjacency, cell_to_region, up_extremities, down_extremities)

final_boundaries, new_cell_to_region = qrings.merge_by_cells(connected_regions, cell_to_region, boundaries, expanded_labels)

cell_to_region, region_to_cells = qrings.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)

viewer.add_labels(final_boundaries, name="New Boundary Segments", scale=[pix_to_um, pix_to_um])


###############################################################################
#cell_to_region = right_to_region

cells_in_regions = set(cell_to_region.keys())
cells_not_in_regions = set(celldata["label"]) - cells_in_regions

missing_cells_df = celldata[
    celldata["label"].isin(cells_not_in_regions)
].copy()

missing_cells_df["ratio"] = (
    missing_cells_df["diameter_rad"] /
    missing_cells_df["prev_diameter_rad"]
)

bad_cells = missing_cells_df[
    missing_cells_df["ratio"] >= 2.1
].copy()

bad_labels = bad_cells["label"].values

bad_mask = np.zeros_like(expanded_labels, dtype=bool)

bad_mask[np.isin(expanded_labels, bad_labels)] = True

viewer.add_image(
    bad_mask,
    name="Bad Missing Cells (Left Ratio)",
    opacity=0.5,
    colormap="chartreuse",
    scale=[pix_to_um, pix_to_um]
)

###############################################################################
# This function allows to get a list of regions containing at least one cell of
# the same radial file. Can be used to avoid merging of boundary segments belonging
# to different files
up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)


incompatible_region_pairs = qrings.incompatible_regions(celldata, cell_to_region)

# Finally, we find each up_extremity's nearest down_extremity and vice versa.
# We keep pairs of up and down that are mutually the nearest for each other
# When a region has only one cell that is thus both the up and down extremity,
# nearest extremities are the same point and they are excluded from the merging
# This avoid merging potential region falsely identified as boundary
nearest_extremity, _ = qrings.get_nearest_extremity(rightcells_df, cell_to_region, up_extremities, down_extremities, incompatible_region_pairs)

# This step could be repeat iteratively to add new connections
# But we will still have non connected regions where
# - up and down extremities are in the same radial files
# - there are several up and down extremities in a small zone, this could introduce errors

pairs_df, valid, excluded = qrings.analyze_pairs_angles(celldata, nearest_extremity)

nearest_extremity = valid

lines = []
for up_label, down_label in nearest_extremity:
    up_coords = rightcells_df[rightcells_df["label"] == up_label][["centroid-0", "centroid-1"]].values[0]
    down_coords = rightcells_df[rightcells_df["label"] == down_label][["centroid-0", "centroid-1"]].values[0]
    lines.append([up_coords, down_coords])

viewer.add_shapes(lines, shape_type='line', edge_color='magenta', name='Mutual Nearest Pairs', edge_width=3)

###############################################################################
# Rightcells extremity connection path

neighbor_lookup = (
    celldata[["label", "up_neighbor", "down_neighbor"]]
    .set_index("label")
    .to_dict("index")
)

right_to_region = cell_to_region   # rightcells only

up_ext_cells   = {int(v): int(k) for k, v in up_extremities.items() if pd.notna(v)}
down_ext_cells = {int(v): int(k) for k, v in down_extremities.items() if pd.notna(v)}

all_up_ext   = set(up_ext_cells.keys())
all_down_ext = set(down_ext_cells.keys())

bad_cells_set = set(bad_cells["label"])

def find_extremity_path(start_cell,
                       direction,
                       neighbor_lookup,
                       up_ext_set,
                       down_ext_set,
                       max_steps=20):

    current = int(start_cell)
    path = [current]

    for step in range(1, max_steps + 1):

        next_cell = neighbor_lookup.get(current, {}).get(f"{direction}_neighbor")

        if next_cell is None or pd.isna(next_cell):
            return None, path

        next_cell = int(next_cell)
        path.append(next_cell)

        # success condition
        if direction == "up" and next_cell in down_ext_set:
            return next_cell, path

        if direction == "down" and next_cell in up_ext_set:
            return next_cell, path

        current = next_cell

    return None, path

def find_badcell_path(start_cell,
                     direction,
                     neighbor_lookup,
                     bad_cells_set,
                     up_ext_set,
                     down_ext_set,
                     max_steps=20):

    current = int(start_cell)
    path = [current]

    for step in range(1, max_steps + 1):

        next_cell = neighbor_lookup.get(current, {}).get(f"{direction}_neighbor")

        if next_cell is None or pd.isna(next_cell):
            return current, path  # dead end

        next_cell = int(next_cell)

        # CASE 1: reach opposite extremity → valid connection
        if direction == "up" and next_cell in down_ext_set:
            path.append(next_cell)
            return next_cell, path

        if direction == "down" and next_cell in up_ext_set:
            path.append(next_cell)
            return next_cell, path

        # CASE 2: next cell NOT bad → stop BEFORE including it
        if next_cell not in bad_cells_set:
            return current, path  # last valid = last bad cell

        # otherwise continue
        path.append(next_cell)
        current = next_cell

    return None, path

extremity_connections = {}

for region in region_to_cells.keys():

    extremity_connections[region] = {
        "up_to_down": [],
        "down_to_up": []
    }

    up_cell = up_extremities.get(region)
    down_cell = down_extremities.get(region)

    # UP → DOWN
    if pd.notna(up_cell):

        target_cell, path = find_badcell_path(
            up_cell,
            "up",
            neighbor_lookup,
            bad_cells_set,
            all_up_ext,
            all_down_ext,
            max_steps=20
        )

        if target_cell is not None:

            # CASE 1: real connection to extremity
            if target_cell in down_ext_cells:
        
                target_region = down_ext_cells[target_cell]
        
                extremity_connections[region]["up_to_down"].append({
                    "target_region": target_region,
                    "path": path,
                    "type": "connection"
                })
        
            # CASE 2: stopped on bad cell (not an extremity)
            else:
        
                extremity_connections[region]["up_to_down"].append({
                    "target_region": None,
                    "path": path,
                    "type": "dead_end"
                })

    # DOWN → UP
    if pd.notna(down_cell):

        target_cell, path = find_badcell_path(
            down_cell,
            "down",
            neighbor_lookup,
            bad_cells_set,
            all_up_ext,
            all_down_ext,
            max_steps=20
        )

        if target_cell is not None:

            if target_cell in up_ext_cells:
        
                target_region = up_ext_cells[target_cell]
        
                extremity_connections[region]["down_to_up"].append({
                    "target_region": target_region,
                    "path": path,
                    "type": "connection"
                })
        
            else:
        
                extremity_connections[region]["down_to_up"].append({
                    "target_region": None,
                    "path": path,
                    "type": "dead_end"
                })
            
paths_coords = []

centroid_lookup = dict(
    zip(celldata["label"],
        zip(celldata["centroid-0"], celldata["centroid-1"]))
)

for region_data in extremity_connections.values():

    for direction in ["up_to_down", "down_to_up"]:

        for conn in region_data[direction]:

            coords = [
                centroid_lookup[cell]
                for cell in conn["path"]
                if cell in centroid_lookup
            ]

            if len(coords) > 1:
                paths_coords.append(np.array(coords))

viewer.add_shapes(
    paths_coords,
    shape_type="path",
    edge_color="magenta",
    edge_width=4,
    name="Extremity-to-Extremity paths (≤10)"
)

centroid_lookup = dict(
    zip(
        celldata["label"],
        zip(celldata["centroid-0"], celldata["centroid-1"])
    )
)

up_neighbor_lines = []
down_neighbor_lines = []

for region, up_label in up_extremities.items():

    if pd.isna(up_label):
        continue

    up_label = int(up_label)

    up_neighbor = neighbor_lookup.get(up_label, {}).get("up_neighbor")

    if up_neighbor is None or pd.isna(up_neighbor):
        continue

    up_neighbor = int(up_neighbor)

    if up_label in centroid_lookup and up_neighbor in centroid_lookup:

        coords = [
            centroid_lookup[up_label],
            centroid_lookup[up_neighbor]
        ]

        up_neighbor_lines.append(coords)
        
for region, down_label in down_extremities.items():

    if pd.isna(down_label):
        continue

    down_label = int(down_label)

    down_neighbor = neighbor_lookup.get(down_label, {}).get("down_neighbor")

    if down_neighbor is None or pd.isna(down_neighbor):
        continue

    down_neighbor = int(down_neighbor)

    if down_label in centroid_lookup and down_neighbor in centroid_lookup:

        coords = [
            centroid_lookup[down_label],
            centroid_lookup[down_neighbor]
        ]

        down_neighbor_lines.append(coords)

if len(up_neighbor_lines) > 0:
    viewer.add_shapes(
        up_neighbor_lines,
        shape_type="line",
        edge_color="blue",
        edge_width=3,
        name="Up neighbors"
    )

if len(down_neighbor_lines) > 0:
    viewer.add_shapes(
        down_neighbor_lines,
        shape_type="line",
        edge_color="green",
        edge_width=3,
        name="Down neighbors"
    )
#################
up_neighbor_lines = []

for region, up_label in up_extremities.items():

    if pd.isna(up_label):
        continue

    up_label = int(up_label)

    up_neighbor = neighbor_lookup.get(up_label, {}).get("up_neighbor")

    if up_neighbor is None or pd.isna(up_neighbor):
        continue

    up_neighbor = int(up_neighbor)

    # NEW CONDITION: neighbor must be in a region
    if up_neighbor not in cells_in_regions:
        continue

    if up_label in centroid_lookup and up_neighbor in centroid_lookup:

        coords = [
            centroid_lookup[up_label],
            centroid_lookup[up_neighbor]
        ]

        up_neighbor_lines.append(coords)
        
down_neighbor_lines = []

for region, down_label in down_extremities.items():

    if pd.isna(down_label):
        continue

    down_label = int(down_label)

    down_neighbor = neighbor_lookup.get(down_label, {}).get("down_neighbor")

    if down_neighbor is None or pd.isna(down_neighbor):
        continue

    down_neighbor = int(down_neighbor)

    # NEW CONDITION
    if down_neighbor not in cells_in_regions:
        continue

    if down_label in centroid_lookup and down_neighbor in centroid_lookup:

        coords = [
            centroid_lookup[down_label],
            centroid_lookup[down_neighbor]
        ]

        down_neighbor_lines.append(coords)
        
if len(up_neighbor_lines) > 0:
    viewer.add_shapes(
        up_neighbor_lines,
        shape_type="line",
        edge_color="cyan",
        edge_width=3,
        name="Up neighbors (in regions)"
    )

if len(down_neighbor_lines) > 0:
    viewer.add_shapes(
        down_neighbor_lines,
        shape_type="line",
        edge_color="cyan",
        edge_width=3,
        name="Down neighbors (in regions)"
    )
#################
upward_points = []
downward_points = []

# Extract centroid coordinates from celldata
for region, up_label in up_extremities.items():
    up_cell = celldata[celldata["label"] == up_label]
    if not up_cell.empty:
        upward_points.append((up_cell["centroid-0"].values[0], up_cell["centroid-1"].values[0]))

for region, down_label in down_extremities.items():
    down_cell = celldata[celldata["label"] == down_label]
    if not down_cell.empty:
        downward_points.append((down_cell["centroid-0"].values[0], down_cell["centroid-1"].values[0]))

# Convert to numpy arrays for Napari
upward_points = np.array(upward_points)
downward_points = np.array(downward_points)

# Add the points to Napari
if len(upward_points) > 0:
    viewer.add_points(upward_points, name="Upward Earlywood Cells", size=5, face_color="blue", border_color="white")

if len(downward_points) > 0:
    viewer.add_points(downward_points, name="Downward Earlywood Cells", size=5, face_color="green", border_color="white")

#################
import networkx as nx

G = nx.Graph()
G.add_nodes_from(region_to_cells.keys())

for region, directions in extremity_connections.items():

    for direction in ["up_to_down", "down_to_up"]:

        for conn in directions[direction]:

            if conn["type"] == "connection":

                target_region = conn["target_region"]

                if target_region is not None and target_region != region:
                    G.add_edge(region, target_region)
                    
for region, up_label in up_extremities.items():

    if pd.isna(up_label):
        continue

    up_label = int(up_label)
    up_neighbor = neighbor_lookup.get(up_label, {}).get("up_neighbor")

    if up_neighbor is None or pd.isna(up_neighbor):
        continue

    up_neighbor = int(up_neighbor)

    if up_neighbor not in cell_to_region:
        continue

    target_region = cell_to_region[up_neighbor]

    # EXCLUDE up → up extremity connections
    if up_neighbor in up_ext_cells:
        continue

    if target_region != region:
        G.add_edge(region, target_region)
        
for region, down_label in down_extremities.items():

    if pd.isna(down_label):
        continue

    down_label = int(down_label)
    down_neighbor = neighbor_lookup.get(down_label, {}).get("down_neighbor")

    if down_neighbor is None or pd.isna(down_neighbor):
        continue

    down_neighbor = int(down_neighbor)

    if down_neighbor not in cell_to_region:
        continue

    target_region = cell_to_region[down_neighbor]

    # EXCLUDE down → down extremity
    if down_neighbor in down_ext_cells:
        continue

    if target_region != region:
        G.add_edge(region, target_region)
        
merged_groups = list(nx.connected_components(G))

new_boundaries = boundaries.copy()
region_mapping = {}
new_label_base = new_boundaries.max() + 1

for group in merged_groups:

    new_label = new_label_base

    for region in group:
        region_mapping[region] = new_label

    new_label_base += 1
    
for region, directions in extremity_connections.items():

    base_region = region_mapping.get(region, region)

    for direction in ["up_to_down", "down_to_up"]:

        for conn in directions[direction]:

            for cell in conn["path"]:

                # ✅ include ALL bad cells
                if cell in bad_cells_set:
                    new_boundaries[expanded_labels == cell] = base_region
                    
for old_region, new_region in region_mapping.items():
    new_boundaries[boundaries == old_region] = new_region
                        
viewer.add_labels(new_boundaries, name="Boundaries", scale=[pix_to_um, pix_to_um])

boundaries = new_boundaries.copy()

cell_to_region, region_to_cells = qrings.map_cell_to_region(boundaries > 0, boundaries, expanded_labels)

# Find missing cells
missing_cells = set(cell_to_region.keys()) - set(rightcells_df["label"])

# Extract corresponding rows
new_rows = celldata[celldata["label"].isin(missing_cells)].copy()

# Assign region
new_rows["boundary_region"] = new_rows["label"].map(cell_to_region)

# Append
rightcells_df = pd.concat([rightcells_df, new_rows], ignore_index=False)
rightcells_df["boundary_region"] = rightcells_df["label"].map(cell_to_region)

