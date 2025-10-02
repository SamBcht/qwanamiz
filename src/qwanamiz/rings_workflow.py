# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:10:42 2025

@author: sambo
"""
import networkx as nx
import rings_functions

##############################################################################
# Detection of tree-ring transitions by comparing successive cells properties
# (radial diameter and early-latewood classification)
# Get lastcells in rings based on diameter and woodzone cell features
lastcells_labels, rightcells_labels, leftcells_labels = rings_functions.get_lastcells(celldata, adjacency)

# Create an empty mask with the same shape as expanded_labels
lastcells_mask = np.zeros_like(expanded_labels, dtype=bool)
rightcells_mask = np.zeros_like(expanded_labels, dtype=bool)
left_neighbors_mask = np.zeros_like(expanded_labels, dtype=bool)

# Get the labels of lastcells and their right_neighbors
#lastcell_labels = lastcells_df["label"].values
#rightcells_labels = lastcells_df["right_neighbor"].values
#left_neighbor_labels = lastcells_df["left_neighbor"].values

# Retain labels that are both lastcell and a right neighbor
#lastcells_inter = set(lastcell_labels) & set(right_neighbor_labels)

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
lastcells_df = celldata[celldata["label"].isin(lastcells_labels)].copy()

# Filter using MultiIndex levels
adjacency_lastcells = adjacency[
    adjacency.index.get_level_values("label1").isin(lastcells_labels) &
    adjacency.index.get_level_values("label2").isin(lastcells_labels)
].copy()

# Visualize edges connecting 2 lastcells
lines = []

for index, row in adjacency_lastcells.iterrows():
    coords1 = rings_functions.parse_centroid(row['centroid1'])
    coords2 = rings_functions.parse_centroid(row['centroid2'])
    
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
    coords1 = rings_functions.parse_centroid(row['centroid1'])
    coords2 = rings_functions.parse_centroid(row['centroid2'])
    
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
    coords1 = rings_functions.parse_centroid(row['centroid1'])
    coords2 = rings_functions.parse_centroid(row['centroid2'])
    lines.append([coords1, coords2])

viewer.add_shapes(
    lines,
    shape_type='line',
    edge_color='yellow',
    name='Neighbors Edges'
)

###############################################################################
#### RING BOUNDARY GRAPH & CONNECTED COMPONENTS ####

#### The method use in this section could be an elegant way to handle several
#following steps of the scripts.

# This could allow us to write the workflow independantly from the image arrays
# (expanded_labels), just taking the informations from the adajcency and cells dataframes

# The idea would be to add or remove nodes and edges if they can or not be 
# considered with sufficient confidence to belong to a ring boundary or to be excluded

#### We could use it to separate problematic regions by finding edges to remove
# to avoid connecting cells of the same radial file

#### Similarly we could use it to integrate successively cells and edges we can
# confidently attribute to ring boundaries but that are not detected at first
# because of too restricting criteria
# See Common Neighbors & Up-Down Pairs sections of the script

# Now we can construct the graph using previously filtered nodes and edges
graph = rings_functions.boundary_graph(celldata, adjacency, lastcells_labels, rightcells_labels)

# Find connected components (as sets of nodes)
# This will group all cells that are connected by a path along retained edges
# So it will segregate ring boundaries and group togeteher cells belonging to
# a same boundary
connected_components = list(nx.connected_components(graph))



# Create a mapping from node to component ID
node_to_component = {}
for i, component in enumerate(connected_components):
    for node in component:
        node_to_component[node] = i + 1 # i + 1 to avoid the 0 label reserved for background

# Now we can assign component (boundary) IDs as a node attribute
nx.set_node_attributes(graph, node_to_component, 'component_id')

# Finally we can visualize the results of the grouping
# We thus have a first image with labels corresponding to cells groups of cells
# expected to belong to a ring transition

# Prepare a mask of pixels whose cell label is in label_to_region
target_labels = np.array(list(node_to_component.keys()))
target_mask = np.isin(expanded_labels, target_labels)

# Get region IDs for those cell labels
label_array = expanded_labels[target_mask]
region_array = np.vectorize(node_to_component.get)(label_array)

# Make a copy to avoid modifying in-place unless you want to
boundary_labeled = np.zeros_like(expanded_labels, dtype=int)

# Update boundary-labeled values at those positions
boundary_labeled[target_mask] = region_array

viewer.add_labels(boundary_labeled, name="Final Boundary Corrected", scale=[pix_to_um, pix_to_um])

# Now we will work mostly with rightcells
# The earlywood nature of rightcells gives clearer adjacencies and we avoid 
# unwanted groupings by using a single line of cells
right_to_region, region_to_right = rings_functions.map_cell_to_region(rightcells_mask, boundary_labeled, expanded_labels)


###### FIND PROBLEMATIC BOUNDARY REGIONS
# Problematic regions are those where there is more than one rightcell (or lastcell)
# for a unique radial file

# It's not a big deal for most of the image as ring boundaries are sufficiently separated
# But it can be a bigger problem for images with very narrow rings

# Problems very often arrive when 2 rightcells in radial files above each other
# are adjacent by a little corner touching

# Step 1: Map lastcell labels to their corresponding boundary region
rightcells_df["boundary_region"] = rightcells_df["label"].map(right_to_region)

# Step 2: Count unique lastcell labels per (radial_file, boundary_region)
region_counts = rightcells_df.groupby(["radial_file", "boundary_region"])["label"].nunique()

# Step 3: Filter for regions with more than one lastcell in the same radial_file
problematic_regions = region_counts[region_counts > 1].reset_index()["boundary_region"].unique()

print("Regions with multiple lastcells in the same radial_file:", problematic_regions)

# Step 1: Create an empty mask
problematic_mask = np.zeros_like(boundary_labeled, dtype=bool)

# Step 2: Set pixels belonging to problematic regions to True
problematic_mask[np.isin(boundary_labeled, problematic_regions)] = True

viewer.add_image(problematic_mask, name="Problematic Regions", opacity=0.5, colormap="magenta", scale=[pix_to_um, pix_to_um])  # Highlighted problem regions

# Here we can define a function to correct the problematic regions based on
# the subgraph of the region. The idea would be to find the minimum edges to 
# remove to resolve the problem
###############################################################################
#### Now the objective will be to add progressively new cells at the extremities
# ring boundary groups. We can then use these cells and their adjacencies to group
# ring boundary segments that we can think with confidence they belong to a same 
# ring boundary

# We update the boundary_labeled image image to keep only rightcells
rightcells_boundary = rings_functions.update_boundary_labels(np.zeros_like(expanded_labels, dtype=int), right_to_region, expanded_labels)

viewer.add_labels(rightcells_boundary, name="Rightcells Boundary", scale=[pix_to_um, pix_to_um])

#### Now we find the most up- and downward cells in each ring boundary segments
up_extremities, down_extremities = rings_functions.get_extremities(region_to_right, rightcells_df)

# Visualize
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


###############################################################################
#### Then we search in the extremity cell adjacencies if they can unite ring 
# boundary segments by their respective up and down extremities

# Common Neighbors are cells adjacent to both the up extremity of one boundary
# segment and the down extremity of another segment

# Up and Down Pairs are pairs of adjacent cells where one is adjacent to the 
# up extremity of one segment and the other is adjacent to the down extremity of
# another segment

# We also keep the remaining cells for further use
common_neighbors, up_down_pairs, remaining_labels, upward_neighbors, downward_neighbors = rings_functions.get_extremity_neighbors(up_extremities, down_extremities, celldata)


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

###############################################################################
#### INTEGRATION OF NEW CELLS

#### The integration could be implemented in a more elegant way using the method
# of the graph by adding nodes and edges involving common neighbors, up-down pairs etc...

# Empirically, it seems to me that we can confidently integrate common neighbors
# and up and down pairs and use them to merge boundary segments without 
# introducing too much errors

# To avoid errors, it's important to always keep the direct adjacency link 
# between the new integrated cell and its specific up and/or down extremity
# as well as the label of the extremity ring segment

# We start by integrating common neighbors and merge ring segments accordingly
updated_boundaries = rings_functions.integrate_commons(upward_neighbors, 
                                       downward_neighbors, 
                                       common_neighbors, 
                                       rightcells_boundary, 
                                       expanded_labels)

#viewer.add_labels(updated_boundaries, name="Updated Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

# We then integrate up and down pairs and also merge regions accordingly
# An update of the cell_to_region mapping is done internally
final_boundaries = rings_functions.integrate_updown(upward_neighbors, 
                                       downward_neighbors, 
                                       up_down_pairs, 
                                       updated_boundaries, 
                                       expanded_labels)

#viewer.add_labels(final_boundaries, name="Final Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

# We update the mapping of cells to their boundary region
cell_to_region, region_to_cells = rings_functions.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)
###############################################################################
#### INTEGRATION OF CELLS AT THE EXTREMITIES

# We find in the remaining cells adjacent to extremities the ones that show
# characteristics of ring transition
labels_to_integrate = rings_functions.get_candidate_cells(celldata, remaining_labels, lastcells_labels, diameter_factor = 1.8)

integration_mask = np.zeros_like(expanded_labels, dtype=bool)
integration_mask[np.isin(expanded_labels, list(labels_to_integrate))] = True

viewer.add_image(integration_mask, name="Integrated cells", opacity=0.5, colormap="orange", scale = [pix_to_um, pix_to_um])


# Cells retained for integration are the ones with their direct left neighbor
# showing a X times lower diameter
# or a transition between earlywood and latewood
boundaries = rings_functions.integrate_candidates(final_boundaries, 
                                  expanded_labels, 
                                  labels_to_integrate, 
                                  cell_to_region, 
                                  upward_neighbors, 
                                  downward_neighbors)

viewer.add_labels(boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

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

cell_to_region, region_to_cells = rings_functions.map_cell_to_region(boundaries > 0, boundaries, expanded_labels)


# Find the extrmities of the new ring segments
up_extremities, down_extremities = rings_functions.get_extremities(region_to_cells, rightcells_df)

# Visualize
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

############################################################################
# FIND ADJACENCIES BETWEEN RING SEGMENTS AFTER ADDITION OF CELLS

# Here we extend the adjacency research to all types instead of only radial_sel
# Possibility to filter the dataframe to restrict the research

# The function return a list of tuples with labels of the CONNECTED CELLS
connected_regions = rings_functions.get_segment_adjacency(adjacency, cell_to_region, up_extremities, down_extremities)


# Visualisation with lines
# Step 1: Create a dictionary for quick centroid lookup
label_to_centroid = {
    row["label"]: (row["centroid-0"], row["centroid-1"])
    for _, row in rightcells_df.iterrows()
}

# Step 2: Build list of line coordinates for each adjacency pair
adjacency_lines = []

for label1, label2 in connected_regions:
    if label1 in label_to_centroid and label2 in label_to_centroid:
        point1 = label_to_centroid[label1]
        point2 = label_to_centroid[label2]
        line = [point1, point2]  # Each line is a pair of 2D points
        adjacency_lines.append(line)

# Step 3: Add to Napari as a shapes layer
viewer.add_shapes(
    adjacency_lines,
    shape_type='line',
    edge_color='magenta',
    edge_width=2,
    name='Connected Regions'
)

final_boundaries, new_cell_to_region = rings_functions.merge_by_cells(connected_regions, cell_to_region, boundaries, expanded_labels)

viewer.add_labels(final_boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

cell_to_region, region_to_cells = rings_functions.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)

# Find the extrmities of the new ring segments
up_extremities, down_extremities = rings_functions.get_extremities(region_to_cells, rightcells_df)

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

# This function allows to get a list of regions containing at least one cell of
# the same radial file. Can be used to avoid merging of boundary segments belonging
# to different files
incompatible_region_pairs = rings_functions.incompatible_regions(celldata, cell_to_region)

# Finally, we find each up_extremity's nearest down_extremity and vice versa.
# We keep pairs of up and down that are mutually the nearest for each other
# When a region has only one cell that is thus both the up and down extremity,
# nearest extremities are the same point and they are excluded from the merging
# This avoid merging potential region falsely identified as boundary
nearest_extremity = rings_functions.get_nearest_extremity(rightcells_df, cell_to_region, up_extremities, down_extremities, incompatible_region_pairs)

# This step could be repeat iteratively to add new connections
# But we will still have non connected regions where
# - up and down extremities are in the same radial files
# - there are several up and down extremities in a small zone, this could introduce errors

pairs_df, valid, excluded = rings_functions.analyze_pairs_angles(celldata, nearest_extremity)

nearest_extremity = valid

lines = []
for up_label, down_label in nearest_extremity:
    up_coords = rightcells_df[rightcells_df["label"] == up_label][["centroid-0", "centroid-1"]].values[0]
    down_coords = rightcells_df[rightcells_df["label"] == down_label][["centroid-0", "centroid-1"]].values[0]
    lines.append([up_coords, down_coords])

viewer.add_shapes(lines, shape_type='line', edge_color='chartreuse', name='Mutual Nearest Pairs', edge_width=3)

new_boundaries, new_cell_to_region = rings_functions.merge_by_cells(nearest_extremity, cell_to_region, final_boundaries, expanded_labels)


viewer.add_labels(new_boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])



cell_to_region, region_to_cells = rings_functions.map_cell_to_region(new_boundaries > 0, new_boundaries, expanded_labels)

# At this stage we can remove spurious regions by excluding those with fewer than a given number of cells
cell_to_region, region_to_cells = rings_functions.filter_boundaries(cell_to_region, region_to_cells, mincells = 4)
new_boundaries = rings_functions.update_boundary_labels(np.zeros_like(expanded_labels, dtype = int), cell_to_region, expanded_labels)





###############################################################################
#### SECOND SEARCH OF NEAREST EXTREMITY
# This step does the same as before but without regions containing few cells and with new regions merged
# Find the extrmities of the new ring segments
up_extremities, down_extremities = rings_functions.get_extremities(region_to_cells, rightcells_df)

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


incompatible_region_pairs = rings_functions.incompatible_regions(celldata, cell_to_region)

nearest_extremity = rings_functions.get_nearest_extremity(rightcells_df, cell_to_region, up_extremities, down_extremities, incompatible_region_pairs)

pairs_df, valid, excluded = rings_functions.analyze_pairs_angles(celldata, nearest_extremity)

nearest_extremity = valid

lines = []
for up_label, down_label in nearest_extremity:
    up_coords = rightcells_df[rightcells_df["label"] == up_label][["centroid-0", "centroid-1"]].values[0]
    down_coords = rightcells_df[rightcells_df["label"] == down_label][["centroid-0", "centroid-1"]].values[0]
    lines.append([up_coords, down_coords])

viewer.add_shapes(lines, shape_type='line', edge_color='chartreuse', name='Mutual Nearest Pairs', edge_width=3)

new_boundaries, new_cell_to_region = rings_functions.merge_by_cells(nearest_extremity, cell_to_region, new_boundaries, expanded_labels)


cell_to_region, region_to_cells = rings_functions.map_cell_to_region(new_boundaries > 0, new_boundaries, expanded_labels)

# At this stage we can remove spurious regions by excluding those with fewer than a given number of cells
cell_to_region, region_to_cells = rings_functions.filter_boundaries(cell_to_region, region_to_cells, mincells = 5)
new_boundaries = rings_functions.update_boundary_labels(np.zeros_like(expanded_labels, dtype = int), cell_to_region, expanded_labels)

viewer.add_labels(new_boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])


###############################################################################
# FIND REGION EXTREMITIES NEAR THE BORDERS OF THE IMAGE
up_extremities, down_extremities = rings_functions.get_extremities(region_to_cells, rightcells_df)

all_border_cells, upper_region_sequence, lower_region_sequence, matched_up, matched_down, unjustified = rings_functions.get_border_cells(rightcells_df, 
                                                                                                                         cell_to_region, 
                                                                                                                         up_extremities,
                                                                                                                         down_extremities,
                                                                                                                         image_height = expanded_labels.shape[0], 
                                                                                                                         image_width = expanded_labels.shape[1], 
                                                                                                                         border_margin = 75, 
                                                                                                                         pix_to_um = pix_to_um)

# Create an empty mask the same shape as expanded_labels
border_mask = np.zeros_like(expanded_labels, dtype=bool)

# Mark upward border cells with 1, downward border cells with 2
border_mask[np.isin(expanded_labels, list(all_border_cells))] = True
#border_mask[np.isin(expanded_labels, list(downward_border_cells))] = True

# Display in Napari
viewer.add_image(border_mask, name="Border Up/Down Cells", scale=[pix_to_um, pix_to_um])


# Result
print("Upper border regions (left to right):", upper_region_sequence)
print("Lower border regions (left to right):", lower_region_sequence)
print("Matching upper regions :", matched_up)
print("Matching lower regions :", matched_down)
print("Unmatched :", unjustified)
# Intersection: regions that have both an upward and a downward border cell
regions_topdown = (set(upper_region_sequence) | set(matched_down)) & (set(lower_region_sequence) | set(matched_up))
print(f"{len(regions_topdown)} regions touch both the top and bottom borders.")
print("Valid regions :", regions_topdown)

#import numpy as np

def find_merge_candidates_from_sequences(upper_sequence, lower_sequence, matched_up=None, matched_down=None):
    """
    Identify candidate regions to merge based on index differences 
    between upper and lower sequences. Regions already in matched_up 
    or matched_down are removed from their respective sequences.
    
    Parameters
    ----------
    upper_sequence : list[int]
        Regions touching the top of the image (sorted by x).
    lower_sequence : list[int]
        Regions touching the bottom of the image (sorted by x).
    matched_up : set[int], optional
        Set of region IDs already matched from upper sequence.
    matched_down : set[int], optional
        Set of region IDs already matched from lower sequence.
    
    Returns
    -------
    merge_candidates : list[tuple[int,int]]
        Candidate (upper_region, lower_region) pairs.
    corrected_upper : list[int]
        Upper sequence after removing matched regions.
    corrected_lower : list[int]
        Lower sequence after removing matched regions.
    """

    if matched_up is None:
        matched_up = set()
    if matched_down is None:
        matched_down = set()

    # --- Step 1: remove matched regions ---
    corrected_upper = [r for r in upper_sequence if r not in matched_up]
    corrected_lower = [r for r in lower_sequence if r not in matched_down]

    # --- Step 2: Find regions appearing in both sequences (anchors) ---
    anchors = set(corrected_upper) & set(corrected_lower)

    upper_indices = [i for i, val in enumerate(corrected_upper) if val in anchors]
    lower_indices = [i for i, val in enumerate(corrected_lower) if val in anchors]

    # Add virtual index at start
    upper_indices.insert(0, -1)
    lower_indices.insert(0, -1)

    upper_diff = [b - a - 1 for a, b in zip(upper_indices, upper_indices[1:])]
    lower_diff = [b - a - 1 for a, b in zip(lower_indices, lower_indices[1:])]

    matching_indices = np.where([a == b and a != 0 for a, b in zip(lower_diff, upper_diff)])[0].tolist()

    # --- Step 3: Collect candidate pairs between anchors ---
    merge_candidates = []
    for idx in matching_indices:
        for n in range(lower_diff[idx]):
            up_val = corrected_upper[upper_indices[idx] + 1 + n]
            down_val = corrected_lower[lower_indices[idx] + 1 + n]
            merge_candidates.append((up_val, down_val))

    return merge_candidates, corrected_upper, corrected_lower



candidates, cu, cl = find_merge_candidates_from_sequences(
    upper_region_sequence, lower_region_sequence, matched_up, matched_down
)

print("Corrected upper:", cu)
print("Corrected lower:", cl)
print("Merge candidates:", candidates)



# Identifying true ring boundaries from the upper and lower sequences
ring_lines = rings_functions.find_ring_lines(rightcells_df, region_to_cells, upper_region_sequence, lower_region_sequence)

lines = []

# Make sure celldata is indexed by label
cells_indexed = celldata.set_index("label")

# Prepare the lines
for region, labels in ring_lines.items():
    if not labels:  # skip empty lists
        continue
    region_cells = cells_indexed.loc[labels]
    coords = list(zip(region_cells["centroid-0"], region_cells["centroid-1"]))
    lines.append(coords)

# Add lines as shapes to the viewer
viewer.add_shapes(
    lines,
    shape_type="path",
    edge_color="red",   # or try "region" for per-region colors
    edge_width=2,
    name="Ring boundaries"
)




def get_region_sequences(new_boundaries, n_lines=10, matched_up=None, matched_down=None):
    """
    Scan horizontal lines across the image and extract ordered region sequences.
    Regions in matched_up or matched_down are removed from the sequences.

    Parameters
    ----------
    new_boundaries : np.ndarray
        2D array where each pixel has a region ID (0 = background).
    n_lines : int
        Number of horizontal probing lines between top and bottom.
    matched_up : set[int], optional
        Regions to exclude from the sequences (upper matches).
    matched_down : set[int], optional
        Regions to exclude from the sequences (lower matches).

    Returns
    -------
    y_positions : list of int
        Vertical positions (rows) where sequences were sampled.
    sequences : list of list[int]
        Region ID sequences (ordered left→right, without duplicates).
    """
    if matched_up is None:
        matched_up = set()
    if matched_down is None:
        matched_down = set()
    excluded = matched_up | matched_down

    height, width = new_boundaries.shape
    step = height // (n_lines + 1)

    sequences = []
    y_positions = []

    for i in range(1, n_lines + 1):
        y = i * step
        row_regions = new_boundaries[y, :]

        # Keep order, remove background (0), duplicates, and excluded regions
        seq = []
        seen = set()
        for region in row_regions:
            if region == 0 or region in excluded:
                continue
            if region not in seen:
                seq.append(int(region))  # cast to int for consistency
                seen.add(region)

        sequences.append(seq)
        y_positions.append(y)

    return y_positions, sequences

y_positions, sequences = get_region_sequences(new_boundaries, n_lines=20, matched_up=matched_up, matched_down=matched_down)

def validate_merge_candidates(sequences, merge_candidates):
    """
    Check that regions in merge candidate pairs do not appear together
    in any sequence. Returns only valid candidates.

    Parameters
    ----------
    sequences : list of list[int]
        The aligned or raw region sequences.
    merge_candidates : list of tuple[int, int]
        Candidate pairs (upper_region, lower_region) to potentially merge.

    Returns
    -------
    valid_candidates : list of tuple[int, int]
        Subset of merge_candidates that are safe to merge.
    invalid_candidates : list of tuple[int, int]
        Candidate pairs that cannot be merged because they appear together.
    """
    valid_candidates = []
    invalid_candidates = []

    for up, down in merge_candidates:
        conflict = any(up in seq and down in seq for seq in sequences)
        if conflict:
            invalid_candidates.append((up, down))
        else:
            valid_candidates.append((up, down))

    return valid_candidates, invalid_candidates

valid_pairs, invalid_pairs = validate_merge_candidates(sequences, candidates)

print("Valid to merge:", valid_pairs)
print("Cannot merge:", invalid_pairs)

from collections import Counter
from collections import defaultdict, deque

from collections import defaultdict, deque

def align_region_sequences(sequences, gap_value=None, upper_seq=None, lower_seq=None):
    """
    Align multiple region sequences while preserving left→right order in each sequence,
    using upper_seq and lower_seq as references for top and bottom.

    Parameters
    ----------
    sequences : list[list[int]]
        Each list is a row sequence of region IDs.

    gap_value : any
        Value to fill when a region is missing in a row.

    upper_seq : list[int], optional
        Sequence of regions at the top of the image (enforced first row).

    lower_seq : list[int], optional
        Sequence of regions at the bottom of the image (enforced last row).

    Returns
    -------
    aligned : list[list[int]]
        Aligned sequences including upper and lower sequences if provided.
    all_regions : list[int]
        Final column order of regions.
    """
    # Step 1: Build a directed graph of precedence relationships
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    all_regions = set()

    # Include the sequences
    for seq in sequences:
        for i, region in enumerate(seq):
            all_regions.add(region)
            for j in range(i+1, len(seq)):
                next_region = seq[j]
                if next_region not in graph[region]:
                    graph[region].add(next_region)
                    in_degree[next_region] += 1
                in_degree.setdefault(region, 0)

    # Include upper_seq as precedence constraints (if given)
    if upper_seq is not None:
        for i, region in enumerate(upper_seq):
            all_regions.add(region)
            for j in range(i+1, len(upper_seq)):
                next_region = upper_seq[j]
                if next_region not in graph[region]:
                    graph[region].add(next_region)
                    in_degree[next_region] += 1
                in_degree.setdefault(region, 0)

    # Include lower_seq as precedence constraints (if given)
    if lower_seq is not None:
        for i, region in enumerate(lower_seq):
            all_regions.add(region)
            for j in range(i+1, len(lower_seq)):
                next_region = lower_seq[j]
                if next_region not in graph[region]:
                    graph[region].add(next_region)
                    in_degree[next_region] += 1
                in_degree.setdefault(region, 0)

    # Step 2: Topological sort to determine column order
    queue = deque([r for r in all_regions if in_degree[r] == 0])
    ordered_regions = []

    while queue:
        r = queue.popleft()
        ordered_regions.append(r)
        for nbr in graph[r]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    # Step 3: Align sequences
    aligned = []
    
    # Add upper_seq as the first row if given
    if upper_seq is not None:
        seq_set = set(upper_seq)
        row = [r if r in seq_set else gap_value for r in ordered_regions]
        aligned.append(row)

    # Align main sequences
    for seq in sequences:
        seq_set = set(seq)
        row = [r if r in seq_set else gap_value for r in ordered_regions]
        aligned.append(row)

    # Add lower_seq as the last row if given
    if lower_seq is not None:
        seq_set = set(lower_seq)
        row = [r if r in seq_set else gap_value for r in ordered_regions]
        aligned.append(row)

    return aligned, ordered_regions


aligned, regions = align_region_sequences(sequences, gap_value=None, upper_seq=cu, lower_seq=cl)

def check_sequence_order_per_line(aligned_sequences, y_positions, new_boundaries):
    """
    Check that regions in each aligned sequence are ordered left-to-right
    based on x-coordinates at each y-position (line).
    
    Parameters
    ----------
    aligned_sequences : list of list[int]
        Aligned sequences of region IDs.
    y_positions : list[int]
        Row indices corresponding to each line.
    new_boundaries : np.ndarray
        2D array with region labels (0 = background).
    """
    for line_idx, (seq, y) in enumerate(zip(aligned_sequences, y_positions)):
        misordered = []
        prev_x = None
        prev_region = None

        for region in seq:
            # get all x positions of this region in this line
            xs = np.where(new_boundaries[y, :] == region)[0]
            if len(xs) == 0:
                continue
            x = xs.mean()  # mean x of this region along the line
            if prev_x is not None and x < prev_x:
                misordered.append((prev_region, prev_x, region, x))
            prev_x = x
            prev_region = region

        if misordered:
            print(f"⚠️ Line {line_idx} (y={y}) is NOT ordered left-to-right:")
            for a, xa, b, xb in misordered:
                print(f"   {a} (x={xa:.2f}) comes before {b} (x={xb:.2f})")
        else:
            print(f"✅ Line {line_idx} (y={y}) is correctly ordered left-to-right")

check_sequence_order_per_line(aligned, y_positions, new_boundaries)


def get_x_positions_per_line(aligned_sequences, y_positions, new_boundaries, pix_to_um=1.0):
    """
    Compute x positions for each region per line from aligned sequences.
    Automatically adds 75 µm offsets at top and bottom if needed.
    
    Parameters
    ----------
    aligned_sequences : list of list[int or None]
        Aligned sequences of region IDs including upper/lower sequences if any.
    y_positions : list[int]
        Original y indices corresponding to the core lines (without upper/lower).
    new_boundaries : np.ndarray
        2D array with region labels (0 = background).
    pix_to_um : float
        Microns per pixel. Default=1.0.
        
    Returns
    -------
    x_positions_matrix : list[list[float or None]]
        Same shape as aligned_sequences, x positions for each region in each line.
    """
    # --- detect if we have upper and lower sequences ---
    n_rows = len(aligned_sequences)
    n_y = len(y_positions)

    # if we’re missing two y positions but have two extra sequences, add top and bottom automatically
    if n_rows == n_y + 2:
        height = new_boundaries.shape[0]
        offset_px = int(round(75 / pix_to_um))
        y_top = offset_px
        y_bottom = height - offset_px
        y_positions = [y_top] + y_positions + [y_bottom]

    elif n_rows == n_y + 1:  # just one extra line
        height = new_boundaries.shape[0]
        offset_px = int(round(75 / pix_to_um))
        # decide whether to add at top or bottom based on which one is missing
        if y_positions[0] > offset_px:  # assume top missing
            y_positions = [offset_px] + y_positions
        else:  # assume bottom missing
            y_positions = y_positions + [height - offset_px]

    # --- now compute x positions ---
    x_positions_matrix = []
    for seq, y in zip(aligned_sequences, y_positions):
        line_positions = []
        for region in seq:
            if region is None:
                line_positions.append(None)
                continue
            xs = np.where(new_boundaries[y, :] == region)[0]
            if len(xs) == 0:
                line_positions.append(None)
            else:
                line_positions.append(xs.mean())  # mean x of this region along the line
        x_positions_matrix.append(line_positions)

    return x_positions_matrix

x_positions_matrix = get_x_positions_per_line(
    aligned,  # including upper and lower sequences
    y_positions,        # only the original y_positions without borders
    new_boundaries,
    pix_to_um=pix_to_um
)


for row in aligned:
    print(row)

print("Conflicts:", conflicts)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_alignment(aligned, region_order, names=None):
    """
    Visualize alignment as a matrix with a unique color per region.
    
    Parameters:
    - aligned: list of lists, each sublist is a sequence of regions (None for gaps)
    - region_order: list of all regions in the alignment
    - names: optional list of sequence names
    """
    if names is None:
        names = [f"Seq{i+1}" for i in range(len(aligned))]

    n_seq = len(aligned)
    n_cols = len(region_order)

    # Map each region to an integer
    region_to_int = {region: i+1 for i, region in enumerate(region_order)}
    
    # Build integer matrix
    data = np.zeros((n_seq, n_cols), dtype=int)
    for i, row in enumerate(aligned):
        for j, region in enumerate(row):
            if region is not None:
                data[i, j] = region_to_int[region]

    # Create a colormap: 0 (gaps) will be white, regions get unique colors
    n_regions = len(region_order)
    cmap_colors = plt.cm.gist_ncar(np.linspace(0, 1, n_regions))
    
    # Shuffle colors
    rng = np.random.default_rng(4)
    shuffled_indices = rng.permutation(n_regions)
    shuffled_colors = cmap_colors[shuffled_indices]
    cmap = ListedColormap(np.vstack(([1,1,1,1], shuffled_colors)))  # 0 = white

    fig, ax = plt.subplots(figsize=(n_cols*0.5, n_seq*0.5))
    im = ax.imshow(data, cmap=cmap, aspect='auto')

    # Add region IDs as text
    for i in range(n_seq):
        for j in range(n_cols):
            val = data[i, j]
            if val != 0:
                region_id = region_order[val-1]
                ax.text(j, i, str(region_id), ha='center', va='center', fontsize=7, color='black')

    ax.set_yticks(range(n_seq))
    ax.set_yticklabels(names)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(region_order, rotation=90)
    ax.set_xlim(-0.5, n_cols-0.5)
    ax.set_ylim(n_seq-0.5, -0.5)  # invert y-axis
    plt.tight_layout()
    plt.show()


plot_alignment(aligned, regions, names=None)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_alignment_with_xpos(aligned, x_positions_matrix, region_order, names=None):
    """
    Visualize alignment as a matrix with a unique color per region,
    but display the x-position value inside each cell instead of the region ID.
    
    Parameters
    ----------
    aligned : list[list[int or None]]
        Each sublist is a sequence of regions (None for gaps).
    x_positions_matrix : list[list[float or None]]
        Same shape as `aligned`, contains x positions of each region.
    region_order : list[int]
        List of all regions in the alignment (defines color mapping).
    names : list[str], optional
        Sequence names.
    """
    if names is None:
        names = [f"Seq{i+1}" for i in range(len(aligned))]

    n_seq = len(aligned)
    n_cols = len(region_order)

    # Map each region to an integer
    region_to_int = {region: i+1 for i, region in enumerate(region_order)}

    # Build integer matrix for colors
    data = np.zeros((n_seq, n_cols), dtype=int)
    for i, row in enumerate(aligned):
        for j, region in enumerate(row):
            if region is not None:
                data[i, j] = region_to_int[region]

    # Create a colormap: 0 (gaps) will be white, regions get unique colors
    n_regions = len(region_order)
    cmap_colors = plt.cm.gist_ncar(np.linspace(0, 1, n_regions))

    rng = np.random.default_rng(4)
    shuffled_indices = rng.permutation(n_regions)
    shuffled_colors = cmap_colors[shuffled_indices]
    cmap = ListedColormap(np.vstack(([1, 1, 1, 1], shuffled_colors)))  # 0 = white

    fig, ax = plt.subplots(figsize=(n_cols * 0.5, n_seq * 0.5))
    im = ax.imshow(data, cmap=cmap, aspect='auto')

    # Add x-position text (rounded)
    for i in range(n_seq):
        for j in range(n_cols):
            val = data[i, j]
            xpos = x_positions_matrix[i][j] if i < len(x_positions_matrix) and j < len(x_positions_matrix[i]) else None
            if val != 0 and xpos is not None:
                ax.text(j, i, f"{xpos:.1f}", ha='center', va='center', fontsize=6, color='black')

    ax.set_yticks(range(n_seq))
    ax.set_yticklabels(names)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(region_order, rotation=90)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_seq - 0.5, -0.5)  # invert y-axis
    plt.tight_layout()
    plt.show()

plot_alignment_with_xpos(aligned, x_positions_matrix, regions)


def compute_gap_distances_matrix(x_positions_matrix):
    """
    Compute the horizontal distance from each cell to the next
    non-None cell in the same line.

    Parameters
    ----------
    x_positions_matrix : list of list[float or None]
        x-coordinate of each region at each line/column.

    Returns
    -------
    gap_matrix : list of list[float or None]
        Same shape as input. Each cell contains:
            distance to next non-None region on the right,
            or None if no such region exists.
    """
    n_rows = len(x_positions_matrix)
    n_cols = len(x_positions_matrix[0])

    gap_matrix = []
    for i in range(n_rows):
        row_gaps = [None] * n_cols
        x_row = x_positions_matrix[i]

        for j in range(n_cols):
            x_current = x_row[j]
            if x_current is None:
                continue
            # find next non-None x to the right
            for k in range(j+1, n_cols):
                x_next = x_row[k]
                if x_next is not None:
                    row_gaps[j] = x_next - x_current
                    break
        gap_matrix.append(row_gaps)

    return gap_matrix

gap_matrix = compute_gap_distances_matrix(x_positions_matrix)

plot_alignment_with_xpos(aligned, gap_matrix, regions)

def compute_prev_gap_distances_matrix(x_positions_matrix):
    """
    Compute the horizontal distance from each cell to the previous
    non-None cell in the same line (to the left).

    Parameters
    ----------
    x_positions_matrix : list of list[float or None]
        x-coordinate of each region at each line/column.

    Returns
    -------
    prev_gap_matrix : list of list[float or None]
        Same shape as input. Each cell contains:
            distance to previous non-None region on the left,
            or None if no such region exists.
    """
    n_rows = len(x_positions_matrix)
    n_cols = len(x_positions_matrix[0])

    prev_gap_matrix = []
    for i in range(n_rows):
        row_gaps = [None] * n_cols
        x_row = x_positions_matrix[i]

        for j in range(n_cols):
            x_current = x_row[j]
            if x_current is None:
                continue
            # find previous non-None x to the left
            for k in range(j-1, -1, -1):  # go leftwards
                x_prev = x_row[k]
                if x_prev is not None:
                    row_gaps[j] = x_current - x_prev
                    break
        prev_gap_matrix.append(row_gaps)

    return prev_gap_matrix

# usage
prev_gap_matrix = compute_prev_gap_distances_matrix(x_positions_matrix)

plot_alignment_with_xpos(aligned, prev_gap_matrix, regions)



def reorder_columns_by_x(region_matrix, x_matrix):
    """
    Reorder the columns of region_matrix and x_matrix
    by the mean x position across rows.

    Parameters
    ----------
    region_matrix : list of list[int or None]
        Region IDs per row/column.
    x_matrix : list of list[float or None]
        x positions corresponding to each region ID.

    Returns
    -------
    region_matrix_sorted : list of list[int or None]
        Columns reordered by x position.
    x_matrix_sorted : list of list[float or None]
        Columns reordered by x position.
    """
    n_rows = len(region_matrix)
    n_cols = len(region_matrix[0])

    # Convert to np.array for convenience
    x_arr = np.array(x_matrix, dtype=float)
    region_arr = np.array(region_matrix, dtype=object)

    # Compute mean x per column ignoring None/nan
    mean_x_per_col = []
    for j in range(n_cols):
        col_vals = [x_matrix[i][j] for i in range(n_rows) if x_matrix[i][j] is not None]
        mean_x = np.nan if len(col_vals) == 0 else np.mean(col_vals)
        mean_x_per_col.append(mean_x)

    # Replace nans with large number to push them to the end
    mean_x_clean = [val if not np.isnan(val) else np.inf for val in mean_x_per_col]

    # Get sort order
    sort_indices = np.argsort(mean_x_clean)

    # Reorder both matrices
    region_matrix_sorted = [[row[j] for j in sort_indices] for row in region_matrix]
    x_matrix_sorted = [[row[j] for j in sort_indices] for row in x_matrix]

    return region_matrix_sorted, x_matrix_sorted

region_matrix_sorted, x_matrix_sorted = reorder_columns_by_x(aligned, x_positions_matrix)

plot_alignment_with_xpos(region_matrix_sorted, x_matrix_sorted, regions)

gap_matrix = compute_gap_distances_matrix(x_matrix_sorted)

plot_alignment_with_xpos(region_matrix_sorted, gap_matrix, regions)

prev_gap_matrix = compute_prev_gap_distances_matrix(x_matrix_sorted)

plot_alignment_with_xpos(region_matrix_sorted, prev_gap_matrix, regions)



def find_merge_by_x(region_matrix, x_matrix, x_threshold=10):
    """
    Find pairs of regions (by their IDs) where the bottom x of the upper region
    is within ±x_threshold of the top x of the lower region.

    Parameters
    ----------
    region_matrix : list of list[int or None]
        Matrix of region IDs.
    x_matrix : list of list[float or None]
        Matrix of x positions corresponding to region IDs.
    x_threshold : float
        Fixed allowed difference in x between bottom of upper region and top of lower region.

    Returns
    -------
    merge_candidates : list of tuple(int, int)
        List of region ID pairs that can be merged.
    """
    n_rows = len(region_matrix)
    n_cols = len(region_matrix[0])
    x_arr = np.array(x_matrix, dtype=object)
    reg_arr = np.array(region_matrix, dtype=object)
    merge_candidates = []

    for j in range(n_cols - 1):
        # Skip columns that overlap vertically
        overlap = any(reg_arr[i, j] is not None and reg_arr[i, j+1] is not None
                      for i in range(n_rows))
        if overlap:
            continue

        # Rows with values in each column
        rows1 = [i for i in range(n_rows) if x_arr[i, j] is not None]
        rows2 = [i for i in range(n_rows) if x_arr[i, j+1] is not None]
        if not rows1 or not rows2:
            continue

        top1, bottom1 = rows1[0], rows1[-1]
        top2, bottom2 = rows2[0], rows2[-1]

        # Determine which column is above
        if top1 < top2:  # col j above col j+1
            x_bottom_upper = x_arr[bottom1, j]
            x_top_lower = x_arr[top2, j+1]
            region_upper = reg_arr[top1, j]
            region_lower = reg_arr[top2, j+1]
        else:  # col j+1 above col j
            x_bottom_upper = x_arr[bottom2, j+1]
            x_top_lower = x_arr[top1, j]
            region_upper = reg_arr[top2, j+1]
            region_lower = reg_arr[top1, j]

        if x_bottom_upper is None or x_top_lower is None:
            continue

        # Check if bottom x is inside top x ± threshold
        if abs(x_bottom_upper - x_top_lower) <= x_threshold:
            merge_candidates.append((region_upper, region_lower))

    return merge_candidates


def find_mergeable_columns_auto(region_matrix, x_matrix, gap_next_matrix, gap_prev_matrix,
                                gap_ratio_thresh=0.5, thresh_factor=0.5,
                                use_median=True, max_x_thresh=200):
    n_rows = len(region_matrix)
    n_cols = len(region_matrix[0])

    x_arr = np.array(x_matrix, dtype=object)
    gap_next_arr = np.array(gap_next_matrix, dtype=object)
    gap_prev_arr = np.array(gap_prev_matrix, dtype=object)

    merge_candidates = []

    for j in range(n_cols-1):
        # Check no overlap per row
        overlap = any(region_matrix[i][j] is not None and region_matrix[i][j+1] is not None
                      for i in range(n_rows))
        if overlap:
            continue

        # Find rows with values in each column
        rows1 = [i for i in range(n_rows) if x_arr[i,j] is not None]
        rows2 = [i for i in range(n_rows) if x_arr[i,j+1] is not None]
        if not rows1 or not rows2:
            continue

        top1, bottom1 = rows1[0], rows1[-1]
        top2, bottom2 = rows2[0], rows2[-1]
        x_top1, x_bottom1 = x_arr[top1,j], x_arr[bottom1,j]
        x_top2, x_bottom2 = x_arr[top2,j+1], x_arr[bottom2,j+1]

        # gather all gaps
        gaps_next1 = [gap_next_arr[i,j] for i in rows1 if gap_next_arr[i,j] is not None]
        gaps_next2 = [gap_next_arr[i,j+1] for i in rows2 if gap_next_arr[i,j+1] is not None]
        gaps_prev1 = [gap_prev_arr[i,j] for i in rows1 if gap_prev_arr[i,j] is not None]
        gaps_prev2 = [gap_prev_arr[i,j+1] for i in rows2 if gap_prev_arr[i,j+1] is not None]
        if not gaps_next1 or not gaps_next2 or not gaps_prev1 or not gaps_prev2:
            continue

        # define x threshold from min of all gaps
        min_all_gaps = min(min(gaps_next1), min(gaps_next2),
                           min(gaps_prev1), min(gaps_prev2))
        x_thresh = thresh_factor * min_all_gaps
        if x_thresh > max_x_thresh:
            x_thresh = max_x_thresh

        # ✅ Compare bottom of upper region to top of lower region
        if top1 < top2:
            x_upper_bottom = x_bottom2
            x_lower_top = x_top1
        else:
            x_upper_bottom = x_bottom1
            x_lower_top = x_top2

        x_distance = abs(x_upper_bottom - x_lower_top) if (
            x_upper_bottom is not None and x_lower_top is not None) else np.inf

        if x_distance > x_thresh:
            continue

        # Compare distance to next and prev symmetrically
        if use_median:
            g_next1 = np.median(gaps_next1)
            g_next2 = np.median(gaps_next2)
            g_prev1 = np.median(gaps_prev1)
            g_prev2 = np.median(gaps_prev2)
        else:
            g_next1 = np.min(gaps_next1)
            g_next2 = np.min(gaps_next2)
            g_prev1 = np.min(gaps_prev1)
            g_prev2 = np.min(gaps_prev2)

        ratio_next = abs(g_next1 - g_next2) / max(g_next1, g_next2)
        ratio_prev = abs(g_prev1 - g_prev2) / max(g_prev1, g_prev2)

        if ratio_next > gap_ratio_thresh or ratio_prev > gap_ratio_thresh:
            continue

        merge_candidates.append((j, j+1))

    return merge_candidates


merge_candidates = find_mergeable_columns_auto(
    region_matrix_sorted,
    x_matrix_sorted,
    gap_matrix,
    prev_gap_matrix,
    gap_ratio_thresh=0.5,
    thresh_factor=0.5,
    use_median=True
)
print("Merge candidates:", merge_candidates)

def compact_nonoverlapping_columns(region_matrix):
    """
    Compact columns by merging non-overlapping columns into the leftmost possible
    without changing row order.

    Parameters
    ----------
    region_matrix : list of list
        Matrix of region IDs or None (rows x cols).

    Returns
    -------
    merged_matrix : list of list
        Same shape but with fewer columns where possible.
    """
    import copy
    matrix = copy.deepcopy(region_matrix)
    n_rows = len(matrix)
    n_cols = len(matrix[0])

    # Work with columns
    cols = [[matrix[r][c] for r in range(n_rows)] for c in range(n_cols)]
    used = [False] * n_cols
    merged_cols = []

    for i in range(n_cols):
        if used[i]:
            continue
        col_i = cols[i][:]  # copy
        for j in range(i + 1, n_cols):
            if used[j]:
                continue
            col_j = cols[j]

            # Check overlap row by row
            overlap = any(a is not None and b is not None for a, b in zip(col_i, col_j))
            if not overlap:
                # Fill col_i’s None positions with col_j values row by row
                for r in range(n_rows):
                    if col_i[r] is None and col_j[r] is not None:
                        col_i[r] = col_j[r]
                used[j] = True
        merged_cols.append(col_i)
        used[i] = True

    # Convert back to rows × cols
    merged_matrix = [[merged_cols[c][r] for c in range(len(merged_cols))]
                     for r in range(n_rows)]
    return merged_matrix


merged = compact_nonoverlapping_columns(region_matrix_sorted)

plot_alignment(merged, regions, names=None)

from collections import defaultdict, deque

def align_region_sequences_with_merges(sequences, valid_pairs=None, gap_value=None, upper_seq=None, lower_seq=None):
    """
    Align region sequences, preserving left→right order, and allowing valid merge candidates
    to occupy the same column.

    Parameters
    ----------
    sequences : list[list[int]]
        Each list is a row sequence of region IDs.
    valid_pairs : set[tuple[int,int]], optional
        Set of allowed merge candidate pairs, e.g. {(3,5), (7,8)}.
        Both (a,b) and (b,a) are considered equivalent.
    gap_value : any
        Value to fill when a region is missing in a row.
    upper_seq : list[int], optional
        Sequence of regions at the top of the image (enforced first row).
    lower_seq : list[int], optional
        Sequence of regions at the bottom of the image (enforced last row).

    Returns
    -------
    aligned : list[list[tuple|int]]
        Aligned sequences. Each cell is either a single region ID or a tuple of merged IDs.
    ordered_columns : list[tuple|int]
        Final column representation (with merges).
    """
    if valid_pairs is None:
        valid_pairs = set()

    # Step 1: Collect all precedence constraints
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    all_regions = set()

    def add_constraints(seq):
        for i, region in enumerate(seq):
            all_regions.add(region)
            for j in range(i+1, len(seq)):
                next_region = seq[j]
                if next_region not in graph[region]:
                    graph[region].add(next_region)
                    in_degree[next_region] += 1
                in_degree.setdefault(region, 0)

    for seq in sequences:
        add_constraints(seq)
    if upper_seq is not None:
        add_constraints(upper_seq)
    if lower_seq is not None:
        add_constraints(lower_seq)

    # Step 2: Topological sort to get base order
    queue = deque([r for r in all_regions if in_degree[r] == 0])
    ordered_regions = []
    while queue:
        r = queue.popleft()
        ordered_regions.append(r)
        for nbr in graph[r]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    # Step 3: Merge valid candidates in column representation
    merged_columns = []
    skip = set()
    for i, r in enumerate(ordered_regions):
        if r in skip:
            continue
        # look ahead to see if next can merge
        if i+1 < len(ordered_regions):
            nxt = ordered_regions[i+1]
            if (r, nxt) in valid_pairs or (nxt, r) in valid_pairs:
                merged_columns.append((r, nxt))
                skip.add(nxt)
                continue
        merged_columns.append(r)

    # Step 4: Align sequences
    def build_row(seq):
        seq_set = set(seq)
        row = []
        for col in merged_columns:
            if isinstance(col, tuple):
                # merged column: check if any are present
                present = tuple(sorted([c for c in col if c in seq_set]))
                row.append(present if present else gap_value)
            else:
                row.append(col if col in seq_set else gap_value)
        return row

    aligned = []
    if upper_seq is not None:
        aligned.append(build_row(upper_seq))
    for seq in sequences:
        aligned.append(build_row(seq))
    if lower_seq is not None:
        aligned.append(build_row(lower_seq))

    return aligned, merged_columns

aligned_m, regions_m = align_region_sequences_with_merges(sequences, valid_pairs=valid_pairs, gap_value=None, upper_seq=cu, lower_seq=cl)

plot_alignment(aligned_m, regions_m, names=None)

from collections import Counter

def fill_columns(aligned_matrix, merge_candidates=set(), min_fraction=0.7):
    """
    Fill None values in aligned matrix columns based on majority values.
    
    Parameters
    ----------
    aligned_matrix : list[list[int | None]]
        Aligned sequences matrix (rows = sequences, columns = regions).
    merge_candidates : set of tuples
        Pairs of regions that may be merged, we avoid filling None with these.
    min_fraction : float
        Minimum fraction of occurrences for a value to fill None safely.
    
    Returns
    -------
    filled_matrix : list[list[int]]
        New matrix with some None values filled where safe.
    """
    import copy
    filled_matrix = copy.deepcopy(aligned_matrix)
    n_rows = len(aligned_matrix)
    n_cols = len(aligned_matrix[0])

    for col_idx in range(n_cols):
        col_vals = [filled_matrix[row][col_idx] for row in range(n_rows) if filled_matrix[row][col_idx] is not None]
        
        # Skip column if empty
        if not col_vals:
            continue
        
        # Remove values that appear in merge_candidates
        candidate_values = set()
        for pair in merge_candidates:
            candidate_values.update(pair)
        safe_vals = [v for v in col_vals if v not in candidate_values]

        if not safe_vals:
            continue

        # Count occurrences
        counts = Counter(safe_vals)
        most_common_val, count = counts.most_common(1)[0]
        fraction = count / n_rows

        if fraction >= min_fraction:
            # Replace None with the most common safe value
            for row in range(n_rows):
                if filled_matrix[row][col_idx] is None:
                    filled_matrix[row][col_idx] = most_common_val

    return filled_matrix

filled = fill_columns(aligned, candidates, 0.7)

plot_alignment(filled, regions, names=None)

from typing import List, Tuple, Any, Union

def remove_weak_columns(
    aligned: List[List[Any]],
    columns: List[Union[int, Tuple[int, ...]]],
    merge_candidates: List[Tuple[int, int]],
    gap_value=None,
    min_complete_fraction: float = 1.0,
    require_both_neighbors: bool = True,
):
    """
    Remove columns that:
      - have only a single distinct non-gap value across rows, AND
      - are strictly between two 'strong' neighbor columns.

    A "strong" column is:
      - a merged column (tuple in `columns`), OR
      - a column containing any member of `merge_candidates`, OR
      - a column whose fraction of non-gap entries >= min_complete_fraction.
    """
    if not aligned or not columns:
        return aligned, columns, {"removed_indices": [], "removed_columns": []}

    n_rows = len(aligned)
    n_cols = len(columns)
    assert all(len(row) == n_cols for row in aligned), "All rows must have same length"

    # Precompute stats per column
    col_non_gap_counts = []
    col_non_gap_sets = []
    for j in range(n_cols):
        non_gap_entries = [aligned[r][j] for r in range(n_rows) if aligned[r][j] != gap_value]
        col_non_gap_counts.append(len(non_gap_entries))
        col_non_gap_sets.append(set(non_gap_entries))

    # Flatten merge candidates for easier lookup
    merge_region_set = {r for pair in merge_candidates for r in pair}

    def is_strong_col(idx):
        col_def = columns[idx]
        # merged column (tuple) is always strong
        if isinstance(col_def, (tuple, list)):
            return True
        # any member of merge_candidates is strong
        if col_def in merge_region_set:
            return True
        # otherwise check coverage
        coverage = col_non_gap_counts[idx] / float(n_rows) if n_rows > 0 else 0.0
        return coverage >= float(min_complete_fraction)

    to_remove = [False] * n_cols
    for j in range(n_cols):
        if isinstance(columns[j], (tuple, list)):
            continue
        unique_sets = col_non_gap_sets[j]
        if len(unique_sets) != 1 or col_non_gap_counts[j] == 0:
            continue
        left_idx, right_idx = j - 1, j + 1
        if left_idx < 0 or right_idx >= n_cols:
            continue
        left_strong, right_strong = is_strong_col(left_idx), is_strong_col(right_idx)
        cond = (left_strong and right_strong) if require_both_neighbors else (left_strong or right_strong)
        if cond:
            to_remove[j] = True

    removed_indices = [i for i, rem in enumerate(to_remove) if rem]
    removed_columns = [columns[i] for i in removed_indices]

    if not removed_indices:
        return aligned, columns, {"removed_indices": [], "removed_columns": []}

    keep_mask = [not rem for rem in to_remove]
    new_columns = [col for col, keep in zip(columns, keep_mask) if keep]
    new_aligned = [[val for val, keep in zip(row, keep_mask) if keep] for row in aligned]

    return new_aligned, new_columns, {
        "removed_indices": removed_indices,
        "removed_columns": removed_columns,
    }

def remove_singleton_columns(region_matrix):
    """
    Remove columns that contain only one non-None value.

    Parameters
    ----------
    region_matrix : list of list
        Matrix of region IDs or None (rows x cols).

    Returns
    -------
    cleaned_matrix : list of list
        Same number of rows, fewer columns.
    """
    n_rows = len(region_matrix)
    n_cols = len(region_matrix[0])
    
    # Identify columns to keep
    keep_cols = []
    for c in range(n_cols):
        col = [region_matrix[r][c] for r in range(n_rows)]
        non_none_count = sum(1 for val in col if val is not None)
        if non_none_count > 1:
            keep_cols.append(c)
    
    # Build cleaned matrix
    cleaned_matrix = [[region_matrix[r][c] for c in keep_cols] for r in range(n_rows)]
    return cleaned_matrix


aligned_clean, merged_columns_clean, info = remove_weak_columns(
    filled,
    regions,
    merge_candidates=valid_pairs,
    gap_value=None
)

plot_alignment(aligned_clean, merged_columns_clean, names=None)

from collections import Counter

def normalize_pair(pair):
    """Return the pair as an ordered tuple (smaller first)."""
    a, b = pair
    return (min(a, b), max(a, b))

from collections import Counter

def find_reliable_pairs(list1, list2, match_results, min_count=2):
    """
    Combine pairs from two lists and match_results dict, 
    return pairs that appear at least min_count times.

    Parameters
    ----------
    list1, list2 : list of (regionA, regionB)
        Candidate region pairs from two sources.
    match_results : dict
        Dictionary with keys (1..4) for polynomial degrees, 
        each containing 'mutual_matches' and 'one_sided_matches'.
    min_count : int
        Minimum number of sources where a pair must appear.

    Returns
    -------
    reliable_pairs : list of tuple
        Pairs appearing at least min_count times.
    mutual_pairs : list of tuple
        All mutual pairs extracted from match_results.
    one_sided_pairs : list of tuple
        All one-sided pairs extracted from match_results.
    """

    # ---- 1) extract mutual & one-sided pairs from match_results ----
    mutual_pairs = []
    one_sided_pairs = []

    for deg, matches in match_results.items():
        if 'mutual_matches' in matches:
            for r1, r2 in matches['mutual_matches'].items():
                r1 = int(r1) if isinstance(r1, (np.integer,)) else r1
                r2 = int(r2) if isinstance(r2, (np.integer,)) else r2
                mutual_pairs.append(tuple(sorted((r1, r2))))

        if 'one_sided_matches' in matches:
            for r1, r2 in matches['one_sided_matches'].items():
                r1 = int(r1) if isinstance(r1, (np.integer,)) else r1
                r2 = int(r2) if isinstance(r2, (np.integer,)) else r2
                one_sided_pairs.append(tuple(sorted((r1, r2))))

    # deduplicate inside each category
    mutual_pairs = list(set(mutual_pairs))
    one_sided_pairs = list(set(one_sided_pairs))

    # ---- 2) normalize list1 and list2 as well ----
    list1_clean = [tuple(sorted((int(a), int(b)))) for a,b in list1]
    list2_clean = [tuple(sorted((int(a), int(b)))) for a,b in list2]

    # ---- 3) count how many times each pair appears across all sources ----
    all_pairs = list1_clean + list2_clean + mutual_pairs + one_sided_pairs
    counts = Counter(all_pairs)

    reliable_pairs = [p for p, c in counts.items() if c >= min_count]

    return reliable_pairs, mutual_pairs, one_sided_pairs



def merge_region_columns(region_matrix, reliable_pairs):
    """
    Merge columns for reliable pairs and fill None values.

    Parameters
    ----------
    region_matrix : 2D array-like
        Matrix of region IDs or None values.
    reliable_pairs : list of tuple
        Pairs of region IDs to merge.

    Returns
    -------
    merged_matrix : np.ndarray
        Updated matrix with merged columns.
    """

    # convert to numpy array if needed
    region_matrix = np.array(region_matrix, dtype=object)

    # Get all unique region IDs present
    all_regions = np.unique(region_matrix[region_matrix != None])

    # Build a mapping from region_id to column indices
    region_to_cols = {}
    for j in range(region_matrix.shape[1]):
        col_vals = region_matrix[:, j]
        non_none = col_vals[col_vals != None]
        if len(non_none) > 0:
            region_id = non_none[0]  # assume first non-None is the region id
            region_to_cols.setdefault(region_id, []).append(j)

    merged = region_matrix.copy()

    for r1, r2 in reliable_pairs:
        # find the columns corresponding to r1 and r2
        cols_r1 = region_to_cols.get(r1, [])
        cols_r2 = region_to_cols.get(r2, [])
        if not cols_r1 or not cols_r2:
            continue  # skip if either region not found

        # take the first column of r1 and r2 (most common)
        c1 = cols_r1[0]
        c2 = cols_r2[0]

        # for each row: if c1 is None but c2 not None, copy c2; vice versa
        for i in range(merged.shape[0]):
            v1 = merged[i, c1]
            v2 = merged[i, c2]
            if v1 is None and v2 is not None:
                merged[i, c1] = v2
            elif v2 is None and v1 is not None:
                merged[i, c2] = v1

        merged[:, c1] = np.where(merged[:, c1] == None, merged[:, c2], merged[:, c1])
        merged[:, c2] = None

    # Remove completely empty columns
    non_empty_cols = [j for j in range(merged.shape[1]) if any(merged[:, j] != None)]
    merged_compact = merged[:, non_empty_cols]


    return merged_compact

import numpy as np
from collections import defaultdict

def merge_region_chains(region_matrix, reliable_pairs):
    """
    Merge columns based on reliable pairs, including chains of regions.
    Fill None values and remove completely empty columns.

    Parameters
    ----------
    region_matrix : 2D array-like
        Matrix of region IDs or None values.
    reliable_pairs : list of tuple
        Pairs of region IDs to merge.

    Returns
    -------
    merged_matrix : np.ndarray
        Updated matrix with merged chains and empty columns removed.
    """

    # Step 1: Build connected components of regions
    parent = {}

    def find(x):
        # Union-find with path compression
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for r1, r2 in reliable_pairs:
        union(r1, r2)

    # Map each region to its component
    components = defaultdict(list)
    for r in parent:
        components[find(r)].append(r)

    # Step 2: Convert to numpy array for easy manipulation
    merged = np.array(region_matrix, dtype=object)

    # Step 3: For each component, merge all columns
    region_to_cols = {}
    n_rows, n_cols = merged.shape
    for j in range(n_cols):
        col_vals = merged[:, j]
        non_none = col_vals[col_vals != None]
        if len(non_none) > 0:
            region_id = non_none[0]
            region_to_cols.setdefault(region_id, []).append(j)

    for comp_regions in components.values():
        # Find all columns involved
        cols = []
        for r in comp_regions:
            cols.extend(region_to_cols.get(r, []))
        if not cols:
            continue
        cols = sorted(cols)
        main_col = cols[0]

        # Merge all other columns into main_col
        for c in cols[1:]:
            for i in range(n_rows):
                if merged[i, main_col] is None and merged[i, c] is not None:
                    merged[i, main_col] = merged[i, c]
            merged[:, c] = None  # blank merged columns

    # Step 4: Remove completely empty columns
    non_empty_cols = [j for j in range(n_cols) if any(merged[:, j] != None)]
    merged_compact = merged[:, non_empty_cols]

    return merged_compact


def clean_and_validate_matrix(aligned_matrix, valid_pairs=set()):
    """
    Remove None values from rows, check row length consistency,
    and validate column values with respect to valid merge pairs.
    
    Parameters
    ----------
    aligned_matrix : list[list[int | None]]
        Aligned sequences matrix (rows = sequences, columns = regions).
    valid_pairs : set of tuple(int, int)
        Allowed pairs of region IDs that can coexist in the same column.
    
    Returns
    -------
    cleaned_matrix : list[list[int]]
        Updated matrix without None values, validated for consistency.
    
    Raises
    ------
    ValueError : if row lengths differ after cleaning or if a column
                 contains invalid multiple region IDs.
    """
    # Step 1: remove None from each row
    cleaned = [[val for val in row if val is not None] for row in aligned_matrix]

    # Step 2: check that all rows have same length
    row_lengths = {len(r) for r in cleaned}
    if len(row_lengths) != 1:
        raise ValueError(f"Inconsistent row lengths after cleaning: {row_lengths}")

    n_rows = len(cleaned)
    n_cols = len(cleaned[0])

    # Step 3: validate columns
    for col_idx in range(n_cols):
        col_vals = {cleaned[row][col_idx] for row in range(n_rows)}
        
        if len(col_vals) > 1:
            # must correspond to a valid pair
            pairs = {(a, b) for a in col_vals for b in col_vals if a != b}
            valid = any(p in valid_pairs or (p[::-1] in valid_pairs) for p in pairs)
            if not valid:
                raise ValueError(
                    f"Invalid column {col_idx}: contains {col_vals}, "
                    f"which is not in valid_pairs"
                )

    return cleaned

cleaned = clean_and_validate_matrix(filled, valid_pairs)


def find_missing_regions(new_boundaries, aligned_matrix):
    """
    Check which regions in the labeled image are not represented in the matrix.
    
    Parameters
    ----------
    new_boundaries : np.ndarray
        2D array where each pixel has a region ID (0 = background).
    aligned_matrix : list[list[int]]
        Final aligned matrix of region IDs (no None values).
    
    Returns
    -------
    missing : set of int
        Region IDs that appear in the labeled image but not in the matrix.
    """
    # All regions in the labeled image (excluding background = 0)
    image_regions = set(np.unique(new_boundaries)) - {0}
    
    # All regions in the matrix
    matrix_regions = {val for row in aligned_matrix for val in row}
    
    # Missing = regions in image but not in matrix
    missing = image_regions - matrix_regions
    
    return missing

missing = find_missing_regions(new_boundaries, cleaned)
print("Missing regions:", missing)


    # Insert matched_up into lower sequence at the right relative position
    for region in matched_up:
        if region not in lower_region_sequence and region in upper_region_sequence:
            idx = upper_region_sequence.index(region)

        # Find the closest neighbor already present in lower_sequence
            inserted = False
        # Try to place before the next region that exists in lower
            for next_region in upper_region_sequence[idx+1:]:
                if next_region in lower_region_sequence:
                    insert_idx = lower_region_sequence.index(next_region)
                    lower_region_sequence.insert(insert_idx, region)
                    inserted = True
                    break
        # If no "next" neighbor, append at the end
            if not inserted:
                lower_region_sequence.append(region)

# Insert matched_down into upper sequence at the right relative position
    for region in matched_down:
        if region not in upper_region_sequence and region in lower_region_sequence:
            idx = lower_region_sequence.index(region)

            inserted = False
            for next_region in lower_region_sequence[idx+1:]:
                if next_region in upper_region_sequence:
                    insert_idx = upper_region_sequence.index(next_region)
                    upper_region_sequence.insert(insert_idx, region)
                    inserted = True
                    break
            if not inserted:
                upper_region_sequence.append(region)

    # ---- ORDER CHECK ----
    def check_sequence_order(sequence, region_coords, name="sequence"):
        misordered = []
        for a, b in zip(sequence, sequence[1:]):
            xa = region_coords.get(a, None)
            xb = region_coords.get(b, None)
            if xa is not None and xb is not None and xa > xb:
                misordered.append((a, xa, b, xb))
        if misordered:
            print(f"⚠️ {name} is NOT ordered left-to-right:")
            for a, xa, b, xb in misordered:
                xa_str = f"{xa:.2f}" if xa is not None else "NA"
                xb_str = f"{xb:.2f}" if xb is not None else "NA"
                print(f"   {a} (x={xa_str}) comes before {b} (x={xb_str})")
        else:
            print(f"✅ {name} is correctly ordered left-to-right.")

    # Run check for both sequences
    upper_x = dict(zip(upper_region_coords["region"], upper_region_coords["centroid-1"]))
    lower_x = dict(zip(lower_region_coords["region"], lower_region_coords["centroid-1"]))
    check_sequence_order(upper_region_sequence, upper_x, name="Upper")
    check_sequence_order(lower_region_sequence, lower_x, name="Lower")





import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx

def merge_columns(region_mat, x_mat, dist_next, dist_prev, 
                  max_x_diff=50, max_dist_diff=5, min_coverage=0.3):
    """
    Merge columns in matrices that likely represent the same region.
    """
    n_rows, n_cols = region_mat.shape
    # Filter columns with very low coverage first
    coverage = np.sum(region_mat>0, axis=0) / n_rows
    keep_cols = coverage >= min_coverage
    region_mat = region_mat[:, keep_cols]
    x_mat = x_mat[:, keep_cols]
    dist_next = dist_next[:, keep_cols]
    dist_prev = dist_prev[:, keep_cols]
    n_cols = region_mat.shape[1]

    # Build similarity graph
    G = nx.Graph()
    for i in range(n_cols):
        G.add_node(i)

    for i in range(n_cols):
        for j in range(i+1, n_cols):
            # overlapping rows
            valid = (region_mat[:,i]>0) & (region_mat[:,j]>0)
            if not np.any(valid):
                # maybe complementary rows? still allow merge if x similar where overlaps exist
                valid = (~np.isnan(x_mat[:,i])) & (~np.isnan(x_mat[:,j]))
            if np.sum(valid)==0:
                continue
            dx = np.nanmean(np.abs(x_mat[valid,i]-x_mat[valid,j]))
            ddn = np.nanmean(np.abs(dist_next[valid,i]-dist_next[valid,j]))
            ddp = np.nanmean(np.abs(dist_prev[valid,i]-dist_prev[valid,j]))
            if dx<max_x_diff and ddn<max_dist_diff and ddp<max_dist_diff:
                G.add_edge(i,j)

    # Connected components = merged columns
    components = list(nx.connected_components(G))

    # Build new merged matrices
    new_cols = len(components)
    merged_region = np.zeros((n_rows,new_cols),dtype=int)
    merged_x = np.full((n_rows,new_cols),np.nan)
    merged_dn = np.full((n_rows,new_cols),np.nan)
    merged_dp = np.full((n_rows,new_cols),np.nan)

    for k, comp in enumerate(components):
        comp = list(comp)
        for c in comp:
            for r in range(n_rows):
                if region_mat[r,c]>0:   # region present
                    merged_region[r,k] = region_mat[r,c]
                    merged_x[r,k] = x_mat[r,c]
                    merged_dn[r,k] = dist_next[r,c]
                    merged_dp[r,k] = dist_prev[r,c]

    return merged_region, merged_x, merged_dn, merged_dp, components

merged_region, merged_x, merged_dn, merged_dp, groups = merge_columns(
    aligned, x_positions_matrix, gap_matrix, prev_gap_matrix,
    max_x_diff=500, max_dist_diff=500, min_coverage=0.3)

print("Number of merged columns:", merged_region.shape[1])
print("Groups of merged original columns:", groups)



import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev

def fit_curves_for_regions(cells, region_to_cells, smoothing=0.0):
    """
    Fit a smooth curve (x as function of y) for each region using cell centroids.
    
    Parameters
    ----------
    cells : pd.DataFrame
        Must contain 'label', 'centroid-0' (y) and 'centroid-1' (x).
    region_to_cells : dict
        Mapping region_id -> list of cell labels in that region.
    smoothing : float
        Smoothing factor for spline fitting (passed to splrep).
        
    Returns
    -------
    dict
        region_id -> dict with:
            'points': original (x,y) points sorted by y,
            'spline': (tck) tuple returned by splrep,
            'y_range': (y_min, y_max) of the region.
    """
    curves = {}

    for region, labels in region_to_cells.items():
        region_cells = cells[cells['label'].isin(labels)]
        if region_cells.empty:
            continue
        
        # Extract coordinates (y,x)
        y = region_cells['centroid-0'].values
        x = region_cells['centroid-1'].values
        
        # Sort by y for consistent fitting
        sort_idx = np.argsort(y)
        y_sorted = y[sort_idx]
        x_sorted = x[sort_idx]
        
        # Need at least 4 points for splrep (cubic spline)
        if len(y_sorted) < 4:
            # fallback: just store points without spline
            curves[region] = {
                'points': np.column_stack((x_sorted, y_sorted)),
                'spline': None,
                'y_range': (y_sorted.min(), y_sorted.max())
            }
            continue
        
        # Fit a spline x(y)
        tck = splrep(y_sorted, x_sorted, s=smoothing)
        
        curves[region] = {
            'points': np.column_stack((x_sorted, y_sorted)),
            'spline': tck,
            'y_range': (y_sorted.min(), y_sorted.max())
        }

    return curves

curves = fit_curves_for_regions(celldata, region_to_cells, smoothing=2.0)

from napari.utils import colormaps

from numpy.polynomial import Polynomial



all_curves = []  # store coordinates for all curves
all_names = []   # optional, store names per curve

for region, cell_ids in region_to_cells.items():
    region_cells = celldata[celldata['label'].isin(cell_ids)]
    y = region_cells['centroid-0'].to_numpy()
    x = region_cells['centroid-1'].to_numpy()

    if len(x) < 3:
        continue  # skip tiny regions

    # Fit a low-degree polynomial: x = f(y)
    degree = 2
    coeffs = np.polyfit(y, x, deg=degree)
    poly = np.poly1d(coeffs)

    # Predict x positions along the y-range
    y_vals = np.linspace(y.min(), y.max(), 200)
    x_vals = poly(y_vals)

    coords = np.column_stack((y_vals, x_vals))
    all_curves.append(coords)
    all_names.append(region)

# Add all curves at once as one Shapes layer
viewer.add_shapes(all_curves, shape_type='path',
                  edge_color='yellow', edge_width=2,
                  name='all_region_curves')

endpoints = {}

for region, cell_ids in region_to_cells.items():
    region_cells = celldata[celldata['label'].isin(cell_ids)]
    y = region_cells['centroid-0'].to_numpy()
    x = region_cells['centroid-1'].to_numpy()

    if len(x) < 3:
        continue

    # Fit polynomial
    degree = 2
    coeffs = np.polyfit(y, x, degree)
    poly = np.poly1d(coeffs)
    y_vals = np.linspace(y.min(), y.max(), 200)
    x_vals = poly(y_vals)

    # Tangent: derivative of polynomial
    deriv = np.polyder(poly)
    
    # Store top and bottom endpoints with tangent
    endpoints[region] = {
        'top': (y_vals[0], x_vals[0], deriv(y_vals[0])),
        'bottom': (y_vals[-1], x_vals[-1], deriv(y_vals[-1])),
        'poly': poly
    }

def find_best_match(endpoint, candidates, max_dist=250, angle_thresh=1):
    """
    Find the best candidate to connect to `endpoint`.
    - endpoint: (y, x, tangent)
    - candidates: list of (y, x, tangent, region)
    - max_dist: maximum allowed Euclidean distance
    - angle_thresh: maximum allowed difference in tangent slope
    """
    y0, x0, slope0 = endpoint
    best_score = -np.inf
    best_candidate = None
    
    for y1, x1, slope1, region in candidates:
        dist = np.hypot(x1-x0, y1-y0)
        if dist > max_dist:
            continue
        angle_diff = abs(slope1 - slope0)
        if angle_diff > angle_thresh:
            continue
        score = -dist - angle_diff*50  # simple scoring function
        if score > best_score:
            best_score = score
            best_candidate = region
            
    return best_candidate

connections = {}  # region -> next region
for region, data in endpoints.items():
    bottom_endpoint = (data['bottom'][0], data['bottom'][1], data['bottom'][2])
    
    # Build candidate list: top endpoints of all other regions
    candidates = [(v['top'][0], v['top'][1], v['top'][2], r) 
                  for r, v in endpoints.items() if r != region]
    
    next_region = find_best_match(bottom_endpoint, candidates)
    if next_region is not None:
        connections[region] = next_region

connections_top = {}  # region -> next region
for region, data in endpoints.items():
    top_endpoint = (data['top'][0], data['top'][1], data['top'][2])
    
    # Build candidate list: top endpoints of all other regions
    candidates = [(v['bottom'][0], v['bottom'][1], v['bottom'][2], r) 
                  for r, v in endpoints.items() if r != region]
    
    next_region = find_best_match(top_endpoint, candidates)
    if next_region is not None:
        connections_top[region] = next_region

print(connections)
print(connections_top)



def make_cone(y0, x0, slope, length=200, half_angle_deg=15, n_points=30, flip=False):
    """
    Build a cone polygon from an endpoint along a slope.

    Parameters
    ----------
    y0, x0 : float
        Endpoint coordinates
    slope : float
        Derivative dx/dy at endpoint (from polyder)
    length : float
        How far to extend the cone
    half_angle_deg : float
        Half angle of the cone (in degrees)
    n_points : int
        Points along the cone arc
    flip : bool
        If True, flip direction (useful for top vs bottom)
    """
    # Build direction vector in (dy, dx)
    dy = 1.0
    dx = slope * dy
    direction = np.array([dy, dx])
    direction = direction / np.linalg.norm(direction)

    if flip:
        direction = -direction  # flip for top endpoints

    # Direction angle in image coords
    base_angle = np.arctan2(direction[0], direction[1])  # atan2(dy, dx)

    # Two edge angles of the cone
    half_angle = np.deg2rad(half_angle_deg)
    angles = np.linspace(base_angle - half_angle, base_angle + half_angle, n_points)

    # Build cone tip to arc
    tip = np.array([y0, x0])
    arc = np.column_stack([
        y0 + length * np.sin(angles),
        x0 + length * np.cos(angles)
    ])

    # Polygon: tip + arc + back to tip
    cone = np.vstack([tip, arc, tip])
    return cone

# Example usage:
# Build cones for all endpoints and add to napari as one shapes layer:
cones = []
for region, data in endpoints.items():
    yb, xb, slopeb = data['bottom']
    yt, xt, slopet = data['top']

    # Bottom endpoint (looking downward in image)
    cone_b = make_cone(yb, xb, slopeb, length=250, half_angle_deg=15, flip=False)
    cones.append(cone_b)

    # Top endpoint (looking upward in image)
    cone_t = make_cone(yt, xt, slopet, length=250, half_angle_deg=15, flip=True)
    cones.append(cone_t)

# In napari:
viewer.add_shapes(cones, shape_type='polygon', face_color='red', opacity=0.2, name='search_cones')


def point_in_polygon(x, y, polygon):
    """
    Ray casting algorithm to test if (x,y) lies inside polygon (array Nx2).
    """
    poly = polygon
    inside = False
    n = len(poly)
    px, py = poly[:,1], poly[:,0]  # polygon coords
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def find_best_match_with_cones(endpoint, candidates, length=250, half_angle_deg=15, flip=False):
    """
    Return list of candidate regions whose endpoint lies inside the cone
    of the current endpoint.

    endpoint: (y,x,slope)
    candidates: list of (y,x,slope,region)
    """
    y0, x0, slope0 = endpoint
    # Build cone polygon
    cone = make_cone(y0, x0, slope0, length=length, half_angle_deg=half_angle_deg, flip=flip)

    in_cone = []
    for y1, x1, slope1, region in candidates:
        if point_in_polygon(x1, y1, cone):
            in_cone.append(region)
    return in_cone

connections = {}  # region -> next region

# 1) bottom cones of each region
for region, data in endpoints.items():
    bottom_endpoint = (data['bottom'][0], data['bottom'][1], data['bottom'][2])
    # top endpoints of others
    candidates = [(v['top'][0], v['top'][1], v['top'][2], r)
                  for r, v in endpoints.items() if r != region]

    # Who falls inside region's bottom cone?
    inside_bottom_cone = find_best_match_with_cones(
        bottom_endpoint, candidates, length=250, half_angle_deg=15, flip=False
    )

    for candidate_region in inside_bottom_cone:
        # Mutual check: candidate’s top cone includes our bottom point?
        cand_data = endpoints[candidate_region]
        top_endpoint_cand = (cand_data['top'][0], cand_data['top'][1], cand_data['top'][2])
        # our bottom point in candidate’s top cone?
        inside_candidate_top = find_best_match_with_cones(
            top_endpoint_cand,
            [(data['bottom'][0], data['bottom'][1], data['bottom'][2], region)],
            length=250, half_angle_deg=15, flip=True
        )
        if region in inside_candidate_top:
            # Mutual match confirmed
            connections[region] = candidate_region

print(connections)


# --- main ---
def find_matches_by_polynomial(region_to_cells, celldata,
                               degrees=(1,2,3,4),
                               length=250,
                               half_angle_deg=15):
    """
    For each polynomial degree, compute endpoints, cones, and mutual matches.
    Returns:
      results[degree] = {
         'endpoints': dict(region -> endpoints),
         'mutual_matches': dict(region -> region),
         'one_sided_matches': dict(region -> region)
      }
    """
    results = {}

    for degree in degrees:
        # 1. Fit polynomials
        endpoints = {}
        for region, cell_ids in region_to_cells.items():
            region_cells = celldata[celldata['label'].isin(cell_ids)]
            y = region_cells['centroid-0'].to_numpy()
            x = region_cells['centroid-1'].to_numpy()
            if len(x) < degree + 1:
                continue
            coeffs = np.polyfit(y, x, degree)
            poly = np.poly1d(coeffs)
            y_vals = np.linspace(y.min(), y.max(), 200)
            x_vals = poly(y_vals)
            deriv = np.polyder(poly)
            endpoints[region] = {
                'top': (y_vals[0], x_vals[0], deriv(y_vals[0])),
                'bottom': (y_vals[-1], x_vals[-1], deriv(y_vals[-1])),
                'poly': poly
            }

        # 2. Cones and matches
        mutual_matches = {}
        one_sided_matches = {}

        for region, data in endpoints.items():
            # bottom cone of current region
            yb, xb, slopeb = data['bottom']
            bottom_cone = make_cone(yb, xb, slopeb, length, half_angle_deg, flip=False)

            # all other regions top endpoints
            for other_region, other_data in endpoints.items():
                if other_region == region:
                    continue
                yt, xt, slopet = other_data['top']
                # Check if other region’s top endpoint in current bottom cone
                in_my_bottom = point_in_polygon(xt, yt, bottom_cone)
                if not in_my_bottom:
                    continue

                # Candidate’s top cone
                top_cone_other = make_cone(yt, xt, slopet, length, half_angle_deg, flip=True)
                # Check if our bottom endpoint in their top cone
                in_their_top = point_in_polygon(xb, yb, top_cone_other)

                if in_my_bottom and in_their_top:
                    # Mutual match
                    mutual_matches[region] = other_region
                elif in_my_bottom:
                    # Only one sided (region → other_region)
                    one_sided_matches[region] = other_region

        results[degree] = {
            'endpoints': endpoints,
            'mutual_matches': mutual_matches,
            'one_sided_matches': one_sided_matches
        }

    return results

match_results = find_matches_by_polynomial(region_to_cells, celldata,
                                           degrees=(1,2,3,4),
                                           length=250,
                                           half_angle_deg=15)

for degree in range(1, 5):
    print(f"Degree {degree} mutual matches:")
    print(match_results[degree]['mutual_matches'])
    print(f"Degree {degree} one-sided matches:")
    print(match_results[degree]['one_sided_matches'])
    print("-" * 40)


##### RENDU LA !!!!! #####

### regions top-down should accurately define a valid tree-ring boundary

### regions that are not in borders seem mostly falsely recognized lastcells-rightcells
# But it can be segments able to connect up and down border regions
# a lot of these regions are resin ducts regions
# For images with less well defined ring boundaries we could still have a lot of
# little unmatched valid boundary segments

### upper_region_sequence & lower region_sequence should align on regions topdown
## insertion can happen (e.g. a region inserted between two successive ones in the upper or lower
# sequence)
#
# - if an insertion in the upper sequence match an insertion in the lower sequence,
# the two region have chances to belong to the same ring
# but it can also be two unwanted regions that fell randomly between two same rings
#
# - if an insertion is find in one of the sequence but not in the other, it can be
# an unwanted little region to delete
# or a valid boundary segment that doesn't reach the border
#
# - if two regions appear at different places in the upper and lower sequence,
# something went wrong. It can indicate that two rings are crossing each other,
# that is biologically impossible

### TO SUMMARIZE :
## At this point we want to ensure that we didn't forget a valid ring boundary in
# the middle
# There's no more little unwanted regions dispersed in the image

#### Then we will be able to :

## Date all cells : it could be done
# as in the R package by drawing polygons from the coordinates of cells
# or using radial file and cell rank in files along with the sequence of boundary regions

## Get the ring boundary lines :
# using contour finding functions we can have a very precise set of coordinates
# for each ring boundary

## Measure ring width & other ring level parameters (area, number of cells...) :
# we can be very precise using ring boundary lines and contour findings

## Get distance of tracheid from and to previous and next ring boundary :
# we can also be very precise and it will be useful for intra-annual analysis

## Get ring sectorization with different methods possibilities :
# Fixed width sectors or % of the ringwidth sectors

## Filter valid tracheids and radial files






### AND THE GAME IS OVER !!!

###############################################################################
