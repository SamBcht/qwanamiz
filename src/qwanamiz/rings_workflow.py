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
#lastcells_mask = np.zeros_like(expanded_labels, dtype=bool)
rightcells_mask = np.zeros_like(expanded_labels, dtype=bool)
#left_neighbors_mask = np.zeros_like(expanded_labels, dtype=bool)

# Get the labels of lastcells and their right_neighbors
#lastcell_labels = lastcells_df["label"].values
#rightcells_labels = lastcells_df["right_neighbor"].values
#left_neighbor_labels = lastcells_df["left_neighbor"].values

# Retain labels that are both lastcell and a right neighbor
#lastcells_inter = set(lastcell_labels) & set(right_neighbor_labels)

# Create a mask where pixels belong to lastcells or their right_neighbors
#lastcells_mask[np.isin(expanded_labels, lastcells_labels)] = True
rightcells_mask[np.isin(expanded_labels, rightcells_labels)] = True
#left_neighbors_mask[np.isin(expanded_labels, leftcells_labels)] = True

#viewer.add_image(lastcells_mask, name="LastCells Mask", opacity=0.5, colormap="red", scale = [pix_to_um, pix_to_um])
#viewer.add_image(rightcells_mask, name="Right-N Mask", opacity=0.5, colormap="orange", scale = [pix_to_um, pix_to_um])
#viewer.add_image(left_neighbors_mask, name="Left-N Mask", opacity=0.5, colormap="yellow", scale = [pix_to_um, pix_to_um])

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

#viewer.add_labels(boundary_labeled, name="Final Boundary Corrected", scale=[pix_to_um, pix_to_um])

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
#problematic_mask = np.zeros_like(boundary_labeled, dtype=bool)

# Step 2: Set pixels belonging to problematic regions to True
#problematic_mask[np.isin(boundary_labeled, problematic_regions)] = True

#viewer.add_image(problematic_mask, name="Problematic Regions", opacity=0.5, colormap="magenta", scale=[pix_to_um, pix_to_um])  # Highlighted problem regions

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

#viewer.add_labels(rightcells_boundary, name="Rightcells Boundary", scale=[pix_to_um, pix_to_um])

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
#if len(upward_points) > 0:
#    viewer.add_points(upward_points, name="Upward Earlywood Cells", size=5, face_color="blue", border_color="white")

#if len(downward_points) > 0:
#    viewer.add_points(downward_points, name="Downward Earlywood Cells", size=5, face_color="green", border_color="white")


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

#viewer.add_labels(boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

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
#if len(upward_points) > 0:
#    viewer.add_points(upward_points, name="Upward Earlywood Cells", size=5, face_color="blue", border_color="white")

#if len(downward_points) > 0:
#    viewer.add_points(downward_points, name="Downward Earlywood Cells", size=5, face_color="green", border_color="white")

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
nearest_extremity, _ = rings_functions.get_nearest_extremity(rightcells_df, cell_to_region, up_extremities, down_extremities, incompatible_region_pairs)

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


#viewer.add_labels(new_boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])



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

nearest_extremity, _ = rings_functions.get_nearest_extremity(rightcells_df, cell_to_region, up_extremities, down_extremities, incompatible_region_pairs)

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

#viewer.add_labels(new_boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

up_extremities, down_extremities = rings_functions.get_extremities(region_to_cells, rightcells_df)

incompatible_pairs = set()

nearest_extremity, all_regions = rings_functions.get_nearest_extremity(rightcells_df, 
                                                                       cell_to_region, 
                                                                       up_extremities, 
                                                                       down_extremities, 
                                                                       incompatible_pairs,
                                                                       new_boundaries.shape,
                                                                       75,
                                                                       pix_to_um)

valid_pairs, excluded_pairs = rings_functions.filter_isolated_pairs(nearest_extremity, all_regions)

print(f"✅ Valid (isolated) pairs: {len(valid_pairs)}")
print(f"❌ Excluded (crowded) pairs: {len(excluded_pairs)}")

lines = []
for up_label, down_label in valid_pairs:
    up_coords = rightcells_df[rightcells_df["label"] == up_label][["centroid-0", "centroid-1"]].values[0]
    down_coords = rightcells_df[rightcells_df["label"] == down_label][["centroid-0", "centroid-1"]].values[0]
    lines.append([up_coords, down_coords])

viewer.add_shapes(lines, shape_type='line', edge_color='magenta', name='Mutual Nearest Pairs', edge_width=3)

new_boundaries, new_cell_to_region = rings_functions.merge_by_cells(valid_pairs, cell_to_region, new_boundaries, expanded_labels)


cell_to_region, region_to_cells = rings_functions.map_cell_to_region(new_boundaries > 0, new_boundaries, expanded_labels)

# At this stage we can remove spurious regions by excluding those with fewer than a given number of cells
cell_to_region, region_to_cells = rings_functions.filter_boundaries(cell_to_region, region_to_cells, mincells = 5)
new_boundaries = rings_functions.update_boundary_labels(np.zeros_like(expanded_labels, dtype = int), cell_to_region, expanded_labels)

viewer.add_labels(new_boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

###############################################################################
# COMPUTE RINGS ELLIPSE
# Example: new_boundaries is your labeled image
props = skimage.measure.regionprops(label_image=new_boundaries, spacing=pix_to_um)

shapes_data = []
shapes_colors = []

for prop in props:
    label_id = prop.label
    #if label_id not in candidate_regions:
        #continue
    y0, x0 = prop.centroid
    orientation = prop.orientation
    major_len = prop.major_axis_length
    minor_len = prop.minor_axis_length

    # major axis
    x_major1 = x0 + np.cos(orientation) * 0.5 * minor_len
    y_major1 = y0 - np.sin(orientation) * 0.5 * minor_len
    x_major2 = x0 - np.cos(orientation) * 0.5 * minor_len
    y_major2 = y0 + np.sin(orientation) * 0.5 * minor_len
    shapes_data.append(np.array([[y_major1, x_major1], [y_major2, x_major2]]))
    shapes_colors.append('chartreuse')

    # minor axis
    x_minor1 = x0 + np.sin(orientation) * 0.5 * major_len
    y_minor1 = y0 + np.cos(orientation) * 0.5 * major_len
    x_minor2 = x0 - np.sin(orientation) * 0.5 * major_len
    y_minor2 = y0 - np.cos(orientation) * 0.5 * major_len
    shapes_data.append(np.array([[y_minor1, x_minor1], [y_minor2, x_minor2]]))
    shapes_colors.append('red')

viewer.add_shapes(shapes_data, shape_type='line', edge_color=shapes_colors, name='Ellipse Axes', edge_width=10
#                  , scale=[pix_to_um, pix_to_um]
                  )
################################################################################
# FIND REGION EXTREMITIES NEAR THE BORDERS OF THE IMAGE WITH ELLIPSE

region_classes, ring_regions, seq = rings_functions.classify_regions_by_axis(new_boundaries, pix_to_um)

for label, cls in region_classes.items():
    print(label, ":", cls)

print("Top sequence:", seq["top"])
print("Bottom sequence:", seq["bottom"])

up_extremities, down_extremities = rings_functions.get_extremities(region_to_cells, rightcells_df)



all_border_cells, upper_region_sequence, lower_region_sequence, matched_up, matched_down, unjustified = rings_functions.get_border_cells(rightcells_df, 
                                                                                                                         cell_to_region, 
                                                                                                                         up_extremities,
                                                                                                                         down_extremities,
                                                                                                                         image_height = expanded_labels.shape[0], 
                                                                                                                         image_width = expanded_labels.shape[1], 
                                                                                                                         border_margin = 75, 
                                                                                                                         pix_to_um = pix_to_um)


y_positions, sequences = rings_functions.get_region_sequences(new_boundaries, n_lines=20)


aligned, regions = rings_functions.align_region_sequences(sequences, gap_value=None, upper_seq=seq["top"], lower_seq=seq["bottom"])

#plot_alignment(aligned, regions, names=None)

candidates, cu, cl = rings_functions.find_merge_candidates(
    seq["top"], seq["bottom"]
)

print("Corrected upper:", cu)
print("Corrected lower:", cl)
print("Merge candidates:", candidates)

cleaned_matrix = rings_functions.remove_singleton_columns(aligned)

#plot_alignment(cleaned_matrix, regions, names=None)

filled = rings_functions.fill_columns(cleaned_matrix, candidates, 0.79, region_classes)

rings_functions.plot_alignment(filled, regions, names=None)

###############################################################################


incomplete = rings_functions.find_incomplete_regions(filled)

for region_id, info in incomplete.items():
    print(f"Region {region_id}: missing in rows {info['missing_rows']}")


final_merge = rings_functions.filter_incomplete_regions(incomplete_info=incomplete, 
                                     classifications=region_classes,
                                     merge_candidates=candidates, 
                                     matched_up=matched_up, 
                                     matched_down=matched_down)


valid, duplicates = rings_functions.filter_pairs_overlap(final_merge, region_classes, filled)
print(valid)
print(duplicates)


pair_extremities = {}

for r1, r2 in valid:
    cell1 = rings_functions.get_extremity_cell(r1, up_extremities, down_extremities, region_classes)
    cell2 = rings_functions.get_extremity_cell(r2, up_extremities, down_extremities, region_classes)
    coord1 = rings_functions.get_coordinates(cell1, rightcells_df)
    coord2 = rings_functions.get_coordinates(cell2, rightcells_df)
    pair_extremities[(r1, r2)] = (coord1, coord2)

print(pair_extremities)


all_merge_pairs = rings_functions.select_regions_to_merge(pair_extremities, candidates, final_merge)


aligned_top, aligned_bottom = rings_functions.build_aligned_sequences(filled, all_merge_pairs, final_merge)

print("Top   →", aligned_top)
print("Bottom→", aligned_bottom)

# Identifying true ring boundaries from the upper and lower sequences
ring_lines = rings_functions.find_ring_lines(rightcells_df, region_to_cells, aligned_top, aligned_bottom)

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

# Getting polygon coordinates defining tree rings from the ring lines
ring_polygons = rings_functions.draw_polygons(cells = celldata, ring_lines = ring_lines, upper_sequence = aligned_top, image_width = expanded_labels.shape[1] * pix_to_um)


first_year=1
# Assigning rings to years based on the polygon coordinates
celldata = rings_functions.assign_years(cells = celldata, polygons = ring_polygons, year0 = first_year)

# And ring polygons
viewer.add_shapes(ring_polygons,
                  shape_type='polygon',
                  edge_color='black',
                  face_color=[['red', 'green', 'blue', 'coral', 'black'][i % 5] for i in range(len(ring_polygons))],
                  opacity = 0.3,
                  name='Tree-ring polygons')

celltemp = celldata.copy()
celltemp.set_index('label', inplace = True)
year_dict = celltemp['year'].to_dict()
year_image = np.vectorize(year_dict.get)(expanded_labels)

viewer.add_labels(year_image.astype(int), name="Tree-ring year", opacity=0.7, scale=[pix_to_um, pix_to_um])



boundaries = rings_functions.extract_ring_boundaries(year_image, pix_to_um)

# Visualize in Napari
viewer.add_shapes(
    boundaries,
    shape_type="path",
    edge_color="magenta",
    edge_width=2,
    name="Refined ring boundaries"#,
    #scale=[pix_to_um, pix_to_um]
)





rw = rings_functions.measure_ringwidth(boundaries)

# Example: print distances in micrometers
for i, d in enumerate(rw, start=1):
    print(f"Mean distance between ring {i} and ring {i+1}: {d:.2f} μm")


distances_df = rings_functions.compute_cell_distances(celldata, boundaries, year_col="year")
###############################################################################
# RINGWIDTH & DISTANCES CHECK
# Example: inspect distances
#print(distances_df[["dist_to_current", "dist_to_previous", "prev_ring"]].tail())

# Convert distances to micrometers if needed
# Uncomment if ring distances were already in µm
# distances_df["dist_to_current"] *= pix_to_um
# distances_df["dist_to_previous"] *= pix_to_um

# Add total boundary distance per cell
distances_df["cell_ring_width"] = distances_df["dist_to_next"] + distances_df["dist_to_prev"]

# Compute mean ring width from cell distances
mean_ringwidth_from_cells = distances_df.groupby("year")["cell_ring_width"].mean()

# Convert mean_ring_distances (list from earlier) to a Series for comparison
# Those were based on skeleton boundaries (already in pixel or µm?)
ringwidth_from_boundaries = pd.Series(rw, index=range(2, len(rw)+2))

# Compare both
comparison_df = pd.DataFrame({
    "ringwidth_from_boundaries": ringwidth_from_boundaries,
    "ringwidth_from_cells": mean_ringwidth_from_cells
})

# Remove first ring if incomplete
comparison_df = comparison_df.iloc[:]

print(comparison_df)
##############################################################################
celldata = distances_df.copy()

ringprops_df = pd.DataFrame(
    skimage.measure.regionprops_table(
        year_image.astype(int),
        spacing = pix_to_um,
        properties = (
            'label',
            'area',
            'area_filled',
            'major_axis_length',
            'minor_axis_length',
            'centroid',
            'orientation',
            'perimeter_crofton')))

# --- Add ringwidth from boundaries ---
# `ringwidth_from_boundaries` is a Series with index = year, value = ringwidth
ringprops_df["ringwidth"] = ringprops_df["label"].map(ringwidth_from_boundaries)

# --- Add ringwidth from cells ---
# `mean_ringwidth_from_cells` is a Series with index = year, value = mean cell-based ring width
ringprops_df["rw_from_cells"] = ringprops_df["label"].map(mean_ringwidth_from_cells)

# Assuming your celldata has:
# 'year'       → the ring ID
# 'radial_file' → radial file ID
# 'file_rank'  → current rank of the cell in the radial file

# Compute new rank per radial_file in each ring
celldata["file_rank_scaled"] = (
    celldata
    .sort_values(["year", "radial_file", "file_rank"])  # make sure sorted
    .groupby(["year", "radial_file"])
    .cumcount() + 1  # starts from 1
)



celldata = rings_functions.filter_radial_files(celldata)


filtered_celldata = celldata[celldata["valid_radial_file"]].copy()
# Optional: reset index if you want
#filtered_celldata.reset_index(drop=True, inplace=True)

# Make a blank mask same size as your labeled image
filtered_mask = np.zeros_like(expanded_labels, dtype=bool)
filtered_labels = filtered_celldata.index
filtered_mask[np.isin(expanded_labels, filtered_labels)] = True
# Add as a labels layer to Napari
viewer.add_image(
    filtered_mask,
    name="Filtered radial files",
    opacity=0.7,
    scale=[pix_to_um, pix_to_um]
)

ringprops_df = rings_functions.add_radialfile_stats(celldata, ringprops_df)

ringprops_df = rings_functions.early_latewood_width(celldata, ringprops_df)


celldata = celldata.drop(
    columns = [
        'next_diameter_rad',
        'prev_diameter_rad',
        'next_woodzone',
        'prev_ring'])