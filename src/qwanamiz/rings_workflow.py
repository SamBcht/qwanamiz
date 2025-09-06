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

viewer.add_labels(final_boundaries, name="Final Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

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

lines = []
for up_label, down_label in nearest_extremity:
    up_coords = rightcells_df[rightcells_df["label"] == up_label][["centroid-0", "centroid-1"]].values[0]
    down_coords = rightcells_df[rightcells_df["label"] == down_label][["centroid-0", "centroid-1"]].values[0]
    lines.append([up_coords, down_coords])

viewer.add_shapes(lines, shape_type='line', edge_color='chartreuse', name='Mutual Nearest Pairs', edge_width=3)

new_boundaries, new_cell_to_region = rings_functions.merge_by_cells(nearest_extremity, cell_to_region, final_boundaries, expanded_labels)


cell_to_region, region_to_cells = rings_functions.map_cell_to_region(new_boundaries > 0, new_boundaries, expanded_labels)

# At this stage we can remove spurious regions by excluding those with fewer than a given number of cells
cell_to_region, region_to_cells = rings_functions.filter_boundaries(cell_to_region, region_to_cells, mincells = 5)
new_boundaries = rings_functions.update_boundary_labels(np.zeros_like(expanded_labels, dtype = int), cell_to_region, expanded_labels)

viewer.add_labels(new_boundaries, name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

# Find the extrmities of the new ring segments
up_extremities, down_extremities = rings_functions.get_extremities(region_to_cells, rightcells_df)

###############################################################################
# FIND REGION EXTREMITIES NEAR THE BORDERS OF THE IMAGE

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




def get_region_sequences(new_boundaries, n_lines=10):
    """
    Scan horizontal lines across the image and extract ordered region sequences.

    Parameters
    ----------
    new_boundaries : np.ndarray
        2D array where each pixel has a region ID (0 = background).
    n_lines : int
        Number of horizontal probing lines between top and bottom.

    Returns
    -------
    y_positions : list of int
        Vertical positions (rows) where sequences were sampled.
    sequences : list of list[int]
        Region ID sequences (ordered left→right, without duplicates).
    """
    height, width = new_boundaries.shape
    step = height // (n_lines + 1)

    sequences = []
    y_positions = []

    for i in range(1, n_lines + 1):
        y = i * step
        row_regions = new_boundaries[y, :]

        # Keep order, remove background (0) and duplicates
        seq = []
        seen = set()
        for region in row_regions:
            if region == 0:
                continue
            if region not in seen:
                seq.append(int(region))  # cast to int for consistency
                seen.add(region)

        sequences.append(seq)
        y_positions.append(y)

    return y_positions, sequences

y_positions, sequences = get_region_sequences(new_boundaries, n_lines=10)


def align_region_sequences(sequences, gap_value=None, upper_seq=None, lower_seq=None):
    """
    Align multiple region sequences, ensuring no region changes position.
    If a region appears in different positions across sequences -> flag as conflict.
    
    Parameters:
    - sequences: list of sequences to align
    - gap_value: value to insert for missing regions
    - upper_seq: optional sequence to add at top
    - lower_seq: optional sequence to add at bottom
    """
    # Step 1: establish a reference order using the first sequence
    reference = sequences[0]
    all_regions = list(reference)  # copy
    
    # Step 2: insert new regions into reference order when encountered in others
    for seq in sequences[1:]:
        for i, region in enumerate(seq):
            if region not in all_regions:
                # try to insert based on neighbors if possible
                prev_r = seq[i-1] if i > 0 else None
                next_r = seq[i+1] if i < len(seq)-1 else None
                
                if prev_r in all_regions:
                    insert_idx = all_regions.index(prev_r) + 1
                elif next_r in all_regions:
                    insert_idx = all_regions.index(next_r)
                else:
                    insert_idx = len(all_regions)
                all_regions.insert(insert_idx, region)

    # Step 3: build aligned sequences
    aligned = []
    conflicts = []
    for seq in sequences:
        aligned_seq = []
        seq_positions = {r: i for i, r in enumerate(seq)}
        for idx, region in enumerate(all_regions):
            if region in seq_positions:
                # check position consistency (relative order only)
                pos = seq_positions[region]
                # if region order is inconsistent -> conflict
                if seq[pos] != region:
                    conflicts.append((region, seq))
                aligned_seq.append(region)
            else:
                aligned_seq.append(gap_value)
        aligned.append(aligned_seq)
        
    if upper_seq is not None:
        aligned.insert(0, [region if region in upper_seq else gap_value for region in all_regions])
    
    # Step 5: add lower_seq at the bottom
    if lower_seq is not None:
        aligned.append([region if region in lower_seq else gap_value for region in all_regions])
       

    return aligned, all_regions, conflicts

aligned, regions, conflicts = align_region_sequences(sequences, upper_seq=upper_region_sequence, lower_seq=lower_region_sequence)

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
