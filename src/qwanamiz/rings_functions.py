# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:05:51 2025

@author: sambo
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from collections import Counter, deque
import copy
#from collections import defaultdict, deque
from skimage.measure import regionprops, regionprops_table
import math
from skimage.morphology import skeletonize

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from typing import Dict, List, Tuple, Any

def morks_index(cell_df):
    """
    Classify cells as earlywood or latewood based on Mork's index.

    Parameters:
    cell_df (pd.DataFrame): Dataframe with 'WallThickness' and 'LumenLength' columns.

    Returns:
    pd.DataFrame: The same dataframe with an added 'woodzone' column.
    """
    cell_df = cell_df.copy()  # Avoid modifying original dataframe
    cell_df["woodzone"] = np.where(
        (cell_df["WallThickness"] * 4 >= cell_df["diameter_rad"]), "latewood", "earlywood"
    )
    return cell_df

def get_lastcells(celldata, adjacency, diameter_factor = 2.5, diameter_factor_prev = 8):
    
    # Create a lookup dictionary for cellID -> diameter_rad
    diameter_lookup = celldata.set_index("label")["diameter_rad"].to_dict()

    # Map the right_neighbor and left_neighbor to get their diameter_rad values
    celldata["next_diameter_rad"] = celldata["right_neighbor"].map(diameter_lookup)
    celldata["prev_diameter_rad"] = celldata["left_neighbor"].map(diameter_lookup)

    # Create a lookup dictionary for cellID -> diameter_rad
    woodzone_lookup = celldata.set_index("label")["woodzone"].to_dict()

    celldata["next_woodzone"] = celldata["right_neighbor"].map(woodzone_lookup)

    # Define the diameter ratio factor (e.g., 2.5)
    diameter_factor = 2.5
    diameter_factor_prev = 8

    # Apply the conditions
    celldata["condition_met"] = (
        (celldata["woodzone"] == "latewood") &
        (celldata["next_woodzone"] == "earlywood") &
        (celldata["diameter_rad"] * diameter_factor < celldata["next_diameter_rad"]) &
        (celldata["diameter_rad"] * diameter_factor_prev > celldata["prev_diameter_rad"])
    )

    # Filter the cells where condition is met
    lastcells = celldata[celldata["condition_met"]].copy()


    # Extract the coordinates (centroid-0 and centroid-1) of the flagged cells
    #coordinates = lastcells[['centroid-0', 'centroid-1']].values

    # Get the labels of lastcells and their right_neighbors
    lastcell_labels = lastcells["label"].values

    ##### STRICT LATEWOOD TO EARLYWOOD TRANSITION
    # Apply the conditions
    celldata["lw-ew_transition"] = (
        (celldata["woodzone"] == "latewood") &
        (celldata["next_woodzone"] == "earlywood")
    )

    # Filter the cells where condition is met
    woodzone_transition = celldata[celldata["lw-ew_transition"]].copy()

    # Exclude labels that are in lastcells from woodzone_transition
    woodzone_transition = woodzone_transition[~woodzone_transition["label"].isin(set(lastcell_labels))]
    # Extract the labels from woodzone_transition
    transition_labels = set(woodzone_transition["label"])

    # Filter the adjacency dataframe where either label1 or label2 is in transition_labels
    transition_adjacency = adjacency[
        adjacency.index.get_level_values("label1").isin(transition_labels) |
        adjacency.index.get_level_values("label2").isin(transition_labels)
    ]


    # Start with direct neighbors of lastcells, using transition_adjacency
    direct_neighbors = transition_adjacency[
        transition_adjacency.index.get_level_values("label1").isin(set(lastcell_labels)) |
        transition_adjacency.index.get_level_values("label2").isin(set(lastcell_labels))
    ]

    # Get the labels of direct_neighbors, keeping only those in transition_labels
    direct_neighbor_labels = set(direct_neighbors.index.get_level_values("label1")) | \
                             set(direct_neighbors.index.get_level_values("label2"))

    # Keep only the labels that are in transition_labels
    direct_neighbor_labels &= transition_labels  # Intersect with transition_labels


    # Initialize the set with direct neighbors
    connected_transition_labels = set(direct_neighbor_labels)

    # Iteratively expand by finding adjacent transition cells
    while True:
        # Find neighbors of the current connected transition cells
        new_neighbors = transition_adjacency[
            transition_adjacency.index.get_level_values("label1").isin(connected_transition_labels) |
            transition_adjacency.index.get_level_values("label2").isin(connected_transition_labels)
        ]

        # Extract labels from both columns
        new_labels = set(new_neighbors.index.get_level_values("label1")) | set(new_neighbors.index.get_level_values("label2"))

        # Keep only those in woodzone_transition
        new_labels &= transition_labels  # Exclude lastcells
        
        # Remove direct_neighbors from new labels
        new_labels -= set(direct_neighbor_labels)

        # Stop if no new labels are found
        if new_labels.issubset(connected_transition_labels):
            break

        # Add new labels to the set
        connected_transition_labels |= new_labels

    connected_transition_labels -= set(direct_neighbor_labels)
    # Final set contains all transition cells connected to lastcells (directly or indirectly)

    ######### UNITE ALL LASTCELLS
    # Create a backup of lastcells
    new_lastcells = lastcells.copy()

    # Convert sets to lists for indexing
    direct_neighbor_labels = list(direct_neighbor_labels)
    connected_transition_labels = list(connected_transition_labels)

    # Extract direct neighbors from woodzone_transition
    direct_neighbors_df = woodzone_transition[woodzone_transition["label"].isin(direct_neighbor_labels)]

    # Extract connected transition labels from woodzone_transition
    connected_transition_df = woodzone_transition[woodzone_transition["label"].isin(connected_transition_labels)]

    # Append direct neighbors first, then connected transition labels
    new_lastcells = pd.concat([new_lastcells, direct_neighbors_df, connected_transition_df], ignore_index=True)
    
    # Get the labels of lastcells and their right_neighbors
    lastcells_labels = new_lastcells["label"].values
    rightcells_labels = new_lastcells["right_neighbor"].values
    leftcells_labels = new_lastcells["left_neighbor"].values
    
    return lastcells_labels, rightcells_labels, leftcells_labels

def parse_centroid(centroid_str):
    return eval(centroid_str)  # Safe here because it's internal and always np.float64

def boundary_graph(celldata, adjacency, lastcells_labels, rightcells_labels):
    
    
    # Keep only cells whose label is in right_neighbor_labels
    lastcells_df = celldata[celldata["label"].isin(lastcells_labels)].copy()
    rightcells_df = celldata[celldata["label"].isin(rightcells_labels)].copy()
    
    # Filter using MultiIndex levels
    adjacency_lastcells = adjacency[
        adjacency.index.get_level_values("label1").isin(lastcells_labels) &
        adjacency.index.get_level_values("label2").isin(lastcells_labels)
    ].copy()
    
    # Filter using MultiIndex levels
    adjacency_rightcells = adjacency[
        adjacency.index.get_level_values("label1").isin(rightcells_labels) &
        adjacency.index.get_level_values("label2").isin(rightcells_labels)
    ].copy()
    
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
    
    # Initialize an empty graph
    G = nx.Graph()

    # Helper function to add nodes with attributes
    def add_nodes_from_df(df, node_type):
        for _, row in df.iterrows():
            label = row['label']
            if label not in G:
                G.add_node(label, node_type=node_type,
                           centroid=(row['centroid-0'], row['centroid-1']))

    # Add nodes for lastcells and rightcells
    add_nodes_from_df(lastcells_df, 'lastcell')
    add_nodes_from_df(rightcells_df, 'rightcell')

    # Helper function to add edges from adjacency dataframe with edge_type attribute
    def add_edges_from_adjacency(adj_df, edge_type):
        for idx, row in adj_df.iterrows():
            label1, label2 = idx[0], idx[1]
            G.add_edge(label1, label2, edge_type=edge_type)

    # Add edges for each adjacency dataframe
    add_edges_from_adjacency(adjacency_lastcells, 'lastcell-lastcell')
    add_edges_from_adjacency(adjacency_rightcells, 'rightcell-rightcell')
    add_edges_from_adjacency(adjacency_neighbors, 'lastcell-rightcell')

    
    return G


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

def update_boundary_labels(boundary_labeled, label_to_region, cell_labels):
    # Make a copy to avoid modifying in-place unless you want to
    boundary_corrected = boundary_labeled.copy()

    # Prepare a mask of pixels whose cell label is in label_to_region
    target_labels = np.array(list(label_to_region.keys()))
    target_mask = np.isin(cell_labels, target_labels)

    # Get region IDs for those cell labels
    label_array = cell_labels[target_mask]
    region_array = np.vectorize(label_to_region.get)(label_array)

    # Update boundary-labeled values at those positions
    boundary_corrected[target_mask] = region_array
    
    return boundary_corrected


def get_extremities(region_to_cell, cells_df):
    upward_cells = {}
    downward_cells = {}

    # Filter earlywood cells
    #earlywood_cells = celldata[celldata["woodzone"] == "earlywood"]

    for region, cell_labels in region_to_cell.items():
        # Select only the cells belonging to this region
        region_cells = cells_df[cells_df["label"].isin(cell_labels)]
        
        if not region_cells.empty:
            # Find the most upward and downward cells based on y-coordinate
            upward_cell = region_cells.loc[region_cells["centroid-0"].idxmin(), "label"]
            downward_cell = region_cells.loc[region_cells["centroid-0"].idxmax(), "label"]
            
            # Store in dictionaries
            upward_cells[region] = upward_cell
            downward_cells[region] = downward_cell
            
    return upward_cells, downward_cells

def get_extremity_neighbors(upward_cells, downward_cells, cells_df):
    # Initialize dictionaries to store the neighbors
    upward_neighbors = {}
    downward_neighbors = {}

    # Iterate through the upward cells and get their up neighbors
    for region, up_label in upward_cells.items():
        # Retrieve the row corresponding to the upward cell in earlywood_cells dataframe
        up_cell_row = cells_df[cells_df["label"] == up_label]
        
        if not up_cell_row.empty:
            up_neighbor = up_cell_row["up_neighbor"].values[0]
            # Store the region and its up neighbor
            upward_neighbors[region] = {"upward_cell": up_label, "up_neighbor": up_neighbor}

    # Iterate through the downward cells and get their down neighbors
    for region, down_label in downward_cells.items():
        # Retrieve the row corresponding to the downward cell in earlywood_cells dataframe
        down_cell_row = cells_df[cells_df["label"] == down_label]
        
        if not down_cell_row.empty:
            down_neighbor = down_cell_row["down_neighbor"].values[0]
            # Store the region and its down neighbor
            downward_neighbors[region] = {"downward_cell": down_label, "down_neighbor": down_neighbor}

    # The resulting dictionaries will contain region-to-cell mappings for upward and downward cells with their respective neighbors

    # Extract the set of up neighbors from upward_neighbors
    up_neighbors_set = {data["up_neighbor"] for data in upward_neighbors.values() if data["up_neighbor"] is not None}

    # Extract the set of down neighbors from downward_neighbors
    down_neighbors_set = {data["down_neighbor"] for data in downward_neighbors.values() if data["down_neighbor"] is not None}

    ###########################################################################
    # COMMON NEIGHBORS
    # Find common neighbors appearing in both sets
    common_neighbors = up_neighbors_set & down_neighbors_set  # Intersection of both sets
    # Convert common_neighbors labels to int (or use the type of labels in expanded_labels)
    common_neighbors = {int(label) for label in common_neighbors}
    # Remove the background (label = 0) from the common_neighbors set
    common_neighbors = {label for label in common_neighbors if label != 0}
    
    ###########################################################################
    # UP AND DOWN PAIRS
    # Find the "up" unique neighbors (in up_neighbors_set but not in down_neighbors_set)
    up_unique_neighbors = up_neighbors_set - down_neighbors_set

    # Find the "down" unique neighbors (in down_neighbors_set but not in up_neighbors_set)
    down_unique_neighbors = down_neighbors_set - up_neighbors_set

    # Convert to int and remove background label (0)
    up_unique_neighbors = {int(label) for label in up_unique_neighbors if label != 0}
    down_unique_neighbors = {int(label) for label in down_unique_neighbors if label != 0}
    
    # Step 1: Retrieve up and down neighbors for all up_unique_neighbors and down_unique_neighbors
    up_neighbors_dict = {}
    down_neighbors_dict = {}

    # Iterate over up_unique_neighbors and store their up_neighbors
    for up_label in up_unique_neighbors:
        up_cell_row = cells_df[cells_df["label"] == up_label]
        if not up_cell_row.empty:
            up_neighbors_dict[up_label] = up_cell_row["up_neighbor"].values[0]

    # Iterate over down_unique_neighbors and store their down_neighbors
    for down_label in down_unique_neighbors:
        down_cell_row = cells_df[cells_df["label"] == down_label]
        if not down_cell_row.empty:
            down_neighbors_dict[down_label] = down_cell_row["down_neighbor"].values[0]

    # Step 2: Find the correct matches between up_neighbor and down_neighbor
    up_down_pairs = []

    # Iterate through all up_neighbors and down_neighbors and check for mutual relationships
    for up_label, up_neighbor in up_neighbors_dict.items():
        for down_label, down_neighbor in down_neighbors_dict.items():
            if up_label == down_neighbor and up_neighbor == down_label:  # Corrected condition
                up_down_pairs.append((up_label, down_label))
    
    ###########################################################################
    # REMAINING NEIGHBORS            
    # Step 1: Gather all labels already accounted for
    integrated_labels = set(common_neighbors)  # Labels in both up and down neighbor sets
    integrated_labels.update(label for pair in up_down_pairs for label in pair)  # Labels in up-down pairs

    # Step 2: Find the remaining labels
    remaining_up_neighbors = up_neighbors_set - integrated_labels
    remaining_down_neighbors = down_neighbors_set - integrated_labels

    # Step 3: Combine the remaining labels
    remaining_labels = remaining_up_neighbors | remaining_down_neighbors
    
    return common_neighbors, up_down_pairs, remaining_labels, upward_neighbors, downward_neighbors



def integrate_commons(upward_neighbors, downward_neighbors, common_neighbors, last_labeled, expanded_labels):
    # Step 1: Create an empty graph
    G = nx.Graph()

    # Step 1a: define offset for common neighbors
    offset = int(last_labeled.max()) + 1000
    def encode_common(cn): return cn + offset
    def decode_common(cn): return cn - offset

    # Step 2: Build the graph by connecting regions via common_neighbors
    for region, upward_data in upward_neighbors.items():
        up_neighbor = upward_data["up_neighbor"]
        if up_neighbor in common_neighbors:
            G.add_edge(region, encode_common(up_neighbor))

    for region, downward_data in downward_neighbors.items():
        down_neighbor = downward_data["down_neighbor"]
        if down_neighbor in common_neighbors:
            G.add_edge(region, encode_common(down_neighbor))

    # Step 3: Find connected components
    connected_components = list(nx.connected_components(G))

    # Step 4: Copy mask
    updated_boundary_labeled = last_labeled.copy()

    # Initialize new label counter
    new_label = int(last_labeled.max()) + 2

    # Track regions
    all_regions = set(np.unique(last_labeled))
    merged_regions = set()
    merged_region_mapping = {}

    # Step 5: Merge components
    for component in connected_components:
        # Separate into regions + commons
        regions_in_comp = [n for n in component if isinstance(n, (int, np.integer)) and n <= last_labeled.max()]
        commons_in_comp = [decode_common(n) for n in component if n not in regions_in_comp]

        if len(regions_in_comp) > 1:  # Only merge if multiple regions
            component_label = new_label
            new_label += 1

            merged_region_mapping[component_label] = {
                "original_regions": regions_in_comp,
                "common_neighbors": commons_in_comp,
            }

            merged_regions.update(regions_in_comp)

            # Assign merged label
            for region in regions_in_comp:
                updated_boundary_labeled[last_labeled == region] = component_label

    # Step 6: Keep original labels for non-merged regions
    for region in all_regions - merged_regions:
        updated_boundary_labeled[last_labeled == region] = region

    # Step 7: Integrate common neighbors
    new_boundary_labeled = updated_boundary_labeled.copy()

    for merged_label, mapping in merged_region_mapping.items():
        for common_neighbor in mapping["common_neighbors"]:
            new_boundary_labeled[expanded_labels == common_neighbor] = merged_label

    return new_boundary_labeled


def integrate_updown(upward_neighbors, downward_neighbors, up_down_pairs, last_labeled, expanded_labels):
    # Store the results
    up_down_neighbors = []

    for up_neighbor_label, down_neighbor_label in up_down_pairs:
        # Find the corresponding upward_cell whose up_neighbor matches
        up_match = [v["upward_cell"] for v in upward_neighbors.values() if v["up_neighbor"] == up_neighbor_label]
        # Find the corresponding downward_cell whose down_neighbor matches
        down_match = [v["downward_cell"] for v in downward_neighbors.values() if v["down_neighbor"] == down_neighbor_label]

        # Add only if both matches found
        if up_match and down_match:
            up_down_neighbors.append((up_match[0], down_match[0]))
            
            
    cell_to_region, region_to_cells = map_cell_to_region(last_labeled > 0, last_labeled, expanded_labels)
    
    
    # Store the results
    up_down_regions = []
        
    # Iterate over each (up_neighbor, down_neighbor) pair
    for up_neighbor, down_neighbor in up_down_neighbors:
        # Get the corresponding region labels from cell_to_region
        up_region = cell_to_region.get(up_neighbor, None)
        down_region = cell_to_region.get(down_neighbor, None)

        # Append to the list if both neighbors have a mapped region
        #if up_region is not None and down_region is not None:
        up_down_regions.append((up_region, down_region))

    # Filter out pairs with None values before adding to the graph
    valid_up_down_regions = [(up, down) for up, down in up_down_regions if up is not None and down is not None]


    # Step 1: Create an undirected graph
    G = nx.Graph()
    G.add_edges_from(valid_up_down_regions)  # Each pair is an edge connecting two regions

    # Step 2: Find connected components (groups of regions that should be merged)
    merged_region_groups = list(nx.connected_components(G))

    # Step 3: Create a mapping from old region labels to new merged labels
    region_merge_map = {}
    max_existing_label = last_labeled.max()
    for new_region_label, region_group in enumerate(merged_region_groups, start=max_existing_label+1):
        for region in region_group:
            region_merge_map[region] = new_region_label  # Assign new merged label

    # Print example of mapping
    #print("Region Merge Map (First 10 entries):", list(region_merge_map.items())[:10])

    # Step 4: Apply the region merge map to update the labeled image
    updated_boundary_labeled = last_labeled.copy()

    # Replace old region labels with the new merged labels
    for old_region, new_region in region_merge_map.items():
        updated_boundary_labeled[last_labeled == old_region] = new_region


    # Step 7: Add the updated labels to the viewer
    #viewer.add_labels(updated_boundary_labeled, name="Up-Down Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

    # ADD THE UP AND DOWN CELLS AND THEIR LEFT NEIGHBORS
    # Step 1: Create a mapping from up/down labels to merged regions
    pair_to_merged_region = {}

    # Corrected mapping for pairs
    for (up_label, down_label), (old_region_up, old_region_down) in zip(up_down_pairs, up_down_regions):
        # Find the merged region for both old regions
        merged_region_up = region_merge_map.get(old_region_up, old_region_up)
        merged_region_down = region_merge_map.get(old_region_down, old_region_down)
        
        # Ensure they map to the same final merged region
        #final_merged_region = min(merged_region_up, merged_region_down)  # Ensure consistency

        # Assign both up and down labels to the final merged region
        pair_to_merged_region[up_label] = merged_region_up
        pair_to_merged_region[down_label] = merged_region_down

    # Create a copy of the labeled image to modify
    final_boundary_labeled = updated_boundary_labeled.copy()

    # Update labels in the image based on pair_to_merged_region
    for cell_label, merged_region in pair_to_merged_region.items():
        if cell_label is not None and merged_region is not None:
            final_boundary_labeled[expanded_labels == cell_label] = merged_region
            
            
    return final_boundary_labeled


def get_candidate_cells(celldata, remaining_labels, lastcells_labels, diameter_factor = 1.8):
    
    # Create a lookup dictionary for cellID -> diameter_rad
    diameter_lookup = celldata.set_index("label")["diameter_rad"].to_dict()

    # Filter the cells where condition is met
    woodzone_transition = celldata[celldata["lw-ew_transition"]].copy()

    # Exclude labels that are in lastcells from woodzone_transition
    woodzone_transition = woodzone_transition[~woodzone_transition["label"].isin(set(lastcells_labels))]
    # Extract the labels from woodzone_transition
    transition_labels = set(woodzone_transition["label"])
    
    # Step 1: Identify remaining_labels in transition_labels
    remaining_in_transition = remaining_labels & set(lastcells_labels)

    # Step 2: Map left_neighbor and diameter_rad for remaining labels
    left_neighbors = celldata.loc[celldata["label"].isin(remaining_labels), ["label", "left_neighbor", "diameter_rad"]]

    # Step 3: Find labels where left_neighbor is in transition_labels
    left_neighbor_in_transition = set(left_neighbors[left_neighbors["left_neighbor"].isin(transition_labels)]["label"])

    # Step 4: Find labels where left_neighbor has a smaller diameter (1.5x or less)
    # Map left neighbor diameters using the lookup dictionary
    left_neighbors["left_diameter_rad"] = left_neighbors["left_neighbor"].map(diameter_lookup)

    # Filter where left neighbor diameter is ≤ 1.5x smaller
    left_neighbor_small_diameter = set(left_neighbors[left_neighbors["diameter_rad"] >= diameter_factor * left_neighbors["left_diameter_rad"]]["label"])

    # Step 5: Ensure there is no overlap between groups
    left_neighbor_in_transition -= remaining_in_transition
    left_neighbor_small_diameter -= (remaining_in_transition | left_neighbor_in_transition)
    
    # Combine all labels to integrate
    labels_to_integrate = left_neighbor_in_transition | left_neighbor_small_diameter
    
    return labels_to_integrate

def integrate_candidates(last_labeled, expanded_labels, labels_to_integrate, cell_to_region, upward_neighbors, downward_neighbors):
    # Final mapping from label to region
    label_to_region = {}

    # Check upward_neighbors
    for region, entry in upward_neighbors.items():
        up_neighbor = entry["up_neighbor"]
        if up_neighbor in labels_to_integrate:
            upward_cell = entry["upward_cell"]
            region_id = cell_to_region.get(upward_cell)
            if region_id is not None:
                label_to_region[up_neighbor] = region_id

    # Check downward_neighbors
    for region, entry in downward_neighbors.items():
        down_neighbor = entry["down_neighbor"]
        if down_neighbor in labels_to_integrate:
            downward_cell = entry["downward_cell"]
            region_id = cell_to_region.get(downward_cell)
            if region_id is not None:
                label_to_region[down_neighbor] = region_id

    # Make a copy to avoid modifying in-place unless you want to
    final_boundary_corrected = last_labeled.copy()

    # Prepare a mask of pixels whose cell label is in label_to_region
    target_labels = np.array(list(label_to_region.keys()))
    target_mask = np.isin(expanded_labels, target_labels)

    # Get region IDs for those cell labels
    label_array = expanded_labels[target_mask]
    region_array = np.vectorize(label_to_region.get)(label_array)

    # Update boundary-labeled values at those positions
    final_boundary_corrected[target_mask] = region_array
    
    return final_boundary_corrected


def get_segment_adjacency(adjacency, cell_to_region, upward_cells, downward_cells):
    
    # Build sets for faster lookup
    upward_labels = set(upward_cells.values())
    downward_labels = set(downward_cells.values())

    # List to store conflicting adjacency pairs
    adjacent_regions = []

    # Iterate through adjacency dataframe (MultiIndex: label1, label2)
    for (label1, label2), _ in adjacency.iterrows():
        # Check if one is upward and the other is downward
        if (label1 in upward_labels and label2 in downward_labels) or (label2 in upward_labels and label1 in downward_labels):
            region1 = cell_to_region.get(label1)
            region2 = cell_to_region.get(label2)

            if region1 is not None and region2 is not None and region1 != region2:
                adjacent_regions.append((label1, label2))
                
    return adjacent_regions


def merge_by_cells(cell_pairs, cell_to_region, last_labeled, expanded_labels):
    
    region_pairs = []
    
    for up_label, down_label in cell_pairs:
        up_region = cell_to_region[up_label]
        down_region = cell_to_region[down_label]
        region_pair = tuple(sorted((up_region, down_region)))
        
        region_pairs.append(region_pair)
    
    # Step 2: Create a graph and add edges
    region_graph = nx.Graph()
    region_graph.add_edges_from(region_pairs)

    # Step 3: Find connected components
    components = list(nx.connected_components(region_graph))

    # Step 4: Build mapping from old region → new merged region
    region_merge_map = {}
    for i, comp in enumerate(components, start=1):
        for region in comp:
            region_merge_map[region] = i

    # Include any regions not involved in merging
    all_regions = set(cell_to_region.values())
    unmerged_regions = all_regions - set(region_merge_map.keys())
    next_region_id = max(region_merge_map.values(), default=0) + 1
    for region in unmerged_regions:
        region_merge_map[region] = next_region_id
        next_region_id += 1

    # Step 5: Update cell_to_region
    cell_to_region_merged = {
        label: region_merge_map[region]
        for label, region in cell_to_region.items()
    }

    # Make a copy to avoid in-place modification
    final_boundary = last_labeled.copy()

    # Get all cell labels involved in merging
    target_labels = np.array(list(cell_to_region_merged.keys()))

    # Create a mask for those labels in the expanded_labels
    target_mask = np.isin(expanded_labels, target_labels)

    # Extract the matching label values from expanded_labels
    label_array = expanded_labels[target_mask]

    # Apply the mapping to get the merged region values
    region_array = np.vectorize(cell_to_region_merged.get)(label_array)

    # Update final_boundary_corrected only at those positions
    final_boundary[target_mask] = region_array

    # Optional: ensure background stays 0 (if some overwriting happened)
    final_boundary[expanded_labels == 0] = 0
    
    return final_boundary, cell_to_region_merged



def incompatible_regions(celldata, cell_to_region):
    # Ensure cell_to_region is applied
    celldata = celldata.copy()
    celldata["region"] = celldata["label"].map(cell_to_region)
    
    # Dictionary to store incompatible region pairs
    incompatible_pairs = set()

    # Group by radial file
    for rf, group in celldata.groupby("radial_file"):
        unique_regions = group["region"].dropna().unique()
        
        # Add all combinations of regions within this radial file as incompatible
        for r1, r2 in combinations(sorted(unique_regions), 2):
            incompatible_pairs.add(tuple(sorted((r1, r2))))

    return incompatible_pairs

def get_nearest_extremity(cells_df, 
                          cell_to_region, 
                          upward_cells, 
                          downward_cells, 
                          incompatible_regions, 
                          image_shape=None, 
                          border_margin=10.0,
                          pix_to_um=1):
    
    if image_shape is not None:
        height = image_shape[0]*pix_to_um
        width=image_shape[1]*pix_to_um
                
    else:
        height=None
        width=None
    
    # Step 2: Extract centroids and track label-to-region
    up_labels, up_centroids, up_regions = [], [], []
    for region, label in upward_cells.items():
        row = cells_df[cells_df["label"] == label]
        if not row.empty:
            y, x = row["centroid-0"].values[0], row["centroid-1"].values[0]
            # ✅ exclude extremities near any image border
            if image_shape is not None:
                if (y < border_margin) or (y >  height - border_margin) \
                   or (x < border_margin) or (x > width - border_margin):
                    continue  # skip this extremity
            up_labels.append(label)
            up_centroids.append((row["centroid-0"].values[0], row["centroid-1"].values[0]))
            up_regions.append(region)

    down_labels, down_centroids, down_regions = [], [], []
    for region, label in downward_cells.items():
        row = cells_df[cells_df["label"] == label]
        if not row.empty:
            row = cells_df[cells_df["label"] == label]
            if not row.empty:
                y, x = row["centroid-0"].values[0], row["centroid-1"].values[0]
                # ✅ exclude extremities near any image border
                if image_shape is not None:
                    if (y < border_margin) or (y >  height - border_margin) \
                       or (x < border_margin) or (x > width - border_margin):
                        continue  # skip this extremity
            down_labels.append(label)
            down_centroids.append((row["centroid-0"].values[0], row["centroid-1"].values[0]))
            down_regions.append(region)
            
    if not up_labels or not down_labels:
        return [], {}

    # Step 3: Compute distance matrix
    dist_matrix = cdist(np.array(up_centroids), np.array(down_centroids))

    # Step 4: Find nearest neighbors
    up_to_down = {up_labels[i]: down_labels[j] for i, j in enumerate(np.argmin(dist_matrix, axis=1))}
    down_to_up = {down_labels[j]: up_labels[i] for j, i in enumerate(np.argmin(dist_matrix.T, axis=1))}

    # Step 5: Find mutual nearest pairs with different regions
    region_pairs = []
    mutual_pairs = []
    neighborhoods = {}
    
        
    for up_label, down_label in up_to_down.items():
        if down_to_up.get(down_label) == up_label:
            up_region = cell_to_region[up_label]
            down_region = cell_to_region[down_label]
            if up_region != down_region:
                region_pair = tuple(sorted((up_region, down_region)))
                
                # Skip incompatible region pairs
                if region_pair not in incompatible_regions:
                    mutual_pairs.append((up_label, down_label))
                    region_pairs.append(region_pair)
                    
                    up_idx = up_labels.index(up_label)
                    down_idx = down_labels.index(down_label)

                    p_up = np.array(up_centroids[up_idx])
                    p_down = np.array(down_centroids[down_idx])
                    segment_len = np.linalg.norm(p_up - p_down)

                    # Radius threshold
                    radius = 3 * segment_len

                    # Compute distances from both endpoints
                    all_labels = up_labels + down_labels
                    all_centroids = np.vstack([up_centroids, down_centroids])

                    dist_up = np.linalg.norm(all_centroids - p_up, axis=1)
                    dist_down = np.linalg.norm(all_centroids - p_down, axis=1)

                    # Combine and find neighbors within radius
                    nearby_idx = np.where((dist_up < radius) | (dist_down < radius))[0]
                    nearby_labels = [all_labels[i] for i in nearby_idx if all_labels[i] not in (up_label, down_label)]
                    
                    if image_shape is not None:
                        valid_nearby_labels = []
                        for lbl in nearby_labels:
                            row = cells_df[cells_df["label"] == lbl]
                            if not row.empty:
                                y, x = row["centroid-0"].values[0], row["centroid-1"].values[0]
                                if (
                                    border_margin <= x <= width - border_margin
                                    and border_margin <= y <= height - border_margin
                                ):
                                    valid_nearby_labels.append(lbl)
                        nearby_labels = valid_nearby_labels

                    neighborhoods[(up_label, down_label)] = {
                        "segment_length": segment_len,
                        "radius": radius,
                        "nearby_labels": nearby_labels,
                    }
                    
    return mutual_pairs, neighborhoods

def filter_isolated_pairs(mutual_pairs, neighborhoods):
    """
    Keep only mutual pairs that have no nearby extremities in their neighborhood.

    Parameters
    ----------
    mutual_pairs : list of tuple
        Pairs of (up_label, down_label).
    neighborhoods : dict
        Output of get_nearest_extremity() where each key is a mutual pair.

    Returns
    -------
    valid_pairs : list of tuple
        Pairs with no nearby extremities.
    excluded_pairs : list of tuple
        Pairs that have at least one nearby extremity.
    """
    
    valid_pairs = []
    excluded_pairs = []

    for pair in mutual_pairs:
        info = neighborhoods.get(pair, {})
        nearby = info.get("nearby_labels", [])
        num_nearby = len(nearby)

        if num_nearby == 0:
            valid_pairs.append(pair)
            
        else:
            excluded_pairs.append(pair)
                    
    return valid_pairs, excluded_pairs


def normalize_angle_deg(angle):
    """Normalize any angle (deg) to range (-90, 90]."""
    # First bring to (-180, 180]
    ang = (angle + 180) % 360 - 180
    # Fold to (-90, 90]
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    return ang


def angle_diff_deg(a, b):
    """
    Compute smallest difference between two angles in degrees, 
    assuming angles are in (-90, 90] range.
    Returns a value in [0, 90].
    """
    # Normalize both angles to -90..90
    a = normalize_angle_deg(a)
    b = normalize_angle_deg(b)
    
    diff = abs(a - b)
    # If difference > 90, take the complement
    if diff > 90:
        diff = 180 - diff
    return diff

def analyze_pairs_angles(cells_df, mutual_pairs):
    """
    For each mutual pair (in degrees):
      - cell mean angles
      - their perpendicular angles
      - segment angle left→right
      - differences to mean and to perpendicular
      - passed flag if segment closer to perpendicular than to mean
    """
    records = []
    
    cells_df = cells_df.copy()

    for label1, label2 in mutual_pairs:
        row1 = cells_df[cells_df["label"] == label1].iloc[0]
        row2 = cells_df[cells_df["label"] == label2].iloc[0]

        # centroids
        x1, y1 = row1["centroid-1"], row1["centroid-0"]
        x2, y2 = row2["centroid-1"], row2["centroid-0"]

        # Force left→right
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            label1, label2 = label2, label1
            row1, row2 = row2, row1

        # Segment angle in degrees: 0 = horizontal, +90 = up, -90 = down
        seg_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        seg_angle = normalize_angle_deg(seg_angle)

        ang1 = normalize_angle_deg(row1["mean_angle"])
        ang2 = normalize_angle_deg(row2["mean_angle"])

        # Perpendicular angles
        perp1 = normalize_angle_deg(ang1 + 90)
        perp2 = normalize_angle_deg(ang2 + 90)

        # Differences (absolute)
        diff1 = angle_diff_deg(seg_angle, ang1)
        diff2 = angle_diff_deg(seg_angle, ang2)
        diff1_perp = angle_diff_deg(seg_angle, perp1)
        diff2_perp = angle_diff_deg(seg_angle, perp2)

        # Passed condition: closer to perpendicular than to mean
        passed = (diff1_perp < diff1) and (diff2_perp < diff2)

        records.append({
            "label1": label1,
            "label2": label2,
            "mean_angle1_deg": ang1,
            "mean_angle2_deg": ang2,
            "perp_angle1_deg": perp1,
            "perp_angle2_deg": perp2,
            "segment_angle_deg": seg_angle,
            "diff1_deg": diff1,
            "diff2_deg": diff2,
            "diff1_perp_deg": diff1_perp,
            "diff2_perp_deg": diff2_perp,
            "passed": passed
        })

    df = pd.DataFrame.from_records(records)
    valid_pairs = df[df['passed']].apply(lambda r: (r['label1'], r['label2']), axis=1).tolist()
    excluded_pairs = df[~df['passed']].apply(lambda r: (r['label1'], r['label2']), axis=1).tolist()
    
    return df, valid_pairs, excluded_pairs





def get_border_cells(cells_df, cell_to_region_merged, upward_cells, downward_cells, image_height, image_width, border_margin = 75, pix_to_um = 0.55042690590734):
    
    # Build sets for faster lookup
    upward_labels = set(upward_cells.values())
    downward_labels = set(downward_cells.values())

    
    # Filter rightcells_df for upward and downward labels
    upward_df = cells_df[cells_df["label"].isin(upward_labels)]
    downward_df = cells_df[cells_df["label"].isin(downward_labels)]

    # Upward cells touching top of the image
    upward_border_df = upward_df[upward_df["centroid-0"] <= border_margin*pix_to_um]
    upward_border_cells = set(upward_border_df["label"])

    # Downward cells touching bottom of the image
    downward_border_df = downward_df[downward_df["centroid-0"] >= (image_height - border_margin)*pix_to_um]
    downward_border_cells = set(downward_border_df["label"])

    # -------- LEFT/RIGHT borders --------
    # Cells near the left edge (from both upward and downward sets)
    left_border_cells_up = upward_df[upward_df["centroid-1"] <= border_margin * pix_to_um]["label"]
    left_border_cells_down = downward_df[downward_df["centroid-1"] <= border_margin * pix_to_um]["label"]
    left_border_cells = set(left_border_cells_up).union(left_border_cells_down)

    # Cells near the right edge (from both upward and downward sets)
    right_border_cells_up = upward_df[upward_df["centroid-1"] >= (image_width - border_margin) * pix_to_um]["label"]
    right_border_cells_down = downward_df[downward_df["centroid-1"] >= (image_width - border_margin) * pix_to_um]["label"]
    right_border_cells = set(right_border_cells_up).union(right_border_cells_down)
    
    # ---- FIX: remove cells that are already vertical borders (corner cases) ----
    left_border_cells -= (upward_border_cells | downward_border_cells)
    right_border_cells -= (upward_border_cells | downward_border_cells)

    # Optional: Combine all border cell sets
    vertical_border_cells = upward_border_cells.union(downward_border_cells)
    horizontal_border_cells = left_border_cells.union(right_border_cells)
    all_border_cells = vertical_border_cells.union(horizontal_border_cells)
    
    
    # Convert border cell labels to their corresponding regions
    regions_with_upward_border = {cell_to_region_merged[label] for label in upward_border_cells if label in cell_to_region_merged}
    regions_with_downward_border = {cell_to_region_merged[label] for label in downward_border_cells if label in cell_to_region_merged}

    # Intersection: regions that have both an upward and a downward border cell
    regions_topdown = regions_with_upward_border & regions_with_downward_border

    
    
    # Step 1: Identify unmatched border regions (i.e., not in top-down intersection)
    unmatched_up = regions_with_upward_border - regions_topdown
    unmatched_down = regions_with_downward_border - regions_topdown

    # Step 3: Map cell labels to regions
    left_border_regions = set(cell_to_region_merged[label] for label in left_border_cells if label in cell_to_region_merged)
    right_border_regions = set(cell_to_region_merged[label] for label in right_border_cells if label in cell_to_region_merged)

    # Step 2: See if these unmatched regions touch left or right borders
    left_matched_up = unmatched_up & left_border_regions
    right_matched_up = unmatched_up & right_border_regions
    
    matched_up = left_matched_up | right_matched_up

    left_matched_down = unmatched_down & left_border_regions
    right_matched_down = unmatched_down & right_border_regions
    
    matched_down = left_matched_down | right_matched_down

    # Optionally, collect the remaining unmatched regions that were not justified
    unjustified_up = unmatched_up - (left_matched_up | right_matched_up)
    unjustified_down = unmatched_down - (left_matched_down | right_matched_down)

    unjustified = {
        "unjustified_up": unjustified_up,
        "unjustified_down": unjustified_down
    }
    
    # --- Get region for each border cell ---
    upward_border_df["region"] = upward_border_df["label"].map(cell_to_region_merged)
    downward_border_df["region"] = downward_border_df["label"].map(cell_to_region_merged)

    # --- Upper border: group by region and find min x-coordinate ---
    upper_region_coords = (
        upward_border_df.groupby("region")["centroid-1"]
        .mean()
        .reset_index()
        .sort_values("centroid-1")
    )

    upper_region_sequence = upper_region_coords["region"].tolist()

    # --- Lower border: same process ---

    lower_region_coords = (
        downward_border_df.groupby("region")["centroid-1"]
        .mean()
        .reset_index()
        .sort_values("centroid-1")
    )

    lower_region_sequence = lower_region_coords["region"].tolist()
    
    # Insert matched_up into lower sequence
    matched_up = {int(r) for r in matched_up}
    matched_down = {int(r) for r in matched_down}


    return all_border_cells, upper_region_sequence, lower_region_sequence, matched_up, matched_down, unjustified

# A function that filters the dictionaries linking region IDs to cell IDs by removing
# regions with less than mincells cells
def filter_boundaries(cell_to_region, region_to_cell, mincells = 5):
    # We define a set of regions that have < mincells cells
    region_to_cell = {key: value for key, value in region_to_cell.items() if len(value) >= mincells}

    # We use those regions to filter the cell_to_region dictionary
    cell_to_region = {key: value for key, value in cell_to_region.items() if value in region_to_cell.keys()}

    return cell_to_region, region_to_cell

# A function that identifies the cells forming true ring boundaries
# cells: a DataFrame of cells found at ring boundaries
# region_to_cells: a dictionary with keys corresponding to boundary regions and values being the cells in those regions
# upper_sequence: a list (sorted by x-coordinates) with the order of boundary regions touching the top of the image
# lower_sequence: a list (sorted by x-coordinates) with the order of boundary regions touching the bottom of the image
# return value: a dictionary with region IDs as keys and an list of cells in those region order by y-coordinates
def find_ring_lines(cells, region_to_cells, upper_sequence, lower_sequence):
    # Sorting the cells DataFrame by y-coordinate to guarantee the right order in output
    sorted_cells = cells.sort_values(by = 'centroid-0')

    # First we identify ring boundaries that are found in both the lower and upper sequence
    ring_regions = set(upper_sequence) & set(lower_sequence)

    # We loop over those regions to create the output dictionary
    rings = {}

    for region in ring_regions:
        rings[region] = sorted_cells[sorted_cells["label"].isin(region_to_cells[region])]["label"].to_list()

    # We then need to find spots in-between ring boundaries with unlinked boundaries

    # These lists define the indexes where the well-formed boundaries are found in the upper and lower sequence
    upper_indices = [index for index,value in enumerate(upper_sequence) if value in rings]
    lower_indices = [index for index,value in enumerate(lower_sequence) if value in rings]

    # Then we find how many "free spots" are in between every index, but we need to insert a virtual index -1 to account for the beginning of the image
    upper_indices.insert(0, -1)
    lower_indices.insert(0, -1)

    # Next we find the differences between neighboring indices (-1 because we want the number of intervening boundaries)
    lower_diff = [b - a - 1 for a, b in zip(lower_indices, lower_indices[1:])]
    upper_diff = [b - a - 1 for a, b in zip(upper_indices, upper_indices[1:])]

    # This leads us to identifying indices where unassigned boundaries match
    matching_indices = np.where([a == b and a != 0 for a,b in zip(lower_diff, upper_diff)])[0].tolist()

    # Then we loop over the matching indices to add them to the rings dictionary
    for index in matching_indices:
        for n in range(lower_diff[index]):
            upper_value = upper_sequence[upper_indices[index] + 1 + n]
            lower_value = lower_sequence[lower_indices[index] + 1 + n]
            # We assign the value of the upper sequence to the ring region
            rings[upper_value] = sorted_cells[sorted_cells["label"].isin(region_to_cells[upper_value])]["label"].to_list()
            rings[upper_value] += sorted_cells[sorted_cells["label"].isin(region_to_cells[lower_value])]["label"].to_list()

    return rings

# This function draws polygons from the ring boundaries so that they can be used
# for assigning cells to tree rings
# cells: a pandas DataFrame with information on the cells present in the dataset
# ring_lines: a dictionary with keys representing boundary region IDs and values as lists with cell labels in the right order for drawing ring lines
# upper_sequence: a list with the order of the region IDs along the upper border of the picture. All keys in ring lines should be in that list.
# image_height: the height of the image, in micrometers
# return value: a list of numpy arrays with one 2D array per element representing each polygon. The polygons are in left-to-right order.
def draw_polygons(cells, ring_lines, upper_sequence, image_width):
    # The output is a list of 2D arrays that will hold the coordinates of the polygons
    polygons = list()

    # Creating a copy because we want to use the label as index
    cell_copy = cells.copy()
    cell_copy.set_index("label", inplace = True)

    # We subset the upper sequence to only those regions that are proper ring boundaries
    sequence = [i for i in upper_sequence if i in ring_lines]

    for index, region in enumerate(sequence):
        # Extracting the cells in that region
        # Using the labels in ring_lines already extracts them in sorted y-coordinates
        i_cells = cell_copy.loc[ring_lines[region]]

        # Extracting the x- and y-coordinates of the current boundary region
        xcoords = i_cells["centroid-1"].tolist()
        ycoords = i_cells["centroid-0"].tolist()
        coords = np.array(list(zip(ycoords, xcoords)))

        # Adding the x- and y-coordinates of the previous boundary region
        # if this is not the first region
        if index > 0:
            # We flip the coordinates of the second array to ensure proper polygon
            i_polygon = np.concatenate([coords, np.flip(prev_coords, axis = 0)])
        else:
            # Otherwise we need to add the corners of the image
            top_y = coords[0][0]
            bottom_y = coords[-1][0]

            corners = np.array([[bottom_y, 0], [top_y, 0]])
            i_polygon = np.concatenate([coords, corners])
            
       
        # We need to wrap back to the first vertex for a true polygon
        i_polygon = np.concatenate([i_polygon, i_polygon[0:1, :]])

        # Assigning that polygon to the list of polygons
        polygons.append(i_polygon)
        
        # Setting the previous coordinates to the current ones before restarting the loop
        prev_coords = coords
        
    if prev_coords is not None:
        right_x = image_width - 1
        top_y = prev_coords[0][0]
        bottom_y = prev_coords[-1][0]
        
        right_top = np.array([[top_y, right_x]])
        right_bottom = np.array([[bottom_y, right_x]])

        # Generate intermediate support vertices (exclude endpoints!)
        support_ys = np.linspace(top_y, bottom_y, 18)  # 18 points *between* endpoints
        support_ys = support_ys[1:-1]  # remove duplicates of top_y and bottom_y

        support_points = np.column_stack((support_ys, np.full_like(support_ys, right_x)))

        # Final right boundary with preserved order: go DOWN from top → bottom
        right_side = np.vstack([right_top, support_points, right_bottom])


        #right_side = np.array([[bottom_y, right_x], [top_y, right_x]])
        last_polygon = np.concatenate([prev_coords, np.flip(right_side, axis = 0)])
        last_polygon = np.concatenate([last_polygon, last_polygon[0:1, :]])
        
        polygons.append(last_polygon)


    return polygons

# A function that takes metadata about cells as well as ring polygons and assigns years to cells
# cells: a pandas DataFrame of cell metadata
# polygons: a list of polygons, such as returned by draw_polygons
# year0: the year of the first ring
# magic_shift: a small value added to the cell coordinates to make sure that cells are assigned to the right year. Defaults to 0.001
# threshold_sum: the angle sum value above which a cell is considered to be part of the ring (defaults to 6, theoretical expectation = 2*pi)
# return value: a DataFrame similar to the cells input but with an added column 'year' for the year when a cell was formed
def assign_years(cells, polygons, year0 = 0, magic_shift = 0.001, threshold_sum = 6):

    # We create a new 'year' column for the year that a cell was formed
    cells['year'] = np.nan

    # An index for the polygon being considered
    for i in range(len(polygons)):
        i_polygon = polygons[i]

        # A bounding box for the polygon
        # y1, y2, x1, x2
        bbox = [np.min(i_polygon[:, 0]), np.max(i_polygon[:, 0]), np.min(i_polygon[:, 1]), np.max(i_polygon[:, 1])]

        # We subset the cells that are within the bounding box so we only need to test those
        cell_indices = np.where((cells['centroid-0'] >= bbox[0]) & (cells['centroid-0'] <= bbox[1]) & (cells['centroid-1'] >= bbox[2]) & (cells['centroid-1'] <= bbox[3]))[0]

        # We introduce a small shift to the left in x-coordinates because we want the cells to be included in the current ring
        xcoords = np.array(cells.loc[cell_indices]['centroid-1'].tolist()) + magic_shift
        ycoords = np.array(cells.loc[cell_indices]['centroid-0'].tolist())

        # We need to get vectors that point from each cell to each vertex of the polygon
        # because we need to find the sum of angles linking each cell to each edge
        # x1: x-component of vector from cell to vertex 1
        x1 = np.subtract.outer(xcoords, i_polygon[:, 1])
        # y1: y-component of vector from cell to vertex 1
        y1 = np.subtract.outer(ycoords, i_polygon[:, 0])
        # x2: x-component of vector from cell to vertex 2
        x2 = np.concatenate([x1[:, 1:], x1[:,:1]], axis = 1)
        # y2: y-component of vector from cell to vertex 2
        y2 = np.concatenate([y1[:, 1:], y1[:,:1]], axis = 1)

        # Compute the angles using np.arctan2
        # See https://math.stackexchange.com/questions/317874/calculate-the-angle-between-two-vectors
        angles = np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
        angle_sum = np.sum(angles, axis = 1)
        polygon_indices = cell_indices[abs(angle_sum) > threshold_sum]

        # We consider a cell with an angle sum > 6 to be part of that polygon
        cells.loc[polygon_indices, 'year'] = i + year0

    return cells


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






def classify_regions_by_axis(new_boundaries, pix_to_um=1.0):
    """
    Classify regions based on where their major axis crosses the image borders.
    Returns dict: label -> classification ("top_bottom", "top", "bottom",
                                           "left", "right", "middle")
    """
    props = regionprops(new_boundaries, spacing=pix_to_um)
    height, width = new_boundaries.shape
    classifications = {}
    top_points = []
    bottom_points = []

    for prop in props:
        label_id = prop.label
        y0, x0 = prop.centroid
        orientation = prop.orientation
        major_len = prop.major_axis_length

        # ✅ Major axis endpoints (half-length each side)
        x1 = x0 + np.sin(orientation) * 0.5 * major_len
        y1 = y0 + np.cos(orientation) * 0.5 * major_len
        x2 = x0 - np.sin(orientation) * 0.5 * major_len
        y2 = y0 - np.cos(orientation) * 0.5 * major_len

        # Check border crossings
        crosses_top = (y1 <= 0) or (y2 <= 0)
        crosses_bottom = (y1 >= height*pix_to_um) or (y2 >= height*pix_to_um)
        crosses_left = (x1 <= 0) or (x2 <= 0)
        crosses_right = (x1 >= width*pix_to_um) or (x2 >= width*pix_to_um)

        if crosses_top and crosses_bottom:
            cls = "top_bottom"
        elif crosses_top and crosses_left:
            cls = "top_left"        
        elif crosses_top and crosses_right:
            cls = "top_right"
        elif crosses_top:
            cls = "top"
        elif crosses_bottom and crosses_left:
            cls = "bottom_left"
        elif crosses_bottom and crosses_right:
            cls = "bottom_right"
        elif crosses_bottom:
            cls = "bottom"
        elif crosses_left:
            cls = "left"
        elif crosses_right:
            cls = "right"
        else:
            cls = "middle"

        classifications[label_id] = cls
        
        if "top" in cls:
            # use the topmost endpoint (smallest y)
            if y1 < y2:
                top_points.append((label_id, x1, y1))
            else:
                top_points.append((label_id, x2, y2))
                
        if "bottom" in cls:
            # use the bottommost endpoint (largest y)
            if y1 > y2:
                bottom_points.append((label_id, x1, y1))
            else:
                bottom_points.append((label_id, x2, y2))

    # --- order sequences ---
    top_sorted = sorted(top_points, key=lambda t: t[1])       # sort by x
    bottom_sorted = sorted(bottom_points, key=lambda t: t[1]) # sort by x

    sequences = {
        "top": [lab for lab, _, _ in top_sorted],
        "bottom": [lab for lab, _, _ in bottom_sorted],
    }

    return classifications, props, sequences


def find_merge_candidates(upper_sequence, lower_sequence, matched_up=None, matched_down=None):
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




def fill_columns(aligned_matrix, merge_candidates=set(), min_fraction=0.7, region_classes=None):
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
    region_classes : dict[int, str], optional
        Mapping {region_id: classification}
    
    Returns
    -------
    filled_matrix : list[list[int]]
        New matrix with some None values filled where safe.
    """
    filled_matrix = copy.deepcopy(aligned_matrix)
    n_rows = len(aligned_matrix)
    n_cols = len(aligned_matrix[0])

    # Regions in merge candidates → exclude from filling
    candidate_values = set()
    for pair in merge_candidates:
        candidate_values.update(pair)

    # Region classes that should always be filled
    edge_classes = {"top_left", "top_right", "bottom_left", "bottom_right"}

    for col_idx in range(n_cols):
        # Collect existing values in the column
        col_vals = [filled_matrix[row][col_idx] for row in range(n_rows)
                    if filled_matrix[row][col_idx] is not None]
        if not col_vals:
            continue

        # Identify region class if possible
        main_region = col_vals[0]
        region_class = None
        if region_classes and main_region in region_classes:
            region_class = region_classes[main_region]

        # Remove unsafe values (that appear in merge candidates)
        safe_vals = [v for v in col_vals if v not in candidate_values]

        if not safe_vals:
            continue

        counts = Counter(safe_vals)
        most_common_val, count = counts.most_common(1)[0]
        fraction = count / n_rows

        # Condition 1: fill if majority threshold met
        # Condition 2: or region belongs to a corner classification
        should_fill = (
            fraction >= min_fraction
            or (region_class in edge_classes)
        )

        if should_fill:
            for row in range(n_rows):
                if filled_matrix[row][col_idx] is None:
                    filled_matrix[row][col_idx] = most_common_val

    return filled_matrix



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


def find_incomplete_regions(filled_matrix):
    """
    Identify regions (by ID) that still have None values in their column.

    Parameters
    ----------
    filled_matrix : list[list[int | None]]
        Aligned matrix (rows = sequences, columns = regions).

    Returns
    -------
    incomplete_info : dict
        Keys = region IDs (from column values),
        Values = list of row indices with missing entries.
    """
    n_rows = len(filled_matrix)
    n_cols = len(filled_matrix[0]) if n_rows > 0 else 0

    incomplete_info = {}

    for col_idx in range(n_cols):
        col_vals = [filled_matrix[row][col_idx] for row in range(n_rows)]
        missing_rows = [row_idx for row_idx, val in enumerate(col_vals) if val is None]

        # Skip fully filled columns
        if not missing_rows:
            continue

        # Identify dominant region ID among non-None values
        non_none_vals = [v for v in col_vals if v is not None]
        if non_none_vals:
            most_common_region = Counter(non_none_vals).most_common(1)[0][0]
        else:
            most_common_region = None

        incomplete_info[most_common_region] = {
            "column_index": col_idx,
            "missing_rows": missing_rows,
        }

    return incomplete_info




def filter_incomplete_regions(incomplete_info, merge_candidates, matched_up, matched_down, classifications):
    
    incomplete_regions = list(incomplete_info.keys())
    already_merged = set(r for pair in merge_candidates for r in pair)
    filtered = []
    for region in incomplete_regions:
        cls = classifications.get(region)
        if region in already_merged:
            continue
        if cls in ("top", "bottom") and (region in matched_up or region in matched_down):
            continue
        filtered.append(region)
    return filtered


def filter_pairs_overlap(region_pairs, classifications, filled_matrix):
    """
    Filter region pairs based on classification rules:
    - Reject pairs where both regions are only 'top' types.
    - Reject pairs where both regions are only 'bottom' types.
    - Accept if one is top-related and the other is bottom-related or middle.

    Parameters
    ----------
    region_pairs : list[tuple]
        List of region ID pairs.
    classifications : dict
        Region classification from classify_regions_by_axis().

    Returns
    -------
    valid_pairs : list[tuple]
        Filtered list of region pairs.
    """
    pairs = list(combinations(set(region_pairs), 2))
    duplicates = []
    valid_pairs = []

    top_types = {"top", "top_left", "top_right"}
    bottom_types = {"bottom", "bottom_left", "bottom_right"}

    for r1, r2 in pairs:
        c1 = classifications.get(r1)
        c2 = classifications.get(r2)

        # Check top/bottom classification logic
        both_top = c1 in top_types and c2 in top_types
        both_bottom = c1 in bottom_types and c2 in bottom_types

        if both_top or both_bottom:
            continue  # reject
            
        overlap_found = False
        for row in filled_matrix:
        # Count how many times the two regions appear in this row
            count_r1 = row.count(r1)
            count_r2 = row.count(r2)
            if count_r1 > 0 and count_r2 > 0:
                overlap_found = True
                break  # stop at first overlap

        if overlap_found:
            continue

        valid_pairs.append((r1, r2))
              
        
        region_count = defaultdict(list)  # region -> list of pairs containing it

        for pair in valid_pairs:
            r1, r2 = pair
            region_count[r1].append(pair)
            region_count[r2].append(pair)

        # Step 2: Identify regions appearing in multiple pairs
        for region, reg_pairs in region_count.items():
            if len(reg_pairs) > 1:
                duplicates.append(region)

    return valid_pairs, duplicates



def get_extremity_cell(region, up_extremities, down_extremities, region_classes):
    """Return the extremity cell ID based on region class."""
    zone = region_classes.get(region)
    if zone == "top":
        return down_extremities.get(region)  # bottom-most cell of top region
    elif zone == "bottom":
        return up_extremities.get(region)    # top-most cell of bottom region
    else:
        return None



def get_coordinates(cell_id, cells_df):
    """Return (x, y) coordinates for a given cell ID from rightcells_df."""
    row = cells_df[cells_df["label"] == cell_id]
    if row.empty:
        return None
    return (float(row["centroid-1"].iloc[0]), float(row["centroid-0"].iloc[0]))

def select_regions_to_merge(pair_extremities, candidates, final_merge):
    
    pair_distances = {}
    for pair, (p1, p2) in pair_extremities.items():
        if p1 is None or p2 is None:
            distance = float('inf')  # ignore if missing coordinates
        else:
            distance = math.dist(p1, p2)  # Euclidean distance
        pair_distances[pair] = distance

    #print(pair_distances)

    from collections import defaultdict

    # Step 2: group pairs by involved regions
    region_to_pairs = defaultdict(list)
    for pair, dist in pair_distances.items():
        r1, r2 = pair
        region_to_pairs[r1].append((pair, dist))
        region_to_pairs[r2].append((pair, dist))

    # Step 3: choose best pair per region
    selected_pairs = set()
    used_regions = set()

    for pair, dist in sorted(pair_distances.items(), key=lambda x: x[1]):  # sort by shortest distance
        r1, r2 = pair
        if r1 not in used_regions and r2 not in used_regions:
            selected_pairs.add(pair)
            used_regions.update([r1, r2])

    #print("Selected pairs:", selected_pairs)

    # Collect all regions used in selected_pairs
    used_regions = set()
    for r1, r2 in selected_pairs:
        used_regions.add(r1)
        used_regions.add(r2)

    # Regions that remain alone (not in any selected pair)
    remaining_solo_regions = [r for r in final_merge if r not in used_regions]

    print("Selected pairs:", selected_pairs)
    print("Solo regions:", remaining_solo_regions)

    # selected_pairs is already a set of tuples
    selected_pair_list = list(selected_pairs)

    # Combine candidates and selected_pairs, then deduplicate
    merge_pairs = list({tuple(sorted(p)) for p in (candidates + selected_pair_list)})

    
    return merge_pairs



def build_aligned_sequences(filled, merge_pairs, final_regions):
    # 1. Extract first and last non-empty rows
    top_seq = next(row for row in filled if any(x is not None for x in row))
    bottom_seq = next(row for row in reversed(filled) if any(x is not None for x in row))

    # Remove None
    top_seq = [x for x in top_seq if x is not None]
    bottom_seq = [x for x in bottom_seq if x is not None]

    # 2. Build pair lookup dictionary
    pair_lookup = {}
    for a, b in merge_pairs:
        pair_lookup[a] = b
        pair_lookup[b] = a

    # 3. Build aligned sequences preserving order
    aligned_top = []
    aligned_bottom = []

    # Use two pointers to traverse top_seq and bottom_seq
    i_top = 0
    i_bottom = 0

    while i_top < len(top_seq) or i_bottom < len(bottom_seq):
        top_val = top_seq[i_top] if i_top < len(top_seq) else None
        bottom_val = bottom_seq[i_bottom] if i_bottom < len(bottom_seq) else None

        if top_val == bottom_val:
            # Same region in both sequences
            aligned_top.append(top_val)
            aligned_bottom.append(bottom_val)
            i_top += 1
            i_bottom += 1
        elif top_val and (top_val not in bottom_seq):
            # Region only in top
            aligned_top.append(top_val)
            paired = pair_lookup.get(top_val, top_val)
            aligned_bottom.append(paired)
            i_top += 1
        elif bottom_val and (bottom_val not in top_seq):
            # Region only in bottom
            paired = pair_lookup.get(bottom_val, bottom_val)
            aligned_top.append(paired)
            aligned_bottom.append(bottom_val)
            i_bottom += 1
        else:
            # Different regions in top and bottom
            aligned_top.append(top_val)
            aligned_bottom.append(bottom_val)
            i_top += 1
            i_bottom += 1
            
    def unique_order(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    aligned_top = unique_order(aligned_top)
    aligned_bottom = unique_order(aligned_bottom)

    # 5. ✅ Sanity check to enforce same length
    if len(aligned_top) != len(aligned_bottom):
        raise ValueError(
            f"Aligned sequences differ in length: top={len(aligned_top)}, bottom={len(aligned_bottom)}"
        )

    return aligned_top, aligned_bottom


def extract_ring_boundaries(year_image, pix_to_um):
    """
    Extract boundary pixels between successive rings using adjacency shifts.
    Returns a list of polylines (in micrometers) for each ring pair.
    """
    ring_ids = sorted([int(x) for x in np.unique(year_image) if not np.isnan(x)])
    boundaries = []
    image_shape = year_image.shape

    # Replace NaN with a placeholder to avoid errors in comparison
    temp_image = np.nan_to_num(year_image, nan=-999999)

    for r1, r2 in zip(ring_ids[:-1], ring_ids[1:]):
        # Vertical adjacency
        vshift1 = temp_image[:-1, :]
        vshift2 = temp_image[1:, :]
        v_adj = ((vshift1 == r1) & (vshift2 == r2)) | ((vshift1 == r2) & (vshift2 == r1))
        v_coords = np.column_stack(np.where(v_adj))
        
        # Horizontal adjacency
        hshift1 = temp_image[:, :-1]
        hshift2 = temp_image[:, 1:]
        h_adj = ((hshift1 == r1) & (hshift2 == r2)) | ((hshift1 == r2) & (hshift2 == r1))
        h_coords = np.column_stack(np.where(h_adj))
        
        # Combine vertical and horizontal adjacency pixels
        all_coords = np.vstack([v_coords, h_coords])
        
        # Create binary image
        mask = np.zeros(image_shape, dtype=bool)
        mask[all_coords[:,0], all_coords[:,1]] = True
    
        # Skeletonize
        skeleton = skeletonize(mask)
    
        # Get skeleton coordinates
        y_skel, x_skel = np.where(skeleton)
        
        polyline = [(y * pix_to_um, x * pix_to_um) for y, x in zip(y_skel, x_skel)]
        
        # Optional: sort by Y to get rough polyline order
        polyline.sort(key=lambda p: p[0])
        
        boundaries.append(polyline)
        

    return boundaries

def one_ringwidth(line1, line2, trim_fraction=0.1):
    """
    Compute mean ring width between two polylines (lists of (y,x) coordinates),
    removing the top and bottom trim_fraction of nearest-neighbor distances.
    
    Parameters
    ----------
    line1, line2 : list of (y, x)
        Polylines for two successive ring boundaries.
    trim_fraction : float
        Fraction of extreme values to remove from both ends of distance arrays (0-0.5).
    
    Returns
    -------
    mean_dist : float
        Symmetric trimmed mean distance between the two lines.
    """
    if len(line1) == 0 or len(line2) == 0:
        return np.nan
    
    line1 = np.array(line1)
    line2 = np.array(line2)
    
    # Build KD-trees
    tree1 = cKDTree(line1)
    tree2 = cKDTree(line2)
    
    # Nearest neighbor distances
    d1, _ = tree2.query(line1)  # line1 -> line2
    d2, _ = tree1.query(line2)  # line2 -> line1
    
    # Trim outliers
    if trim_fraction > 0:
        def trim_array(arr):
            arr_sorted = np.sort(arr)
            n = len(arr_sorted)
            k = int(trim_fraction * n)
            if k >= n // 2:
                return arr_sorted  # don't trim if too small
            return arr_sorted[k : n - k]
        
        d1 = trim_array(d1)
        d2 = trim_array(d2)
    
    # Symmetric mean distance
    mean_dist = (d1.mean() + d2.mean()) / 2.0
    return mean_dist

def measure_ringwidth(lines, trim_fraction=0.1):
    
    mean_ring_distances = [
        one_ringwidth(lines[i], lines[i+1], trim_fraction)
        for i in range(len(lines)-1)
    ]
    
    return mean_ring_distances



def compute_cell_distances(celldata, skeleton_boundaries, year_col="year"):
    """
    Compute distance of each cell centroid to its current and previous ring boundary.
    The first ring is excluded since it has no previous boundary.
    """

    # Ensure a copy and index by label
    df = celldata.copy().set_index("label")

    # Initialize columns
    df["dist_to_next"] = np.nan
    df["dist_to_prev"] = np.nan
    #df["prev_ring"] = np.nan  # store previous ring index
    
    # Loop over boundaries
    for i, boundary in enumerate(skeleton_boundaries):
        if boundary is None or len(boundary) == 0:
            continue  # skip empty boundaries

        boundary = np.array(boundary)
        tree = cKDTree(boundary)

        # Cells in current ring i+1
        cells_in_ring = df[df[year_col] == (i + 1)]
        if len(cells_in_ring) == 0:
            continue

        centroids = cells_in_ring[["centroid-0", "centroid-1"]].values

        # Distance to CURRENT ring line
        d_curr, _ = tree.query(centroids)
        df.loc[cells_in_ring.index, "dist_to_next"] = d_curr

        # Distance to PREVIOUS ring line (only if exists)
        if i > 0:
            prev_boundary = np.array(skeleton_boundaries[i - 1])
            tree_prev = cKDTree(prev_boundary)
            d_prev, _ = tree_prev.query(centroids)
            df.loc[cells_in_ring.index, "dist_to_prev"] = d_prev
            #df.loc[cells_in_ring.index, "prev_ring"] = i  # index previous ring

    # Remove first ring (no previous reference)
    #df = df[df["prev_ring"].notna()]

    return df


def filter_radial_files(celldata):
    
        # Step 1: Sort cells by file_rank_scaled within each radial_file in each year
    celldata = celldata.copy()    
    celldata_sorted = celldata.sort_values(["year", "radial_file", "file_rank_scaled"])
    
    # Step 2: Identify first and last cell in each radial file
    first_last_cells = celldata_sorted.groupby(["year", "radial_file"]).agg(
        first_idx=("file_rank_scaled", "idxmin"),
        last_idx=("file_rank_scaled", "idxmax")
    )
    
    # Step 3: Check distances for validity
    valid_radial_files = []
    
    for (year, radial_file), row in first_last_cells.iterrows():
        first_cell = celldata_sorted.loc[row["first_idx"]]
        last_cell = celldata_sorted.loc[row["last_idx"]]
        
        cond_first = first_cell["dist_to_prev"] < first_cell["diameter_rad"]
        cond_last = last_cell["dist_to_next"] < (last_cell["diameter_rad"] + 2*last_cell["WallThickness"])
        
        if cond_first and cond_last:
            valid_radial_files.append((year, radial_file))
    
    # Step 4: Filter celldata based on year × radial_file combinations
    valid_mask = celldata_sorted.set_index(["year", "radial_file"]).index.isin(valid_radial_files)
    
    celldata_sorted["valid_radial_file"] = valid_mask
    
    return celldata_sorted

def add_radialfile_stats(celldata, ringprops_df):
    """
    Compute ring-level radial file statistics and merge into ringprops_df.
    
    Parameters
    ----------
    celldata : pd.DataFrame
        Must contain columns: 'year', 'radial_file', 'woodzone'.
        Each cell should have a radial_file and ring (year) assignment.
    ringprops_df : pd.DataFrame
        Must contain column 'label' corresponding to year.
    
    Returns
    -------
    ringprops_df : pd.DataFrame
        Original DataFrame with added columns:
        - most_freq_n_cells
        - most_freq_earlywood
        - most_freq_latewood
    """
    celldata=celldata.copy()
    
    nb_cells = (
        celldata.groupby("year").size()
        .rename("nb_cells")
    )
    
    nb_rfiles = (
        celldata.groupby(["year", "radial_file"]).size()
        .reset_index()
        .groupby("year")
        .size()
        .rename("nb_rfiles")
    )
    
    df = celldata[celldata["valid_radial_file"]].copy()
    
    # Group by ring and radial_file
    grouped = df.groupby(["year", "radial_file"])
    
    n_valid_cells = (
        df.groupby("year").size()
        .rename("nb_cells_val")
    )
    
    n_valid_files = (
        grouped.size()
        .reset_index()
        .groupby("year")
        .size()
        .rename("nb_rfiles_val")
    )
    
    # Most frequent total number of cells per radial_file in a ring
    cells_per_file = grouped.size().reset_index(name="n_cells_per_file")
    most_freq_cells = (
        cells_per_file.groupby("year")["n_cells_per_file"]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("nb_cells_mode")
    )

    # Most frequent number of earlywood cells per radial_file in a ring
    ew_counts = grouped["woodzone"].apply(lambda x: (x == "earlywood").sum()).reset_index(name="n_ew")
    most_freq_ew = (
        ew_counts.groupby("year")["n_ew"]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("nb_ewcells_mode")
    )

    # Most frequent number of latewood cells per radial_file in a ring
    lw_counts = grouped["woodzone"].apply(lambda x: (x == "latewood").sum()).reset_index(name="n_lw")
    most_freq_lw = (
        lw_counts.groupby("year")["n_lw"]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("nb_lwcells_mode")
    )

    # Merge all statistics
    stats_df = pd.concat([nb_cells, 
                          nb_rfiles, 
                          n_valid_cells, 
                          n_valid_files, 
                          most_freq_cells, 
                          most_freq_ew, 
                          most_freq_lw], 
                         axis=1).reset_index()
    
    # Merge with ringprops_df
    ringprops_df = ringprops_df.merge(stats_df, left_on="label", right_on="year", how="left")
    
    # Optionally drop the temporary 'year' column
    #ringprops_df = ringprops_df.drop(columns=["year"])
    
    return ringprops_df

def early_latewood_width(celldata, ringprops_df):
    """
    Adds earlywood and latewood width per ring based on valid radial files.
    """

    # Work only with valid radial file cells
    df = celldata[celldata["valid_radial_file"]].copy()

    results = []

    for year, group in df.groupby("year"):
        ew_widths = []
        lw_widths = []

        for radial_file, rf_group in group.groupby("radial_file"):

            # --- Earlywood width ---
            ew_cells = rf_group[rf_group["woodzone"] == "earlywood"]
            if not ew_cells.empty:
                last_ew = ew_cells.loc[ew_cells["file_rank_scaled"].idxmax()]
                ew_width = last_ew["dist_to_prev"] + 0.5 * last_ew["diameter_rad"]
                ew_widths.append(ew_width)

            # --- Latewood width ---
            lw_cells = rf_group[rf_group["woodzone"] == "latewood"]
            if not lw_cells.empty:
                first_lw = lw_cells.loc[lw_cells["file_rank_scaled"].idxmin()]
                lw_width = (
                    first_lw["dist_to_next"]
                    + 0.5 * first_lw["diameter_rad"]
                    + first_lw["WallThickness"]
                )
                lw_widths.append(lw_width)

        # Compute means for the ring
        results.append({
            "year": year,
            "earlywood_width": np.mean(ew_widths) if ew_widths else np.nan,
            "latewood_width": np.mean(lw_widths) if lw_widths else np.nan,
        })

    width_df = pd.DataFrame(results)

    # Merge into ringprops_df
    ringprops_df = ringprops_df.merge(width_df, left_on="label", right_on="year", how="left")

    return ringprops_df
