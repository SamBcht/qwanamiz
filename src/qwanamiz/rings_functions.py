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

    # Step 2: Build the graph by connecting regions via common_neighbors
    for region, upward_data in upward_neighbors.items():
        up_neighbor = upward_data["up_neighbor"]
        if up_neighbor in common_neighbors:
            G.add_edge(region, up_neighbor)

    for region, downward_data in downward_neighbors.items():
        down_neighbor = downward_data["down_neighbor"]
        if down_neighbor in common_neighbors:
            G.add_edge(region, down_neighbor)

    # Step 3: Find connected components (i.e., groups of regions that are connected through common neighbors)
    connected_components = list(nx.connected_components(G))

    # Step 4: Create a copy of the cleaned_boundary_labeled mask to store updated labels
    updated_boundary_labeled = last_labeled.copy()

    # Initialize new label counter
    new_label = np.max(last_labeled) + 1  # Start labeling from max+1 (to avoid overlap with existing labels)

    # Track original regions that do not merge
    all_regions = set(np.unique(last_labeled))
    merged_regions = set()  # Regions that were merged

    # Store the mapping: new_label → contributing common_neighbors
    merged_region_mapping = {}

    # Step 5: Iterate over each connected component
    for component in connected_components:
        if len(component) > 1:  # Only merge if there are multiple regions
            # Assign a new label for this merged component
            component_label = new_label
            new_label += 1  # Increment for next merged component

            # Get common_neighbors linked to the regions in this component
            contributing_common_neighbors = set()

            for region in component:
                if region in upward_neighbors and upward_neighbors[region]["up_neighbor"] in common_neighbors:
                    contributing_common_neighbors.add(upward_neighbors[region]["up_neighbor"])
                if region in downward_neighbors and downward_neighbors[region]["down_neighbor"] in common_neighbors:
                    contributing_common_neighbors.add(downward_neighbors[region]["down_neighbor"])

            # Store the mapping (merged label → original labels + common neighbors)
            merged_region_mapping[component_label] = {
                "original_regions": list(component),  # Original merged regions
                "common_neighbors": list(contributing_common_neighbors),  # Common neighbors used for merging
            }

            # Mark merged regions
            merged_regions.update(component)

            # Assign the new label to all regions in the current component
            for region in component:
                updated_boundary_labeled[last_labeled == region] = component_label

    # Step 6: Keep original labels for non-merged regions
    for region in all_regions - merged_regions:
        updated_boundary_labeled[last_labeled == region] = region
        
    ######## COMMON NEIGHBORS INTEGRATION

    new_boundary_labeled = updated_boundary_labeled.copy()

    # Step 2: Track newly integrated cells
    #newly_integrated_cells = set()

    # Step 3: Iterate over merged regions and integrate common_neighbors
    for merged_label, mapping in merged_region_mapping.items():
        common_neighbors_to_integrate = mapping["common_neighbors"]
        
        for common_neighbor in common_neighbors_to_integrate:
            # Assign the common_neighbor the same label as the merged region
            new_boundary_labeled[expanded_labels == common_neighbor] = merged_label
            #newly_integrated_cells.add(common_neighbor)
            
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

def get_nearest_extremity(cells_df, cell_to_region, upward_cells, downward_cells, incompatible_regions):
    # Step 2: Extract centroids and track label-to-region
    up_labels, up_centroids, up_regions = [], [], []
    for region, label in upward_cells.items():
        row = cells_df[cells_df["label"] == label]
        if not row.empty:
            up_labels.append(label)
            up_centroids.append((row["centroid-0"].values[0], row["centroid-1"].values[0]))
            up_regions.append(region)

    down_labels, down_centroids, down_regions = [], [], []
    for region, label in downward_cells.items():
        row = cells_df[cells_df["label"] == label]
        if not row.empty:
            down_labels.append(label)
            down_centroids.append((row["centroid-0"].values[0], row["centroid-1"].values[0]))
            down_regions.append(region)

    # Step 3: Compute distance matrix
    dist_matrix = cdist(np.array(up_centroids), np.array(down_centroids))

    # Step 4: Find nearest neighbors
    up_to_down = {up_labels[i]: down_labels[j] for i, j in enumerate(np.argmin(dist_matrix, axis=1))}
    down_to_up = {down_labels[j]: up_labels[i] for j, i in enumerate(np.argmin(dist_matrix.T, axis=1))}

    # Step 5: Find mutual nearest pairs with different regions
    region_pairs = []
    mutual_pairs = []
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
                    
    return mutual_pairs



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
        .min()
        .reset_index()
        .sort_values("centroid-1")
    )

    upper_region_sequence = upper_region_coords["region"].tolist()

    # --- Lower border: same process ---

    lower_region_coords = (
        downward_border_df.groupby("region")["centroid-1"]
        .min()
        .reset_index()
        .sort_values("centroid-1")
    )

    lower_region_sequence = lower_region_coords["region"].tolist()
    
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
def draw_polygons(cells, ring_lines, upper_sequence, image_height):
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
            corners = np.array([[image_height, 0], [0, 0]])
            i_polygon = np.concatenate([coords, corners])

        # We need to wrap back to the first vertex for a true polygon
        i_polygon = np.concatenate([i_polygon, i_polygon[0:1, :]])

        # Assigning that polygon to the list of polygons
        polygons.append(i_polygon)
        
        # Setting the previous coordinates to the current ones before restarting the loop
        prev_coords = coords

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
        polygon_indices = cell_indices[angle_sum > threshold_sum]

        # We consider a cell with an angle sum > 6 to be part of that polygon
        cells.loc[polygon_indices, 'year'] = i + year0

    return cells

