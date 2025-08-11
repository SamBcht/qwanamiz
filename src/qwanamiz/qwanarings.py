
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 14:40:54 2025

@author: sambo
"""

# Generic python imports
import os
import datetime
from collections import defaultdict
import argparse
import pickle

# Application library imports
import numpy as np
import pandas as pd
import networkx as nx

# qwanamiz-specific imports
import qwanamiz
import rings_functions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", help = """The prefix of the files to use for the analysis. Suffixes '_imgs.npz', '_cells.csv' and
                                              '_adjacency.csv' will be added to that prefix to obtain the input files.""")

    parser.add_argument("--pixel-size", dest = "pixel", type = float, default = 0.55042690590734,
                        help = """Size of a pixel in the wanted measurement unit. Defaults to 0.55042690590734 micrometers.""")

    parser.add_argument("--minimum-cells", dest = "mincells", type = int, default = 5,
                        help = """The minimum number of cells in a ring-boundary region to consider it. Defaults to 5.""")

    args = parser.parse_args()

    # Reading the input data
    images = np.load(f"{args.prefix}_imgs.npz")
    celldata = pd.read_csv(f"{args.prefix}_cells.csv")
    adjacency = pd.read_csv(f"{args.prefix}_adjacency.csv")

    # Explicitly setting the double index on the adjacency DataFrame
    adjacency.set_index(['label1', 'label2'], inplace=True)

    # Giving an explicit variable name to the expanded labels image
    expanded_labels = images['explabs']

    pix_to_um = args.pixel

    ##############################################################################
    # Detection of tree-ring transitions by comparing successive cells properties
    # (radial diameter and early-latewood classification)
    # Get lastcells in rings based on diameter and woodzone cell features
    lastcells_labels, rightcells_labels, leftcells_labels = rings_functions.get_lastcells(celldata, adjacency)

    # Create a mask where pixels belong to lastcells or their right_neighbors
    rightcells_mask = np.zeros_like(expanded_labels, dtype=bool)
    rightcells_mask[np.isin(expanded_labels, rightcells_labels)] = True

    ###############################################################################
    # Now we can filter the cell and adjacency dataframes based on cell classification
    # This allow us to filter the edges (adjacencies) and nodes (cells) involved in
    # a ring transition

    # Keep only cells whose label is in right_neighbor_labels
    rightcells_df = celldata[celldata["label"].isin(rightcells_labels)].copy()
    # Extract the lastcell labels as a set
    rightcells_labels = set(rightcells_df["label"])

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

    #### Now we find the most up- and downward cells in each ring boundary segments
    up_extremities, down_extremities = rings_functions.get_extremities(region_to_right, rightcells_df)

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

    # We then integrate up and down pairs and also merge regions accordingly
    # An update of the cell_to_region mapping is done internally
    final_boundaries = rings_functions.integrate_updown(upward_neighbors, 
                                                        downward_neighbors, 
                                                        up_down_pairs, 
                                                        updated_boundaries, 
                                                        expanded_labels)

    # We update the mapping of cells to their boundary region
    cell_to_region, region_to_cells = rings_functions.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)
    ###############################################################################
    #### INTEGRATION OF CELLS AT THE EXTREMITIES

    # We find in the remaining cells adjacent to extremities the ones that show
    # characteristics of ring transition
    labels_to_integrate = rings_functions.get_candidate_cells(celldata, remaining_labels, lastcells_labels, diameter_factor = 1.8)

    # Cells retained for integration are the ones with their direct left neighbor
    # showing a X times lower diameter
    # or a transition between earlywood and latewood
    boundaries = rings_functions.integrate_candidates(final_boundaries, 
                                                      expanded_labels, 
                                                      labels_to_integrate, 
                                                      cell_to_region, 
                                                      upward_neighbors, 
                                                      downward_neighbors)

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

    ############################################################################
    # FIND ADJACENCIES BETWEEN RING SEGMENTS AFTER ADDITION OF CELLS

    # Here we extend the adjacency research to all types instead of only radial_sel
    # Possibility to filter the dataframe to restrict the research

    # The function return a list of tuples with labels of the CONNECTED CELLS
    connected_regions = rings_functions.get_segment_adjacency(adjacency, cell_to_region, up_extremities, down_extremities)

    final_boundaries, new_cell_to_region = rings_functions.merge_by_cells(connected_regions, cell_to_region, boundaries, expanded_labels)

    cell_to_region, region_to_cells = rings_functions.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)

    # Find the extrmities of the new ring segments
    up_extremities, down_extremities = rings_functions.get_extremities(region_to_cells, rightcells_df)

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

    new_boundaries, new_cell_to_region = rings_functions.merge_by_cells(nearest_extremity, cell_to_region, final_boundaries, expanded_labels)

    cell_to_region, region_to_cells = rings_functions.map_cell_to_region(new_boundaries > 0, new_boundaries, expanded_labels)

    # At this stage we can remove spurious regions by excluding those with fewer than a given number of cells
    cell_to_region, region_to_cells = rings_functions.filter_boundaries(cell_to_region, region_to_cells, mincells = args.mincells)
    new_boundaries = rings_functions.update_boundary_labels(np.zeros_like(expanded_labels, dtype = int), cell_to_region, expanded_labels)

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

    # Result
    print("Upper border regions (left to right):", upper_region_sequence)
    print("Lower border regions (left to right):", lower_region_sequence)
    print("Matching upper regions :", matched_up)
    print("Matching lower regions :", matched_down)

    # Identifying true ring boundaries from the upper and lower sequences
    ring_lines = rings_functions.find_ring_lines(rightcells_df, region_to_cells, upper_region_sequence, lower_region_sequence)

    # Intersection: regions that have both an upward and a downward border cell
    regions_topdown = (set(upper_region_sequence) | set(matched_up)) & (set(lower_region_sequence) | set(matched_down))
    print(f"{len(regions_topdown)} regions touch both the top and bottom borders.")
    print("Valid regions :", regions_topdown)

    # Saving the images of interest to file for later retrieval by ringview.py
    output_path = f"{args.prefix}_ring_imgs"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path,
                        new_boundaries = new_boundaries)

    # Saving native python objects by serializing with pickle
    with open(f'{args.prefix}_rings.pkl', 'wb') as file:
        pickle.dump(ring_lines, file)

