
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 14:40:54 2025

@author: sambo
"""

# Generic python imports
import os
import datetime
import argparse
import pickle

# Application library imports
import numpy as np
import pandas as pd
import networkx as nx
from skimage.measure import regionprops_table

# qwanamiz-specific imports
from qwanamiz import qwanamiz as qmiz
from qwanamiz import rings_functions as qrings
from qwanamiz import qwanaplots as qplots

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_dir", required=True,
                    help="""Path to the main directory containing subfolders for each processed image. 
                    Suffixes '_imgs.npz', '_cells.csv' and '_adjacency.csv' must be in the subfolders to obtain the input files.""")

    parser.add_argument("--pixel-size", dest = "pixel", type = float, default = 0.55042690590734,
                        help = """Size of a pixel in the wanted measurement unit. Defaults to 0.55042690590734 micrometers.""")

    parser.add_argument("--minimum-cells", dest = "mincells", type = int, default = 4,
                        help = """The minimum number of cells in a ring-boundary region to consider it. Defaults to 4.""")

    parser.add_argument("--first-year", dest = "firstyear", type = int, default = 1,
                        help = """The calendar year when the first ring was formed, used for assigning cells to years. Defaults to 1 (year unknown).""")

    args = parser.parse_args()

    pix_to_um = args.pixel
    
    # --- Find all directories ending with "_outputs" ---
    output_dirs = sorted([
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir)
        if d.endswith("_outputs") and os.path.isdir(os.path.join(args.input_dir, d))
    ])
    
    if not output_dirs:
        print("No directories ending with '_outputs' found in input directory.")
        exit()
        
    # --- Process each directory ---
    for outdir in output_dirs:
    
        start = datetime.datetime.now()

        base_name = os.path.basename(outdir).replace("_outputs", "")
        print(f"Processing {base_name}...")

        # Reading the data needed for the analysis
        print("Reading input files")
        prefix = os.path.join(outdir, base_name)
        celldata, adjacency, expanded_labels, prediction = qrings.read_qwanarings_inputs(prefix)
        qmiz.update_runtime(start)
    
        ##############################################################################
        # Detection of tree-ring transitions by comparing successive cells properties
        # (radial diameter and early-latewood classification)
        print("Find boundary cells & connected components")
        celldata = qrings.morks_index(celldata)
        
        # Get lastcells in rings based on diameter and woodzone cell features
        lastcells, rightcells = qrings.get_lastcells(celldata, adjacency)
    
        # Create a mask where pixels belong to lastcells or their right_neighbors
        rightcells_mask = np.zeros_like(expanded_labels, dtype=bool)
        rightcells_mask[np.isin(expanded_labels, list(rightcells))] = True
    
        ###############################################################################
        # Now we can filter the cell and adjacency dataframes based on cell classification
        # This allow us to filter the edges (adjacencies) and nodes (cells) involved in
        # a ring transition
    
        # Keep only cells whose label is in right_neighbor_labels
        rightcells_df = celldata[celldata["label"].isin(rightcells)].copy()
        # Extract the lastcell labels as a set
    
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
        graph = qrings.boundary_graph(celldata, adjacency, lastcells, rightcells)
    
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
        right_to_region, region_to_right = qrings.map_cell_to_region(rightcells_mask, boundary_labeled, expanded_labels)
    
        qmiz.update_runtime(start)
        
        ###### FIND PROBLEMATIC BOUNDARY REGIONS
        # Problematic regions are those where there is more than one rightcell (or lastcell)
        # for a unique radial file
    
        # It's not a big deal for most of the image as ring boundaries are sufficiently separated
        # But it can be a bigger problem for images with very narrow rings
    
        # Problems very often arrive when 2 rightcells in radial files above each other
        # are adjacent by a little corner touching
        
        print("Find boundary segments to merge with adjacency")
    
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
        rightcells_boundary = qrings.update_boundary_labels(np.zeros_like(expanded_labels, dtype=int), right_to_region, expanded_labels)
    
        #### Now we find the most up- and downward cells in each ring boundary segments
        up_extremities, down_extremities = qrings.get_extremities(region_to_right, rightcells_df)
    
        ###############################################################################
        #### Then we search in the extremity cell adjacencies if they can unite ring 
        # boundary segments by their respective up and down extremities
    
        # Common Neighbors are cells adjacent to both the up extremity of one boundary
        # segment and the down extremity of another segment
    
        # Up and Down Pairs are pairs of adjacent cells where one is adjacent to the 
        # up extremity of one segment and the other is adjacent to the down extremity of
        # another segment
    
        # We also keep the remaining cells for further use
        common_neighbors, up_down_pairs, remaining_labels, upward_neighbors, downward_neighbors = qrings.get_extremity_neighbors(up_extremities, down_extremities, celldata)
    
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
        ###############################################################################
        #### INTEGRATION OF CELLS AT THE EXTREMITIES
    
        # We find in the remaining cells adjacent to extremities the ones that show
        # characteristics of ring transition
        labels_to_integrate = qrings.get_candidate_cells(celldata, remaining_labels, lastcells, diameter_factor = 1.8)
    
        # Cells retained for integration are the ones with their direct left neighbor
        # showing a X times lower diameter
        # or a transition between earlywood and latewood
        boundaries = qrings.integrate_candidates(final_boundaries, 
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
    
        cell_to_region, region_to_cells = qrings.map_cell_to_region(boundaries > 0, boundaries, expanded_labels)
    
    
        # Find the extrmities of the new ring segments
        up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)
        
        qmiz.update_runtime(start)
    
        ############################################################################
        # FIND ADJACENCIES BETWEEN RING SEGMENTS AFTER ADDITION OF CELLS
    
        # Here we extend the adjacency research to all types instead of only radial_sel
        # Possibility to filter the dataframe to restrict the research
        
        print("Iterative search of boundary segments to merge from distances")
    
        # The function return a list of tuples with labels of the CONNECTED CELLS
        connected_regions = qrings.get_segment_adjacency(adjacency, cell_to_region, up_extremities, down_extremities)
    
        final_boundaries, new_cell_to_region = qrings.merge_by_cells(connected_regions, cell_to_region, boundaries, expanded_labels)
    
        cell_to_region, region_to_cells = qrings.map_cell_to_region(final_boundaries > 0, final_boundaries, expanded_labels)
    
        # Find the extrmities of the new ring segments
        up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)
    
        # This function allows to get a list of regions containing at least one cell of
        # the same radial file. Can be used to avoid merging of boundary segments belonging
        # to different files
        incompatible_region_pairs = qrings.incompatible_regions(celldata, cell_to_region)
    
        # Finally, we find each up_extremity's nearest down_extremity and vice versa.
        # We keep pairs of up and down that are mutually the nearest for each other
        # When a region has only one cell that is thus both the up and down extremity,
        # nearest extremities are the same point and they are excluded from the merging
        # This avoid merging potential region falsely identified as boundary
        nearest_extremity,_ = qrings.get_nearest_extremity(rightcells_df, 
                                                                  cell_to_region, 
                                                                  up_extremities, 
                                                                  down_extremities, 
                                                                  incompatible_region_pairs)
        
        pairs_df, valid, excluded = qrings.analyze_pairs_angles(celldata, nearest_extremity)
    
        nearest_extremity = valid
    
    
        # This step could be repeat iteratively to add new connections
        # But we will still have non connected regions where
        # - up and down extremities are in the same radial files
        # - there are several up and down extremities in a small zone, this could introduce errors
    
        new_boundaries, new_cell_to_region = qrings.merge_by_cells(nearest_extremity, cell_to_region, final_boundaries, expanded_labels)
    
        cell_to_region, region_to_cells = qrings.map_cell_to_region(new_boundaries > 0, new_boundaries, expanded_labels)
    
        # At this stage we can remove spurious regions by excluding those with fewer than a given number of cells
        cell_to_region, region_to_cells = qrings.filter_boundaries(cell_to_region, region_to_cells, mincells = args.mincells)
        new_boundaries = qrings.update_boundary_labels(np.zeros_like(expanded_labels, dtype = int), cell_to_region, expanded_labels)
    
        
        ###############################################################################
        #### SECOND SEARCH OF NEAREST EXTREMITY
        # This step does the same as before but without regions containing few cells and with new regions merged
        # Find the extrmities of the new ring segments
        up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)
        
        incompatible_region_pairs = qrings.incompatible_regions(celldata, cell_to_region)
    
        nearest_extremity, _ = qrings.get_nearest_extremity(rightcells_df, cell_to_region, up_extremities, down_extremities, incompatible_region_pairs)
    
        pairs_df, valid, excluded = qrings.analyze_pairs_angles(celldata, nearest_extremity)
    
        nearest_extremity = valid
        
        new_boundaries, new_cell_to_region = qrings.merge_by_cells(nearest_extremity, cell_to_region, new_boundaries, expanded_labels)
    
    
        cell_to_region, region_to_cells = qrings.map_cell_to_region(new_boundaries > 0, new_boundaries, expanded_labels)
    
        # At this stage we can remove spurious regions by excluding those with fewer than a given number of cells
        cell_to_region, region_to_cells = qrings.filter_boundaries(cell_to_region, region_to_cells, mincells = 5)
        new_boundaries = qrings.update_boundary_labels(np.zeros_like(expanded_labels, dtype = int), cell_to_region, expanded_labels)
    
        up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)
    
        incompatible_pairs = set()
    
        nearest_extremity, all_regions = qrings.get_nearest_extremity(rightcells_df, 
                                                                               cell_to_region, 
                                                                               up_extremities, 
                                                                               down_extremities, 
                                                                               incompatible_pairs,
                                                                               new_boundaries.shape,
                                                                               75,
                                                                               pix_to_um)
    
        valid_pairs, excluded_pairs = qrings.filter_isolated_pairs(nearest_extremity, all_regions)
        
        new_boundaries, new_cell_to_region = qrings.merge_by_cells(valid_pairs, cell_to_region, new_boundaries, expanded_labels)
    
    
        cell_to_region, region_to_cells = qrings.map_cell_to_region(new_boundaries > 0, new_boundaries, expanded_labels)
    
        # At this stage we can remove spurious regions by excluding those with fewer than a given number of cells
        cell_to_region, region_to_cells = qrings.filter_boundaries(cell_to_region, region_to_cells, mincells = 5)
        new_boundaries = qrings.update_boundary_labels(np.zeros_like(expanded_labels, dtype = int), cell_to_region, expanded_labels)
        
        qmiz.update_runtime(start)

        ################################################################################
        # FIND REGION EXTREMITIES NEAR THE BORDERS OF THE IMAGE WITH ELLIPSE
        
        print("Find ring sequences & draw rings")
    
        region_classes, ring_regions, seq = qrings.classify_regions_by_axis(new_boundaries, pix_to_um)
        
        ###############################################################################
        # FIND REGION EXTREMITIES NEAR THE BORDERS OF THE IMAGE
        up_extremities, down_extremities = qrings.get_extremities(region_to_cells, rightcells_df)
    
    
        all_border_cells, upper_region_sequence, lower_region_sequence, matched_up, matched_down, unjustified = qrings.get_border_cells(rightcells_df, 
                                                                                                                                 cell_to_region, 
                                                                                                                                 up_extremities,
                                                                                                                                 down_extremities,
                                                                                                                                 image_height = expanded_labels.shape[0], 
                                                                                                                                 image_width = expanded_labels.shape[1], 
                                                                                                                                 border_margin = 75, 
                                                                                                                                 pix_to_um = pix_to_um)
    
        # Result
        y_positions, sequences = qrings.get_region_sequences(new_boundaries, n_lines=20)
    
    
        aligned, regions = qrings.align_region_sequences(sequences, gap_value=None, upper_seq=seq["top"], lower_seq=seq["bottom"])
    
        candidates, cu, cl = qrings.find_merge_candidates(
            seq["top"], seq["bottom"]
        )
        
        print("Merge candidates:", candidates)
    
        cleaned_matrix = qrings.remove_singleton_columns(aligned)
    
        filled = qrings.fill_columns(cleaned_matrix, candidates, 0.79, region_classes)
    
        incomplete = qrings.find_incomplete_regions(filled)
        
        final_merge = qrings.filter_incomplete_regions(incomplete_info=incomplete, 
                                             classifications=region_classes,
                                             merge_candidates=candidates, 
                                             matched_up=matched_up, 
                                             matched_down=matched_down)
        
        valid, duplicates = qrings.filter_pairs_overlap(final_merge, region_classes, filled)
        print(valid)
        print(duplicates)
        
        pair_extremities = {}
    
        for r1, r2 in valid:
            cell1 = qrings.get_extremity_cell(r1, up_extremities, down_extremities, region_classes)
            cell2 = qrings.get_extremity_cell(r2, up_extremities, down_extremities, region_classes)
            coord1 = qrings.get_coordinates(cell1, rightcells_df)
            coord2 = qrings.get_coordinates(cell2, rightcells_df)
            pair_extremities[(r1, r2)] = (coord1, coord2)
            
        all_merge_pairs = qrings.select_regions_to_merge(pair_extremities, candidates, final_merge)
    
    
        aligned_top, aligned_bottom = qrings.build_aligned_sequences(filled, all_merge_pairs, final_merge)
    
        print("Top   →", aligned_top)
        print("Bottom→", aligned_bottom)
    
        # Identifying true ring boundaries from the upper and lower sequences
        ring_lines = qrings.find_ring_lines(rightcells_df, region_to_cells, aligned_top, aligned_bottom)
    
        # Getting polygon coordinates defining tree rings from the ring lines
        ring_polygons = qrings.draw_polygons(cells = celldata, ring_lines = ring_lines, upper_sequence = aligned_top, image_width = expanded_labels.shape[1] * pix_to_um)
    
        # Assigning rings to years based on the polygon coordinates
        celldata = qrings.assign_years(cells = celldata, polygons = ring_polygons, year0 = args.firstyear)
    
        # Create an image of cell assignment for display
        celltemp = celldata.copy()
        celltemp.set_index('label', inplace = True)
        year_dict = celltemp['year'].to_dict()
        year_image = np.vectorize(lambda x: np.nan if year_dict.get(x) is None else year_dict.get(x))(expanded_labels)
        
        qmiz.update_runtime(start)

        ###############################################################################
        # RINGWIDTH & RING-LEVEL MEASUREMENTS
        
        print("Refine ring boundaries & calculate ring properties")
        
        # Get exact ring boundaries
        boundaries = qrings.extract_ring_boundaries(year_image, pix_to_um)
        
        # Measure ringwidth from ring boundary lines
        rw = qrings.measure_ringwidth(boundaries)
        
        # Get cells distances from previous and next ring boundaries
        distances_df = qrings.compute_cell_distances(celldata, boundaries, year_col="year")
        
        ## Calculate ring width from cells as checkpoint
        # Add total boundary distance per cell
        distances_df["cell_ring_width"] = distances_df["dist_to_next"] + distances_df["dist_to_prev"]
    
        # Compute mean ring width from cell distances
        mean_ringwidth_from_cells = distances_df.groupby("year")["cell_ring_width"].mean()
    
        # Convert mean_ring_distances (list from earlier) to a Series for comparison
        # Those were based on skeleton boundaries (already in pixel or µm?)
        ringwidth_from_boundaries = pd.Series(rw, index=range(2, len(rw)+2))
        
        ##############################################################################
        celldata = distances_df.copy()
    
        ringprops_df = pd.DataFrame(
            regionprops_table(
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
    
        celldata = qrings.filter_radial_files(celldata)
        
        ringprops_df = qrings.add_radialfile_stats(celldata, ringprops_df)
    
        ringprops_df = qrings.early_latewood_width(celldata, ringprops_df)
        
        ringprops_df = ringprops_df.drop(
            columns = [
                'year_y'])
        
        # Add SampleId column to ring dataframe
        ## TO DO : Add correspondance check with base_name : if "SampleId" in celldata:
            #assert celldata["SampleId"].nunique() == 1
            #assert celldata["SampleId"].iloc[0] == base_name
        sampleID = celldata["SampleId"].unique()
        ringprops_df['SampleId'] = sampleID[0]
    
        celldata = celldata.drop(
            columns = [
                'next_diameter_rad',
                'prev_diameter_rad',
                'next_woodzone'])
        
        filtered_celldata = celldata[celldata["valid_radial_file"]].copy()
    
        # Make a blank mask same size as your labeled image
        filtered_mask = np.zeros_like(expanded_labels, dtype=bool)
        filtered_labels = filtered_celldata.index
        filtered_mask[np.isin(expanded_labels, filtered_labels)] = True
    
        qmiz.update_runtime(start)
    
        ###########################################################################
        print("save outputs")
        # Saving the images of interest to file for later retrieval by ringview.py
        os.makedirs(outdir, exist_ok=True)

        # Build base prefix for files inside this folder
        base_prefix = os.path.join(outdir, base_name)  # e.g., output/image1_outputs/image1
        
        # --- Save images for later retrieval ---
        output_path = f"{base_prefix}_ring_imgs.npz"
        np.savez_compressed(output_path,
                            new_boundaries=new_boundaries,
                            year_image=year_image)
        
        # --- Save python objects with pickle ---
        with open(f"{base_prefix}_rings.pkl", "wb") as file:
            pickle.dump(ring_lines, file)
        
        with open(f"{base_prefix}_polygons.pkl", "wb") as file:
            pickle.dump(ring_polygons, file)
        
        # --- Save DataFrames as CSV ---
        celldata.to_csv(f"{base_prefix}_ringcells.csv", index=True)
        ringprops_df.to_csv(f"{base_prefix}_rings.csv", index=False)
        
        # --- Draw rings to an image ---
        qplots.draw_rings(
            prediction=prediction,
            year_image=year_image,
            filtered_mask=filtered_mask,
            celldata=celldata,
            output_path=f"{base_prefix}_img.png",
            pix_to_um=pix_to_um
        )
        
        print(f"Saved workflow output to {outdir}")
        
        qmiz.update_runtime(start)

