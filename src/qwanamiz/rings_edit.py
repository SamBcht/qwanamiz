# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:32:17 2026

@author: sambo
"""

import os
import datetime
import argparse
import pickle

# Application library imports
import numpy as np
import pandas as pd
#import networkx as nx
from skimage.measure import regionprops_table
#from skimage.measure import regionprops
#from skimage.morphology import convex_hull_image
#from skimage import measure
# scipy imports
#from scipy import ndimage as ndi

#import napari

#from qwanamiz import qwanamiz as qmiz
from qwanamiz import rings_functions as qrings
from qwanamiz import qwanaplots as qplots
#import rings_functions as qrings
#import qwanaplots as qplots
#import vonmisesmix as qwanamiz.vonmisesmix

# Define script specific functions
def load_manual_ring_regions(filepath):
    ring_regions = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            regions = [int(x) for x in line.split()]
            ring_regions.append(regions)

    return ring_regions

def find_ring_lines_manual(
    cells,
    region_to_cells,
    ring_regions,
    exclude_same_radial_file=False,
    radial_col="radial_file"
):
    """
    Manual version of find_ring_lines using user-defined region groups.

    Parameters
    ----------
    cells : DataFrame
        Boundary cells (rightcells_df)
    region_to_cells : dict
        Mapping region -> list of cell labels
    ring_regions : list of list
        Each sublist = regions forming one ring
    exclude_same_radial_file : bool
        If True, keep only one cell per radial file per ring
    radial_col : str
        Column name for radial file

    Returns
    -------
    rings : dict
        {representative_region: [cell labels]}
    final_top_sequence : list
        List of representative regions (first region of each line)
    """

    # Sort once globally (like your original function)
    sorted_cells = cells.sort_values(by="centroid-0")

    rings = {}
    final_top_sequence = []

    for regions in ring_regions:

        if len(regions) == 0:
            continue

        # Representative region = first one in the line
        rep_region = regions[0]

        # --- Collect all labels from all regions
        all_labels = []
        for r in regions:
            if r not in region_to_cells:
                print(f"Warning: region {r} not found")
                continue
            all_labels.extend(region_to_cells[r])

        # --- Extract corresponding cells
        ring_cells = sorted_cells[
            sorted_cells["label"].isin(all_labels)
        ].copy()

        # --- Optional filtering: one cell per radial file
        if exclude_same_radial_file:
            counts = ring_cells[radial_col].value_counts()
            valid_radials = counts[counts == 1].index
            ring_cells = ring_cells[ring_cells[radial_col].isin(valid_radials)]

        # --- Store labels
        rings[rep_region] = ring_cells["label"].to_list()
        final_top_sequence.append(rep_region)

    return rings, final_top_sequence


def main():

    parser = argparse.ArgumentParser(description="Manual ring correction pipeline")

    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--only",
        nargs="+",
        help="List of sampleIDs to process (space-separated)"
    )
    parser.add_argument("--pixel_size", type=float, default=0.55)
    parser.add_argument("--first-year", dest = "firstyear", type = int, default = 1,
                        help = """The calendar year when the first ring was formed, used for assigning cells to years. Defaults to 1 (year unknown).""")
    parser.add_argument("--exclude_radial_duplicates", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable Napari viewer (interactive mode)"
    )

    args = parser.parse_args()

    base_folder = args.input_dir
    pix_to_um = args.pixel_size
    
    # --- Find all *_outputs folders ---
    output_dirs = sorted([
        os.path.join(base_folder, d)
        for d in os.listdir(base_folder)
        if d.endswith("_outputs") and os.path.isdir(os.path.join(args.input_dir, d))
    ])
    
    #print("BASE FOLDER:", base_folder)
    #print("FOUND OUTPUTS:", output_dirs)
    #print("ONLY FILTER:", args.only)
    
    if args.only:
        output_dirs = [
            d for d in output_dirs
            if os.path.basename(d).replace("_outputs", "") in args.only
        ]
    
    if not output_dirs:
        print("No *_outputs folders found")
        return
    
    # --- Loop over samples ---
    for output_folder in output_dirs:
    
        start = datetime.datetime.now()
    
        sampleID = os.path.basename(output_folder).replace("_outputs", "")
        base_prefix = os.path.join(output_folder, sampleID)
    
        print(f" Processing {sampleID}")
    
        # --- Check edit file ---
        edit_path = os.path.join(output_folder, f"{sampleID}_edit.txt")
        if not os.path.exists(edit_path):
            print(f"Skipping {sampleID} (no edit file)")
            continue
    
        try:
            # ---------------- LOAD ----------------
            images = np.load(f"{base_prefix}_imgs.npz")
            ring_images = np.load(f"{base_prefix}_ring_imgs.npz")
    
            celldata = pd.read_csv(f"{base_prefix}_ringcells.csv")
            celldata.set_index('label', drop=False, inplace=True)
    
            prediction = images['bw_img']
            expanded_labels = images['explabs']
            new_boundaries = ring_images['new_boundaries']
    
            # ---------------- VIEWER ----------------
            if args.viewer:
                import napari
                print(f"Opening Napari for {sampleID}...")
            
                viewer = napari.Viewer()
                viewer.add_image(prediction, name='Original B&W', scale=[pix_to_um]*2)
                viewer.add_labels(new_boundaries, name='Boundaries', scale=[pix_to_um]*2)
            
                print("Edit your file if needed.")
                print("Then close Napari and press ENTER to continue...\n")
            
                napari.run()  # blocks until viewer is closed
            
                input("Press ENTER to continue processing...")
    
            # ---------------- MAPPING ----------------
            cell_to_region, region_to_cells = qrings.map_cell_to_region(
                new_boundaries > 0, new_boundaries, expanded_labels
            )
    
            # clean mapping
            valid_labels = set(celldata.index)
            region_to_cells = {
                r: [c for c in cells if c in valid_labels]
                for r, cells in region_to_cells.items()
            }
    
            # ---------------- MANUAL EDIT ----------------
            ring_regions = load_manual_ring_regions(edit_path)
    
            ring_lines, final_top = find_ring_lines_manual(
                celldata,
                region_to_cells,
                ring_regions,
                exclude_same_radial_file=args.exclude_radial_duplicates
            )
    
            # ---------------- POLYGONS ----------------
            ring_polygons = qrings.draw_polygons(
                cells=celldata,
                ring_lines=ring_lines,
                upper_sequence=final_top,
                image_width=expanded_labels.shape[1] * pix_to_um
            )
    
            # ---------------- YEARS ----------------
            celldata = qrings.assign_years(
                cells=celldata,
                polygons=ring_polygons,
                year0=args.firstyear
            )
    
            celldata, suspect_labels = qrings.correct_large_lastcells(celldata)
    
            # ---------------- YEAR IMAGE ----------------
            year_dict = celldata.set_index("label")["year"].to_dict()
            year_image = np.vectorize(lambda x: year_dict.get(x, np.nan))(expanded_labels)
    
            # ---------------- SAVE ----------------
            np.savez_compressed(
                f"{base_prefix}_ring_imgs.npz",
                new_boundaries=new_boundaries,
                year_image=year_image
            )
            
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
            #mean_ringwidth_from_cells = distances_df.groupby("year")["cell_ring_width"].mean()

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
            #ringprops_df["rw_from_cells"] = ringprops_df["label"].map(mean_ringwidth_from_cells)

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
            #sampleID = celldata["SampleId"].unique()
            ringprops_df['SampleId'] = sampleID
            
            filtered_celldata = celldata[celldata["valid_radial_file"]].copy()

            # Make a blank mask same size as your labeled image
            filtered_labels = filtered_celldata.index
            filtered_mask = np.isin(expanded_labels, filtered_labels)

            #qmiz.update_runtime(start)

            ###########################################################################
            print("save outputs")
            # Saving the images of interest to file for later retrieval by ringview.py
            #os.makedirs(outdir, exist_ok=True)

            # Build base prefix for files inside this folder
            #base_prefix = os.path.join(outdir, base_name)
            
            # --- Save images for later retrieval ---
            #output_path = f"{base_prefix}_ring_imgs.npz"
            #np.savez_compressed(output_path,new_boundaries=new_boundaries,year_image=year_image)
            
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
                pix_to_um=pix_to_um,
                ring_boundaries=boundaries
            )
            
            print(f"Saved workflow output to {output_folder}")
            
            #qmiz.update_runtime(start)

    
        except Exception as e:
            print(f"Error in {sampleID}: {e}")
            continue

if __name__ == "__main__":
    main()
