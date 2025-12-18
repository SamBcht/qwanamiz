# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:46:49 2024

@author: sambo
"""

import os
import argparse
import napari
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ast
import pickle

def qwa_napari_view(img_path, cells_path, ring_path, ring_pickle, polygon_pickle, pix_to_um):
    
    # Loading the data from qwanaflow
    images = np.load(img_path)
    cells = pd.read_csv(cells_path)
    cells.set_index('label', inplace = True, drop = False)

    # Loading the data from qwanarings if the files exist
    if os.path.exists(ring_path):
        ring_images = np.load(ring_path)

    if os.path.exists(ring_pickle):
        with open(ring_pickle, "rb") as file:
            rings = pickle.load(file)

    if os.path.exists(polygon_pickle):
        with open(polygon_pickle, "rb") as file:
            polygons = pickle.load(file)

    # Launch Napari viewer
    viewer = napari.Viewer()

    # Drawing the binarized image
    viewer.add_image(images['bw_img'], name='Original B&W', scale = [pix_to_um, pix_to_um])

    # Add the labeled image
    viewer.add_labels(images['labs'], 
                      name = 'Lumens', 
                      scale = [pix_to_um, pix_to_um])

    # If you have the original prediction image, you can also add it
    #viewer.add_image(prediction, name='Prediction', scale = [pix_to_um, pix_to_um])

    # Add the distance map to the viewer
    viewer.add_image(images['dmap'], 
                     name = 'Distance Map', 
                     colormap = 'magma', 
                     scale = [pix_to_um, pix_to_um])

    # Add the expanded labels image
    viewer.add_labels(images['explabs'], 
                      name = 'Cells', 
                      scale = [pix_to_um, pix_to_um])
    
    # Add the watershed result image
    viewer.add_labels(images['watershed'],
                      name = 'Watershed segmentation',
                      scale = [pix_to_um, pix_to_um])

    # Add centroids as points to the viewer directly from the DataFrame
    viewer.add_points(cells[['centroid-0', 'centroid-1']], 
                      name = 'Centroids', 
                      size = 5, 
                      face_color = 'red', 
                      border_color = 'white')

    ## RADIAL FILES LAYER
    # Prepare the lines and colors for visualization
    lines = []
    colors = []
    unique_radial_files = cells['radial_file'].dropna().unique()
    unique_radial_files = unique_radial_files[unique_radial_files != 0]

    # Create a colormap
    cmap = plt.get_cmap('tab20', len(unique_radial_files))
    color_map = {rf: cmap(i) for i, rf in enumerate(unique_radial_files)}

    # Prepare the lines and corresponding colors
    for i in unique_radial_files:
        radial_file_df = cells[cells["radial_file"] == i]

        if radial_file_df.shape[0] == 1:
            continue

        radial_file_df = radial_file_df.sort_values("centroid-1")
        coords = list(zip(radial_file_df["centroid-0"], radial_file_df["centroid-1"]))
        
        # Append line coordinates and color to respective lists
        lines.append(coords)
        colors.append(color_map[i])


    # Add lines as shapes to the viewer
    viewer.add_shapes(lines, 
                      shape_type='path', 
                      edge_color=colors, 
                      edge_width=3, 
                      name='Radial Files')


    ## DIAMETERS LAYERS
    diam_df = cells.copy()
    diam_df = diam_df.dropna(subset=['extr_rad'])

    # Removing the 'np.float64' function calls from the input otherwise it will not be parsed properly
    diam_df['extr_rad'] = diam_df['extr_rad'].str.replace("np.float64", "")
    diam_df['extr_tan'] = diam_df['extr_tan'].str.replace("np.float64", "")
    diam_df['extr_rad'] = diam_df['extr_rad'].apply(ast.literal_eval)
    diam_df['extr_tan'] = diam_df['extr_tan'].apply(ast.literal_eval)

    # Prepare the lines and corresponding colors
    diameters_rad = []
    for cell, cell_data in diam_df.iterrows():
        #cell_data['extr_rad'] = cell_data['extr_rad'].apply(ast.literal_eval)
        if cell_data['extr_rad'] != 0: 
            coords1, coords2 = cell_data['extr_rad']
            
        elif cell_data['extr_rad'] == 0: 
            coords1 = [cell_data['centroid-0'], cell_data['centroid-1']]
            coords2 = [cell_data['centroid-0'], cell_data['centroid-1']]
        #coords2 = edge_data['centroid2']

        
        # Append line coordinates and color to respective lists
        diameters_rad.append([coords1, coords2])
        
    # Add lines as shapes to the viewer
    viewer.add_shapes(diameters_rad, 
                      shape_type='line', 
                      edge_color='blue', 
                      edge_width=1, 
                      name='Radial Diameters')

    # Prepare the lines and corresponding colors
    diameters_tan = []
    for cell, cell_data in diam_df.iterrows():
        #cell_data['extr_rad'] = cell_data['extr_rad'].apply(ast.literal_eval)
        if cell_data['extr_tan'] != 0: 
            coords1, coords2 = cell_data['extr_tan']
            
        elif cell_data['extr_tan'] == 0: 
            coords1 = [cell_data['centroid-0'], cell_data['centroid-1']]
            coords2 = [cell_data['centroid-0'], cell_data['centroid-1']]
        #coords2 = edge_data['centroid2']

        
        # Append line coordinates and color to respective lists
        diameters_tan.append([coords1, coords2])
        
    # Add lines as shapes to the viewer
    viewer.add_shapes(diameters_tan, 
                      shape_type='line', 
                      edge_color='green', 
                      edge_width=1, 
                      name='Tangential Diameters')

    # Drawing the set of boundaries found by qwanarings.py
    if os.path.exists(ring_path):
        viewer.add_labels(ring_images['new_boundaries'], name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

    # DRAWING THE TREE-RING BOUNDARIES FOUND BY qwanarings.py
    # Prepare the lines visualization
    if os.path.exists(ring_pickle):
        lines = []

        # Prepare the lines
        for i,region in enumerate(rings):
            region_cells = cells.loc[rings[region]]
            coords = list(zip(region_cells["centroid-0"], region_cells["centroid-1"]))
            lines.append(coords)

        # Add lines as shapes to the viewer
        viewer.add_shapes(lines,
                          shape_type='path',
                          edge_color='black',
                          edge_width=10,
                          name='Ring boundaries')

    # And ring polygons
    if os.path.exists(polygon_pickle):
        viewer.add_shapes(polygons,
                          shape_type='polygon',
                          edge_color='black',
                          face_color=[['red', 'green', 'blue', 'coral', 'black'][i % 5] for i in range(len(polygons))],
                          opacity = 0.3,
                          name='Tree-ring polygons')

    # Add cell year assignment
    if os.path.exists(ring_path):
        viewer.add_labels(ring_images['year_image'].astype(int), name="Tree-ring year", opacity=0.7, scale=[pix_to_um, pix_to_um])

    napari.run()
    
    return viewer

def main():


    # Set the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix", help = """The prefix of the sample to use qwanamiz.py with. qwanamiz will look for
                                            file paths corresponding to 'prefix + _imgs.npz', 'prefix + _cells.csv',
                                            'prefix + _ring_imgs.npz', 'prefix + _rings.pkl', and 'prefix + _polygons.pkl'
                                            These files should all be output by qwanaflow or qwanarings. Files output
                                            by the qwanarings utility are optional and will not be drawn if only qwanaflow
                                            has been run.""")

    parser.add_argument("--pixel-size", dest = "pixel", type = float, default = 0.55042690590734,
                        help = """Size of a pixel in the wanted measurement unit. Defaults to 0.55042690590734 micrometers.""")

    args = parser.parse_args()

    # Paths to the files produced by qwanaflow
    imgs = args.prefix + "_imgs.npz"
    cells_df = args.prefix + '_cells.csv'

    # Paths to the files produced by qwanarings
    ring_path = args.prefix + "_ring_imgs.npz"
    ring_pickle = args.prefix + "_rings.pkl"
    polygon_pickle = args.prefix + "_polygons.pkl"
    
    qwa_napari_view(img_path = imgs, 
                    cells_path = cells_df,
                    ring_path = ring_path,
                    ring_pickle = ring_pickle,
                    polygon_pickle = polygon_pickle,
                    pix_to_um = args.pixel)
