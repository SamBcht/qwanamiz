# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:46:49 2024

@author: sambo
"""

import argparse
import napari
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ast

def qwa_napari_view(img_path, cells_path):
    
    
    images = np.load(img_path)

    cells = pd.read_csv(cells_path)

    pix_to_um = 0.55042690590734

    # Launch Napari viewer
    viewer = napari.Viewer()

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
    
    # Add the expanded labels image
    viewer.add_labels(images['rd_map'], 
                      name = 'Rays & Resin Ducts', 
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

    napari.run()
    
    return viewer

if __name__ == '__main__':


    # Set the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix", help = """The prefix of the sample to use qwanamiz.py with. qwanamiz will look for
                                            file paths corresponding to 'prefix + _imgs.npz' and 'prefix + _cells.csv'.
                                            These files should all be output by qwanaflow.py.""")

    args = parser.parse_args()
    
    imgs = args.prefix + "_imgs.npz"
    cells_df = args.prefix + '_cells.csv'
    
    qwa_napari_view(img_path = imgs, 
                    cells_path = cells_df)
