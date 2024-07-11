# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:46:49 2024

@author: sambo
"""

import napari
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ast

def qwa_napari_view(img_path, cells_path, edges_path):
    
    
    images = np.load(img_path)

    cells = pd.read_csv(cells_path)

    edges = pd.read_csv(edges_path,
                        converters={"centroid1": ast.literal_eval,
                                    "centroid2": ast.literal_eval})

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

    # Add centroids as points to the viewer directly from the DataFrame
    viewer.add_points(cells[['centroid-0', 'centroid-1']], 
                      name = 'Centroids', 
                      size = 5, 
                      face_color = 'red', 
                      edge_color = 'white')

    ## RADIAL FILES LAYER
    # Prepare the lines and colors for visualization
    # Filter the DataFrame
    final_df = edges.dropna(subset=['radial_file'])
        #(edges['radial_file'] != 'nan')]
    lines = []
    colors = []
    unique_radial_files = final_df['radial_file'].unique()

    # Create a colormap
    cmap = plt.get_cmap('tab20', len(unique_radial_files))
    color_map = {rf: cmap(i) for i, rf in enumerate(unique_radial_files)}

    # Prepare the lines and corresponding colors
    for edge, edge_data in final_df.iterrows():
        coords1 = edge_data['centroid1']
        coords2 = edge_data['centroid2']
        radial_file_id = edge_data['radial_file']
        
        # Append line coordinates and color to respective lists
        lines.append([coords1, coords2])
        colors.append(color_map[radial_file_id])


    # Add lines as shapes to the viewer
    viewer.add_shapes(lines, 
                      shape_type='line', 
                      edge_color=colors, 
                      edge_width=3, 
                      name='Radial Files')


    ## DIAMETERS LAYERS
    diam_df = cells.dropna(subset=['extr_rad'])

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
    
    return viewer

if __name__ == '__main__':
    
    imgs = r'C:\Users\sambo\Desktop\QWAnamiz_store\array_test.npz'
    cells_df = r'C:\Users\sambo\Desktop\QWAnamiz_store\output_measures\L20_F07-1M1-Sc1_cells.csv'
    edges_df = r'C:\Users\sambo\Desktop\QWAnamiz_store\output_measures\L20_F07-1M1-Sc1_adjacency.csv'
    
    qwa_napari_view(img_path = imgs, 
                    cells_path = cells_df, 
                    edges_path = edges_df)
    