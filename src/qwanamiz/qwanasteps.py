# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:12:33 2024

@author: sambo
"""


import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.color
import skimage.metrics

from scipy.ndimage import distance_transform_edt
#import matplotlib
import matplotlib.pyplot as plt

import skimage.graph
import skimage.util
import qwanamiz
#from tools import histogram
#from mixture import density, vonmises_pdfit, mixture_pdfit, pdfit
#from typing import Tuple 

#from scipy.stats import vonmises
import datetime
#import ast
import napari

#####################################################################
# TEST if there is a difference between the png and the numpy array
#img_path = 'C:/Users/sambo/Desktop/QWAnamiz_store/final_outputs/L20_F02-1M2-Sc5_array.npy'

#png_test = np.load(img_path)

png_path = 'C:/Users/sambo/Desktop/QWAnamiz_store/final_outputs/L20_F13-1M1-Sc1_segmented.png'

prediction = skimage.io.imread(png_path)

#(score, diff) = skimage.metrics.structural_similarity(prediction, png_test, full=True, data_range = 255)
#print("Image Similarity: {:.4f}%".format(score * 100))
# -> The only difference is that the numpy array is of type float64
# whereas the png is of type uint8


## IMAGE METADATA AND RESOLUTION :
# 10x scans have a resolution of 46146 dpi. We can define a scaling factor
# with conversion : 1 Pixel = 0.55042690590734 Microns
pix_to_um = 0.55042690590734

############################################################################
# Launch the NAPARI viewer

# Launch Napari viewer
viewer = napari.Viewer()

# Add the segmented image layer
viewer.add_image([prediction, prediction[::2, ::2]], 
                 name='Prediction', 
                 scale = [pix_to_um, pix_to_um])


#################################################################################
#### Step 1: Cell Labeling and Measurements

## CELL LUMEN DETECTION : the label function from scikit.image package
# compute a connected components analysis on the binary image.
# Two pixels are connected e.g. belong to the same cell lumen
# when they are neighbors and have the same value (here 'black' or 'white').
# See scikit-image documentation for more information
# Adjust the connectivity if needed. By default, connectivity is defined by 
# prediction.ndim = 2
labeled_image = skimage.measure.label(prediction)

# Add the labeled image
viewer.add_labels([labeled_image, labeled_image[::2, ::2]], 
                  name = 'Lumens', 
                  scale = [pix_to_um, pix_to_um])

## CELL LUMEN MEASUREMENTS : the function 'regionprops_table measure the lumen 
# traits listed in properties. Return a pandas dataframe with measurements
# in microns if spacing is set with the scaling factor.
# See scikit-image documentation for more information
# See also additionnal properties that could be computed in the documentation
regionprops_df = pd.DataFrame(
    skimage.measure.regionprops_table(
        labeled_image,
        spacing = pix_to_um,
        properties = (
            'label',
            'area',
            'major_axis_length',
            'minor_axis_length',
            'centroid',
            'orientation',
            'perimeter_crofton',
            'image',
            'bbox')))


## DISTANCE MAP OF CELL WALLS : Compute the distance map of cell walls pixels,
# e.g. background pixels. Return an image where each back ground pixel takes
# the value of the distance to the nearest cell lumen in microns with the 
# sampling parameter set to the scaling factor.
distance_map, nearest_label_coords = distance_transform_edt(labeled_image == 0,
                                      sampling = pix_to_um,
                                      return_indices = True)

# Add the distance map to the viewer
viewer.add_image([distance_map, distance_map[::2, ::2]],
                 name = 'Distance Map', 
                 colormap = 'magma', 
                 scale = [pix_to_um, pix_to_um])

## EXPAND CELL LUMENS UNTIL JUNCTION WITH ADJACENT CELLS : Expand labels in 
# label image by distance pixels without overlapping.
# the expand_cells function is a modified version of the original expand_labels
# to avoid calculating the distance map a second time.
# The distance parameter can be considered as a cell wall thickness threshold
# Two lumens separated by more than two times the distance won't be considered
# as adjacent.
expanded_labels = qwanamiz.expand_cells(labeled_image,
                                        distance_map,
                                        nearest_label_coords,
                                        distance = 10,
                                        spacing = pix_to_um)

## MEASURE PROPERTIES OF WHOLE CELLS
# Other interesting properties can be added
# Should be noted that cell are approximate and the properties value
# can have consequent biases (ex : cells touching a ray have an overestimated
# area)
expandprops_df = pd.DataFrame(
    skimage.measure.regionprops_table(
        expanded_labels,
        spacing = pix_to_um,
        properties = (
            'label',
            'area')
        )
    )

# Join the newly measured cell properties to the lumen dataframe
regionprops_df = regionprops_df.join(expandprops_df.set_index('label'), 
                    on = 'label',  
                    lsuffix = '_lumen',
                    rsuffix = '_cell',
                    validate = '1:1')

# Add the expanded labels image
viewer.add_labels([expanded_labels, expanded_labels[::2, ::2]], 
                  name = 'Cells', 
                  scale = [pix_to_um, pix_to_um])

# Add centroids as points to the viewer
viewer.add_points(regionprops_df[['centroid-0', 'centroid-1']], 
                  name = 'Centroids', 
                  size = 5, 
                  face_color = 'red', 
                  edge_color = 'white')

####################################################################################

##################################################################################
#### Step 2 : Cell adjacency analysis

## REGION ADJACENCY GRAPH : The get_adjacent_labels function compute a simplified
# Region Adjacency Graph. Return a set of adjacent cells pairs e.g. each pair
# of labels that share a common border.
adj_graph = qwanamiz.get_adjacent_labels(expanded_labels)

# Transform the set of label pairs in a dataframe, retrieve label centroid
# coordinates and measure edge angle, length and center
adjacency = qwanamiz.adjacency_dataframe(adj_graph, regionprops_df)

###############################################################################

##############################################################################
## DIRECTIONNALITY ANALYSIS

# Determine the number of subsamples based on the 
# image size
# The calculate_grid function is designed to automatically 
# return the number of subdivision
# of the original image to be analysed at the next step.
# It is based on a fixed scaled size of the subsamples
# Defaults : 480 to 750 µm height X 1250 to 2000 µm width
img_height, img_width = prediction.shape

nb_rows, nb_cols = qwanamiz.calculate_grid(image_width = img_width, 
                                    image_height = img_height, 
                                    pixel_to_micron = pix_to_um)

adjacency, vm_parameters = qwanamiz.directionnality(
    adjacency,
    image_height = img_height, 
    image_width = img_width,
    spacing = pix_to_um,
    num_rows = nb_rows,
    num_cols = nb_cols,
    convergence_threshold = 0.001)

angle_plot = qwanamiz.plot_angles(params = vm_parameters,
                                  num_rows = nb_rows,
                                  num_cols = nb_cols)

######################################################################################

######################################################################################
## CELL WALL THICKNESS MEASUREMENTS

# RUNNING TIME IMPROVEMENT :
# ((This step could be moved further and 
# applied only to the edges we are interested in))

# Compute cell wall thickness between centroids of adjacent cells

startTime = datetime.datetime.now()

adjacency = qwanamiz.measure_wallthickness(adjacency, distance_map, scan_width = 20, scale = pix_to_um)

endTime = datetime.datetime.now()
print(f'runtime : {endTime - startTime}')     
############################################################################

################################################################################
#### EDGE CLASSIFICATION ON DIRECTIONNALITY

## Edge classification and filtering based on directionnality

# The classify_edges function does a first general classification
# of edges based on their angles compared to the main direction 
# determined in previous steps.
# Edges in the inerval of the peak are classified as tangential
# Edges not in the interval but in a 15 degrees tolerance around
# are classified as in_doubt
# Other edges are classified as radial (for radial walls)
qwanamiz.classify_edges(adjacency, tolerance = 15)

# Add lines as shapes to the viewer
lines = []
colors = []
# Define color mappings for each classification
color_map = {
    'radial': 'blue',
    'tangential': 'orange',
    'indoubt': 'red'
}

for index, row in adjacency.iterrows():
    coords1 = row['centroid1']
    coords2 = row['centroid2']
    classification = row['wall_classification']
    
    # Append line coordinates and color to respective lists
    lines.append([coords1, coords2])
    colors.append(color_map[classification])

viewer.add_shapes(lines, 
                  shape_type = 'line', 
                  edge_color = colors, 
                  name = 'Edges')

###################################################################################

###################################################################################
#### EDGE CLASSIFICATION ON CONNECTIVITY

# Count the total number of neighbors of each cell
regionprops_df = qwanamiz.count_neighbors(regionprops_df, adjacency)

## Refining of edges classification based on their neighborhood caracteristics
# The following functions aim at improving the continuity in the
# successive edges of the same radial files by examining the connectivity
# (number & types of neighbors) of in_doubt edges
# A decision tree is used to reclassify or not the edges in radial files
qwanamiz.find_neighbors(adjacency) 

qwanamiz.refine_neighbors(adjacency)

qwanamiz.update_neighbors(adjacency)

# Final classification after refining
# Prepare the lines and colors for visualization
lines = []
colors = []
unique_situation = adjacency['wall_classification'].unique()

# Create a colormap
cmap = plt.get_cmap('tab20', len(unique_situation))
color_map = {rf: cmap(i) for i, rf in enumerate(unique_situation)}

# Prepare the lines and corresponding colors
for edge, edge_data in adjacency.iterrows():
    coords1 = edge_data['centroid1']
    coords2 = edge_data['centroid2']
    situation = edge_data['wall_classification']
    
    # Append line coordinates and color to respective lists
    lines.append([coords1, coords2])
    colors.append(color_map[situation])


# Add lines as shapes to the viewer
viewer.add_shapes(lines, 
                  shape_type='line', 
                  edge_color=colors, 
                  edge_width=3, 
                  name='Classified edges')

# Create an image for the legend
legend_texts = [f"{situation}" for situation in unique_situation]
legend_colors = [color_map[situation] for situation in unique_situation]

fig, ax = plt.subplots(figsize=(2, len(unique_situation) * 0.5))
for i, (text, color) in enumerate(zip(legend_texts, legend_colors)):
    ax.text(0.1, 1 - (i + 1) / (len(unique_situation) + 1), text, fontsize=12, color=color, ha='left', va='center')
ax.axis('off')

# Save the figure as an image
fig.canvas.draw()
legend_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
legend_image = legend_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

plt.show(fig)
###########################################################################

###########################################################################
#### RADIAL FILES ASSIGNMENT


# A walking algorithm is implemented to find the straightest 
# & longest horizontal path possible along neighbor edges
qwanamiz.assign_radial_files(adjacency)

## Radial files layer
# Prepare the lines and colors for visualization
# Filter the DataFrame
final_df = adjacency.dropna(subset=['radial_file'])
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
                  shape_type = 'line', 
                  edge_color = colors, 
                  edge_width = 3, 
                  name = 'Radial Files')

######################################################################
#### FIND POTENTIAL RAYS & RESIN DUCTS

## The remaining background of the expanded_cells is exploited to identify
# possible rays and resin ducts & to flag tracheids adjacent to them
rays_ducts_table, rays_ducts_map = qwanamiz.rays_and_ducts(expanded_labels, 
                                                  scale = pix_to_um, 
                                                  min_duct_area = 80, 
                                                  min_duct_width = 10, 
                                                  min_ray_ecc = 0.8, 
                                                  min_ray_aspect = 2.5)   

viewer.add_labels([rays_ducts_map, rays_ducts_map[::2, ::2]], 
                  name="Rays & Resin Ducts", 
                  scale = [pix_to_um, pix_to_um])

## Flag tracheids adjacent to a possible ray or resin duct
rays_adj, duct_adj, unk_adj = qwanamiz.artefact_adjacent(expanded_labels, 
                                                rays_ducts_map)

duct_neighbors = np.isin(expanded_labels, duct_adj)
viewer.add_labels([duct_neighbors, duct_neighbors[::2, ::2]], 
                  name="Resin Ducts Neighbors", 
                  scale = [pix_to_um, pix_to_um])

# Add the flagged tracheids to the dataframe
regionprops_df = qwanamiz.adjacent_type_column(regionprops_df, 
                             rays_adj, 
                             duct_adj, 
                             unk_adj)
######################################################################

######################################################################
#### MAP CELLS AND EDGES & MEASURE DIAMETERS

## Retrieve information of each indivual cell from the adjacency dataframe
qwanamiz.get_cell_walls(regionprops_df, adjacency)

## Measure tangential & radial diameters adjusted by the 
# tracheid angle
qwanamiz.measure_diameters(regionprops_df, spacing = pix_to_um)

## Diameters layer
diam_df = regionprops_df.dropna(subset=['extr_rad'])

#diam_df['extr_rad'] = diam_df['extr_rad'].apply(ast.literal_eval)
#diam_df['extr_tan'] = diam_df['extr_tan'].apply(ast.literal_eval)

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

## Get the radial wall measurement values for each cell
qwanamiz.get_radial_walls(regionprops_df, adjacency)
#########################################################################

#########################################################################
#### SAVE THE FINAL DATAFRAME

## The final dataframe should be formatted to keep only useful information
output_path = 'C:/Users/sambo/Desktop/QWAnamiz_store/final_outputs/L20_F02-1M2-Sc5_cellstest.csv'
regionprops_df.to_csv(output_path, index=False)
