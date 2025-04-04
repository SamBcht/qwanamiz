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

sampleID = "L20_F33-1M2-Sc7"

png_path = 'C:/Users/sambo/Desktop/QWAnamiz_store/final_outputs/L20_F33-1M2-Sc7_segmented.png'

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
            'bbox',
            'solidity')))


## Splitting merged cells using watershed segmentation
#  This step needs to return updated cell measurements (regionprops_df)
#  and labeled_image. It also returns an array that contains the result of the watershed segmentation
labeled_image, regionprops_df, watershed_result = qwanamiz.adjust_labels(labeled_image, regionprops_df, scale = pix_to_um,
                                                                         area_threshold = 500, solidity_threshold = 0.95)

# Add the labeled image
viewer.add_labels([labeled_image, labeled_image[::2, ::2]], 
                  name = 'Lumens', 
                  scale = [pix_to_um, pix_to_um])


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
qwanamiz.classify_edges(adjacency, tolerance = 5)

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


###########################################################################
#### RADIAL FILES ASSIGNMENT

# Radial files grouping
start = datetime.datetime.now()
print("Radial files detection")

regionprops_df, adjacency = qwanamiz.assign_radial_files(regionprops_df, adjacency, stitch_angle_tolerance = 20)

endTime = datetime.datetime.now()
print(f'runtime : {endTime - start}')

######################################################################

## RADIAL FILES LAYER
# Prepare the lines and colors for visualization
lines = []
colors = []
unique_radial_files = regionprops_df['radial_file'].dropna().unique()
unique_radial_files = unique_radial_files[unique_radial_files != 0]

# Create a colormap
cmap = plt.get_cmap('tab20', len(unique_radial_files))
color_map = {rf: cmap(i) for i, rf in enumerate(unique_radial_files)}

# Prepare the lines and corresponding colors
for i in unique_radial_files:
    radial_file_df = regionprops_df[regionprops_df["radial_file"] == i]

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
#qwanamiz.get_cell_walls(regionprops_df, adjacency)

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

######################################################################################
## CELL WALL THICKNESS MEASUREMENTS


# Compute cell wall thickness between centroids of adjacent cells

startTime = datetime.datetime.now()

regionprops_df = qwanamiz.measure_wallthickness(regionprops_df, adjacency, distance_map, auto_pixelwidth=True, scan_width = 75, scale = pix_to_um)

endTime = datetime.datetime.now()
print(f'runtime : {endTime - startTime}')


regionprops_df['SampleId'] = sampleID

regionprops_df = regionprops_df.drop(
    columns = [
        'image',
        'bbox-0',
        'bbox-1',
        'bbox-2',
        'bbox-3'])

regionprops_df["WallThickness"] = regionprops_df[["left_wall_thickness", "right_wall_thickness"]].mean(axis=1, skipna=True)


# Filter "isolated" cells and those without radial_file
filtered_data = regionprops_df[(regionprops_df['classification'] == 'isolated') | (regionprops_df['radial_file'].isna())]

celldata = regionprops_df.copy()
# Remove "isolated" cells and those without radial_file from the main dataframe
celldata = celldata.dropna(subset=['radial_file'])

# Filter out "isolated" cells
celldata = celldata[celldata['classification'] != 'isolated']

celldata = qwanamiz.morks_index(celldata)


############################################################################

#########################################################################
#### SAVE THE FINAL DATAFRAME

## The final dataframe should be formatted to keep only useful information
output_folder = "C:/Users/sambo/Desktop/QWAnamiz_store/qwanamiz_dev"

celldata.to_csv(f"{output_folder}/{sampleID}_cells.csv", index=False)

adjacency.to_csv(f"{output_folder}/{sampleID}_adjacency.csv", index=True)

filtered_data.to_csv(f"{output_folder}/{sampleID}_filtered.csv", index=False)

np.savez_compressed(f"{output_folder}/{sampleID}_imgs.npz",
                    bw_img = prediction,
                    explabs = expanded_labels, 
                    labs = labeled_image)