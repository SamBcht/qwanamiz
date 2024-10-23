# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:32:21 2024

@author: sambo
"""

import numpy as np
import skimage.measure
import pandas as pd
from skimage.draw import line
from scipy.stats import circmean
import matplotlib.pyplot as plt
#from tools import histogram
from vonmisesmix import histogram, density, vonmises_pdfit, mixture_pdfit, pdfit, vonmises_density
from scipy.stats import vonmises
from multiprocessing import Pool
from functools import partial
##########################################################################
# Simili expand_labels function to avoid calculation of the distance map a second time
def expand_cells(label_image, distances, indices, distance = 1, spacing = 1,):
    
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask] for dimension_indices in indices
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out





###########################################################################
# Function to get the adjacent cells
def get_adjacent_labels(labeled_image, background_label=0):
    adjacent_labels = set()

    # Create shifted image to check vertical adjacencies
    vshift1 = labeled_image[:-1,] # All but the last row
    vshift2 = labeled_image[1:,]  # All but the first row

    # We have an adjacency if
    # - The corresponding values are different
    # - None of the values correspond to the background label
    v_adj = vshift1 != vshift2
    v_adj = np.logical_and(v_adj, vshift1 != background_label)
    v_ind = np.where(np.logical_and(v_adj, vshift2 != background_label))

    # We do the same thing to check for horizontal adjacencies
    hshift1 = labeled_image[:,:-1] # All but the last column
    hshift2 = labeled_image[:,1:]  # All but the first column

    h_adj = hshift1 != hshift2
    h_adj = np.logical_and(h_adj, hshift1 != background_label)
    h_ind = np.where(np.logical_and(h_adj, hshift2 != background_label))

    # Now we add the adjacent label tuples based on the positions of the matches
    for i in range(len(v_ind[0])):
        adjacent_labels.add(tuple(sorted((vshift1[v_ind[0][i], v_ind[1][i]], vshift2[v_ind[0][i], v_ind[1][i]]))))

    for i in range(len(h_ind[0])):
        adjacent_labels.add(tuple(sorted((hshift1[h_ind[0][i], h_ind[1][i]], hshift2[h_ind[0][i], h_ind[1][i]]))))

    return adjacent_labels

############################################################################
# Function to compute angle, center, and length
def compute_edge_properties(centroid1, centroid2):
    # Ensure calculation is from rightmost to leftmost centroid
    if centroid1[1] < centroid2[1]:  # centroid1 is to the left of centroid2
        centroid1, centroid2 = centroid2, centroid1
    
    dy = centroid1[0] - centroid2[0]
    dx = centroid1[1] - centroid2[1]
    angle = np.arctan2(dy, dx)  # Angle in radians
    angle_degrees = np.degrees(angle)  # Convert to degrees

    center = ((centroid1[0] + centroid2[0]) / 2, (centroid1[1] + centroid2[1]) / 2)
    length = np.sqrt(dx**2 + dy**2)

    return angle_degrees, center, length

########################################################################
# Arrange the adjacency graph in a dataframe with edges characteristics
def adjacency_dataframe(rag, lumen_props):
    
    # Create a DataFrame from the set of label tuples
    adj_df = pd.DataFrame(rag, columns=['label1', 'label2'])
    
    # Merge to get coordinates for label1# Merge to get coordinates for label1
    adj_df = adj_df.merge(lumen_props,
                 left_on = 'label1',
                 right_on = 'label',
                 suffixes = ('', '_label1')
                          ).drop(
                              columns = [
                                  'label',
                                  'area_lumen',
                                  'major_axis_length',
                                  'minor_axis_length',
                                  'orientation',
                                  'perimeter_crofton',
                                  'image',
                                  'bbox-0',
                                  'bbox-1',
                                  'bbox-2',
                                  'bbox-3',
                                  'area_cell'])
    
    # Merge to get coordinates for label2
    adj_df = adj_df.merge(lumen_props,
                 left_on = 'label2',
                 right_on = 'label',
                 suffixes = ('', '_label2')
                          ).drop(
                              columns = [
                                  'label',
                                  'area_lumen',
                                  'major_axis_length',
                                  'minor_axis_length',
                                  'orientation',
                                  'perimeter_crofton',
                                  'image',
                                  'bbox-0',
                                  'bbox-1',
                                  'bbox-2',
                                  'bbox-3',
                                  'area_cell'])
    
    # Use assign to create centroid1 and centroid2 columns as tuples
    adj_df = adj_df.assign(
        centroid1 = lambda df: df.apply(
            lambda row: (row['centroid-0'], row['centroid-1']), axis=1),
        centroid2 = lambda df: df.apply(
            lambda row: (row['centroid-0_label2'], row['centroid-1_label2']), axis=1)
        ).drop(columns = [
            'centroid-0',
            'centroid-1',
            'centroid-0_label2',
            'centroid-1_label2'])
    
    # Apply the function to compute angles, centers, and lengths        
    adj_df = adj_df.assign(
        angle_center_length=lambda df: df.apply(
            lambda row: compute_edge_properties(
                row['centroid1'],
                row['centroid2']),
            axis=1)
    )
    
    # Split the tuple into separate columns
    adj_df[['angle', 'center', 'length']] = pd.DataFrame(
        adj_df['angle_center_length'].tolist(),
        index=adj_df.index)

    # Drop the intermediate column
    adj_df = adj_df.drop(columns=['angle_center_length'])
    
    return adj_df

#############################################################################
# Automatically define the numbers of rows and columns used to divide the image based on the image shape
def calculate_grid(image_width, image_height, pixel_to_micron, row_min_height = 480, row_max_height = 750, col_min_width = 1250, col_max_width = 2000):
    """
    Calculate the number of rows and columns based on given image size and desired micron ranges.

    Args:
    - image_width: width of the image in pixels.
    - image_height: height of the image in pixels.
    - pixel_to_micron: conversion factor (microns per pixel).
    - row_min_height: minimum height of a row in microns.
    - row_max_height: maximum height of a row in microns.
    - col_min_width: minimum width of a column in microns.
    - col_max_width: maximum width of a column in microns.

    Returns:
    - num_rows: number of rows
    - num_cols: number of columns
    """

    # Convert image dimensions from pixels to microns
    image_width_microns = image_width * pixel_to_micron
    image_height_microns = image_height * pixel_to_micron

    # Calculate the number of rows and columns based on desired micron range
    # Number of rows: each row has a height between row_min_height and row_max_height
    row_height = (row_min_height + row_max_height) / 2  # average height
    num_rows = np.ceil(image_height_microns / row_height)

    # Number of columns: each column has a width between col_min_width and col_max_width
    col_width = (col_min_width + col_max_width) / 2  # average width
    num_cols = np.ceil(image_width_microns / col_width)

    return int(num_rows), int(num_cols)


# Directionnality modelisation
def directionnality(adj_df,
                    image_height,
                    image_width,
                    spacing = 1,
                    num_rows = 4,
                    num_cols = 8,
                    # Threshold for acceptable difference between mu and the peak angle
                    mu_threshold = 5,  # in degrees. Adjust this value based on your needs
                    max_iterations = 10,  # Maximum number of iterations to avoid infinite looping
                    convergence_threshold = 0.001,
                    k_threshold = 50):
    
    row_height = (image_height*spacing) / num_rows
    col_width = (image_width*spacing) / num_cols

    # Subsampling image and filtering of edges based on von Mises distributions

    # Dictionary to store the parameters for each subsample
    subsample_params = {}

    for i in range(num_rows):
        for j in range(num_cols):
            # Determine the bounds of the current subsample
            y_min = i * row_height
            y_max = (i + 1) * row_height
            x_min = j * col_width
            x_max = (j + 1) * col_width
            
            # Filter the edges within the current subsample
            subsample_edges = adj_df[
                (adj_df['center'].apply(lambda c: y_min <= c[0] < y_max)) &
                (adj_df['center'].apply(lambda c: x_min <= c[1] < x_max))
            ]
            
            # Convert angles to radians
            angle_rad = np.radians(subsample_edges['angle'])
            
            # Compute the histogram
            x_histo, y_histo = histogram(angle_rad, bins=90)
            
            # Find the angle corresponding to the maximum y value in the histogram
            max_peak_angle = np.degrees(x_histo[np.argmax(y_histo)])

            # Determining starting values to find the von Mises distribution parameters
            # The max peak angle ± 60 degrees are a good starting approximation
            mu_start = [max_peak_angle - 60, max_peak_angle, max_peak_angle + 60]
            mu_start = [i + 180 if i < -90 else i for i in mu_start]
            mu_start = [i - 180 if i > 90 else i for i in mu_start]
            mu_start = np.radians(mu_start)

            # Kappa values roughly similar to those empirically observed
            kappa_start = np.array([10, 150, 10])

            # We use pi values in equal proportions
            pi_start = np.array([1.0, 1.0, 1.0]) / 3
            
            # Fit mixture of von Mises distributions
            iterations = 0
            while iterations < max_iterations:
                
                m = mixture_pdfit(angle_rad, n=3, mu = mu_start, kappa = kappa_start, pi = pi_start, threshold = convergence_threshold)
            
                # Parameters of the horizontal edges distribution
                max_index = np.unravel_index(np.argmax(m, axis=None), m.shape)[1]
                mu = m[1, max_index]
                kappa = m[2, max_index]
                
                # Check if the estimated mu is similar to the maximum peak angle
                if np.abs(np.degrees(mu) - max_peak_angle) < mu_threshold and kappa > k_threshold:
                    break

                iterations += 1

            # Calculate the bounds of the interval
            lower_bound = vonmises.ppf(0.005, kappa, loc=mu)
            upper_bound = vonmises.ppf(0.995, kappa, loc=mu)
            
            # Save the parameters for this subsample
            subsample_params[f'{i+1}_{j+1}'] = {
                'vonmisses_params': m,
                'bounds': (lower_bound, upper_bound),
                'x': (x_min, x_max),
                'y': (y_min, y_max),
                'x_histo': x_histo,
                'y_histo': y_histo,
                'mu': mu,
                'kappa': kappa,
                'nb_cells': len(angle_rad),
                'cell_index': subsample_edges.index}

    # Initialize an empty list to store the rows
    rows = []

    # Iterate over the dictionary items
    for subsample_index, params in subsample_params.items():
        bounds = params['bounds']
        cell_indices = params['cell_index']
        
        # Create a row for each cell index
        for cell_index in cell_indices:
            rows.append({
                'cell_index': cell_index,
                'subsample_index': subsample_index,
                'lower_bound': bounds[0],
                'upper_bound': bounds[1]
            })

    # Create DataFrame from the list of rows
    df = pd.DataFrame(rows)

    # Merge the dataframes based on the index (cell_index)
    merged_df = pd.merge(adj_df, df, left_index=True, right_on='cell_index')
    
    return merged_df, subsample_params

# A function that shows the empirical distribution of angles
# and the one esimated by the von Mises distributions for
# each of the subsets (num_rows x num_cols) of the image
def plot_angles(params, num_rows, num_cols):

    # Create a figure to display the histograms
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

    # Looping over the rows and columns
    for i in range(num_rows):
        for j in range(num_cols):

            # Extracting the relevant data from the set of parameters
            x_histo = params[f'{i+1}_{j+1}']['x_histo']
            y_histo = params[f'{i+1}_{j+1}']['y_histo']
            mu = params[f'{i+1}_{j+1}']['mu']
            kappa = params[f'{i+1}_{j+1}']['kappa']
            m = params[f'{i+1}_{j+1}']['vonmisses_params']
            lower_bound = params[f'{i+1}_{j+1}']['bounds'][0]
            upper_bound = params[f'{i+1}_{j+1}']['bounds'][1]

            # Plotting
            ax = axes[i, j]
            
            # Plot the empirical distribution
            ax.plot(x_histo, y_histo, label='Raw', color='blue')
            
            # Plot the distribution using the parameters obtained from the EM algorithm
            f = np.zeros(len(x_histo))
            for k in range(m.shape[1]):
                f += m[0, k] * density(x_histo, m[1, k], m[2, k])
            ax.plot(x_histo, f / np.sum(f), label='Fit', color='red')

            # Add the 99% interval bounds as vertical lines
            ax.axvline(lower_bound, color='green', linestyle='--')
            ax.axvline(upper_bound, color='green', linestyle='--')
            ax.text(min(x_histo) * 1.3, max(y_histo) * 0.3, f'{np.degrees(lower_bound):.2f}°', color='green', fontsize=8, ha='center')
            ax.text(max(x_histo) * 0.7, max(y_histo) * 0.3, f'{np.degrees(upper_bound):.2f}°', color='green', fontsize=8, ha='center')
            ax.text(max(x_histo) * 0.7, max(y_histo) * 0.8, f'{np.degrees(mu):.2f}°', color='red', fontsize=8, ha='center')
            ax.text(max(x_histo) * 0.7, max(y_histo) * 0.7, f'{kappa:.2f}', color='red', fontsize=8, ha='center')

            # Set limits and title
            ax.set_xlim(-np.pi/2, np.pi/2)
            ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
            ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°'])
            ax.set_title(f"Subsample ({i+1}, {j+1})")

            # Display the legend
            #ax.legend()

    # Adjust layout
    fig.tight_layout()

    return fig 

#########################################################################
# Cell Wall Measurements
def thickness_between_centroids(row, dist_map, scaling = 1, pixelwidth = 10):
    # Define profile line between centroids
    mid_line = skimage.measure.profile_line(
        dist_map,
        row['pix_centroid1'],
        row['pix_centroid2'],
        linewidth = pixelwidth,
        order = 0,
        reduce_func = None)
    
    # Initialize list to store max values
    adjusted_max_vals = []
        
        # Loop through each column to find the maximum and check for duplicates
    for col in range(mid_line.shape[1]):
        max_val = np.max(mid_line[:, col])  # Find the max value in the column
        max_in_col = mid_line[:, col]       # Get all values in the column
            
            # Check if there are multiple values equal to the max
        if np.sum(max_in_col == max_val) > 1:  # If more than one max value exists
            max_val += scaling/2         # Adjust the max value
                
        adjusted_max_vals.append(max_val)
    
    # Calculate the mean of these maximum values
    max_thickness = np.mean(adjusted_max_vals)

    return max_thickness

def measure_wallthickness(adj_df, dist_map, scan_width = 10, scale = 1, nprocesses = 1):

    # Get the centroids coordinates in pixels
    adj_df['pix_centroid1'] = adj_df['centroid1'].apply(lambda x: (x[0] / scale, x[1] / scale))
    adj_df['pix_centroid2'] = adj_df['centroid2'].apply(lambda x: (x[0] / scale, x[1] / scale))

    # The case for multiprocessing
    if(nprocesses > 1):
        with Pool(processes = nprocesses) as p:
            multi_thickness = partial(thickness_between_centroids, dist_map = dist_map, pixelwidth = scan_width, scaling = scale)
            adj_df['wall_thickness'] = p.map(multi_thickness, [row for index,row in adj_df.iterrows()])
    
    # Otherwise with only one process
    else:
        adj_df['wall_thickness'] = adj_df.apply(
                lambda row: thickness_between_centroids(row,
                                                        dist_map = dist_map,
                                                        pixelwidth = scan_width,
                                                        scaling = scale),
                axis=1)

    
    return adj_df

#########################################################################
# Classify cell walls between radial and tangential
def classify_edges(df, tolerance = 5):
    
    # Create a new column 'edge_classification' with initial value 'rad_wall'
    df['wall_classification'] = 'radial'
    
    # Convert tolerance in radians
    tolerance_rad = np.radians(tolerance)

    # Update the classification to 'tangential' for edges with angle between lower and upper bounds
    for index, row in df.iterrows():
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']
        angle = np.radians(row['angle'])
        low_tolerance = lower_bound - tolerance_rad
        up_tolerance = upper_bound + tolerance_rad
        
        if lower_bound <= angle <= upper_bound:
            df.at[index, 'wall_classification'] = 'tangential'
            
        elif (low_tolerance <= angle < lower_bound) or (upper_bound < angle <= up_tolerance):
            df.at[index, 'wall_classification'] = 'indoubt'
               
    return df

############################################################################
# Find tangential neighbors
def find_neighbors(complete_df):
    
    # Reset indexes
    complete_df.set_index(['label1', 'label2'], inplace=True)

    # Initialize the neighbor column
    complete_df['neighbors'] = None
    
    edges_df = complete_df[complete_df['wall_classification'] == 'tangential']

    # Find neighbors for each edge
    for edge, edge_data in edges_df.iterrows():
        label1, label2 = edge
        
        # Find all edges that share a common label with the current edge
        neighbors = edges_df[(edges_df.index.get_level_values('label1') == label1) |
                             (edges_df.index.get_level_values('label2') == label1) |
                             (edges_df.index.get_level_values('label1') == label2) |
                             (edges_df.index.get_level_values('label2') == label2)].index

        # Remove the current edge from its own neighbor list
        neighbors = neighbors.drop(edge)
        
        # Assign the neighbors to the 'neighbors' column
        complete_df.at[edge, 'neighbors'] = list(neighbors)
        
    return complete_df


############################################################################
def refine_neighbors(complete_df):
    
    # Filter starting edges as tangential having 1 or less neighbor
    starting_edge = complete_df[complete_df['wall_classification'] == 'tangential']
    starting_edge = starting_edge[starting_edge['neighbors'].apply(lambda x: len(x) <= 1)]
    
    starting_edge['ext_label'] = None
    for start, start_data in starting_edge.iterrows():
        ext1, ext2 = start
        start_neigh = start_data['neighbors']
        
        if len(start_neigh) == 1:
            if (ext1 not in start_neigh[0]):
                starting_edge.at[start, 'ext_label'] = ext1
            elif (ext2 not in start_neigh[0]):
                starting_edge.at[start, 'ext_label'] = ext2
    
    # Filter tangential edges with more than one neighbor
    tan_edge = complete_df[complete_df['wall_classification'] == 'tangential']
    tan_edge = tan_edge[tan_edge['neighbors'].apply(lambda x: len(x) > 1)]
        
    # Filter ambiguous edges and initialize columns for reclassification
    ambiguous_edge = complete_df.copy()[complete_df['wall_classification'] == 'indoubt']
    ambiguous_edge['amb_neighbors'] = None
    ambiguous_edge['situation'] = None

    # Find neighbors for each edge
    for edge, edge_data in ambiguous_edge.iterrows():
        label1, label2 = edge
        
        # Find all edges that share a common label with the current edge
        neighbors = starting_edge[
            (starting_edge.index.get_level_values('label1') == label1) |
            (starting_edge.index.get_level_values('label2') == label1) |
            (starting_edge.index.get_level_values('label1') == label2) |
            (starting_edge.index.get_level_values('label2') == label2)
        ].index
        
        # Find all edges that share a common label with the current edge
        amb_neighbors = ambiguous_edge[
            (ambiguous_edge.index.get_level_values('label1') == label1) |
            (ambiguous_edge.index.get_level_values('label2') == label1) |
            (ambiguous_edge.index.get_level_values('label1') == label2) |
            (ambiguous_edge.index.get_level_values('label2') == label2)
        ].index


        # Remove the current edge from its own neighbor list
        amb_neighbors = amb_neighbors.drop(edge)
        
        # Assign the neighbors to the 'neighbors' column
        ambiguous_edge.at[edge, 'neighbors'] = list(neighbors)
        ambiguous_edge.at[edge, 'amb_neighbors'] = list(amb_neighbors)
        

        # Connection between two starting points of 2 radial_files
        if len(neighbors) == 2 and len(amb_neighbors) == 0:
            ext_label1 = starting_edge.at[neighbors[0], 'ext_label']
            ext_label2 = starting_edge.at[neighbors[1], 'ext_label']
            if (ext_label1 in edge) and (ext_label2 in edge):
                ambiguous_edge.at[edge, 'situation'] = 'bridge'
            elif (ext_label1 in edge) and (ext_label2 is None):
                ambiguous_edge.at[edge, 'situation'] = 'bridge'
            elif (ext_label2 in edge) and (ext_label1 is None):
                ambiguous_edge.at[edge, 'situation'] = 'bridge'
            else:
                ambiguous_edge.at[edge, 'situation'] = 'unknown'
        
        # End of a file
        elif len(neighbors) == 1 and len(amb_neighbors) == 0:
            if starting_edge.at[neighbors[0], 'ext_label'] in edge:
                end1, end2 = edge
                if end1 == starting_edge.at[neighbors[0], 'ext_label']:
                    end_connex = end2
                else: end_connex = end1
                end_neighb = tan_edge[
                    (tan_edge.index.get_level_values('label1') == end_connex) |
                    (tan_edge.index.get_level_values('label2') == end_connex)
                    ].index
                if len(list(end_neighb)) > 0:
                    ambiguous_edge.at[edge, 'situation'] = 'end_bifurc'
                else:
                    ambiguous_edge.at[edge, 'situation'] = 'bridge_end'
            else: ambiguous_edge.at[edge, 'situation'] = 'false_end'
                
            
        # Start of a long connection (2 or more bridges)
        elif len(neighbors) == 1 and len(amb_neighbors) == 1:
            ambiguous_edge.at[edge, 'situation'] = 'connex'
            
        # Bifurcation
        elif (len(neighbors) == 2 and len(amb_neighbors) == 1):
            ambiguous_edge.at[edge, 'situation'] = 'cross'
            
        # Bifurcation
        elif (len(neighbors) == 1 and len(amb_neighbors) >= 2):
            ambiguous_edge.at[edge, 'situation'] = 'cross2'
            
        # Start of a long connection (2 or more bridges)
        elif (len(neighbors) == 0 and len(amb_neighbors) == 0):
            ambiguous_edge.at[edge, 'situation'] = 'isolated'
            
        elif (len(neighbors) == 0 and len(amb_neighbors) == 2):
            ambiguous_edge.at[edge, 'situation'] = 'in_connex'

    for edge, edge_data in ambiguous_edge.iterrows():
        # label1, label2 = edge
        neighbors = ambiguous_edge.at[edge, 'neighbors']
        amb_neighbors = ambiguous_edge.at[edge, 'amb_neighbors']
        
        if ambiguous_edge.at[edge, 'situation'] == 'connex':
            
            ext_neigh = starting_edge.at[neighbors[0], 'ext_label']
            ext_amb = ambiguous_edge.at[amb_neighbors[0], 'neighbors']
            
            if ambiguous_edge.at[amb_neighbors[0], 'situation'] == 'connex':
                
                if (ext_neigh != starting_edge.at[ext_amb[0], 'ext_label']) and (ext_neigh in edge) and (starting_edge.at[ext_amb[0], 'ext_label'] in amb_neighbors[0]):
                    ambiguous_edge.at[edge, 'situation'] = 'db_bridge'
                    ambiguous_edge.at[amb_neighbors[0], 'situation'] = 'db_bridge'
                
                elif (ext_neigh == starting_edge.at[ext_amb[0], 'ext_label']) and (ext_neigh in edge) and (starting_edge.at[ext_amb[0], 'ext_label'] in amb_neighbors[0]):
                    ambiguous_edge.at[edge, 'situation'] = 'bifurcation'
                    ambiguous_edge.at[amb_neighbors[0], 'situation'] = 'bifurcation'
                
        elif ambiguous_edge.at[edge, 'situation'] == 'in_connex':

            if (ambiguous_edge.at[amb_neighbors[0], 'situation'] == 'connex') and (ambiguous_edge.at[amb_neighbors[1], 'situation'] == 'connex'):
                ambiguous_edge.at[edge, 'situation'] = 'mid_bridge'
                ambiguous_edge.at[amb_neighbors[0], 'situation'] = 'long_bridge'
                ambiguous_edge.at[amb_neighbors[1], 'situation'] = 'long_bridge'
                
            elif (ambiguous_edge.at[amb_neighbors[0], 'situation'] == 'connex') and (ambiguous_edge.at[amb_neighbors[1], 'situation'] == 'in_connex'):
                ambiguous_edge.at[edge, 'situation'] = 'connex'
                ambiguous_edge.at[amb_neighbors[0], 'situation'] = 'long_bridge'
                
            elif (ambiguous_edge.at[amb_neighbors[1], 'situation'] == 'connex') and (ambiguous_edge.at[amb_neighbors[0], 'situation'] == 'in_connex'):
                ambiguous_edge.at[edge, 'situation'] = 'connex'
                ambiguous_edge.at[amb_neighbors[1], 'situation'] = 'long_bridge'
            
        elif ambiguous_edge.at[edge, 'situation'] == 'cross':
            
            if ambiguous_edge.at[amb_neighbors[0], 'situation'] == 'connex':
                # probably a bridge if amb_neighbor is 'connex'
                ext_neigh1 = starting_edge.at[neighbors[0], 'ext_label']
                ext_neigh2 = starting_edge.at[neighbors[1], 'ext_label']
                
                if ext_neigh1 in edge and ext_neigh2 in edge:
                    ambiguous_edge.at[edge, 'situation'] = 'bridge'
            # inside bifurcation if amb_neighbor is also 'cross'
    
    for index,row in ambiguous_edge.iterrows():
        if ambiguous_edge.at[index, 'situation'] is not None:
            complete_df.at[index, 'wall_classification'] = ambiguous_edge.at[index, 'situation']
    
    return complete_df
###################################################################################
# Update neighbors after refining
def update_neighbors(complete_df):
    
    # Filter the DataFrame
    edges_df = complete_df.copy()[
        (complete_df['wall_classification'] == 'tangential') |
        (complete_df['wall_classification'].str.contains('bridge'))
    ]
    
    # Initialize the 'neighbors' column with empty lists where it is None
    edges_df['neighbors'] = edges_df['neighbors'].apply(lambda x: x if isinstance(x, list) else [])


    for edge, edge_data in edges_df.iterrows():
        label1, label2 = edge
        
        if edges_df.at[edge, 'wall_classification'] != 'tangential':
            # Find all edges that share a common label with the current edge
            neighbors = edges_df[(edges_df.index.get_level_values('label1') == label1) |
                                 (edges_df.index.get_level_values('label2') == label1) |
                                 (edges_df.index.get_level_values('label1') == label2) |
                                 (edges_df.index.get_level_values('label2') == label2)].index

            # Remove the current edge from its own neighbor list
            neighbors = neighbors.drop(edge)
            
            # Assign the neighbors to the 'neighbors' column
            edges_df.at[edge, 'neighbors'] = list(neighbors)
            
            # Update the neighbors list of each neighbor
            for neighbor in neighbors:
                edges_df.at[neighbor, 'neighbors'].append(edge)
                
    # Remove duplicates from neighbors list
    edges_df['neighbors'] = edges_df['neighbors'].apply(lambda x: list(set(x)))
    
    for index,row in edges_df.iterrows():
        complete_df.at[index, 'neighbors'] = edges_df.at[index, 'neighbors']
    
    return complete_df

#################################################################################
# Get cell wall thickness and cell neighbors

def get_cell_walls(cells_df, walls_df):
    
    edges_df = walls_df[
        (walls_df['wall_classification'] == 'tangential') |
        (walls_df['wall_classification'].str.contains('bridge'))
    ]
    
    # Iterate over each row in cells_df
    cells_df['left_neighbor'] = 0
    cells_df['left_wall_thickness'] = 0.0
    cells_df['left_angle'] = 0.0
    cells_df['right_neighbor'] = 0
    cells_df['right_wall_thickness'] = 0.0
    cells_df['right_angle'] = 0.0
    cells_df['classification'] = 'regular'
    cells_df['radial_file'] = None

    # Iterate over each row in cells_df
    for idx, row in cells_df.iterrows():
        label = row['label']
        label_centroid = (row['centroid-0'], row['centroid-1'])
        
        # Filter edges_df for rows where the label is either label1 or label2
        filtered_edges = edges_df[(edges_df.index.get_level_values('label1') == label) |
                                  (edges_df.index.get_level_values('label2') == label)]
        
        if len(filtered_edges) == 0:
            # No neighbors
            cells_df.at[idx, 'classification'] = 'isolated'
        
        elif len(filtered_edges) == 1:
            # Single neighbor
            cells_df.at[idx, 'classification'] = 'extremity'
            
            edge = filtered_edges.index[0]
            edge_data = filtered_edges.iloc[0]
            
            label1, label2 = edge
            neighbor_label = label2 if label1 == label else label1
            neighbor_centroid = cells_df.loc[cells_df['label'] == neighbor_label, ['centroid-0', 'centroid-1']].values[0]
            
            cells_df.at[idx, 'radial_file'] = edge_data['radial_file']
            
            if neighbor_centroid[1] < label_centroid[1]:  # Left neighbor
                cells_df.at[idx, 'left_neighbor'] = neighbor_label
                cells_df.at[idx, 'left_wall_thickness'] = edge_data['wall_thickness']
                cells_df.at[idx, 'left_angle'] = edge_data['angle']
            else:  # Right neighbor
                cells_df.at[idx, 'right_neighbor'] = neighbor_label
                cells_df.at[idx, 'right_wall_thickness'] = edge_data['wall_thickness']
                cells_df.at[idx, 'right_angle'] = edge_data['angle']
        
        elif len(filtered_edges) == 2:
            # Two neighbors
            edge1 = filtered_edges.index[0]
            edge2 = filtered_edges.index[1]
            edge_data1 = filtered_edges.iloc[0]
            edge_data2 = filtered_edges.iloc[1]
            
            label1_1, label1_2 = edge1
            label2_1, label2_2 = edge2
            
            neighbor1_label = label1_2 if label1_1 == label else label1_1
            neighbor2_label = label2_2 if label2_1 == label else label2_1
            
            neighbor1_centroid = cells_df.loc[cells_df['label'] == neighbor1_label, ['centroid-0', 'centroid-1']].values[0]
            neighbor2_centroid = cells_df.loc[cells_df['label'] == neighbor2_label, ['centroid-0', 'centroid-1']].values[0]
            
            if edge_data1['radial_file'] == edge_data2['radial_file']:
                cells_df.at[idx, 'radial_file'] = edge_data1['radial_file']
            else: cells_df.at[idx, 'radial_file'] = 0
            
            if neighbor1_centroid[1] < label_centroid[1] and neighbor2_centroid[1] > label_centroid[1]:
                # Proper left and right neighbors
                cells_df.at[idx, 'left_neighbor'] = neighbor1_label
                cells_df.at[idx, 'left_wall_thickness'] = edge_data1['wall_thickness']
                cells_df.at[idx, 'left_angle'] = edge_data1['angle']
                cells_df.at[idx, 'right_neighbor'] = neighbor2_label
                cells_df.at[idx, 'right_wall_thickness'] = edge_data2['wall_thickness']
                cells_df.at[idx, 'right_angle'] = edge_data2['angle']
            
            elif neighbor1_centroid[1] > label_centroid[1] and neighbor2_centroid[1] < label_centroid[1]:
                # Proper left and right neighbors, swapped order
                cells_df.at[idx, 'left_neighbor'] = neighbor2_label
                cells_df.at[idx, 'left_wall_thickness'] = edge_data2['wall_thickness']
                cells_df.at[idx, 'left_angle'] = edge_data2['angle']
                cells_df.at[idx, 'right_neighbor'] = neighbor1_label
                cells_df.at[idx, 'right_wall_thickness'] = edge_data1['wall_thickness']
                cells_df.at[idx, 'right_angle'] = edge_data1['angle']
        
        else:
            # More than two neighbors
            cells_df.at[idx, 'classification'] = 'cross'
    
    return cells_df
    
###################################################################

############################################################################
# Assign radial files ids to continuous straight lines of cells
def assign_radial_files(complete_df):
    
    # Set cell labels column as index
    #edges_df.set_index(['label1', 'label2'], inplace=True)
    
    # Initialize the 'radial_file' column with None
    complete_df['radial_file'] = None
    complete_df['file_rank'] = None
    
    # Filter the DataFrame
    edges_df = complete_df[
        (complete_df['wall_classification'] == 'tangential') |
        (complete_df['wall_classification'].str.contains('bridge'))
    ]
    

    # Initialize radial_file_id to start assigning IDs from 1
    radial_file_id = 1
    
    # Create an empty set visited_edges to keep track of visited edges
    visited_edges = set()
    
    # Create a function to define the next edge in the same radial file
    # that takes the current edge, previous edge, and neighbors as input
    def find_next_edge(current_edge, previous_edge, neighbors):
        
        # If there are no neighbors, return None
        if not neighbors:
            return None
        
        # If there's only one neighbor, return it
        if len(neighbors) == 1:
            return neighbors[0]
        
        # Find the neighbor with the closest angle
        # To do this, calculate the angle difference between the current edge and each neighbor's angle,
        # and select the neighbor with the minimum difference
        current_angle = edges_df.at[current_edge, 'angle']
        previous_angle = edges_df.at[previous_edge, 'angle']
        
        def angle_difference(angle1, angle2):
            return min(abs(angle1 - angle2), 360 - abs(angle1 - angle2))
        
        best_neighbor = min(neighbors, key=lambda neighbor: angle_difference(edges_df.at[neighbor, 'angle'], current_angle))
        
        #DEPRECTATED : if angle_difference(edges_df.at[best_neighbor, 'angle'], previous_angle) <= angle_difference(edges_df.at[best_neighbor, 'angle'], current_angle):
            #return best_neighbor
        
        #return None
        return best_neighbor
    
    # Iterate over each edge in the DataFrame
    for edge in edges_df.index:
    # Find a starting edge (that as only one neighbor and hasn't been visited yet)
        # if the edge has alreaby been assigned to a radial file or
        # if it has already been visited
        # go to the next line
        if edges_df.at[edge, 'radial_file'] is not None or edge in visited_edges:
            continue
        
        # if not retrieve the neighbors
        neighbors = edges_df.at[edge, 'neighbors']
        
        # if it has more than one neighbor, go to the next line
        if len(neighbors) != 1:
            continue
    # here the conditions to be a starting edge have been complied
    
    # Then we initialize the radial file loop with the starting edge
        # Assign radial_file ID to the starting edge
        current_edge = edge
        edges_df.at[current_edge, 'radial_file'] = radial_file_id
        visited_edges.add(current_edge)
        previous_edge = None
        
        # Initialize the file rank counter
        ranking = 1
        edges_df.at[current_edge, 'file_rank'] = ranking

        while True:
            # Update the list of neighbors to exclude those that have already been visited
            neighbors = [n for n in edges_df.at[current_edge, 'neighbors'] 
                         if n not in visited_edges 
                         or edges_df.at[n, 'radial_file'] is None]
            #Use the find_next_edge function to find the next edge
            next_edge = find_next_edge(current_edge, previous_edge, neighbors)
            # If no next edge is found, exit the loop
            if next_edge is None:
                break
            
            # Assign the radial file ID to the next edge and add it to the set of visited edges
            edges_df.at[next_edge, 'radial_file'] = radial_file_id
            visited_edges.add(next_edge)
            
            # Assign the file rank to the edge
            ranking += 1
            edges_df.at[next_edge, 'file_rank'] = ranking
            
            # Mark remaining neighbors as visited
            visited_edges.update(neighbors)
            
            # Update the previous and current edges for the next iteration
            previous_edge, current_edge = current_edge, next_edge
            
        # After completing the loop for a radial file, increment the radial file ID

        radial_file_id += 1
        
    complete_df.update(edges_df)
        
    return complete_df


######################################################################################
# Measure radial and tangential diameters relative to the cell angle

def measure_diameters(complete_df, spacing = 1):
    """
    Measure the diameters of objects along specified angles and their perpendiculars.
    
    Parameters:
    - complete_df: DataFrame containing columns 'label', 'centroid-0', 'centroid-1', and 'angle'.
    - spacing: Conversion factor from pixels to micrometers
    
    Returns:
    - df: Updated DataFrame with additional 'diameter_rad' and 'diameter_tan' columns.
    """
    complete_df['diameter_rad'] = None
    complete_df['diameter_tan'] = None
    complete_df['extr_rad'] = None
    complete_df['extr_tan'] = None
    complete_df['mean_angle'] = None
    
    df = complete_df[
        (complete_df['classification'] == 'extremity') |
        (complete_df['classification'] == 'regular')]

    for index, row in df.iterrows():
        
        label = row['label']
        
        centroid = (row['centroid-0']/spacing, row['centroid-1']/spacing)
        
        bbox = (row['bbox-0'], row['bbox-1'], row['bbox-2']-1, row['bbox-3']-1)
        
        if row['classification'] == 'extremity':
            angle_rad = np.deg2rad(row['left_angle']) if row['right_angle'] == 0 else np.deg2rad(row['right_angle'])
            
        elif row['classification'] == 'regular':
            angle_rad = circmean(np.deg2rad(np.array([row['left_angle'], row['right_angle']])))

        # Create a binary mask for the current object
        binary_mask = row['image']

        # Convert angle to radians
        # angle_rad = np.deg2rad(angle)
        perp_angle = angle_rad + np.pi / 2  # Perpendicular angle in radians

        # Calculate the diameters for both angles
        rad_diameter, rad_extr = calculate_diameter(
            binary_mask,
            centroid,
            angle_rad,
            bbox,
            spacing = spacing)
        tan_diameter, tan_extr = calculate_diameter(
            binary_mask,
            centroid,
            perp_angle,
            bbox,
            spacing = spacing)

        # Add diameters, mean angle and extremities to the dataframe
        complete_df.at[index, 'diameter_rad'] = rad_diameter * spacing
        complete_df.at[index, 'diameter_tan'] = tan_diameter * spacing
        complete_df.at[index, 'extr_rad'] = rad_extr
        complete_df.at[index, 'extr_tan'] = tan_extr
        complete_df.at[index, 'mean_angle'] = np.rad2deg(angle_rad)

    return complete_df

def calculate_diameter(label_image, centroid, angle, bbox, spacing = 1):
    """
    Calculate the diameter of an object along a specified angle.
    
    Parameters:
    - binary_mask: Binary mask of the object.
    - centroid: Tuple of (y, x) coordinates of the centroid.
    - angle_rad: Angle in radians along which to measure the diameter.
    
    Returns:
    - diameter: Measured diameter along the specified angle.
    """
    
    min_row, min_col, max_row, max_col = bbox
    y0, x0 = centroid
    
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    # Calculate intersections with the bounding box
    intersections = []
    intersections_sliced = []

    # Left side (min_col)
    t_left = (min_col - x0) / cos_theta
    y_left = y0 + t_left * sin_theta
    
    
    if min_row <= y_left <= max_row:
        intersections.append((min_col, y_left))
        intersections_sliced.append((0, y_left - min_row))
    else:
        if y_left < min_row:
            y_left = min_row
            x_left = x0 + (y_left - y0) / sin_theta * cos_theta
            if min_col <= x_left <= max_col:
                intersections.append((x_left, min_row))
                intersections_sliced.append((x_left - min_col, 0))
        elif y_left > max_row:
            y_left = max_row
            x_left = x0 + (y_left - y0) / sin_theta * cos_theta
            if min_col <= x_left <= max_col:
                intersections.append((x_left, max_row))
                intersections_sliced.append((x_left - min_col, max_row - min_row))

        
    # Right side (max_col)
    t_right = (max_col - x0) / cos_theta
    y_right = y0 + t_right * sin_theta
    
    if min_row <= y_right <= max_row:
        intersections.append((max_col, y_right))
        intersections_sliced.append((max_col - min_col, y_right - min_row))
    else:
        if y_right < min_row:
            y_right = min_row
            x_right = x0 + (y_right - y0) / sin_theta * cos_theta
            if min_col <= x_right <= max_col:
                intersections.append((x_right, min_row))
                intersections_sliced.append((x_right - min_col, 0))
        elif y_right > max_row:
            y_right = max_row
            x_right = x0 + (y_right - y0) / sin_theta * cos_theta
            if min_col <= x_right <= max_col:
                intersections.append((x_right, max_row))
                intersections_sliced.append((x_right - min_col, max_row - min_row))
                
    if len(intersections_sliced) < 2:
         distance = 0
         diam_coords = 0
    else: 
        (x1_sliced, y1_sliced), (x2_sliced, y2_sliced) = intersections_sliced[:2]
        # Find the points on the line where there is a switch from background to foreground
        rr, cc = line(int(y1_sliced), int(x1_sliced), int(y2_sliced), int(x2_sliced))
        
        line_coords = np.array(list(zip(rr, cc)))
        
        # Detect switches
        switch_points = []
        if label_image[line_coords[0][0], line_coords[0][1]] == 1:
            switch_points.append(line_coords[0])
        if label_image[line_coords[-1][0], line_coords[-1][1]] == 1:
            switch_points.append(line_coords[-1])
        for i in range(1, len(line_coords)):
            if label_image[line_coords[i][0], line_coords[i][1]] != label_image[line_coords[i-1][0], line_coords[i-1][1]]:
                switch_points.append(line_coords[i])
                
        if len(switch_points) == 2:
            (y1, x1), (y2, x2) = switch_points[0:2]
            
        elif len(switch_points) > 2:
            (y1, x1) = switch_points[0]
            (y2, x2) = switch_points[-1]
            
        elif len(switch_points) < 2:
            (y1, x1), (y2, x2) = intersections_sliced[0:2]
            
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        diam_coords = ((y1 + min_row) * spacing, (x1 + min_col) * spacing), ((y2 + min_row) * spacing, (x2 + min_col)* spacing)


    return distance, diam_coords

################################################

# Attribute the correct up and down radial wall measurements to each tracheid
def get_radial_walls(cells_df, walls_df):
    
    edges_df = walls_df[
        (walls_df['wall_classification'] == 'radial')
    ]
    
    # Initialize new columns
    cells_df['up_neighbor'] = 0
    cells_df['up_wall_thickness'] = 0.0
    cells_df['down_neighbor'] = 0
    cells_df['down_wall_thickness'] = 0.0
    
    # Adjust the comparison to handle angles between -90 and 90
    def angle_difference(a, b):
        diff = abs(a - b)
        return min(diff, 180 - diff)  # Handle angle wrapping
    
    # Function to find the edge with angle closest to the perpendicular angle
    def closest_edge(edges, perpendicular_angle):
        if not edges:
            return None
        # Find the index of the closest edge (edges are tuples, so access angle via edge[1]['angle'])
        closest_edge_tuple = min(edges, key=lambda x: angle_difference(x[1]['angle'], perpendicular_angle))
        return closest_edge_tuple[0]  # Return the index of the closest edge        

    # Iterate over each row in cells_df
    for idx, row in cells_df.iterrows():
        
        # Skip rows where left or right wall thickness is None or 0
        if row['classification'] == 'isolated' or (row['right_wall_thickness'] in [None, 0] and row['left_wall_thickness'] in [None, 0]):
            continue
        
        label = row['label']
        label_centroid = (row['centroid-0'], row['centroid-1'])
        # Calculate perpendicular angle to the cell's orientation
        angle_deg = row['mean_angle']
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
            
        perpendicular_angle = (angle_deg + 90)
        # Wrap perpendicular angle to the range -90 to 90
        if perpendicular_angle > 90:
            perpendicular_angle -= 180
        elif perpendicular_angle < -90:
            perpendicular_angle += 180
        
        
        # Filter edges_df for rows where the label is either label1 or label2
        filtered_edges = edges_df[(edges_df.index.get_level_values('label1') == label) |
                                  (edges_df.index.get_level_values('label2') == label)]
        
        # Initialize lists for up and down edges
        up_edges = []
        down_edges = []
        
        # Iterate over each filtered edge
        for edge_idx, edge_row in filtered_edges.iterrows():
            
            # Classify as 'up' or 'down' based on the y-coordinate comparison (centroid-0 is y)
            if edge_row['center'][0] < label_centroid[0]:
                up_edges.append((edge_idx, edge_row))
                
            elif edge_row['center'][0] > label_centroid[0]:
                down_edges.append((edge_idx, edge_row))
                
            # Find the closest up edge and down edge
        closest_up_edge = closest_edge(up_edges, perpendicular_angle)
        closest_down_edge = closest_edge(down_edges, perpendicular_angle)
        
    #return closest_up_edge, closest_down_edge
        
                
        # Assign the neighbors and wall thicknesses based on the closest edges
        if closest_up_edge is not None:
            up_label1, up_label2 = closest_up_edge
            up_neighbor = up_label2 if up_label1 == label else up_label1
            up_thick = walls_df.at[closest_up_edge, 'wall_thickness']
            cells_df.at[idx, 'up_neighbor'] = up_neighbor  # Set up neighbor
            cells_df.at[idx, 'up_wall_thickness'] = up_thick
            walls_df.at[closest_up_edge, 'wall_classification'] = 'radial_sel' # Set up wall thickness

        if closest_down_edge is not None:
            down_label1, down_label2 = closest_down_edge
            down_neighbor = down_label2 if down_label1 == label else down_label1
            down_thick = walls_df.at[closest_down_edge, 'wall_thickness']
            cells_df.at[idx, 'down_neighbor'] = down_neighbor  # Set down neighbor
            cells_df.at[idx, 'down_wall_thickness'] = down_thick  # Set down wall thickness
            walls_df.at[closest_down_edge, 'wall_classification'] = 'radial_sel'
        
        

    return cells_df, walls_df

###############################################################################
# Add the total number of neighbors in a new column
def count_neighbors(complete_df, adjacency_df):
    # Count occurrences of each label in both label1 and label2 columns
    label_counts = pd.concat([adjacency_df['label1'], adjacency_df['label2']]).value_counts()
    
    # Create a new column in the dataframe for the number of neighbors
    complete_df['nb_of_neighbors'] = complete_df['label'].map(label_counts).fillna(0).astype(int)

    return complete_df

#####
def rays_and_ducts(labels, 
                   scale = 1, 
                   min_duct_area = 80, 
                   min_duct_width = 20, 
                   min_ray_ecc = 0.8, 
                   min_ray_aspect = 2.5):
    
    # Define a binary mask for the background (i.e., areas you suspect are either rays or resin ducts)
    background_mask = labels == 0

    # Identify regions (connected components) in the background
    background_labels = skimage.measure.label(background_mask)
    
    properties = skimage.measure.regionprops_table(
        background_labels, 
        properties=['label',
                    'centroid',
                    'area', 
                    'eccentricity', 
                    'minor_axis_length', 
                    'major_axis_length', 
                    'orientation'],
        spacing=scale  # Include spacing if needed for calibration
        )

    # Convert to arrays for faster processing
    labels = properties['label']
    areas = properties['area']
    eccentricities = properties['eccentricity']
    minor_axis_lengths = properties['minor_axis_length']
    major_axis_lengths = properties['major_axis_length']
    orientations = properties['orientation']

    # Pre-calculate constants
    pi_by_3 = np.pi / 3
    inf_aspect_ratio = np.inf

    # Initialize a classification array (numeric for performance reasons)
    classifications = np.full_like(labels, fill_value=3, dtype=int)
    
    # Compute aspect ratios (avoiding division by zero)
    aspect_ratios = np.divide(major_axis_lengths, minor_axis_lengths, out=np.full_like(major_axis_lengths, inf_aspect_ratio), where=minor_axis_lengths != 0)
    
    # Classify "horizontal" regions: orientation more or less horizontal
    horizontal = np.abs(orientations) > np.abs(pi_by_3)
    
    # Rays: horizontal, high aspect ratio, and high eccentricity
    rays_mask = (horizontal) & (aspect_ratios > min_ray_aspect) & (eccentricities > min_ray_ecc)
    classifications[rays_mask] = 1  # 1 for Ray
    
    # Resin ducts: lower aspect ratio and larger area
    resin_ducts_mask = (areas > min_duct_area) & (eccentricities < 0.98) & (minor_axis_lengths > min_duct_width) & (~rays_mask)
    classifications[resin_ducts_mask] = 2  # 2 for Resin Duct
    
    
    unknown_mask = (~resin_ducts_mask) & (~rays_mask)
    
    # Initialize classified_mask (using the labels and classifications)
    classified_mask = np.zeros_like(background_labels)

    # Assign the ray classification (1) and resin duct classification (2)
    classified_mask[np.isin(background_labels, labels[rays_mask])] = 1
    classified_mask[np.isin(background_labels, labels[resin_ducts_mask])] = 2
    classified_mask[np.isin(background_labels, labels[unknown_mask])] = 3
        
    properties['classification'] = classifications
    
    properties_df = pd.DataFrame(properties)
    
    return properties_df, classified_mask


#########
def artefact_adjacent(labeled_image, classified_mask, ray_class=1, resin_duct_class=2, unknown_class=3):
    # Initialize sets to store adjacent labels for each class
    rays_adjacent = set()
    ducts_adjacent = set()
    unknown_adjacent = set()

    # Create shifted classified_mask images to check adjacencies
    vshift1 = classified_mask[:-1, :]
    vshift2 = classified_mask[1:, :]
    hshift1 = classified_mask[:, :-1]
    hshift2 = classified_mask[:, 1:]

    # Corresponding labeled image shifts
    label_vshift1 = labeled_image[:-1, :]
    label_vshift2 = labeled_image[1:, :]
    label_hshift1 = labeled_image[:, :-1]
    label_hshift2 = labeled_image[:, 1:]

    # Vertical adjacencies
    v_adj = vshift1 != vshift2
    v_ind = np.where(v_adj)

    for i in range(len(v_ind[0])):
        class1, class2 = vshift1[v_ind[0][i], v_ind[1][i]], vshift2[v_ind[0][i], v_ind[1][i]]
        label1, label2 = label_vshift1[v_ind[0][i], v_ind[1][i]], label_vshift2[v_ind[0][i], v_ind[1][i]]

        # Add labels based on classified_mask class
        if class1 == ray_class or class2 == ray_class:
            rays_adjacent.update([label1, label2])
        if class1 == resin_duct_class or class2 == resin_duct_class:
            ducts_adjacent.update([label1, label2])
        if class1 == unknown_class or class2 == unknown_class:
            unknown_adjacent.update([label1, label2])

    # Horizontal adjacencies
    h_adj = hshift1 != hshift2
    h_ind = np.where(h_adj)

    for i in range(len(h_ind[0])):
        class1, class2 = hshift1[h_ind[0][i], h_ind[1][i]], hshift2[h_ind[0][i], h_ind[1][i]]
        label1, label2 = label_hshift1[h_ind[0][i], h_ind[1][i]], label_hshift2[h_ind[0][i], h_ind[1][i]]

        # Add labels based on classified_mask class
        if class1 == ray_class or class2 == ray_class:
            rays_adjacent.update([label1, label2])
        if class1 == resin_duct_class or class2 == resin_duct_class:
            ducts_adjacent.update([label1, label2])
        if class1 == unknown_class or class2 == unknown_class:
            unknown_adjacent.update([label1, label2])

    # Convert sets to lists
    return list(rays_adjacent), list(ducts_adjacent), list(unknown_adjacent)

def adjacent_type_column(complete_df, 
                             rays_adjacent, 
                             duct_adjacent, 
                             unk_adjacent):
    # Create a new column 'adj_type' initialized with NaN
    complete_df['adj_type'] = np.nan
    
    # Set the column value to 1 if the label is in rays_adj
    complete_df.loc[complete_df['label'].isin(rays_adjacent), 'adj_type'] = 1
    
    # Set the column value to 2 if the label is in duct_adj
    complete_df.loc[complete_df['label'].isin(duct_adjacent), 'adj_type'] = 2
    
    # Set the column value to 3 if the label is in unk_adj
    complete_df.loc[complete_df['label'].isin(unk_adjacent), 'adj_type'] = 3

    return complete_df