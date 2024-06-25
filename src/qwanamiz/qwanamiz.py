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
from tools import histogram
from mixture import density, vonmises_pdfit, mixture_pdfit, pdfit, vonmises_density
from scipy.stats import vonmises
###########################################################################
# Function to get the adjacent cells
def get_adjacent_labels(labeled_image, background_label=0):
    adjacent_labels = set()

    # Iterate over the labeled image
    for i in range(1, labeled_image.shape[0] - 1):
        for j in range(1, labeled_image.shape[1] - 1):
            label = labeled_image[i, j]
            
            # Skip background pixels
            if label == background_label:
                continue

            # Check neighboring pixels
            neighbors = [labeled_image[i-1, j], labeled_image[i+1, j], labeled_image[i, j-1], labeled_image[i, j+1]]
            neighbors = [neighbor for neighbor in neighbors if neighbor != label and neighbor != background_label]

            # Add unique pairs of neighboring labels
            adjacent_labels.update([tuple(sorted((label, neighbor))) for neighbor in neighbors])

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
                                  'area',
                                  'major_axis_length',
                                  'minor_axis_length',
                                  'orientation',
                                  'perimeter_crofton',
                                  'image',
                                  'bbox-0',
                                  'bbox-1',
                                  'bbox-2',
                                  'bbox-3'])
    
    # Merge to get coordinates for label2
    adj_df = adj_df.merge(lumen_props,
                 left_on = 'label2',
                 right_on = 'label',
                 suffixes = ('', '_label2')
                          ).drop(
                              columns = [
                                  'label',
                                  'area',
                                  'major_axis_length',
                                  'minor_axis_length',
                                  'orientation',
                                  'perimeter_crofton',
                                  'image',
                                  'bbox-0',
                                  'bbox-1',
                                  'bbox-2',
                                  'bbox-3'])
    
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
                    k_threshold = 50):
    
    row_height = (image_height*spacing) / num_rows
    col_width = (image_width*spacing) / num_cols


    # Subsampling image and filtering of edges based on von Mises distributions

    # Create a figure to display the histograms
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    
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
            
            # Fit mixture of von Mises distributions
            iterations = 0
            while iterations < max_iterations:
                
                m = mixture_pdfit(angle_rad,n=3)
            
            
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
                'nb_cells': len(angle_rad),
                'cell_index': subsample_edges.index}
           
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
    fig.show()
    
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
    
    return merged_df, subsample_params, fig


#########################################################################
# Cell Wall Measurements
def thickness_between_centroids(centroid1, centroid2, dist_map, pixelwidth = 10):
    # Define profile line between centroids
    mid_line = skimage.measure.profile_line(
        dist_map,
        centroid1,
        centroid2,
        linewidth = pixelwidth,
        reduce_func = None)
    
    # Measure maximum thickness along profile line
    max_val = np.max(mid_line, axis = 0)
    
    # Calculate the mean of these maximum values
    max_thickness = np.mean(max_val)

    return max_thickness

def measure_wallthickness(adj_df, dist_map, scan_width = 10):
    
    adj_df['wall_thickness'] = adj_df.apply(
        lambda row: thickness_between_centroids(row['centroid1'],
                                                row['centroid2'],
                                                dist_map = dist_map,
                                                pixelwidth = scan_width),
        axis=1)
    
    return adj_df

#########################################################################
# Classify cell walls between radial and tangential
def classify_edges(df, tolerance = 5):
    
    # Create a new column 'edge_classification' with initial value 'rad_wall'
    df['wall_classification'] = 'radial'
    
    # Convert tolerance in radians
    tolerance_rad = np.radians(tolerance)

    # Update the classification to 'tan_wall' for edges with angle between lower and upper bounds
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
        

    # Filter ambiguous edges and initialize columns for reclassification
    ambiguous_edge = complete_df[complete_df['wall_classification'] == 'indoubt']
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
            ambiguous_edge.at[edge, 'situation'] = 'end'
            
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
    
    complete_df['wall_classification'].update(ambiguous_edge['situation'])
    
    return complete_df
###################################################################################
# Update neighbors after refining
def update_neighbors(complete_df):
    
    # Filter the DataFrame
    edges_df = complete_df[
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
    
    complete_df['neighbors'].update(edges_df['neighbors'])
    
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
    cells_df['left_wall_thickness'] = 0
    cells_df['left_angle'] = 0
    cells_df['right_neighbor'] = 0
    cells_df['right_wall_thickness'] = 0
    cells_df['right_angle'] = 0
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
            
            if neighbor_centroid[0] < label_centroid[0]:  # Left neighbor
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
            
            if neighbor1_centroid[0] < label_centroid[0] and neighbor2_centroid[0] > label_centroid[0]:
                # Proper left and right neighbors
                cells_df.at[idx, 'left_neighbor'] = neighbor1_label
                cells_df.at[idx, 'left_wall_thickness'] = edge_data1['wall_thickness']
                cells_df.at[idx, 'left_angle'] = edge_data1['angle']
                cells_df.at[idx, 'right_neighbor'] = neighbor2_label
                cells_df.at[idx, 'right_wall_thickness'] = edge_data2['wall_thickness']
                cells_df.at[idx, 'right_angle'] = edge_data2['angle']
            
            elif neighbor1_centroid[0] > label_centroid[0] and neighbor2_centroid[0] < label_centroid[0]:
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
    - labeled_image: 2D numpy array where each object is labeled with a unique integer.
    - df: DataFrame containing columns 'label', 'centroid-0', 'centroid-1', and 'angle'.
    
    Returns:
    - df: Updated DataFrame with additional 'diameter_rad' and 'diameter_tan' columns.
    """
    complete_df['diameter_rad'] = None
    complete_df['diameter_tan'] = None
    complete_df['extr_rad'] = None
    complete_df['extr_tan'] = None
    
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

        # Add the diameters to the dataframe
        complete_df.at[index, 'diameter_rad'] = rad_diameter * spacing
        complete_df.at[index, 'diameter_tan'] = tan_diameter * spacing
        complete_df.at[index, 'extr_rad'] = rad_extr
        complete_df.at[index, 'extr_tan'] = tan_extr

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
            
    (y1, x1), (y2, x2) = switch_points[:2]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    diam_coords = ((y1 + min_row) * spacing, (x1 + min_col) * spacing), ((y2 + min_row) * spacing, (x2 + min_col)* spacing)


    return distance, diam_coords
