# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:16:40 2024

@author: sambo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:28:00 2024

@author: sambo
"""
import os
import argparse
import glob
import datetime
import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.color
# import cv2
# from PIL import Image
#from skimage import img_as_ubyte
from scipy.ndimage import distance_transform_edt
#import matplotlib
import matplotlib.pyplot as plt
#import networkx as nx
import skimage.graph
import skimage.util
import qwanamiz
#from tools import histogram
#from mixture import density, vonmises_pdfit, mixture_pdfit, pdfit
#from typing import Tuple 
#from scipy.special import i0
#from scipy.special import iv
#from scipy.optimize import fsolve
#from scipy.stats import vonmises

def batch_measurements(img_path, sampleID = "Sample1"):

    start = datetime.datetime.now()


    ##############################################################################
    # Image processing and lumen preliminary measurements
    print("Image processing and lumen measurements")

    ##### Load the input array
    ## INPUT : 'prediction' is a numpy array of float64, output of roxasAI algorithm
    # it is the result of the binarization of the original image.
    # The array has the same size as the original image
    prediction = skimage.io.imread(img_path)

    ## IMAGE METADATA AND RESOLUTION :
    # 10x scans have a resolution of 46146 dpi. We can define a scaling factor
    # with conversion : 1 Pixel = 0.55042690590734 Microns
    pix_to_um = 0.55042690590734

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
                'bbox')))
    
    regionprops_df['SampleId'] = sampleID


    ## DISTANCE MAP OF CELL WALLS : Compute the distance map of cell walls pixels,
    # e.g. background pixels. Return an image where each back ground pixel takes
    # the value of the distance to the nearest cell lumen in microns with the 
    # sampling parameter set to the scaling factor.
    distance_map = distance_transform_edt(labeled_image == 0,
                                          sampling = pix_to_um)

    ## EXPAND CELL LUMENS UNTIL JUNCTION WITH ADJACENT CELLS : Expand labels in 
    # label image by distance pixels without overlapping.
    # The distance parameter can be considered as a cell wall thickness threshold
    # Two lumens separated by more than two times the distance won't be considered
    # as adjacent.
    expanded_labels = skimage.segmentation.expand_labels(labeled_image,
                                                         distance = 10,
                                                         spacing = pix_to_um)
    
    expandprops_df = pd.DataFrame(
        skimage.measure.regionprops_table(
            expanded_labels,
            spacing = pix_to_um,
            properties = (
                'label',
                'area')
            )
        )

    regionprops_df = regionprops_df.join(expandprops_df.set_index('label'), 
                        on = 'label',  
                        lsuffix = '_lumen',
                        rsuffix = '_cell',
                        validate = '1:1')


    endTime = datetime.datetime.now()
    print(f'runtime : {endTime - start}')
    ####################################################################################

    ##################################################################################
    #### Step 2 : Cell adjacency analysis
    print("Adjacency analysis")

    ## REGION ADJACENCY GRAPH : The get_adjacent_labels function compute a simplified
    # Region Adjacency Graph. Return a set of adjacent cells pairs e.g. each pair
    # of labels that share a common border.
    adj_graph = qwanamiz.get_adjacent_labels(expanded_labels)

    # Transform the set of label pairs in a dataframe, retrieve label centroid
    # coordinates and measure edge angle, length and center
    adjacency = qwanamiz.adjacency_dataframe(adj_graph, regionprops_df)

    endTime = datetime.datetime.now()
    print(f'runtime : {endTime - start}')
    ###############################################################################

    ##############################################################################
    # Directionnality analysis
    print("Directionnality analysis")

    # Determine the bounds of the subsamples
    img_height, img_width = prediction.shape

    adjacency, vm_parameters, angle_plot = qwanamiz.directionnality(
        adjacency,
        image_height = img_height, 
        image_width = img_width,
        spacing = pix_to_um)


    endTime = datetime.datetime.now()
    print(f'runtime : {endTime - start}')

    ######################################################################################
    # Compute cell wall thickness between centroids of adjacent cells
    print("Wall thickness measurements")
    adjacency = qwanamiz.measure_wallthickness(adjacency, distance_map)
         

    endTime = datetime.datetime.now()
    print(f'runtime : {endTime - start}')

    ############################################################################

    ################################################################################
    # Edge classification and filtering
    print("Edge classification")


    qwanamiz.classify_edges(adjacency, tolerance = 15)

    endTime = datetime.datetime.now()
    print(f'runtime : {endTime - start}')

    ###################################################################################

    ###################################################################################
    # Radial files grouping
    print("neighbor mapping and radial files detection")

    # Take approx. 1-2min to run
    qwanamiz.find_neighbors(adjacency) 

    qwanamiz.refine_neighbors(adjacency)

    # Assign radial files to the edges
    qwanamiz.update_neighbors(adjacency)

    qwanamiz.assign_radial_files(adjacency)

    endTime = datetime.datetime.now()
    print(f'runtime : {endTime - start}')

    print("labels and edges correspondance")
    qwanamiz.get_cell_walls(regionprops_df, adjacency)

    endTime = datetime.datetime.now()
    print(f'runtime : {endTime - start}')

    print("Measure lumen diameters")
    qwanamiz.measure_diameters(regionprops_df, spacing = pix_to_um)
    
    regionprops_df = regionprops_df.drop(
        columns = [
            'image',
            'bbox-0',
            'bbox-1',
            'bbox-2',
            'bbox-3'])

    endTime = datetime.datetime.now()
    print(f'runtime : {endTime - start}')


    print("successfully run")
    
    return regionprops_df, adjacency, angle_plot, vm_parameters, distance_map, expanded_labels, labeled_image


def get_basename(input_file, remove = '_ring.TIF'):
    base_name = os.path.basename(input_file)
    base_name = base_name.replace(remove, '')
    return base_name


if __name__ == '__main__':

    # Set the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help = """A single directory to process. All png files in that directory will be processed.""")
    parser.add_argument("output", help = """A directory to write output files to.""")

    # Parse the arguments
    args = parser.parse_args()

    # Process each image in the input folder
    # Adapt the parameter to the input file type
    img_paths = glob.glob(os.path.join(args.input, '*.png'))
    
    
    ###------------------------------------------- Process files --------------------------------###
    
    for img_path in img_paths:
        
        # Adapt the parameter to the input file type
        base_name = get_basename(img_path, remove = '_segmented.png')
        
        # Run the workflow script
        print(f"Running workflow on {base_name}")
        regionprops_df, adjacency, angle_plot, vm_parameters, distance_map, expanded_labels, labeled_image = batch_measurements(img_path, 
                                                                                                                                sampleID = base_name)

        print('save output')
        start_save = datetime.datetime.now()
        
        # Save the workflow output images
        output_path = os.path.join(args.output, f"{base_name}_imgs")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, dmap = distance_map, 
                            explabs = expanded_labels, 
                            labs = labeled_image)
        #np.save(output_path, distance_map)
        
        #output_path = os.path.join(args.output, f"{base_name}_explabs.npy")
        #np.save(output_path, expanded_labels)
        
        #output_path = os.path.join(args.output, f"{base_name}_labs.npy")
        #np.save(output_path, labeled_image)
        
        output_path = os.path.join(args.output, f"{base_name}_angles.png")
        angle_plot.savefig(output_path)
        
        # Save the cell measurements dataframe
        output_path = os.path.join(args.output, f"{base_name}_cells.csv")
        regionprops_df.to_csv(output_path, index=False)
        
        # Save the adjacency dataframe
        output_path = os.path.join(args.output, f"{base_name}_adjacency.csv")
        adjacency.to_csv(output_path, index=True)
        
        output_path = os.path.join(args.output, f"{base_name}_params.csv")
        (pd.DataFrame.from_dict(data=vm_parameters, orient='index')
         .to_csv(output_path, header=True))
        
        print(f"Saved workflow output to {output_path}")
        endTime = datetime.datetime.now()
        print(f'runtime : {endTime - start_save}')

