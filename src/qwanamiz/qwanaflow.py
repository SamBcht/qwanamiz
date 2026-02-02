# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:16:40 2024

@author: sambo
"""

# Python base library imports
import os
import sys
import argparse
import glob
import datetime

# scikit-image imports
import skimage.io
import skimage.measure

# local qwanamiz imports
import qwanamiz.qwanamiz as qmiz

def batch_measurements(img_path, sampleID = "Sample1", pixel_size = 1, dir_nrows = None, dir_ncols = None,
                       area_threshold = 500, solidity_threshold = 0.95, max_wall_distance = 10,
                       convergence_threshold = 0.001, angle_tolerance = 5, stitch_angle_tolerance = 20, ncores = 1):

    # Determining start time of the analysis to compute run time
    start = datetime.datetime.now()

    #### Step 1 : Cell labeling and measurements

    # 'prediction' is a numpy array of float64 resulting from the binarization of the original image.
    print("Reading input image")
    prediction = skimage.io.imread(img_path, as_gray = True)
    img_height, img_width = prediction.shape
    qmiz.update_runtime(start)

    ## CELL LUMEN DETECTION : the label function from scikit.image package
    # computes a connected components analysis on the binary image.
    # Two pixels are connected e.g. belong to the same cell lumen
    # when they are neighbors and have the same value (here 'black' or 'white').
    print("Labeling individual cells")
    labeled_image = skimage.measure.label(prediction)
    qmiz.update_runtime(start)

    ## CELL LUMEN MEASUREMENTS : the function 'measure_lumens' measures lumen 
    # traits. Returns a pandas dataframe with measurements in microns if
    # spacing is set with the scaling factor.
    print("Measuring lumen-related features")
    cell_df = qmiz.measure_lumens(labeled_image, spacing = pixel_size, nprocesses = ncores)
    qmiz.update_runtime(start)

    ## Splitting merged cells using watershed segmentation
    #  This step needs to return updated cell measurements (cell_df)
    #  and labeled_image. It also returns an array that contains the result of the watershed segmentation
    print("Splitting cells using watershed algorithm")
    labeled_image, cell_df, watershed_result = qmiz.adjust_labels(labeled_image, cell_df, scale = pixel_size,
                                                                  area_threshold = area_threshold,
                                                                  solidity_threshold = solidity_threshold)
    qmiz.update_runtime(start)

    ## DISTANCE MAP OF CELL WALLS : Compute the distance map of cell wall pixels,
    # e.g. background pixels. Return an image where each background pixel takes
    # the value of the distance to the nearest cell lumen in microns with the 
    # sampling parameter set to the scaling factor.
    print("Compute distance from cell wall pixel to nearest lumen")
    distance_map, nearest_label_coords = qmiz.measure_distance(labeled_image = labeled_image, scaling = pixel_size)
    qmiz.update_runtime(start)

    ## EXPAND CELL LUMENS UNTIL JUNCTION WITH ADJACENT CELLS : Expand labels in 
    # label image by distance pixels without overlapping.
    # The distance parameter can be considered as a cell wall thickness threshold
    # Two lumens separated by more than two times the distance won't be considered
    # as adjacent.
    print("Measuring whole cell-related features")

    # Producing an array with integers corresponding to individual cells (both lumen and wall)
    expanded_labels = qmiz.expand_cells(labeled_image,
                                        distance_map,
                                        nearest_label_coords,
                                        max_distance = max_wall_distance)

    # Measuring properties on whole cells
    cell_df = qmiz.measure_cells(cell_df, expanded_labels, spacing = pixel_size)
    
    qmiz.update_runtime(start)

    #### Step 2 : Cell adjacency analysis and radial files

    ## REGION ADJACENCY GRAPH
    # The returned DataFrame is indexed by label pairs corresponding
    # to edges in the graph, with metadata on the adjacencies in various columns
    print("Producing region adjacency graph")
    adjacency = qmiz.adjacency_dataframe(expanded_labels, cell_df)
    qmiz.update_runtime(start)

    ### Directionality analysis
    print("Analyzing directionality and classifying edges")

    # Automatically determining the number of rows and columns in the image for
    # the directionality analysis unless the parameters were explicitly set
    if dir_nrows is None or dir_ncols is None:
        if dir_nrows is not None:
            print("Warning: --dir-nrows was set but not --dir-ncols. Ignoring --dir-nrows argument")

        if dir_ncols is not None:
            print("Warning: --dir-ncols was set but not --dir-nrows. Ignoring --dir-ncols argument")

        nb_rows, nb_cols = qmiz.calculate_grid(image_width = img_width, 
                                               image_height = img_height, 
                                               pixel_to_micron = pixel_size)

    else:
        nb_rows = dir_nrows
        nb_cols = dir_ncols

    

    # Determining the directionality angle for each part of the image
    adjacency, vm_parameters = qmiz.directionality(adjacency,
                                                   image_height = img_height, 
                                                   image_width = img_width,
                                                   spacing = pixel_size,
                                                   num_rows = nb_rows,
                                                   num_cols = nb_cols,
                                                   convergence_threshold = convergence_threshold)

    # Edge classification and filtering
    qmiz.classify_edges(adjacency, tolerance = angle_tolerance)

    qmiz.update_runtime(start)

    # Assign cells to radial files
    print("Assigning cells to radial files")
    cell_df, adjacency = qmiz.assign_radial_files(cell_df, adjacency, stitch_angle_tolerance = stitch_angle_tolerance)
    qmiz.update_runtime(start)
    
    #### Step 3: Measure lumen diameters and cell walls

    print("Measuring lumen diameters")
    cell_df = qmiz.measure_diameters(cell_df, spacing = pixel_size, nprocesses = ncores)
    qmiz.update_runtime(start)

    # Compute cell wall thickness between centroids of adjacent cells
    print("Measuring wall thickness")
    cell_df, adjacency = qmiz.measure_walls(cell_df,
                                            adjacency,
                                            distance_map,
                                            auto_pixelwidth = True,
                                            scale = pixel_size,
                                            scan_width = 75,
                                            nprocesses = ncores)
    qmiz.update_runtime(start)

    # A function that prepares the cell DataFrame for output
    cell_df = qmiz.prepare_cell_output(cells = cell_df, sampleID = sampleID)

    print("Successfully run")
    
    return cell_df, adjacency, vm_parameters, prediction, distance_map, expanded_labels, labeled_image, watershed_result, nb_rows, nb_cols

def get_basename(input_file, remove = '.png'):
    base_name = os.path.basename(input_file)
    base_name = base_name.replace(remove, '')
    return base_name

def main():

    # Set the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help = """If a single directory, all png files in that directory will be processed.
                                           If a single .png file, process only that file.
                                           If a single .txt file, should be a file containing a list of files to process, with one .png file per line.""")

    parser.add_argument("output", help = """A directory to write output files to.""")
    
    parser.add_argument("--pixel-size", dest = "pixel", type = float, default = 1,
                        help = """Conversion factor from of a single pixel to the desired measurement unit. Defaults to 1 (measurements in pixels).""")


    parser.add_argument("--dir-nrows", "-r", dest = "nrows", type = int, default = None,
                        help = """Number of rows to split the image into for the directionality analysis.
                                  If None (the default), the number of rows and colums is automatically determined.""")

    parser.add_argument("--dir-ncols", "-c", dest = "ncols", type = int, default = None,
                        help = """Number of columns to split the image into for the directionality analysis.
                                  If None (the default), the number of rows and colums is automatically determined.""")

    parser.add_argument("--disable-plots", dest = "noplots", action = "store_true",
                        help = """Specify this flag to disable the generation of angle plots. By default they will be produced.""")

    parser.add_argument("--area-threshold", dest = "area_threshold", type = float, default = 500.0,
                        help = """Lumen area above which a cell can be considered for splitting into several cells using
                                  the watershed segmentation algorithm. For a cell to be selected, it must also be below the
                                  solidity threshold. Defaults to 500.""")

    parser.add_argument("--solidity-threshold", dest = "solidity_threshold", type = float, default = 0.95,
                        help = """Solidity threshold below which a cell can be considered for splitting into several cells
                                  using the watershed segmentation algorithm. Higher values (closer to 1) indicate a more convex shape
                                  whereas lower values (closer to 0) indicate concavity (presence of indentations).
                                  For a cell to be selected, it must also be above the lumen area threshold. Defaults to 0.95.""")

    parser.add_argument("--max-wall-distance", dest = "max_wall_distance", type = float, default = 10.0,
                        help = """The maximum distance (in the target measurement unit defined by --pixel-size) between a cell
                                  wall pixel and the nearest cell lumen for that pixel to be considered as belonging to the cell.
                                  This parameter effectively puts a cap on the maximum possible cell wall thickness. Defaults to 10.""")

    parser.add_argument("--vm-threshold", dest = "vmthreshold", type = float, default = 0.001,
                        help = """The convergence threshold in the search of von Mises distribution parameters.
                                  Lower values result in more precise results but slower convergence.
                                  Defaults to 0.001.""")

    parser.add_argument("--angle-tolerance", dest = "angle", type = float, default = 5,
                        help = """The tolerance (in degrees) around the lower and upper bounds found by the
                                  directionality algorithm in determining which cell adjacencies are tangential and
                                  which are radial. A higher value means potentially longer, but inexact, radial files.""")

    parser.add_argument("--stitch-angle-tolerance", dest = "stitch_angle", type = float, default = 20,
                        help = """The tolerance (in degrees) around the lower and upper bounds found by the
                                  directionality algorithm in determining which cell adjacencies are tangential and
                                  which are radial. This angle is applied after the initial radial file assignment
                                  in stitching together radial files and should therefore use a more permissive
                                  angle threshold.""")

    parser.add_argument("--ncores", dest = "ncores", type = int, default = 1,
                        help = """The number of processes to launch for multiprocessing for computing wall thickness.
                        Defaults to 1 (no multiprocessing).""")

    # Parse the arguments
    args = parser.parse_args()

    # If 'input' is a directory then all images ending in .png in that directory will be processed
    if(os.path.isdir(args.input)):
        # Process each .png image in the input folder
        img_paths = glob.glob(os.path.join(args.input, '*.png'))
    elif(os.path.isfile(args.input) and os.path.splitext(args.input)[1] == ".png"):
        # In this case the list of paths is simply the file itself
        img_paths = [args.input]
    elif(os.path.isfile(args.input) and os.path.splitext(args.input)[1] == ".txt"):
        with(open(args.input, "r") as f):
            img_paths = [line.rstrip() for line in f]

        if not all([os.path.exists(file) and file.endswith(".png") for file in img_paths]):
            sys.exit(f"ERROR: Not all files in {args.input} exist or are.png files")

    else:
        sys.exit(f"ERROR: qwanaflow does not know how to process file {args.input}")
    
    
    ###------------------------------------------- Process files --------------------------------###
    start_save = datetime.datetime.now()
    for img_path in img_paths:
        
        # Adapt the parameter to the input file type
        base_name = get_basename(img_path, remove = '.png')
        
        # Run the workflow script
        print(f"Running workflow on {base_name}")
        cell_df, adjacency, vm_parameters, prediction, distance_map, \
                expanded_labels, labeled_image, watershed_result, nrows, ncols = batch_measurements(img_path, 
                                                                                            sampleID = base_name,
                                                                                            pixel_size = args.pixel,
                                                                                            dir_nrows = args.nrows,
                                                                                            dir_ncols = args.ncols,
                                                                                            area_threshold = args.area_threshold,
                                                                                            solidity_threshold = args.solidity_threshold,
                                                                                            max_wall_distance = args.max_wall_distance,
                                                                                            convergence_threshold = args.vmthreshold,
                                                                                            angle_tolerance = args.angle,
                                                                                            stitch_angle_tolerance = args.stitch_angle,
                                                                                            ncores = args.ncores)
        
        print('Saving outputs')
        
        qmiz.write_qwanaflow_outputs(output = args.output,
                                     base_name = base_name,
                                     prediction = prediction,
                                     distance_map = distance_map,
                                     expanded_labels = expanded_labels,
                                     labeled_image = labeled_image,
                                     watershed_result = watershed_result,
                                     vm_parameters = vm_parameters,
                                     nrows = nrows,
                                     ncols = ncols,
                                     cell_df = cell_df,
                                     adjacency = adjacency,
                                     noplots = args.noplots)
        
        output_dir = os.path.join(args.output, f"{base_name}_outputs")
        print(f"Saved workflow output to {output_dir}")

        qmiz.update_runtime(start_save)

