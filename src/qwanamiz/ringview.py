# Generic python imports
import argparse
import pickle

# Application library imports
import pandas as pd
import numpy as np
import napari

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", help = """The prefix of the files to use for the analysis. Suffixes '_imgs.npz', '_cells.csv',
                                              '_adjacency.csv', and '_ring_imgs.npz' will be added to that prefix to obtain the input files.""")

    parser.add_argument("--pixel-size", dest = "pixel", type = float, default = 0.55042690590734,
                        help = """Size of a pixel in the wanted measurement unit. Defaults to 0.55042690590734 micrometers.""")

    args = parser.parse_args()

    pix_to_um = args.pixel

    cells = pd.read_csv(f"{args.prefix}_cells.csv")
    cells.set_index('label', inplace = True, drop = False)

    qwanamiz_images = np.load(f"{args.prefix}_imgs.npz")
    ring_images = np.load(f"{args.prefix}_ring_imgs.npz")

    with open(f"{args.prefix}_rings.pkl", "rb") as file:
        rings = pickle.load(file)

    with open(f"{args.prefix}_polygons.pkl", "rb") as file:
        polygons = pickle.load(file)

    viewer = napari.Viewer()

    # Drawing the binarized image
    viewer.add_image(qwanamiz_images['bw_img'], name='Original B&W', scale = [pix_to_um, pix_to_um])

    # Drawing the expanded labels (the cells)
    viewer.add_labels(qwanamiz_images['explabs'], name = 'Cells', scale = [pix_to_um, pix_to_um])

    # Drawing the set of boundaries found by qwanarings.py
    viewer.add_labels(ring_images['new_boundaries'], name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

    # DRAWING THE TREE-RING BOUNDARIES FOUND BY qwanarings.py
    # Prepare the lines visualization
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
    viewer.add_shapes(polygons,
                      shape_type='polygon',
                      edge_color='black',
                      face_color=[['red', 'green', 'blue', 'coral', 'black'][i % 5] for i in range(len(polygons))],
                      opacity = 0.3,
                      name='Tree-ring polygons')

    # Add the cells with their year identifier
    # Creating the year as an integer in the first place should be done, for now I haven't understood how
    viewer.add_labels(ring_images['year_image'].astype(int), name="Tree-ring year", opacity=0.7, scale=[pix_to_um, pix_to_um])

    # Enable napari to remain open after the script reaches the end
    napari.run()

