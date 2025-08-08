import argparse
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

    qwanamiz_images = np.load(f"{args.prefix}_imgs.npz")
    ring_images = np.load(f"{args.prefix}_ring_imgs.npz")

    viewer = napari.Viewer()

    # Drawing the binarized image
    viewer.add_image(qwanamiz_images['bw_img'], name='Original B&W', scale = [pix_to_um, pix_to_um])

    # Drawing the expanded labels (the cells)
    viewer.add_labels(qwanamiz_images['explabs'], name = 'Cells', scale = [pix_to_um, pix_to_um])

    # Drawing the set of boundaries found by qwanarings.py
    viewer.add_labels(ring_images['new_boundaries'], name="Boundary Labels", opacity=0.7, scale=[pix_to_um, pix_to_um])

    # Enable napari to remain open after the script reaches the end
    napari.run()

