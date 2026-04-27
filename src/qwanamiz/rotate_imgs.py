# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:01:23 2026

@author: sambo
"""

#!/usr/bin/env python3

import os
import argparse
import numpy as np
import tifffile
from PIL import Image


def rotate_images_in_folder(input_folder, output_folder, direction=-90):
    """
    Rotate images by ±90° without modifying pixel values.
    Preserves TIFF resolution metadata and compression.
    Preserves PNG/JPEG DPI metadata when available.
    """

    os.makedirs(output_folder, exist_ok=True)

    if direction == 90:
        k = 1
        pil_method = Image.Transpose.ROTATE_90
    elif direction == -90:
        k = -1
        pil_method = Image.Transpose.ROTATE_270
    else:
        raise ValueError("Direction must be 90 or -90.")

    for filename in os.listdir(input_folder):

        if not filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        print(f"Processing {filename}...")

        # -------- TIFF --------
        if filename.lower().endswith((".tif", ".tiff")):

            with tifffile.TiffFile(input_path) as tif:
                page = tif.pages[0]

                image = page.asarray()
                rotated = np.rot90(image, k=k)

                compression = page.compression

                xres = page.tags.get("XResolution")
                yres = page.tags.get("YResolution")
                resunit = page.tags.get("ResolutionUnit")

                resolution = None
                resolutionunit = None

                if xres and yres:
                    resolution = (xres.value, yres.value)

                if resunit:
                    resolutionunit = resunit.value

            tifffile.imwrite(
                output_path,
                rotated,
                compression=compression,
                resolution=resolution,
                resolutionunit=resolutionunit,
                photometric="rgb" if rotated.ndim == 3 else None,
            )

        # -------- PNG / JPEG --------
        else:
            with Image.open(input_path) as img:

                rotated = img.transpose(pil_method)
                dpi = img.info.get("dpi", None)

                if dpi:
                    rotated.save(output_path, dpi=dpi)
                else:
                    rotated.save(output_path)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Rotate images by ±90° while preserving metadata."
    )

    parser.add_argument(
        "input_folder",
        help="Path to input folder containing images",
    )

    parser.add_argument(
        "output_folder",
        help="Path to output folder",
    )

    parser.add_argument(
        "direction",
        nargs="?",
        type=int,
        choices=[90, -90],
        default=-90,
        help="Rotation direction: 90 (Couter Clock Wise) or -90 (Clock Wise). Default = -90.",
    )

    args = parser.parse_args()

    rotate_images_in_folder(
        args.input_folder,
        args.output_folder,
        args.direction,
    )


if __name__ == "__main__":
    main()