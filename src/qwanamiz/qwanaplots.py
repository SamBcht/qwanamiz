# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 13:17:22 2025

@author: sambo
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def draw_rings(
    prediction,
    year_image,
    filtered_mask,
    celldata,
    output_path,
    pix_to_um=1.0,
    radial_alpha=0.2,
    colorpal='inferno',
    point_radius=5,
    line_width=3
):
    """
    Create a full-resolution PNG summarizing prediction, year rings, and radial files,
    then overlay cell centroids and radial file connections (no intermediate save).

    Parameters
    ----------
    prediction : np.ndarray
        Grayscale base image (walls black, lumens white).
    year_image : np.ndarray
        Labeled year ring array (integer labels).
    filtered_mask : np.ndarray
        Radial file mask (boolean or labeled).
    celldata : pd.DataFrame
        DataFrame with 'centroid-0', 'centroid-1', and 'radial_file' columns.
    output_path : str
        Path to final PNG.
    pix_to_um : float, optional
        Micrometers per pixel (for coordinate scaling).
    radial_alpha : float, optional
        Transparency of radial file overlay.
    """

    # --- Ensure array types ---
    year_image = np.nan_to_num(year_image, nan=0).astype(int)
    filtered_mask = filtered_mask.astype(bool)
    h, w = prediction.shape

    # --- Create color map for year rings ---
    sample_colors = 8
    base_cmap = plt.colormaps[colorpal]
    colors = base_cmap(np.linspace(0, 1, sample_colors))
    # sequence 1,5,2,6,3,7,4,8 (index offset for zero-based)
    seq = [0,4,1,5,2,6,3,7]
    colors = colors[seq]
    colors = np.vstack([[0, 0, 0, 1], np.tile(colors, (int(np.ceil(year_image.max()/8)), 1))])
    cmap = ListedColormap(colors[:year_image.max()+1])

    # --- Convert base prediction to RGB ---
    bg = np.stack([prediction]*3, axis=-1)
    bg = (bg / bg.max() * 255).astype(np.uint8)

    # --- Year overlay ---
    year_rgba = cmap(year_image)
    year_rgb = (year_rgba[..., :3] * 255).astype(np.uint8)
    year_alpha = (year_image > 0).astype(np.uint8)[..., None] * 120
    year_layer = np.concatenate([year_rgb, year_alpha], axis=-1)

    # --- Radial file overlay ---
    rad_layer = np.zeros((h, w, 4), dtype=np.uint8)
    rad_layer[..., 0] = 255  # Red
    rad_layer[..., 3] = (filtered_mask * (255 * radial_alpha)).astype(np.uint8)

    # --- Combine layers ---
    base_img = Image.fromarray(bg, mode="RGB").convert("RGBA")
    year_img = Image.fromarray(year_layer, mode="RGBA")
    rad_img = Image.fromarray(rad_layer, mode="RGBA")

    combined = Image.alpha_composite(base_img, year_img)
    combined = Image.alpha_composite(combined, rad_img)
    # --- Prepare overlay drawing ---
    draw = ImageDraw.Draw(combined)

    def get_color(row):
        if not row["valid_radial_file"]:
            return (50, 50, 50, 255)  # grey
        wz = str(row.get("woodzone", "")).lower()
        if "early" in wz:
            return (0, 255, 0, 255)      # lime
        elif "late" in wz:
            return (255, 0, 255, 255)    # magenta
        else:
            return (50, 50, 50, 255)  # grey

    celldata = celldata.copy()
    celldata["colors"] = celldata.apply(get_color, axis=1)


    # --- Draw lines (per radial_file, but using cell colors) ---
    for rf, group in celldata.groupby("radial_file"):
        group = group.sort_values("centroid-1")
        x = (group["centroid-1"] / pix_to_um).astype(int).values
        y = (group["centroid-0"] / pix_to_um).astype(int).values
        cols = group["colors"].values
        for i in range(len(x) - 1):
            if 0 <= x[i] < w and 0 <= y[i] < h and 0 <= x[i+1] < w and 0 <= y[i+1] < h:
                draw.line([(x[i], y[i]), (x[i+1], y[i+1])], fill=cols[i+1], width=line_width)
                
    # --- Draw points ---
    for _, row in celldata.iterrows():
        x = int(row["centroid-1"] / pix_to_um)
        y = int(row["centroid-0"] / pix_to_um)
        col = row["colors"]
        if 0 <= x < w and 0 <= y < h:
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=col,
            )
            
        # --- Legend & Info ---
    try:
        sample_id = str(celldata["SampleId"].iloc[0])
    except Exception:
        sample_id = "Unknown"
    valid_years = celldata["year"].dropna().unique()
    n_rings = max(0, len(valid_years) - 2)

    text_lines = [
        f"Sample: {sample_id}",
        f"Detected Rings: {n_rings}"
    ]

    # --- Text placement ---
    margin = 30
    font_size = 180
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    x_text = margin
    y_text = margin

    for line in text_lines:
        draw.text((x_text, y_text), line, fill=(255,255,255,255), font=font, stroke_fill=(0, 0, 0, 255), stroke_width=15)
        y_text += font_size + 4


    # --- Save final image ---
    combined.save(output_path)
    print(f"✅ Final PNG image saved : {output_path}")

# A function that plots the main direction of each panel after running directionality function
# base_image: an numpy array of an image to use as a background
# vm_params: a dictionary of von Mises parameters returned by directionality
# scaling: the size of each pixel in the measurement unit used, for scaling back to image coordinates
# cmap: the color map to use for the background image, defaults to 'gray' (grayscale)
def plot_directionality(base_image, vm_params, scaling, cmap = 'gray'):
    
    # Displaying the background image
    plt.imshow(base_image, cmap = cmap)

    # Looping over each panel on which the von Mises parameters wer fitted
    for _, params in vm_params.items():

        # Drawing a rectangle that shows the extent of the panel
        x = np.array(params['x'])[[0, 1, 1, 0, 0]] / scaling
        y = np.array(params['y'])[[0, 0, 1, 1, 1]] / scaling
        plt.plot(x, y, linewidth = 2, c = 'red')

        # Determining arrow coordinates starting from the middle of each panel
        # Arrows will display the angle determined as the radial angle in this panel
        # Here the arrow length is hard-coded to be 1/4 of the panel width; this could be adjusted if needed
        center = np.array([params['x'][0] + params['x'][1], params['y'][0] + params['y'][1]]) / 2 / scaling
        arrow_length = (params['x'][1] - params['x'][0]) / 4 / scaling
        arrow_end = center + np.array([np.cos(params['mu']), np.sin(params['mu'])]) * arrow_length

        # Drawing the arrow
        plt.annotate('',
                     xytext = np.array([center[0], center[1]]),
                     xy = np.array([arrow_end[0], arrow_end[1]]),
                     arrowprops = dict(arrowstyle = "->", color = 'red', lw = 2))

    return

# A function that plots adjacency edges and allows for selecting adjacency type
# base_image: an numpy array of an image to use as a background
# adjacency: a DataFrame of adjacencies, such as returned
# scaling: the size of each pixel in the measurement unit used, for scaling back to image coordinates
# adj_type: the type of adjacency to display on the plot. If None (default), then all are displayed
# color: the color to use for displaying adjacencies, defaults to 'blue'
# linewidth: the line width to use for displaying adjacencies, defaults to 1
# cmap: the color map to use for the background image, defaults to 'gray' (grayscale)
def plot_adjacencies(base_image, adjacency, scaling, adj_type = None, color = 'blue', linewidth = 1, cmap = 'gray'):

    # Displaying the background image
    plt.imshow(base_image, cmap = cmap)

    # Looping over the adjacencies
    for _, row in adjacency.iterrows():

        # Extracting the endpoints of the adjacencies
        y1, x1 = row["centroid1"]
        y2, x2 = row["centroid2"]

        # Displaying the adjacency as a line if it is the selected type
        if adj_type is None or row["wall_classification"] == adj_type:
            plt.plot(np.array([x1, x2]) / scaling, np.array([y1, y2]) / scaling, c = color, linewidth = linewidth)

    return

# A function that plots lines representing radial files
# base_image: an numpy array of an image to use as a background
# cells: a DataFrame of individual cell measurements, with columns 'radial_file' and 'file_rank'
# scaling: the size of each pixel in the measurement unit used, for scaling back to image coordinates
# linewidth: the line width to use for displaying radial files, defaults to 1
# cmap: the color map to use for the background image, defaults to 'gray' (grayscale)
def plot_radial_files(base_image, cells, scaling, linewidth = 1, cmap = 'gray'):

    # Displaying the background image
    plt.imshow(base_image, cmap = cmap)

    # Extracting the set of radial file IDs
    radial_file_ids = np.unique([i for i in cells['radial_file'] if i is not None])

    # Looping over the radial files
    for i in radial_file_ids:

        # Extracting a DataFrame with only cells in a given radial file
        radial_file_df = cells[cells['radial_file'] == i]

        # It is not worth displaying cells that are the only member of their radial file
        if(len(radial_file_df) == 1):
            continue

        # Sorting the cells by file rank so the lines are drawn in the right order
        radial_file_df = radial_file_df.sort_values(by = 'file_rank')

        # Displaying the lines
        plt.plot(radial_file_df['centroid-1'] / scaling, radial_file_df['centroid-0'] / scaling, linewidth = linewidth)

    return

# A function that plots lines representing the cell diameters that were measured
# base_image: an numpy array of an image to use as a background
# cells: a DataFrame of individual cell measurements, with columns 'diameter_rad', 'diameter_tan', 'extr_rad', and 'extr_tan'
# scaling: the size of each pixel in the measurement unit used, for scaling back to image coordinates
# linewidth: the line width to use for displaying diameters, defaults to 1
# cmap: the color map to use for the background image, defaults to 'gray' (grayscale)
def plot_diameters(base_image, cells, scaling, linewidth = 1, cmap = 'gray'):

    # Displaying the background image
    plt.imshow(base_image, cmap = cmap)

    # Looping over the cells
    for index, row in cells.iterrows():
        
        # We only display the radial diameter if it was measured
        if row['diameter_rad'] is not None:
            point1, point2 = row['extr_rad']
            y1, x1 = point1
            y2, x2 = point2
            plt.plot(np.array([x1, x2]) / scaling, np.array([y1, y2]) / scaling, c = 'blue', linewidth = linewidth)

        # We only display the tangential diameter if it was measured
        if row['diameter_tan'] is not None:
            point1, point2 = row['extr_tan']
            y1, x1 = point1
            y2, x2 = point2
            plt.plot(np.array([x1, x2]) / scaling, np.array([y1, y2]) / scaling, c = 'green', linewidth = linewidth)

    return
