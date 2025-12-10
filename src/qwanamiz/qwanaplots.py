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
