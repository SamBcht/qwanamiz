# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:13:17 2025

@author: sambo
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

# --- Dimensions ---
cm_to_inch = 1 / 2.54
total_width_cm = 14
fig_width_in = total_width_cm * cm_to_inch

# Set widths of subplots in cm
img_width_cm = 9
hmap_width_cm = 5
img_width_frac = img_width_cm / total_width_cm
hmap_width_frac = hmap_width_cm / total_width_cm

# Calculate subplot heights
aspect_img = cropped.shape[0] / cropped.shape[1]
aspect_hmap = cropped_array.shape[0] / cropped_array.shape[1]

# Match height based on image height
img_height_in = img_width_cm * cm_to_inch * aspect_img
hmap_height_in = hmap_width_cm * cm_to_inch * aspect_hmap

# Final figure height: max of both + space for legend (optional)
legend_space_in = 1  # you can reduce this if needed
fig_height_in = max(img_height_in, hmap_height_in) + legend_space_in

# --- Create grid layout ---
fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
gs = gridspec.GridSpec(nrows=2, ncols=2, 
                       width_ratios=[img_width_frac, hmap_width_frac], 
                       height_ratios=[max(img_height_in, hmap_height_in), legend_space_in],
                       #height_space=0.05
                       )

# --- Image plot (left) ---
ax_img = fig.add_subplot(gs[0, 0])
ax_img.imshow(cropped, cmap=cmap, interpolation="nearest", vmin=0.0001, origin='lower')
ax_img.axis("off")

# Add overlays
ax_img.imshow(overlay)
ax_img.imshow(shadow)
for contour in contours:
    ax_img.plot(contour[:, 1], contour[:, 0], color='moccasin', linewidth=0.7, alpha=0.3)
ax_img.plot([x1_rad, x2_rad], [y1_rad, y2_rad], color='cyan', linewidth=1)
ax_img.plot([x1_tan, x2_tan], [y1_tan, y2_tan], color='chartreuse', linewidth=1)
ax_img.add_patch(Polygon(polygon_crop, closed=True, edgecolor='magenta',
                         facecolor='none', linewidth=0.8, linestyle="--"))
ax_img.scatter(*zip(*red_pts), facecolor='red', s=40, zorder=2)
if blue_pts:
    ax_img.scatter(*zip(*blue_pts), edgecolor='blue', facecolor='silver', s=30, linewidth=2)
if orange_pts:
    ax_img.scatter(*zip(*orange_pts), color='orangered', facecolor='silver', s=30, linewidth=2)

# --- Heatmap plot (right) ---
ax_hmap = fig.add_subplot(gs[0, 1])
norm = Normalize(vmin=cropped_array.min(), vmax=cropped_array.max())
sns.heatmap(
    cropped_array, annot=True, fmt=".2f", cmap='magma', cbar=True,
    norm=norm, linewidths=0.2, linecolor="white", square=True,
    annot_kws={"color": "white", "weight": "bold", "fontsize": 4},
    ax=ax_hmap, cbar_kws={"orientation": "vertical"}
)
for i, row in enumerate(cropped_array):
    max_j = np.argmax(row)
    for j, val in enumerate(row):
        idx = i * cropped_array.shape[1] + j
        text_obj = ax_hmap.texts[idx]
        if j != max_j:
            text_obj.set_visible(False)
        else:
            text_obj.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground='black'),
                path_effects.Normal()
            ])
ax_hmap.set_xticks([])
ax_hmap.set_yticks([])

# --- Shared legend (bottom row, spanning both columns) ---
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis("off")  # we use this just to place the legend

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Selected Cell',
           markerfacecolor='red', markersize=5),
    Line2D([0], [0], marker='o', color='blue', label='Up/Down Neighbors',
           markerfacecolor='silver', markersize=5, markeredgewidth=1.5),
    Line2D([0], [0], marker='o', color='orangered', label='Left/Right Neighbors',
           markerfacecolor='silver', markersize=5),
    Line2D([0], [0], color='cyan', lw=1, label='Radial Diameter'),
    Line2D([0], [0], color='chartreuse', lw=1, label='Tangential Diameter'),
    Line2D([0], [0], color='magenta', lw=1, linestyle='--', label='Scan Box'),
]

ax_legend.legend(
    handles=legend_elements,
    loc='center', ncol=3, fontsize=7, frameon=True,
    handletextpad=0.5, columnspacing=1.2
)

# Save
fig.savefig("C:/Users/sambo/Desktop/QWAnamiz_store/figures/cell_and_heatmap_combined.png", bbox_inches="tight", dpi=300)
plt.show()



###############################################################################
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle


# --- Dimensions ---
cm_to_inch = 1 / 2.54
total_width_cm = 14
fig_width_in = total_width_cm * cm_to_inch

img_width_cm = 9
hmap_width_cm = 5
img_width_frac = img_width_cm / total_width_cm
hmap_width_frac = hmap_width_cm / total_width_cm

# Heights in inches
img_height_in = img_width_cm * cm_to_inch * aspect_img
hmap_height_in = hmap_width_cm * cm_to_inch * aspect_hmap
legend_height_in = max(hmap_height_in - img_height_in, 0.6)

fig_height_in = max(hmap_height_in, img_height_in + legend_height_in)

# --- Create grid ---
fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
gs = gridspec.GridSpec(
    nrows=2, ncols=2,
    width_ratios=[img_width_frac, hmap_width_frac],
    height_ratios=[img_height_in, legend_height_in],
    hspace=0.02, wspace=0.01
)

cmap = plt.get_cmap('magma').copy()
cmap.set_under('white')

# === LEFT: Image ===
ax_img = fig.add_subplot(gs[0, 0])
ax_img.imshow(cropped, cmap=cmap, interpolation="nearest", vmin=0.0001, origin='lower')
ax_img.axis("off")

# Overlays and annotations (same as before)
ax_img.imshow(overlay)
ax_img.imshow(shadow)
for contour in contours:
    ax_img.plot(contour[:, 1], contour[:, 0], color='moccasin', linewidth=0.7, alpha=0.3)
ax_img.plot([x1_rad, x2_rad], [y1_rad, y2_rad], color='cyan', linewidth=1)
ax_img.plot([x1_tan, x2_tan], [y1_tan, y2_tan], color='chartreuse', linewidth=1)
ax_img.add_patch(Polygon(polygon_crop, closed=True, edgecolor='magenta',
                         facecolor='none', linewidth=0.8, linestyle="--"))
ax_img.scatter(*zip(*red_pts), facecolor='red', s=40, zorder=2)
if blue_pts:
    ax_img.scatter(*zip(*blue_pts), edgecolor='blue', facecolor='silver', s=30, linewidth=2)
if orange_pts:
    ax_img.scatter(*zip(*orange_pts), color='orangered', facecolor='silver', s=30, linewidth=2)

ax_img.add_artist(scalebar)

# === RIGHT: Heatmap (spans both rows) ===
ax_hmap = fig.add_subplot(gs[:, 1])  # ← spans both rows
norm = Normalize(vmin=cropped_array.min(), vmax=cropped_array.max())
sns.heatmap(
    cropped_array, annot=True, fmt=".2f", cmap='magma', cbar=True,
    norm=norm, linewidths=0.2, linecolor="white", square=True,
    annot_kws={"color": "white", "weight": "bold", "fontsize": 4},
    ax=ax_hmap, cbar_kws={"orientation": "vertical"}
)
for i, row in enumerate(cropped_array):
    max_j = np.argmax(row)
    for j, val in enumerate(row):
        idx = i * cropped_array.shape[1] + j
        text_obj = ax_hmap.texts[idx]
        if j != max_j:
            text_obj.set_visible(False)
        else:
            text_obj.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground='black'),
                path_effects.Normal()
            ])
ax_hmap.set_xticks([])
ax_hmap.set_yticks([])

# === Bottom left: Legend ===
ax_legend = fig.add_subplot(gs[1, 0])
ax_legend.axis("off")
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Selected Cell',
           markerfacecolor='red', markersize=5),
    Line2D([0], [0], marker='o', color='blue', label='Up/Down Neighbors',
           markerfacecolor='silver', markersize=5, markeredgewidth=1.5),
    Line2D([0], [0], marker='o', color='orangered', label='Left/Right Neighbors',
           markerfacecolor='silver', markersize=5),
    Line2D([0], [0], color='cyan', lw=1, label='Radial Diameter'),
    Line2D([0], [0], color='chartreuse', lw=1, label='Tangential Diameter'),
    Line2D([0], [0], color='magenta', lw=1, linestyle='--', label='Scan Box'),
]
ax_legend.legend(handles=legend_elements, loc='center', ncol=2, fontsize=7, frameon=True)

# Save or show
fig.savefig("C:/Users/sambo/Desktop/QWAnamiz_store/figures/combined_image_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

################################################################################
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle

# === Dimensions ===
cm_to_inch = 1 / 2.54
fig_width_cm = 14
img_width_cm = 9
hmap_width_cm = fig_width_cm - img_width_cm

img_width_frac = img_width_cm / fig_width_cm
hmap_width_frac = hmap_width_cm / fig_width_cm

# Heights (keep image height as reference)
fig_width_in = fig_width_cm * cm_to_inch
img_height_in = img_width_cm * cm_to_inch * aspect_img
fig_height_in = img_height_in  # no legend

# === Create figure and grid ===
fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[img_width_frac, hmap_width_frac], wspace=0.01)

cmap = plt.get_cmap('magma').copy()
cmap.set_under('white')

# === Left: Image ===
ax_img = fig.add_subplot(gs[0, 0])
ax_img.imshow(cropped, cmap=cmap, interpolation="nearest", vmin=0.0001, origin='lower')
ax_img.axis("off")

# Overlays
ax_img.imshow(overlay)
ax_img.imshow(shadow)
for contour in contours:
    ax_img.plot(contour[:, 1], contour[:, 0], color='moccasin', linewidth=0.7, alpha=0.3)
ax_img.plot([x1_rad, x2_rad], [y1_rad, y2_rad], color='cyan', linewidth=1)
ax_img.plot([x1_tan, x2_tan], [y1_tan, y2_tan], color='chartreuse', linewidth=1)
ax_img.add_patch(Polygon(polygon_crop, closed=True, edgecolor='magenta',
                         facecolor='none', linewidth=0.8, linestyle="--"))
ax_img.scatter(*zip(*red_pts), facecolor='red', s=40, zorder=2)
if blue_pts:
    ax_img.scatter(*zip(*blue_pts), edgecolor='blue', facecolor='silver', s=30, linewidth=2)
if orange_pts:
    ax_img.scatter(*zip(*orange_pts), color='orangered', facecolor='silver', s=30, linewidth=2)

scalebar = ScaleBar(pix_to_um, "um", length_fraction=0.2, location="lower left")
ax_img.add_artist(scalebar)

# === Right: Heatmap ===
ax_hmap = fig.add_subplot(gs[0, 1])
norm = Normalize(vmin=cropped_array.min()+ 0.001, vmax=cropped_array.max())
cmap.set_under('orange', alpha=0.3)

sns.heatmap(
    cropped_array, annot=True, fmt=".2f", cmap=cmap, cbar=True,
    norm=norm, linewidths=0.2, linecolor="white", square=True,
    annot_kws={"color": "white", "weight": "bold", "fontsize": 4},
    ax=ax_hmap, cbar_kws={"orientation": "vertical", 'shrink': 0.6, "label": "Distance to nearest lumen (µm)"}
)

# set the tick labelsize
cbar = ax_hmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=6)
cbar.ax.yaxis.label.set_size(8)

# Max annotation visibility
for i, row in enumerate(cropped_array):
    max_j = np.argmax(row)
    for j, val in enumerate(row):
        idx = i * cropped_array.shape[1] + j
        text_obj = ax_hmap.texts[idx]
        if j != max_j:
            text_obj.set_visible(False)
        else:
            text_obj.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground='black'),
                path_effects.Normal()
            ])
ax_hmap.set_xticks([])
ax_hmap.set_yticks([])



# Get heatmap axis limits (in data coordinates)
x_min, x_max = ax_hmap.get_xlim()
y_min, y_max = ax_hmap.get_ylim()

# Create a rectangle matching the heatmap boundary
rect = Rectangle(
    (x_min, y_min),
    x_max - x_min,
    y_max - y_min,
    fill=False,
    edgecolor='magenta',        # Match your scan zone color
    linewidth=1,
    linestyle='--'
)

ax_hmap.add_patch(rect)


# === Save ===
fig.savefig("C:/Users/sambo/Desktop/QWAnamiz_store/figures/combined_no_legend.png", dpi=300, bbox_inches="tight")
plt.show()


