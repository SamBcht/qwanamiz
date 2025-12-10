# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:04:56 2025

@author: sambo
"""

from matplotlib.patches import Rectangle
from matplotlib.projections import PolarAxes
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from vonmisesmix import histogram, density, vonmises_pdfit, mixture_pdfit, pdfit, vonmises_density

def create_radial_file_array(expanded_labels, regionprops_df):
    # Initialize array with zeros (same shape as expanded_labels)
    radial_file_array = np.zeros_like(expanded_labels, dtype=np.int32)

    # Drop rows without valid radial_file
    df_valid = regionprops_df.dropna(subset=["radial_file"])

    # Round or convert radial_file to integer if needed
    df_valid = df_valid.copy()
    df_valid["radial_file"] = df_valid["radial_file"].astype(int)

    # Create a mapping: label -> radial_file
    label_to_radial_file = dict(zip(df_valid["label"], df_valid["radial_file"]))

    # Vectorized relabeling
    max_label = expanded_labels.max()
    lut = np.zeros(max_label + 1, dtype=np.int32)  # Lookup table
    for label, rf in label_to_radial_file.items():
        lut[label] = rf

    # Apply lookup table to entire image
    radial_file_array = lut[expanded_labels]

    return radial_file_array

def classify_edges_tolerance(df, tolerance = 5):
    
    # Extracting some variables from the DataFrame for coding convenience
    angle = np.radians(df["angle"])
    lb = df["lower_bound"] - np.radians(tolerance)
    ub = df["upper_bound"] + np.radians(tolerance)
    lower = df["lower_bound"]
    upper = df["upper_bound"]

    # Initialize classification column as "radial" by default
    df["wall_classification"] = "radial"

    # Tangential: angle between lower_bound and upper_bound
    mask_tangential = (angle >= lower) & (angle <= upper)
    df.loc[mask_tangential, "wall_classification"] = "tangential"

    # Indoubt: between (lb and lower_bound) or (upper_bound and ub)
    mask_indoubt = ((angle >= lb) & (angle < lower)) | ((angle > upper) & (angle <= ub))
    df.loc[mask_indoubt, "wall_classification"] = "indoubt"


    # Using np.where to classify the edges in a vectorized way
    #df["wall_classification"] = np.where(np.logical_and(angle >= lb, angle <= ub), 'tangential', 'radial')

    return df

adjacency = classify_edges_tolerance(adjacency, tolerance = 20)


image = create_radial_file_array(expanded_labels, regionprops_df)

tan_color = 'orangered'
rad_color = 'blue'
mid_color = 'gold'

dpi=300
dpi_scale = dpi/300 # Figure first created at 300 dpi

cm_to_inch = 1 / 2.54
fig_width_in = 19 * cm_to_inch  # ~7.48 inches

row_range_um = (110, 400)
col_range_um = (6025, 6600)

# Convert micron coordinates to pixel indices
r0 = int(row_range_um[0] / pix_to_um)
r1 = int(row_range_um[1] / pix_to_um)
c0 = int(col_range_um[0] / pix_to_um)
c1 = int(col_range_um[1] / pix_to_um)

cropped = image[r0:r1, c0:c1]  # your cropped image array
height_px, width_px = cropped.shape[:2]
aspect_ratio = height_px / width_px

fig_height_in = fig_width_in * aspect_ratio

fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # remove margins

# Create alpha mask where both labels match
matching_mask = (labeled_image == expanded_labels) & (labeled_image > 0)
alpha_mask = np.where(matching_mask, 0.3, 1.0)  # more transparent where match
alpha_cropped = alpha_mask[r0:r1, c0:c1]

# colored = 'twilight_shifted'
color_map = 'twilight_shifted'
random_seed = 6

num_labels = cropped.max()

# Get colormap
base_cmap = plt.cm.get_cmap(color_map, num_labels + 1)
colors = base_cmap(np.linspace(0, 1, num_labels + 1))
np.random.seed(random_seed)
shuffled = colors[1:]
np.random.shuffle(shuffled)
shuffled = np.vstack([[0, 0, 0, 1], shuffled])  # Black for background
cmap = ListedColormap(shuffled)

display_image = cropped
norm = plt.Normalize(vmin=display_image.min(), vmax=display_image.max())
rgba_image = cmap(norm(display_image))
rgba_image[..., -1] = alpha_cropped  # Set alpha channel
display_image = rgba_image

ax.imshow(display_image, interpolation="nearest")
ax.axis("off")

r0_px = int(row_range_um[0] / pix_to_um)
r1_px = int(row_range_um[1] / pix_to_um)
c0_px = int(col_range_um[0] / pix_to_um)
c1_px = int(col_range_um[1] / pix_to_um)

in_crop = (
    (regionprops_df["centroid-0"] >= row_range_um[0]) & 
    (regionprops_df["centroid-0"] < row_range_um[1]) &
    (regionprops_df["centroid-1"] >= col_range_um[0]) &
    (regionprops_df["centroid-1"] < col_range_um[1])
)


# PLOT CLASSIFIED EDGES
labels_in_crop = regionprops_df.loc[in_crop, "label"].unique()

adjacency_filtered = adjacency[
    adjacency.index.get_level_values("label1").isin(labels_in_crop) &
    adjacency.index.get_level_values("label2").isin(labels_in_crop)
]


color_map = {
        'radial': rad_color,
        'radial_sel': rad_color,
        'tangential': tan_color,
        'indoubt': mid_color
    }

for _, row in adjacency_filtered.iterrows():
        y1_um, x1_um = row['centroid1']  # centroid1 = (row, col)
        y2_um, x2_um = row['centroid2']
        classification = row['wall_classification']

        # Convert to crop-local pixels
        y1 = (y1_um - row_range_um[0]) / pix_to_um
        x1 = (x1_um - col_range_um[0]) / pix_to_um
        y2 = (y2_um - row_range_um[0]) / pix_to_um
        x2 = (x2_um - col_range_um[0]) / pix_to_um

        # Draw line
        ax.plot([x1, x2], [y1, y2], color=color_map.get(classification, 'gray'), linewidth=1/dpi_scale)

# PLOT CENTROIDS       
centroids_um = regionprops_df.loc[in_crop, ["centroid-0", "centroid-1"]].copy()

centroids_um["row_in_crop"] = (centroids_um["centroid-0"] - row_range_um[0]) / pix_to_um
centroids_um["col_in_crop"] = (centroids_um["centroid-1"] - col_range_um[0]) / pix_to_um

ax.plot(
    centroids_um["col_in_crop"],
    centroids_um["row_in_crop"],
    'ro', markersize=2  # red dots
)

scalebar = ScaleBar(pix_to_um, "um", length_fraction=0.2, location="lower left")
ax.add_artist(scalebar)

# Define legend handles
legend_elements = [
    Line2D([0], [0], color=rad_color, lw=2/dpi_scale, label='Tangential Edges'),
    Line2D([0], [0], color=tan_color, lw=2/dpi_scale, label='Radial Edges'),
    Line2D([0], [0], color=mid_color, lw=2/dpi_scale, label='Connection Edges'),
    #Line2D([0], [0], marker='o', color=tan_color, markerfacecolor='red', markersize=5, label='Cell centroids'),
]

# Add legend below the main axis
fig.legend(
    handles=legend_elements,
    loc='lower right',
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.85, 0.01),
    fontsize=8/dpi_scale,
    fancybox=True,
    #shadow = True,
    framealpha = 0.8
    
)

###############################################################################
# 2. Create inset axis manually using fig.add_axes (for polar projection)
# Define inset box [left, bottom, width, height] in figure fraction
inset_position = [0.25, 0.55, 0.35, 0.35]  # adjust as needed
inset_ax = fig.add_axes(inset_position, projection='polar')

# -- Polar inset anchored to image
# Create a polar inset locked to the main image axis
#inset_ax = inset_axes(ax, width="30%", height="30%",loc='upper left',borderpad=2, axes_class=PolarAxes  # <-- This is key)

subsample_label = adjacency_filtered['subsample_index'].unique()
params = vm_parameters
tolerance = 20
# Extract parameters
p = params['1_5']
mu = p['mu']
lower_bound, upper_bound = p['bounds']

# Compute tolerance bounds
tol_lower_left = lower_bound - np.radians(tolerance)
tol_upper_right = upper_bound + np.radians(tolerance)

# Tangential zone (in orange)
theta_tangential = np.linspace(lower_bound, upper_bound, 100)
r = np.ones_like(theta_tangential)
inset_ax.fill_between(theta_tangential, 0, r,  fc=tan_color, ec=None, alpha=0.5, label='Tangential zone')

# Radial zones (in blue)
theta_left = np.linspace(-np.pi/2, tol_lower_left, 100)
theta_right = np.linspace(tol_upper_right, np.pi/2, 100)
inset_ax.fill_between(theta_left, 0, 1, fc=rad_color, ec=None, alpha=0.3, label='Radial zone')
inset_ax.fill_between(theta_right, 0, 1,  fc=rad_color, ec=None, alpha=0.3)

# Tolerance zones
theta_tol_left = np.linspace(tol_lower_left, lower_bound, 100)
theta_tol_right = np.linspace(upper_bound, tol_upper_right, 100)
inset_ax.fill_between(theta_tol_left, 0, 1,  fc=mid_color, ec=None, alpha=0.3, label='Tolerance zone')
inset_ax.fill_between(theta_tol_right, 0, 1,  fc=mid_color, ec=None, alpha=0.3)

# Arrow for mean direction
inset_ax.annotate('', xy=(mu, 1), xytext=(0, 0),
            arrowprops=dict(facecolor='red', edgecolor='red', width=0.2/dpi_scale, headwidth=2/dpi_scale))

# Format plot
inset_ax.set_theta_zero_location("E")   # 0° = East (horizontal right)
inset_ax.set_theta_direction(-1)         # Counter-clockwise
inset_ax.set_thetalim(-np.pi/2, np.pi/2)  # Limit to -90° to +90° (half-disk)
inset_ax.set_rticks([])                # Remove radial ticks
inset_ax.set_yticklabels([])           # Remove radial labels


# Angle labels
inset_ax.set_xticks(np.radians([-90, -45, 0, 45, 90]))
inset_ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°'],bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.8))
inset_ax.tick_params(axis='x', labelsize=6/dpi_scale)
#ax.set_title(f"Polar Directionality - Subsample {subsample_label}", y=1.05)
#inset_ax.legend(loc='center left', bbox_to_anchor=(0.5, -0.1), ncol=1, title = 'Edges classification')

# Add a semi-transparent rectangle behind the polar plot
bg = Rectangle(
    (0, 0), 1, 1, transform=inset_ax.transAxes,
    facecolor='white', alpha=0.6, zorder=-1
)
inset_ax.add_patch(bg)

#inset_ax.spines['polar'].set_visible(False)
inset_ax.grid(True, color='grey', alpha=0.3)

###############################################################################
insetdist_position = [0.11, 0.55, 0.27, 0.35]  # adjust as needed
insetdist_ax = fig.add_axes(insetdist_position)

#for spine in insetdist_ax.spines.values():
#    spine.set_linewidth(1/dpi_scale) 
#insetdist_ax.contour(X, Y, Z,colors='black', linewidths=1/dpi_scale)

# Parse subsample label like '3_5' -> row=3, col=5
subsample_label = subsample_label[0]
row, col = map(int, subsample_label.split("_"))
p = params[subsample_label]

x_histo = p['x_histo']
y_histo = p['y_histo']
mu = p['mu']
kappa = p['kappa']
m = p['vonmisses_params']
lower_bound, upper_bound = p['bounds']

# Plot empirical histogram
insetdist_ax.plot(x_histo, y_histo, label='Raw dist.', color='dimgrey', linewidth=0.5/dpi_scale)

# Plot von Mises mixture fit
f = np.zeros(len(x_histo))
for k in range(m.shape[1]):
    f += m[0, k] * density(x_histo, m[1, k], m[2, k])
insetdist_ax.plot(x_histo, f / np.sum(f), label='Fitted dist.', color='black', linewidth=0.8/dpi_scale)

# Vertical lines for bounds
insetdist_ax.axvspan(-np.pi/2, tol_lower_left, fc=rad_color, ec = None, alpha=0.3, label='Tan. Edges')
insetdist_ax.axvspan(tol_upper_right, np.pi/2, fc=rad_color, ec = None, alpha=0.3)
insetdist_ax.axvspan(tol_lower_left, lower_bound, fc=mid_color, ec = None, alpha=0.3, label='Conn. Edges')
insetdist_ax.axvspan(upper_bound, tol_upper_right, fc=mid_color, ec = None, alpha=0.3)
insetdist_ax.axvspan(lower_bound, upper_bound, fc=tan_color, ec = None, alpha=0.4, label='Rad. Edges')
insetdist_ax.axvline(lower_bound, color=tan_color, linestyle='--', linewidth=0.4/dpi_scale)
insetdist_ax.axvline(upper_bound, color=tan_color, linestyle='--', linewidth=0.4/dpi_scale)
#ax.text(np.radians(-70), max(y_histo) * 0.3, f'{np.degrees(lower_bound):.2f}°', color='orange', fontsize=8)
#insetdist_ax.text(np.radians(45), max(y_histo) * 0.7, f'99% CI : {np.degrees(lower_bound):.2f}°-{np.degrees(upper_bound):.2f}°', color='orange', fontsize=6)

# Annotations for mean direction and concentration
insetdist_ax.text(np.radians(41), max(y_histo) * 0.9, f'pdir={np.degrees(mu):.2f}°', color='red', fontsize=6/dpi_scale)
insetdist_ax.text(np.radians(41), max(y_histo) * 0.8, f'κ={kappa:.2f}', color='red', fontsize=6/dpi_scale)

# Axis formatting
insetdist_ax.set_xlim(-np.pi/2, np.pi/2)
insetdist_ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
insetdist_ax.set_xticklabels(['-90°', '-45°', '0°', '45°', ''], fontsize=6/dpi_scale,bbox = dict(boxstyle="round", ec='white', fc="white", alpha=0.8))

#insetdist_ax.set_yticklabels(['0.0', '', '0.5', ''], fontsize=6,bbox = dict(boxstyle="round", ec='white', fc="white", alpha=0.8))
# Get current tick locations (radial positions)
#yticks = insetdist_ax.get_yticks()

# Format labels as strings with 1 decimal place (or however you want)
#yticklabels = [f"{ytick:.2f}" for ytick in yticks]

# Apply formatted labels with appearance control
#for label, tick_val in zip(insetdist_ax.set_yticklabels(yticklabels, fontsize=6/dpi_scale), yticks):
#    label.set_bbox(dict(boxstyle="round", ec='white', fc="white", alpha=0.8))
insetdist_ax.yaxis.set_ticklabels([])
insetdist_ax.set_ylabel("Density", fontsize = 7/dpi_scale,bbox = dict(boxstyle="round", ec="grey", fc="white", alpha=0.8))
insetdist_ax.set_xlabel("Angle (degrees)", fontsize=7/dpi_scale,bbox = dict(boxstyle="round", ec="grey", fc="white", alpha=0.8))
#insetdist_ax.set_title(f"Distribution of angles between edges and horizontal axis", fontsize=8)
insetdist_ax.legend(fontsize=6/dpi_scale)



fig.savefig("C:/Users/sambo/Desktop/QWAnamiz_store/figures/rag_files_colors500.png", bbox_inches="tight", dpi=dpi)
#fig.savefig("C:/Users/sambo/Desktop/QWAnamiz_store/figures/rag_files_colors500.tif", bbox_inches="tight", dpi=dpi)
