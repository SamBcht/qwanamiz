# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:14:31 2025

@author: sambo
"""

###############################################################################
# FIND REGION EXTREMITIES NEAR THE BORDERS OF THE IMAGE

all_border_cells, upper_region_sequence, lower_region_sequence, matched_up, matched_down, unjustified = rings_functions.get_border_cells(rightcells_df, 
                                                                                                                         cell_to_region, 
                                                                                                                         up_extremities,
                                                                                                                         down_extremities,
                                                                                                                         image_height = expanded_labels.shape[0], 
                                                                                                                         image_width = expanded_labels.shape[1], 
                                                                                                                         border_margin = 75, 
                                                                                                                         pix_to_um = pix_to_um)

# Create an empty mask the same shape as expanded_labels
border_mask = np.zeros_like(expanded_labels, dtype=bool)

# Mark upward border cells with 1, downward border cells with 2
border_mask[np.isin(expanded_labels, list(all_border_cells))] = True
#border_mask[np.isin(expanded_labels, list(downward_border_cells))] = True

# Display in Napari
viewer.add_image(border_mask, name="Border Up/Down Cells", scale=[pix_to_um, pix_to_um])


# Result
print("Upper border regions (left to right):", upper_region_sequence)
print("Lower border regions (left to right):", lower_region_sequence)
print("Matching upper regions :", matched_up)
print("Matching lower regions :", matched_down)
print("Unmatched :", unjustified)
# Intersection: regions that have both an upward and a downward border cell
regions_topdown = (set(upper_region_sequence) | set(matched_down)) & (set(lower_region_sequence) | set(matched_up))
print(f"{len(regions_topdown)} regions touch both the top and bottom borders.")
print("Valid regions :", regions_topdown)



candidates, cu, cl = find_merge_candidates_from_sequences(
    upper_region_sequence, lower_region_sequence, matched_up, matched_down
)

print("Corrected upper:", cu)
print("Corrected lower:", cl)
print("Merge candidates:", candidates)


y_positions, sequences = get_region_sequences(new_boundaries, n_lines=20, matched_up=matched_up, matched_down=matched_down)


aligned, regions = align_region_sequences(sequences, gap_value=None, upper_seq=cu, lower_seq=cl)

plot_alignment(aligned, regions, names=None)

x_positions_matrix = get_x_positions_per_line(
    aligned,  # including upper and lower sequences
    y_positions,        # only the original y_positions without borders
    new_boundaries,
    pix_to_um=pix_to_um
)


# regions in cell_to_region but not in regions_topdown
candidate_regions = set(cell_to_region.values()) - regions_topdown

###############################################################################

from skimage.measure import regionprops


# Example: new_boundaries is your labeled image
props = regionprops(new_boundaries, spacing=pix_to_um)

shapes_data = []
shapes_colors = []

for prop in props:
    label_id = prop.label
    #if label_id not in candidate_regions:
        #continue
    y0, x0 = prop.centroid
    orientation = prop.orientation
    major_len = prop.major_axis_length
    minor_len = prop.minor_axis_length

    # major axis
    x_major1 = x0 + np.cos(orientation) * 0.5 * minor_len
    y_major1 = y0 - np.sin(orientation) * 0.5 * minor_len
    x_major2 = x0 - np.cos(orientation) * 0.5 * minor_len
    y_major2 = y0 + np.sin(orientation) * 0.5 * minor_len
    shapes_data.append(np.array([[y_major1, x_major1], [y_major2, x_major2]]))
    shapes_colors.append('chartreuse')

    # minor axis
    x_minor1 = x0 + np.sin(orientation) * 0.5 * major_len
    y_minor1 = y0 + np.cos(orientation) * 0.5 * major_len
    x_minor2 = x0 - np.sin(orientation) * 0.5 * major_len
    y_minor2 = y0 - np.cos(orientation) * 0.5 * major_len
    shapes_data.append(np.array([[y_minor1, x_minor1], [y_minor2, x_minor2]]))
    shapes_colors.append('red')

viewer.add_shapes(shapes_data, shape_type='line', edge_color=shapes_colors, name='Ellipse Axes', edge_width=10
#                  , scale=[pix_to_um, pix_to_um]
                  )

from skimage.draw import ellipse

# create an empty mask the same size as your labeled image
ellipse_mask = np.zeros_like(new_boundaries, dtype=np.int32)

for prop in props:
    label_id = prop.label
    
#    if label_id not in candidate_regions:
#        continue

    # center and parameters
    r0, c0 = prop.centroid        # (row, col)
    r0, c0 = int(r0), int(c0)

    # regionprops returns full axis length; ellipse_perimeter expects semi-axes
    semi_r = int(prop.major_axis_length / 2)  # along row direction
    semi_c = int(prop.minor_axis_length / 2)  # along col direction
    orientation = prop.orientation            # radians

    # get perimeter coordinates
    rr, cc = ellipse(r0, c0,
                               semi_r, semi_c,
                               rotation=orientation)  # note the minus sign for image coords

    # clip to image bounds
    rr = np.clip(rr, 0, ellipse_mask.shape[0]-1)
    cc = np.clip(cc, 0, ellipse_mask.shape[1]-1)

    ellipse_mask[rr, cc] = label_id

# Add the mask as a new layer in napari
viewer.add_labels(ellipse_mask.astype(np.uint8), name='Ellipse Perimeter', scale=[pix_to_um, pix_to_um])


def classify_regions_by_axis(new_boundaries, pix_to_um=1.0):
    """
    Classify regions based on where their major axis crosses the image borders.
    Returns dict: label -> classification ("top_bottom", "top", "bottom",
                                           "left", "right", "middle")
    """
    props = regionprops(new_boundaries, spacing=pix_to_um)
    height, width = new_boundaries.shape
    classifications = {}
    top_points = []
    bottom_points = []

    for prop in props:
        label_id = prop.label
        y0, x0 = prop.centroid
        orientation = prop.orientation
        major_len = prop.major_axis_length

        # ✅ Major axis endpoints (half-length each side)
        x1 = x0 + np.sin(orientation) * 0.5 * major_len
        y1 = y0 + np.cos(orientation) * 0.5 * major_len
        x2 = x0 - np.sin(orientation) * 0.5 * major_len
        y2 = y0 - np.cos(orientation) * 0.5 * major_len

        # Check border crossings
        crosses_top = (y1 <= 0) or (y2 <= 0)
        crosses_bottom = (y1 >= height*pix_to_um) or (y2 >= height*pix_to_um)
        crosses_left = (x1 <= 0) or (x2 <= 0)
        crosses_right = (x1 >= width*pix_to_um) or (x2 >= width*pix_to_um)

        if crosses_top and crosses_bottom:
            cls = "top_bottom"
        elif crosses_top and crosses_left:
            cls = "top_left"        
        elif crosses_top and crosses_right:
            cls = "top_right"
        elif crosses_top:
            cls = "top"
        elif crosses_bottom and crosses_left:
            cls = "bottom_left"
        elif crosses_bottom and crosses_right:
            cls = "bottom_right"
        elif crosses_bottom:
            cls = "bottom"
        elif crosses_left:
            cls = "left"
        elif crosses_right:
            cls = "right"
        else:
            cls = "middle"

        classifications[label_id] = cls
        
        if "top" in cls:
            # use the topmost endpoint (smallest y)
            if y1 < y2:
                top_points.append((label_id, x1, y1))
            else:
                top_points.append((label_id, x2, y2))
                
        if "bottom" in cls:
            # use the bottommost endpoint (largest y)
            if y1 > y2:
                bottom_points.append((label_id, x1, y1))
            else:
                bottom_points.append((label_id, x2, y2))

    # --- order sequences ---
    top_sorted = sorted(top_points, key=lambda t: t[1])       # sort by x
    bottom_sorted = sorted(bottom_points, key=lambda t: t[1]) # sort by x

    sequences = {
        "top": [lab for lab, _, _ in top_sorted],
        "bottom": [lab for lab, _, _ in bottom_sorted],
    }

    return classifications, props, sequences

region_classes, ring_regions, seq = classify_regions_by_axis(new_boundaries, pix_to_um)

for label, cls in region_classes.items():
    print(label, ":", cls)

print("Top sequence:", seq["top"])
print("Bottom sequence:", seq["bottom"])

# Identifying true ring boundaries from the upper and lower sequences
ring_lines = rings_functions.find_ring_lines(rightcells_df, region_to_cells, seq["top"], seq["bottom"])

lines = []

# Make sure celldata is indexed by label
cells_indexed = celldata.set_index("label")

# Prepare the lines
for region, labels in ring_lines.items():
    if not labels:  # skip empty lists
        continue
    region_cells = cells_indexed.loc[labels]
    coords = list(zip(region_cells["centroid-0"], region_cells["centroid-1"]))
    lines.append(coords)

# Add lines as shapes to the viewer
viewer.add_shapes(
    lines,
    shape_type="path",
    edge_color="red",   # or try "region" for per-region colors
    edge_width=2,
    name="Ring boundaries"
)

y_positions, sequences = get_region_sequences(new_boundaries, n_lines=20)


aligned, regions = align_region_sequences(sequences, gap_value=None, upper_seq=seq["top"], lower_seq=seq["bottom"])

plot_alignment(aligned, regions, names=None)

candidates, cu, cl = find_merge_candidates_from_sequences(
    seq["top"], seq["bottom"]
)

print("Corrected upper:", cu)
print("Corrected lower:", cl)
print("Merge candidates:", candidates)

cleaned_matrix = remove_singleton_columns(aligned)

plot_alignment(cleaned_matrix, regions, names=None)

filled = fill_columns(cleaned_matrix, candidates, 0.79)

plot_alignment(filled, regions, names=None)

merged_matrix = merge_region_columns(cleaned_matrix, candidates)

plot_alignment(merged_matrix, regions, names=None)


################################################################################
import itertools

def line_intersection(p1, d1, p2, d2):
    """
    Solve for intersection of two lines defined by point+direction.
    Returns (x,y) or None if parallel.
    """
    # Solve:
    # p1 + t d1 = p2 + u d2
    A = np.array([[d1[0], -d2[0]],
                  [d1[1], -d2[1]]], dtype=float)
    b = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=float)

    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return None  # parallel lines

    t, u = np.linalg.solve(A, b)
    x = p1[0] + t*d1[0]
    y = p1[1] + t*d1[1]
    return (x, y)

def find_intersections(props, candidate_regions, image_shape):
    """
    props: regionprops list
    candidate_regions: set of region IDs we care about
    image_shape: (height, width)
    """
    # Build line definitions
    lines = {}
    for prop in props:
        if prop.label not in candidate_regions:
            continue
        x0, y0 = prop.centroid[1], prop.centroid[0]  # (col,row)
        theta = prop.orientation
        dx = np.cos(theta)
        dy = -np.sin(theta)  # minus due to row/col
        lines[prop.label] = ((x0,y0), (dx,dy))

    h, w = image_shape
    valid_pairs = []
    intersection_points = []

    for (label1, line1), (label2, line2) in itertools.combinations(lines.items(), 2):
        p1, d1 = line1
        p2, d2 = line2
        inter = line_intersection(p1, d1, p2, d2)
        if inter is None:
            continue
        x, y = inter
        # Check inside image
        if 0 <= x < w and 0 <= y < h:
            valid_pairs.append((label1, label2))
            intersection_points.append((y,x))  # (row,col) for napari

    return valid_pairs, intersection_points

valid_pairs, intersection_points = find_intersections(props, candidate_regions, new_boundaries.shape)

# Add intersection points as a points layer
viewer.add_points(intersection_points, name='Line Intersections', size=30, face_color='red')

print("Region pairs whose lines intersect:", valid_pairs)

################################################################################
region_matrix_sorted, x_matrix_sorted = reorder_columns_by_x(aligned, x_positions_matrix)

plot_alignment(region_matrix_sorted, regions, names=None)

plot_alignment_with_xpos(region_matrix_sorted, x_matrix_sorted, regions)

merged_by_x = find_merge_by_x(region_matrix_sorted, x_matrix_sorted, x_threshold=100)

print(merged_by_x)



match_results = find_matches_by_polynomial(region_to_cells, celldata,
                                           degrees=(1,2,3,4),
                                           length=250,
                                           half_angle_deg=15)


for degree in range(1, 5):
    print(f"Degree {degree} mutual matches:")
    print(match_results[degree]['mutual_matches'])
    print(f"Degree {degree} one-sided matches:")
    print(match_results[degree]['one_sided_matches'])
    print("-" * 40)
#print(match_results)

reliable, mutual, one_sided = find_reliable_pairs(candidates, merged_by_x, match_results, min_count=2)
print(reliable)  # list of (regionA, regionB) appearing ≥2 times

filled = fill_columns(region_matrix_sorted, reliable, 0.7)

plot_alignment(filled, regions, names=None)

cleaned_matrix = remove_singleton_columns(filled)

plot_alignment(cleaned_matrix, regions, names=None)

merged_matrix = merge_region_columns(cleaned_matrix, reliable)

plot_alignment(merged_matrix, regions, names=None)
