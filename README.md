# qwanamiz

Quantitative Wood Anatomy Measurements and Visualization

## Overview

`qwanamiz` is both a set of command-line utilities and a Python library for
performing quantitative wood anatomy (QWA) analyses on high-resolution
microscopic images of conifer wood cells. The following command-line tools are
available for performing streamlined state-of-the art QWA analyses:

* `qwanaflow` performs cell labelization, measurement of cell properties, and
  identification of radial files through the use of a regional adjacency graph.
* `qwanarings` identifies growth ring boundaries, assigns cells to growth
  rings, and performs various measurements at growth-ring scale, including
  total ring width and the width of earlywood and latewood.
* `qwanaviz` provides a user-friendly and comprehensive viewer based on
  [`napari`](https://napari.org/stable/) for visualizing and troubleshooting
  the outputs produced by `qwanamiz` utilities.

These command-line utilities provide arguments to control critical aspects of
the analysis such that `qwanaflow`, `qwanarings` and `qwanaviz` should be
sufficient for most analytical needs. However, the individual Python functions
that are used by those tools can also be used separately as a library to
develop other tools or analysis workflows tailored to particular needs. If this
fits your use case, please see the section on detailed documentation.

## Installation

The latest developement version of `qwanamiz` can be downloaded from GitHub
using the following command:

```bash
git clone https://github.com/SamBcht/qwanamiz.git
```

We recommend installing `qwanamiz` within a virtual environment so that the
proper versions of dependencies are installed. Once the virtual environment is
activated, move to the `qwanamiz` directory and install it with `pip`:

```bash
# Move into qwanamiz directory
cd qwanamiz

# Install with pip
pip install .
```

To run the examples of the **Quick start** section using the test image shipped
with `qwanamiz`, first move to the test data directory:

```bash
cd tests/data
```

## Quick start

### Image preprocessing

We recommend using [RoxasAI](https://github.com/marckatzenmaier/TowardsRoxasAI)
by Katzenmeier et al. (2023) to preprocess wood anatomy images prior to running
the `qwanamiz` workflow. This tool outputs binarized .png files which
differentiate cell lumens from other features in the image, with cell walls in
black and cell lumens in white. However, any .png file which encodes binarized
wood anatomy images this way is suitable for input to qwanamiz.

### `qwanaflow`

The `qwanaflow` command-line utility measures wood anatomical traits and
assigns cells to radial files.

`qwanaflow` can be launched from the command line as follows:

```bash
qwanaflow test_image.png output
```

where the two mandatory positional arguments are an image to analyze and the
name of the directory to use for output. By default, `qwanaflow` will write the
results of the analysis in a directory with the name of the input file followed
by `_outputs` in the output directory.

### `qwanarings`

The output of `qwanaflow` can be further processed to identify tree-ring
boundaries from the cells and adjacency graph using the `qwanarings`
command-line tool:

```bash
qwanarings --input_dir output
```

where the mandatory `--input_dir` argument is the name of the directory where
outputs were written by `qwanarings`. By default, `qwanarings` will
automatically identify all images whose output was written to that directory
and process them.

### `qwanaviz`

The results produced by `qwanaflow` and `qwanarings` can be visualized using
`qwanaviz`, which uses the [`napari`](https://napari.org/stable/) module for
interactive visualization.  This script can be launched from the command line
using the prefix of the sample to visualize as input:

```bash
qwanaviz output/test_image_outputs/test_image
```

The best way to explain what this script does is to actually try it! Feel free
to explore the different layers displayed, remove or add them back, and zoom in
on the features. By hovering over the image, you will be able to see the values
or IDs of the feature that is currently selected as well as the coordinates in
the measurement unit specified when running `qwanaflow` and `qwanarings`.

Note that you do not need to run `qwanarings` before running `qwanaviz`, as the
`qwanarings`-related outputs are optional.

## Output files

### `qwanaflow` outputs

`qwanaflow` (the QWA measurement tool) produces output files with the following
suffixes in a folder named as `{input.filename}_outputs`:

* `_cells.csv`: DataFrame of individual cell measurements (this is the main
  output of `qwanaflow`).
* `_filtered.csv`: DataFrame of cells that were filtered out because they do
  not belong to any radial file. These filtered cells are probably noisy
  artefacts from the original image.
* `_adjacency.csv`: DataFrame containing pairs of adjacent cells and related
  measurements
* `_imgs.npz`: image arrays in compressed `numpy` format with the
  following keys:
    * `bw_img`: the original black and white image used as input to the program.
    * `dmap`: the distance map from each cell wall pixel to the nearest cell
      lumen.
    * `explabs`: the cell ID attributed to each pixel, both lumen and wall.
    * `labs`: the cell ID attributed to each pixel when only lumens are
      considered.
    * `watershed`: an image indicating which cells were processed by the
      watershed algorithm used to split incorrectly merged cells.
* `_angles.png`: Graphical summary of directionality analysis showing von
  Mises distributions fitted for each sub-image.
* `_params.csv`: a DataFrame with the parameters of the von Mises distributions
  used for determining the adjacency angles; useful for debugging issues with
  directionality.

### `qwanarings` outputs

`qwanarings` (the growth ring detection program) produces output files with the
following suffixes inside the processed folder:

* `_rings.csv`: DataFrame containing global ring measurements and data.
* `_ringcells.csv`: DataFrame of individual cell measurements completed with
  ring attribution and ring-level measurements.
* `_ring_imgs.npz`: processed image rasters in compressed `numpy` format with
  the following keys:
    * `new_boundaries`: an image identifying the cells detected as the first
      earlywood cells of each ring, used in detecting growth-ring boundaries.
    * `year_image`: an image of the year attributed to each pixel in the image
      (both cell lumens and walls)
* `_rings.pkl`: a serialized `pickle` object containing the IDs of the cells
  defining ring boundaries.
* `_polygons.pkl`: a serialized `pickle` object, which contains a list of 2D
  `numpy` arrays with the coordinates of the polygons used for defining growth
  rings.
* `_img.png`: an image summarizing the results of the growth-ring analysis and
  the radial files identified as valid.

## Detailed documentation

### Example vignettes

Users who want to better understand the detailed workflow of `qwanaflow` and
`qwanarings` should have a look at the detailed example vignettes under the
`docs` folder:

* `qwanaflow_example.ipynb`
* `qwanarings_example.ipynb`

These can be run as interactive Jupyter notebooks or compiled by navigating to
the `docs` folder and running the command `make html`. Either way, you first need
to install a few dependencies with `pip`.

```bash
# Installing development dependencies with pip
pip install jupyter myst-nb sphinx-autoapi sphinx-rtd-theme

# Compiling the documentation
make html
```

After compiling the documents, you should be able to access them under
`docs/_build/html/index.html`.

Work is under way to host these example vignettes on the web so users will no
longer need to compile them themselves.

Work is also under way to provide comprehensive documentation for all the
library functions used in `qwanamiz` so they can be adapted for other purposes.

### `qwanaflow` command-line arguments

Running `qwanaflow --help` will provide a list of command-line arguments that
can be used to control the cell measurement workflow:

```plaintext
usage: qwanaflow [-h] [--pixel-size PIXEL] [--dir-nrows NROWS]
                 [--dir-ncols NCOLS] [--disable-plots]
                 [--area-threshold AREA_THRESHOLD]
                 [--solidity-threshold SOLIDITY_THRESHOLD]
                 [--max-wall-distance MAX_WALL_DISTANCE]
                 [--vm-threshold VMTHRESHOLD] [--angle-tolerance ANGLE]
                 [--stitch-angle-tolerance STITCH_ANGLE]
                 [--scan-width SCAN_WIDTH] [--ncores NCORES]
                 input output

positional arguments:
  input                 If a single directory, all png files in that directory
                        will be processed. If a single .png file, process only
                        that file. If a single .txt file, should be a file
                        containing a list of files to process, with one .png
                        file per line.
  output                A directory to write output files to.

options:
  -h, --help            show this help message and exit
  --pixel-size PIXEL    Conversion factor from of a single pixel to the
                        desired measurement unit. Defaults to 1 (measurements
                        in pixels).
  --dir-nrows, -r NROWS
                        Number of rows to split the image into for the
                        directionality analysis. If None (the default), the
                        number of rows and colums is automatically determined.
  --dir-ncols, -c NCOLS
                        Number of columns to split the image into for the
                        directionality analysis. If None (the default), the
                        number of rows and colums is automatically determined.
  --disable-plots       Specify this flag to disable the generation of angle
                        plots. By default they will be produced.
  --area-threshold AREA_THRESHOLD
                        Lumen area above which a cell can be considered for
                        splitting into several cells using the watershed
                        segmentation algorithm. For a cell to be selected, it
                        must also be below the solidity threshold. Defaults to
                        500.
  --solidity-threshold SOLIDITY_THRESHOLD
                        Solidity threshold below which a cell can be
                        considered for splitting into several cells using the
                        watershed segmentation algorithm. Higher values
                        (closer to 1) indicate a more convex shape whereas
                        lower values (closer to 0) indicate concavity
                        (presence of indentations). For a cell to be selected,
                        it must also be above the lumen area threshold.
                        Defaults to 0.95.
  --max-wall-distance MAX_WALL_DISTANCE
                        The maximum distance (in the target measurement unit
                        defined by --pixel-size) between a cell wall pixel and
                        the nearest cell lumen for that pixel to be considered
                        as belonging to the cell. This parameter effectively
                        puts a cap on the maximum possible cell wall
                        thickness. Defaults to 10.
  --vm-threshold VMTHRESHOLD
                        The convergence threshold in the search of von Mises
                        distribution parameters. Lower values result in more
                        precise results but slower convergence. Defaults to
                        0.001.
  --angle-tolerance ANGLE
                        The tolerance (in degrees) around the lower and upper
                        bounds found by the directionality algorithm in
                        determining which cell adjacencies are radial and
                        which are tangential. A higher value means potentially
                        longer, but inexact, radial files. Defaults to 5.
  --stitch-angle-tolerance STITCH_ANGLE
                        The tolerance (in degrees) around the lower and upper
                        bounds found by the directionality algorithm in
                        determining which cell adjacencies are radial and
                        which are tangential. This angle is applied after the
                        initial radial file assignment in stitching together
                        radial files and should therefore use a more
                        permissive angle threshold. Defaults to 20.
  --scan-width SCAN_WIDTH
                        The width (in pixels) of the rectangle to use when
                        computing wall thickness at the boundary between two
                        cells. If None (the default), qwanaflow dynamically
                        computes the scan width for each cell pair to 75% of
                        the average of the two cells' diameter. Explicitly
                        setting the scan width provides faster computation
                        (especially for lower values) but is potentially less
                        accurate.
  --ncores NCORES       The number of processes to launch for multiprocessing
                        during some cell measurement steps. Defaults to 1 (no
                        multiprocessing).
```

### `qwanarings` command-line arguments

Running `qwanarings --help` will provide a list of command-line arguments that
can be used to control the growth-ring detection workflow:

```plaintext
usage: qwanarings [-h] --input_dir INPUT_DIR [--pixel-size PIXEL]
                  [--minimum-cells MINCELLS] [--first-year FIRSTYEAR]

options:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to the main directory containing subfolders for
                        each processed image. Suffixes '_imgs.npz',
                        '_cells.csv' and '_adjacency.csv' must be in the
                        subfolders to obtain the input files.
  --pixel-size PIXEL    Size of a pixel in the wanted measurement unit.
                        Defaults to 0.55042690590734 micrometers.
  --minimum-cells MINCELLS
                        The minimum number of cells in a ring-boundary region
                        to consider it. Defaults to 4.
  --first-year FIRSTYEAR
                        The calendar year when the first ring was formed, used
                        for assigning cells to years. Defaults to 1 (year
                        unknown).
```

### `qwanaviz` command-line arguments

Running `qwanaviz --help` will provide a list of command-line arguments that
can be used to launch the visualization script:

```plaintext
usage: qwanaviz [-h] [--pixel-size PIXEL] prefix

positional arguments:
  prefix              The prefix of the sample to use qwanamiz.py with.
                      qwanamiz will look for file paths corresponding to
                      'prefix + _imgs.npz', 'prefix + _cells.csv', 'prefix +
                      _ring_imgs.npz', 'prefix + _rings.pkl', and 'prefix +
                      _polygons.pkl' These files should all be output by
                      qwanaflow or qwanarings. Files output by the qwanarings
                      utility are optional and will not be drawn if only
                      qwanaflow has been run.

options:
  -h, --help          show this help message and exit
  --pixel-size PIXEL  Size of a pixel in the wanted measurement unit. Defaults
                      to 0.55042690590734 micrometers.
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note
that this project is released with a Code of Conduct. By contributing to this
project, you agree to abide by its terms.

## License

`qwanamiz` is licensed under the terms of the GNU General Public License v3.0
license.

## Credits

`qwanamiz` was developed by the Canadian Research Chair in dendroecology and
dendroclimatology by Samuel Bouchut, Marc-André Lemay, and Fabio Gennaretti.

The Canadian Research Chair in dendroecology and dendroclimatology has been
based at the Groupe de Recherche en Écologie de la MRC Abitibi, Institut de
Recherche sur les Forêts, Université du Québec en Abitibi-Témiscamingue, Amos,
Québec, Canada. The mission of the Chair will end in 2026.

Please refer to the researchers' current affiliation. Fabio Gennaretti is now
affiliated at the Department of Agricultural, Food and Environmental Sciences,
Università Politecnica delle Marche, Area Sistemi Forestali, Via Brecce Bianche
10, 60131, Ancona, Italy.

`qwanamiz` was created with
[`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the
`py-pkgs-cookiecutter`
[template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

Functions used to fit Von Mises distributions have been implemented from
[François Konschelle, Mixture of von Mises distributions](https://framagit.org/fraschelle/mixture-of-von-mises-distributions)

Katzenmaier, M., Garnot, V.S.F., Björklund, J., Schneider, L., Wegner, J.D. and
von Arx, G., 2023. Towards ROXAS AI: Deep learning for faster and more accurate
conifer cell analysis. Dendrochronologia, 81, p.126126.
[doi:10.1016/j.dendro.2023.126126](https://doi.org/10.1016/j.dendro.2023.126126)

