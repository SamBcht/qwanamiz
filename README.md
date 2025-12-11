# qwanamiz

Quantitative Wood Anatomy Measurements and Visualization

## Overview

`qwanamiz` is both a set of command-line utilities and a Python library for
performing quantitative wood anatomy (QWA) analyses on high-resolution
microscopic images of wood cells of conifers. The following command-line tools
are available for performing streamlined state-of-the art QWA analyses:

* `qwanaflow` performs cell labelization, measurement of cell properties, and
  identification of radial files through the use of a regional adjacency graph.
* `qwanarings` identifies growth ring boundaries, assigns cells to growth
  rings, and performs various measurements at growth-ring scale, including
  total ring width and the width of earlywood and latewood.
* `qwanaviz` provides a user-friendly and comprehensive viewer based on `napari`
  for visualizing and troubleshooting the outputs produced by `qwanamiz`
  utilities.

These command-line utilities provide arguments to control critical aspects of
the analysis such that `qwanaflow`, `qwanarings` and `qwanaviz` should be
sufficient for most analytical needs. However, the individual Python functions
that are used in those tools can also be used separately as a library to
develop other tools or analysis workflows tailored to particular needs. If this
fits your use case, please see the section on detailed documentation.

## Installation

The latest developement version of `qwanamiz` can be downloaded from GitHub
using the following command:

```bash
git clone https://github.com/SamBcht/qwanamiz.git
```

We recommend installing `qwanamiz` within a virtual environment so that the
proper versions of dependencies are installed. Once the virtual environment
is activated, move inside the `qwanamiz` directory and install it with `pip`:

```bash
# Move into qwanamiz directory
cd qwanamiz

# Install with pip
pip install .
```

To run the examples of the **Quick start** section using the test image shipped with `qwanamiz`,
first move to the test data directory:

```bash
cd tests/data
```

## Quick start

### Image preprocessing

We recommend using [RoxasAI](https://github.com/marckatzenmaier/TowardsRoxasAI)
by Katzenmeier et al. (2023) to preprocess wood anatomy images prior to running
the `qwanamiz` workflow. This tool outputs binarized .png files which
differentiate cell lumina from other features in the image. However, any .png
file which encodes binarized wood anatomy images this way is suitable for input
to qwanamiz.

### `qwanaflow`

The `qwanaflow` command-line utility measures wood anatomical traits and
assigns cells to radial files.

`qwanaflow` can be launched from the command line as follows:

```bash
python qwanaflow.py test_image.png output
```

where the two mandatory positional arguments are an image to analyze and the
name of the directory to use for output. By default, `qwanaflow` will write the
results of the analysis in a directory with the name of the input file followed
by `_outputs` in the output directory.

# `qwanarings`

The output of `qwanaflow.py` can be further processed to identify tree-ring boundaries
from the cells and adjacency graph using the `qwanarings.py` command-line tool:

## qwanaviz

The results produced by `qwanaflow.py` can be visualized using `qwanaviz.py`,
which requires the [napari](https://napari.org/stable/) module to be installed.
This script can be launched from the command line using the prefix of the sample
to visualize as input:




## Output files

TODO: A description of the files output by `qwanaflow.py` should be written here.

## Detailed documentation

### Example vignettes

### `qwanaflow` command-line arguments

### `qwanarings` command-line arguments

### `qwanaviz` command-line arguments

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note
that this project is released with a Code of Conduct. By contributing to this
project, you agree to abide by its terms.

## License

`qwanamiz` was created by Samuel Bouchut. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`qwanamiz` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

@article{KATZENMAIER2023126126,
    title = {Towards ROXAS AI: Deep learning for faster and more accurate conifer cell analysis},
    author = {Marc Katzenmaier and Vivien Sainte Fare Garnot and Jesper Björklund and Loïc Schneider and Jan Dirk Wegner and Georg {von Arx}}
    journal = {Dendrochronologia},
    year = {2023},
    doi = {https://doi.org/10.1016/j.dendro.2023.126126},
}
