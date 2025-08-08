# qwanamiz

Quantitative Wood Anatomy Measurements and Visualization

## Installation

At the moment, we recommend cloning the repository directly from GitHub:

```bash
git clone https://github.com/SamBcht/qwanamiz.git
```

The two executable files that are needed for analyses are located in `src/qwanamiz`:

- `qwanaflow.py` (cell labeling, measurements, and radial file identification)
- `qwanamiz.py` (visualization of the results produced by `qwanaflow.py`)

### Requirements

In addition to some standard Python modules, qwanamiz requires the following
modules to be available:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-image](https://scikit-image.org/)
- [matplotlib](https://matplotlib.org/)
- [napari](https://napari.org/stable/) (optional, for visualisation with `qwanaviz.py`)

We recommend users to install these modules in a virtual environment. We
provide the environment.yml file to facilitate the installation of dependencies
using conda:

```bash
conda env create -f environment.yml
```

## Image preprocessing

We recommend using [RoxasAI](https://github.com/marckatzenmaier/TowardsRoxasAI)
by Katzenmeier et al. (2023) to preprocess wood anatomy images prior to running
the qwanamiz workflow. This tool outputs binarized .png files which
differentiate cell lumina from other features in the image. However, any .png
file which encodes binarized wood anatomy images this way is suitable for input
to qwanamiz.

## QWAnaflow

The `qwanaflow.py` file contains a script that sequentially runs all the steps
to measure wood anatomical traits. It is an entirely automated quantitative
wood anatomy analysis.

`qwanaflow.py` is designed to be launched from the command line as follows:

```bash
python qwanaflow.py input output
```

The `input` parameter is either a directory containing .png files to be
processed, a single .png file, or a .txt file listing .png files to process.
Only .png files are supported at the moment, but support could be extended to
other file types in the future.

The `output` parameter is a directory to which the output files will be
written. `qwanaflow.py` will strip the input files from their .png extension
and use the resulting prefix to save output files in the `output` directory.

You can see detailed qwanaflow usage by running the `qwanaflow.py --help` command:

```
python qwanaflow.py --help

usage: qwanaflow.py [-h] [--pixel-size PIXEL] [--dir-nrows NROWS] [--dir-ncols NCOLS]
                    [--disable-plots] [--vm-threshold VMTHRESHOLD]
                    [--ncores NCORES]
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
  --pixel_size PIXEL    Size of a pixel in the wanted measurement unit. 
                        Defaults to 0.55042690590734 micrometers
  --dir-nrows NROWS, -r NROWS
                        Number of rows to split the image into for the
                        directionality analysis. Defaults to 4.
  --dir-ncols NCOLS, -c NCOLS
                        Number of columns to split the image into for the
                        directionality analysis. Defaults to 8.
  --disable-plots       Specify this flag to disable the generation of angle
                        plots. By default they will be produced.
  --vm-threshold VMTHRESHOLD
                        The convergence threshold in the search of von Mises
                        distribution parameters. Lower values result in more
                        precise results but slower convergence. Defaults to
                        0.001.
  --ncores NCORES       The number of processes to launch for multiprocessing
                        for computing wall thickness. Defaults to 1 (no
                        multiprocessing).
```

## Output files

TODO: A description of the files output by `qwanaflow.py` should be written here.

## QWAnaviz

The results produced by `qwanaflow.py` can be visualized using `qwanaviz.py`,
which requires the [napari](https://napari.org/stable/) module to be installed.
This script can be launched from the command line using the prefix of the sample
to visualize as input:

```
python qwanaviz.py --help

usage: qwanaviz.py [-h] prefix

positional arguments:
  prefix      The prefix of the sample to use qwanamiz.py with. qwanamiz will
              look for file paths corresponding to 'prefix + _imgs.npz',
              'prefix + _cells.csv', and 'prefix + _adjacency.csv'. These
              files should all be output by qwanaflow.py.

options:
  -h, --help  show this help message and exit
```

# QWAnarings

The output of `qwanaflow.py` can be further processed to identify tree-ring boundaries
from the cells and adjacency graph using the `qwanarings.py` command-line tool:

```
python qwanarings.py --help

usage: qwanarings.py [-h] [--prefix PREFIX] [--pixel-size PIXEL]

options:
  -h, --help          show this help message and exit
  --prefix PREFIX     The prefix of the files to use for the analysis. Suffixes '_imgs.npz', '_cells.csv'
                      and '_adjacency.csv' will be added to that prefix to obtain the input files.
  --pixel-size PIXEL  Size of a pixel in the wanted measurement unit. Defaults to 0.55042690590734 micrometers.
```

The results can be visualized using the `ringview.py` command-line tool:

```
python ringview.py --help

usage: ringview.py [-h] [--prefix PREFIX] [--pixel-size PIXEL]

options:
  -h, --help          show this help message and exit
  --prefix PREFIX     The prefix of the files to use for the analysis. Suffixes '_imgs.npz', '_cells.csv',
                      '_adjacency.csv', and '_ring_imgs.npz' will be added to that prefix to obtain the
                      input files.
  --pixel-size PIXEL  Size of a pixel in the wanted measurement unit. Defaults to 0.55042690590734 micrometers.
```

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
