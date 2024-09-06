# qwanamiz

Quantitative Wood Anatomy Measurements

## Installation

Installlation has not been set up for now. Better download a zip copy of the repo.

```bash
$ pip install qwanamiz
```

## Usage

- TODO

Create an environment using the environment.yml file:

```bash
$ conda env create -f environment.yml
```
For now, I advice to use the IDE Spyder to process the scripts

### RoxasAI:

The "roxasai.py" file contains the script to run the TowardsRoxasAI segmentation model proposed in Katzenmaier et al. (2023).
It transform the original scanned image in a binary white and black image distinguishing cell lumens from cell walls. 

At the end of the "roxas.py" file, set the path of the folder containing the input image(s), the path of the folder that will contain the outputs and the path to the model file.
The input folder must contain images in ".tif" format.
The segmentation model file can be found at https://github.com/marckatzenmaier/TowardsRoxasAI
Run the entire file

### QWAnaflow:

The "qwanaflow.py" file contains a script that run sequentially all the steps to measure wood anatomical traits.
It's an entirely automated quantitative wood anatomy analysis.

`qwanaflow.py` is designed to be launched from the command line as follows

```bash
python qwanaflow.py <input> <output>
```

where `input` is either a directory containing .png files to be processed, a
single .png file, or a .txt file listing .png files to process. Only .png files
are supported at the moment, but support could be extended to other file types
in the future. The `output` parameter is a directory to which the output files
will be written.

You can see detailed qwanaflow usage by running the `qwanaflow.py --help` command:

```bash
python qwanaflow.py --help

usage: qwanaflow.py [-h] [--dir-nrows NROWS] [--dir-ncols NCOLS]
                    [--disable-plots] [--vm-threshold VMTHRESHOLD]
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


```

### QWAnaviz:
At the end of the "qwanaviz.py" file, set the path to:
    - the "imgs.npz" file of the image
    - the "cells.csv" file
    - the "adjacency.csv" file
Run the entire file

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

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
