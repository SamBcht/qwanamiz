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

At the end of the "qwanaflow.py" file, set the path of the folder containing the input and the path of the folder that will contain the outputs.
The input folder must contain a numpy array with a name ending by "_array.npy".
Run the entire file

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