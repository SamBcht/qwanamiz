# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 14:40:54 2025

@author: sambo
"""

import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.color
import skimage.metrics

from scipy.ndimage import distance_transform_edt
#import matplotlib
import matplotlib.pyplot as plt

import skimage.graph
import skimage.util
import qwanamiz
#from tools import histogram
#from mixture import density, vonmises_pdfit, mixture_pdfit, pdfit
#from typing import Tuple 

#from scipy.stats import vonmises
import datetime
#import ast
import napari
from collections import defaultdict

sampleID = "L20_F24-2M1-Sc4"
output_folder = "C:/Users/sambo/Desktop/QWAnamiz_store/qwanamiz_dev"


images = np.load(f"{output_folder}/{sampleID}_imgs.npz")

celldata = pd.read_csv(f"{output_folder}/{sampleID}_cells.csv")

adjacency = pd.read_csv(f"{output_folder}/{sampleID}_adjacency.csv")
adjacency.set_index(['label1', 'label2'], inplace=True)

prediction = images['bw_img']

expanded_labels = images['explabs']

pix_to_um = 0.55042690590734

# Launch Napari viewer
viewer = napari.Viewer()

viewer.add_image(prediction, name='Original B&W', scale = [pix_to_um, pix_to_um])

# Add the expanded labeled image
viewer.add_labels(expanded_labels, 
                  name = 'Cells', 
                  scale = [pix_to_um, pix_to_um])

