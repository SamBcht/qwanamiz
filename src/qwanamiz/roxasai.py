# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:51:08 2024

@author: sambo
"""

import numpy as np
import os, glob, cv2, torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import skimage.measure as skm
from tqdm import tqdm

class SegModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = smp.Unet(encoder_weights=None, classes=2, encoder_name='resnet50')

    def forward(self, x):
        out = self.net(x)
        out = torch.argmax(out, dim=1, keepdim=True).long()
        return out

    def predict(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)  # todo move to function needed more often best would be to have it in the model as predict with the path
        preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')
        img = transforms.ToTensor()(preprocessing_fn(img)).float()
        img = torch.unsqueeze(img, 0)
        r = self(img)
        processed = torch.argmax(torch.squeeze(r.detach()), dim=0).long().numpy()  # process to masks
        labels = skm.label(processed).transpose((1, 0)) # todo why strange transpose
        return labels

def run_model(img_path, model):
    """
    Runs the model on patches of the image and concats the predictions together
    :param img_path: path to the image
    :param model: segmentation model with the forward path specified
    :return: numpy array of predictions with values 0 and 255

    """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_pad = np.zeros((img.shape[0]+224,img.shape[1]+224, img.shape[2]))
    label_pad = np.zeros((img.shape[0]+224,img.shape[1]+224))
    img_pad[112:-112,112:-112] = img
    imgs = []
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')
    for i in range(img_pad.shape[0]//800 + 1): # split image into 1024*1024 chunks
        for j in range(img_pad.shape[1] // 800 + 1):
            x,x1,y,y1 = 800*i, 800*i+1024, 800*j, 800*j+1024
            to_much_x = x1 - img_pad.shape[0]
            to_much_y = y1 - img_pad.shape[1]
            if to_much_x > 0:
                x = img_pad.shape[0] - 1024
                x1 = img_pad.shape[0]
            if to_much_y > 0:
                y = img_pad.shape[1] - 1024
                y1 = img_pad.shape[1]
            input_img = img_pad[x:x1,y:y1].copy()
            img1 = preprocessing_fn(input_img)
            img1 = transforms.ToTensor()(img1).float()
            img1 = torch.unsqueeze(img1, 0)
            r = model.forward(img1)
            processed = torch.squeeze(r.detach()).long().numpy()  # process to masks
            labels = skm.label(processed).transpose((1, 0))  # todo why strange transpose
            result_semseg = (labels.T > 0).astype(int)*255
            label_pad[x+112:x1-112,y+112:y1-112]=result_semseg[112:-112,112:-112]
    return label_pad[112:-112,112:-112]

def get_basename(input_file, remove = '_ring.TIF'):
    base_name = os.path.basename(input_file)
    base_name = base_name.replace(remove, '')
    return base_name


if __name__ == '__main__':
    #import os
    import datetime

    # Set paths
    input_folder = 'C:/Users/sambo/Desktop/QWAnamiz_store/input'
    model_path = 'C:/Users/sambo/Desktop/qwanamiz/qwanamiz/src/qwanamiz/final_semseg_coniferen_model.pth'
    output_folder = 'C:/Users/sambo/Desktop/QWAnamiz_store/input_measures'


    ###------------------------------------------- Prepare Model -------------------------------###
    print('load model')
    model = SegModel()
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    
    # Process each image in the input folder
    img_paths = glob.glob(os.path.join(input_folder, '*.tif'))
    
    
    ###------------------------------------------- Process files --------------------------------###
    
    for img_path in img_paths:
        
        start_save = datetime.datetime.now()
        print('run model')
        prediction = run_model(img_path, model)
        endTime = datetime.datetime.now()
        print(f'runtime : {endTime - start_save}')
        
        print('save output')
        base_name = get_basename(img_path, remove = '.tif')
        
        # Save the segmented image
        output_path = os.path.join(output_folder, f"{base_name}_segmented.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, prediction)
        
        # Save the array
        #output_path = os.path.join(output_folder, f"{base_name}_array.npy")
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #np.save(output_path, prediction)
        print(f"Saved segmentation output to {output_path}")
        endTime = datetime.datetime.now()
        print(f'runtime : {endTime - start_save}')
        

 