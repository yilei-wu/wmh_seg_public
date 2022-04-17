# Author : Yilei Wu Email: yileiwu@outlook.com
# This is a test script for submission to MICCAI WMH Segmentation Challenge

import SimpleITK as sitk  
import numpy as np  
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob
from models.model import Model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True, help='input path')
parser.add_argument('-o', required=True, help='output path')
args = parser.parse_args()

thres_FLAIR = 70
model_path = dir_path = os.path.dirname(os.path.realpath(__file__)) + '/weights'
input_path = args.i
output_path = args.o

def adaptive_preprocessing(image):
    # take in one .nii.gz file and do the preprocess
    # center crop them into 160, 192, 160 and do the gaussian norm
    FLAIR_IMAGE = sitk.ReadImage(image, sitk.sitkFloat32)
    FLAIR_ARRAY = sitk.GetArrayFromImage(FLAIR_IMAGE)

    flair_image = np.repeat(FLAIR_ARRAY, 3, axis=0)

    #  center crop into the shape of 160, 192, 160
    x0, x1, x2 = np.shape(flair_image)
    x0_pad = 160 - x0; x1_pad = 192 - x1; x2_pad = 160 - x2
    x0_a  = x0_pad//2; x0_b = x0_pad//2 + x0_pad%2
    x1_a  = x1_pad//2; x1_b = x1_pad//2 + x1_pad%2
    x2_a  = x2_pad//2; x2_b = x2_pad//2 + x2_pad%2

    if x0_pad > 0:
        flair_image = np.pad(flair_image, ((x0_a, x0_b), (0, 0), (0, 0)), mode='constant')
    else:
        flair_image = flair_image[-x0_a:x0_b, ...]
   
    if x1_pad > 0:
        flair_image = np.pad(flair_image, ((0, 0), (x1_a, x1_b), (0, 0)))
    else:
        flair_image = flair_image[:,-x1_a:x1_b,:]

    if x2_pad > 0:
        flair_image = np.pad(flair_image, ((0, 0), (0, 0), (x2_a, x2_b)))
    else:
        flair_image = flair_image[..., -x2_a:x2_b]
        
    # do the gaussian normalization 
    brain_mask_FLAIR = np.empty((160, 192, 160)) 
    brain_mask_FLAIR[flair_image >= thres_FLAIR] = 1 
    brain_mask_FLAIR[flair_image < thres_FLAIR] = 0 
    for iii in range(np.shape(flair_image)[0]):
        brain_mask_FLAIR[iii, ...] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii, ...])

    flair_image -= np.mean(flair_image[brain_mask_FLAIR==1])
    flair_image /= np.std(flair_image[brain_mask_FLAIR==1])

    return flair_image.transpose(), (x0_a, x0_b, x1_a, x1_b, x2_a, x2_b) 

def adaptive_postprocessing(pred, paddings, output_path):

    pred = pred.transpose()
    x0_a, x0_b, x1_a, x1_b, x2_a, x2_b = paddings

    if x0_a > 0:
        pred = pred[x0_a:-x0_b,...]
    else:
        pred = np.pad(pred, ((-x0_a, -x0_b), (0,0), (0,0)))

    if x1_a > 0:
        pred = pred[:, x1_a:-x1_b, :]
    else:
        pred = np.pad(pred, ((0,0), (-x1_a, -x1_b), (0,0)))

    if x2_a > 0:
        pred = pred[..., x2_a:-x2_b]
    else:
        pred = np.pad(pred, ((0,0), (0,0), (-x2_a, -x2_b)))

    pred_orig_shape = np.empty((np.shape(pred)[0]//3, np.shape(pred)[1], np.shape(pred)[2]))

    for iii in range(np.shape(pred_orig_shape)[0]):
        p0 = pred[iii*3, ...]
        p1 = pred[iii*3 + 1, ...]
        p2 = pred[iii*3 + 2, ...]

        temp_p = (p0 + p1 + p2)/3
        pred_orig_shape[iii, ...] = temp_p

    filename_resultImage = os.path.join(output_path, 'result.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(pred_orig_shape), filename_resultImage)
    
    return pred_orig_shape


if __name__=='__main__':
    FLAIR_image, paddings = adaptive_preprocessing(input_path)
    FLAIR_tensor = torch.unsqueeze(torch.from_numpy(np.expand_dims(FLAIR_image, 0)).float(), 0)
    preds = []

    for each_model in glob.glob(model_path + '/model_*.pth'):
        # construct the model and load the weight
        temp_model = Model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        temp_model.load_state_dict(torch.load(each_model, map_location=device))
        temp_model.to(device)

        # do the inference
        with torch.no_grad():
            if torch.cuda.is_available():
                FLAIR_tensor = FLAIR_tensor.cuda()
            temp_pred = temp_model(FLAIR_tensor)[-1]
            temp_pred = temp_pred.cpu().numpy()

            preds.append(temp_pred)
        
        # for debug, only use one model
        # break

    # get the average of ensemble pred, threshold of 0.1 is applied
    pred_ensemble = np.average(preds, axis=0)
    pred_ensemble[pred_ensemble>0.1]=1
    pred_ensemble[pred_ensemble<0.1]=0
    pred_ensemble = pred_ensemble[0, 0, ...]

    adaptive_postprocessing(pred_ensemble, paddings, output_path)
    print("Complete...")