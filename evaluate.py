import pygame
import random
import time
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
from imgApprox import ImgApprox, sizeX, sizeY, config, lerp_arr

reference_img = Image.open(config['reference_img'])
reference_img = np.asarray(reference_img.resize([sizeX, sizeY]))[:,:,:3].transpose([1,0,2]).astype(np.uint32)

def img_std_dev(arr):
    return (arr**2).mean() - arr.mean()**2

def diff_med(arr):
    return np.abs(2*arr - arr.mean()).mean()

def entropy(arr):
    arr_aux = (arr * 10)//10
    arr_uint = arr.astype(np.int32)
    colors, freq = np.unique(arr_uint.reshape([-1, 3]), axis=0, return_counts=True)
    amount = arr.shape[0] * arr.shape[1]
    freq_rel = freq/amount
    H = -(freq_rel*np.log2(freq_rel)).sum()
    return H

def fitness_color_entropy(img_approx):
    arr = img_approx.img_array()
    arr_hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
    return entropy(arr)/10

def fitness_img(img_approx):
    arr = img_approx.img_array()
    return np.sqrt( ((arr-reference_img)**2).sum(axis=2) ).sum()/(sizeX*sizeY*442)*100

def fitness(img_approx):
    if config['display']:
        img_approx.render()
    p = 0.01
    #return (1-p)*fitness_img(img_approx) + p*fitness_color_entropy(img_approx)
    return -fitness_color_entropy(img_approx)
    #return fitness_img(img_approx)
