import pygame
import random
import time
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
from imgApprox import *

reference_img1 = Image.open(config['reference_img1'])
reference_img1 = np.asarray(reference_img1.resize([sizeX, sizeY]))[:,:,:3].transpose([1,0,2]).astype(np.uint32)

reference_img2 = Image.open(config['reference_img2'])
reference_img2 = np.asarray(reference_img2.resize([sizeX, sizeY]))[:,:,:3].transpose([1,0,2]).astype(np.uint32)

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
    return entropy(arr)/10

def fitness_img(img_approx, reference_img):
    arr = img_approx.img_array()
    return np.sqrt( ((arr-reference_img)**2).sum(axis=2) ).sum()/(sizeX*sizeY*442)*100

def fitness(img_approx):
    p = 0.25
    #return (1-p)*fitness_img(img_approx, reference_img1) - p*fitness_img(img_approx, reference_img2)
    #return -fitness_color_entropy(img_approx)
    return fitness_img(img_approx, reference_img1)
