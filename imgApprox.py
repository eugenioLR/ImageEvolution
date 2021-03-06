import pygame
import random
import time
import numpy as np
import math
import cv2
import json
import os
from copy import deepcopy
from PIL import Image

config_file = open('config.json')
config = json.load(config_file)
config_file.close()

sizeX = config['img_width']
sizeY = config['img_height']

w_width = config['screen_width']
w_height = config['screen_height']

src = None
if config['display']:
    pygame.init()
    src = pygame.display.set_mode([w_width, w_height])
    pygame.display.set_caption("Evo graphics")

def clamp_arr(a, minim, maxim):
    return np.minimum(np.maximum(a, minim), maxim)

def clamp(a, minim, maxim):
    return min(max(a, minim), maxim)

def lerp(t, v0, v1):
    return clamp((1 - t) * v0 + t * v1, v0, v1)

def lerp_arr(t, v0, v1):
    return clamp_arr((1 - t) * v0 + t * v1, v0, v1)

class ImgApprox:
    def __init__(self):
        pass
    
    def clone(self):
        return deepcopy(self)

    def update(self, step):
        pass

    def mutate(self, strength):
        pass

    def render(self):
        pass
    
    def img_array(self):
        pass

    def save_to_image(self, iter = None):
        if not os.path.exists('results'):
            os.makedirs('results')
        
        pathname = config['reference_img1']
        name = pathname.split('.')[0].split('/')[-1]
        alg = config['algorithm']
        if not iter is None:
            alg += '_stop' + str(iter)
        
        filename = 'results/' + name + '_' + alg + '.png'
        
        Image.fromarray(self.data.astype(np.uint8).transpose([1,0,2])).save(filename)


class PixelImage(ImgApprox):
    def __init__(self):
        img_init = config['img_init']

        if img_init == 'black':
            self.data = (np.zeros([sizeX, sizeY, 3])).astype(np.float64)
        elif img_init == 'white':
            self.data = (np.ones([sizeX, sizeY, 3])*255).astype(np.float64)
        elif img_init == 'random':
            self.data = (np.random.random([sizeX, sizeY, 3])*255).astype(np.float64)
        else:
            self.data = (np.zeros([sizeX, sizeY, 3])).astype(np.float64)
    
    def mutate_noise(self, strength):
        noise = np.random.normal(0, strength, [sizeX, sizeY, 3]) * 255
        mask = np.random.random([sizeX, sizeY, 3]) < lerp(random.random(), strength*0.1, strength*10)
        noise = noise * mask
        self.data = clamp_arr(self.data + noise, 0, 255)
    
    def mutate_noise_blocks(self, strength):
        dim = max(sizeX, sizeY)
        block_size = random.randint(1, dim)
        
        noise = np.ones([(dim//block_size + 1)*block_size, (dim//block_size + 1)*block_size, 3])
        noise_blocks = np.random.normal(0, strength, [dim//block_size+1, dim//block_size+1, 3]) * 255
        for i in range(3):
            mask = np.random.random([block_size, block_size]) < strength*40
            noise[:,:,i] = np.kron(noise_blocks[:,:,i], mask)
            noise[:,:,i] = np.roll(noise[:,:,i], -random.randint(0, block_size))
        noise = noise[:sizeX, :sizeY, :]
        self.data = clamp_arr(self.data + noise, 0, 255)
    
    def smoothen(self, strength):
        s = lerp(strength, 0, 0.0003)
        w = s
        w_c = math.sqrt(2) * s
        g_filt = np.array([[w_c,w,w_c],[w,1,w],[w_c,w,w_c]])
        g_filt = g_filt/g_filt.sum()
        for i in range(3):
            self.data[:,:,i] = cv2.filter2D(self.data[:,:,i], -1, g_filt)

    def inc_contrast(self, strength):
        thresh = lerp(strength, 0, 15)
        alpha = 1 + 2*thresh/255
        beta = -thresh
        for i in range(3):
            self.data[:,:,i] = clamp_arr(alpha*self.data[:,:,i]+beta, 0, 255)


    def mutate(self, strength):

        #if random.random() <= thresh:
        #    if random.random() <= 0.6:
        #        self.mutate_noise(strength)
        #    else:
        #        self.mutate_noise_blocks(strength)
        #else:
        #    if random.random() <= 0.5:
        #        self.smoothen(strength)
        #    else:
        #        self.inc_contrast()
        
        decision = random.random()
        thresh = [2/4, 1/4]
        if decision >= thresh[0]:
            self.mutate_noise_blocks(strength)
        elif decision >= thresh[1]:
            self.smoothen(strength)
        else:
            self.inc_contrast(strength)

        #self.mutate_noise_blocks(strength)

    def render(self):
        texture = cv2.resize(self.data.astype(np.uint8), [w_height, w_width], interpolation = cv2.INTER_NEAREST)
        pygame.surfarray.blit_array(src, texture)
        pygame.display.flip()
    
    def img_array(self):
        return self.data
    
    def update(self, step):
        pass