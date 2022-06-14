import pygame
import random
import time
import numpy as np
import math
from copy import deepcopy
from PIL import Image
from evaluate import *
from shapeList import src
from shapeList import ShapeList, Triangle
from imgApprox import ImgApprox, PixelImage, lerp, clamp, config

def iterations(min_v, max_v, inc, change, temp_end, coef=0.93):
    n_iter = int(math.ceil(math.log(temp_end, coef)))
    arr = np.zeros(n_iter)
    val = min_v
    for i in range(n_iter):
        if i%change==0 and i!=0 and val!=max_v:
            val += inc
        arr[i]=int(val)
    return arr

def oneplusone(n_iter, mut_str = 0.01):
    #shapeLst = ShapeList(init_n=1)
    shapeLst = PixelImage()

    fit = fitness(shapeLst)

    j = 0

    clock = pygame.time.Clock()
    for i in range(n_iter):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                exit(0)
        
        if config['display']:
            src.fill('#000000')
        
        # mutate
        mut_str = lerp(i/n_iter, 0.005, 0.02)
        accept = lerp(i/n_iter, 0.006, 0.00001)
        aux = shapeLst.clone()
        aux.mutate(mut_str)

        # new fitness
        fit_aux = fitness(aux)

        if fit_aux <= fit:
            shapeLst = aux
            fit = fit_aux
        elif random.random() <= accept:
            shapeLst = aux
            fit = fit_aux
        pygame.display.update()
        
        #clock.tick(1000)
        j += 1
        if j%500 == 0 and config['verbose']:
            print(f"iteration {j}: {fit}")
    shapeLst.render()
    shapeLst.save_to_image()


def generate_iter(min_v, max_v, temp_fin, coef=0.93):
    n_iter = int(math.ceil(math.log(temp_fin, coef)))
    return np.linspace(min_v, max_v, n_iter).astype(np.int32)

def simulated_annealing(temp_init=1, temp_end=0.00071, mut_str_range = [0.01, 0.001], coef=0.93):
    #shapeLst = ShapeList(init_n=1)
    shapeLst = PixelImage()

    fit = fitness(shapeLst)

    j = 0

    steps = generate_iter(1000, 6000, temp_end, coef)
    if config['verbose']:
        print("number of steps:", steps.sum())

    clock = pygame.time.Clock()

    n_steps = 0
    i_temp = 0
    temp = temp_init
    while temp > temp_end:
        j = steps[i_temp]
        
        mut_str = lerp(1-(n_steps/steps.sum()), mut_str_range[1], mut_str_range[0])
        if config['verbose']:
            print("temp change:", i_temp, "of", len(steps))
        
        for i in range(j):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    exit(0)
            
            if config['display']:
                src.fill('#000000')
            
            # mutate
            aux = shapeLst.clone()
            aux.mutate(mut_str)

            # new fitness
            fit_aux = fitness(aux)

            delta = -(fit-fit_aux)*100
            #delta = (fit-fit_aux)*7500
            if delta <= 0:
                shapeLst = aux
                fit = fit_aux
            elif random.random() <= math.exp(-delta/temp):
                shapeLst = aux
                fit = fit_aux
            pygame.display.update()
            
            j += 1
            if n_steps%500 == 0 and config['verbose']:
                print(f"{n_steps}: {fit}")
            n_steps += 1
        temp = temp*coef
        i_temp += 1
        
    shapeLst.render()
    shapeLst.save_to_image()

if __name__ == '__main__':
    simulated_annealing()