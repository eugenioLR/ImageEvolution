import pygame
import random
import time
import numpy as np
import math
from copy import deepcopy
from PIL import Image
from matplotlib import pyplot as plt
from evaluate import *
from shapeList import src
from shapeList import ShapeList, Triangle
from imgApprox import ImgApprox, PixelImage, lerp, clamp, config

def oneplusone(n_iter, mut_str = 0.01):
    #current_img = ShapeList(init_n=1)
    current_img = PixelImage()

    fit = fitness(current_img)

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
        aux = current_img.clone()
        aux.mutate(mut_str)

        # new fitness
        fit_aux = fitness(aux)

        if fit_aux <= fit:
            current_img = aux
            fit = fit_aux
        elif random.random() <= accept:
            current_img = aux
            fit = fit_aux
        pygame.display.update()
        
        #clock.tick(1000)
        j += 1
        if j%500 == 0 and config['verbose']:
            print(f"iteration {j}: {fit}")
    current_img.render()
    current_img.save_to_image()


def generate_iter(min_v, max_v, temp_fin, coef=0.93):
    n_iter = int(math.ceil(math.log(temp_fin, coef)))
    return np.linspace(min_v, max_v, n_iter).astype(np.int32)

def simulated_annealing(temp_init=1, temp_end=0.00071, mut_str_range = [0.01, 0.001], coef=0.93):
    K=1
    # choose the method used
    if config['method'] == 'triangles':
        current_img = ShapeList(init_n=1)
    elif config['method'] == 'pixels':
        current_img = PixelImage()
    else:
        current_img = PixelImage()

    # generate the steps in each temperature change
    steps = generate_iter(1000, 7000, temp_end, coef)
    

    if config['verbose']:
        fit_history = np.zeros(steps.sum())
        print("number of steps:", steps.sum())

    n_steps = 0
    i_temp = 0
    temp = temp_init
    j = 0
    fit = fitness(current_img)
    while temp > temp_end:
        # new temperature value
        if config['verbose']:
            print("temp change:", i_temp, "of", len(steps))

        # update the mutation strength
        mut_str = lerp(1-(n_steps/steps.sum()), mut_str_range[1], mut_str_range[0])
        
        # do the necessary iterations
        j = steps[i_temp]
        for i in range(j):

            # process GUI events and reset screen
            if config['display']:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        current_img.save_to_image("stopped_gen"+str(n_steps)+".png")
                        exit(0)
                src.fill('#000000')
            
            # update current image
            current_img.update(n_steps)

            # mutate
            aux = current_img.clone()
            aux.mutate(mut_str)

            # new fitness
            fit_aux = fitness(aux)

            # decide whether to accept the new image or not
            delta = fit-fit_aux
            if delta < 0:
                current_img = aux
                fit = fit_aux
            elif random.random() > math.exp(-delta/K*temp):
                current_img = aux
                fit = fit_aux
            
            # display the current image
            if config['display']:
                aux.render()
                pygame.display.update()
            
            # show info about the fitness
            if config['verbose']:
                if n_steps%500 == 0:
                    print(f"{n_steps}: {fit}")
                fit_history[n_steps] = fit
            
            n_steps += 1

        # change temperature
        temp = temp*coef
        i_temp += 1
    
    if config['verbose']:
        plt.plot(fit_history)
        plt.show()
    
    current_img.save_to_image()

if __name__ == '__main__':
    simulated_annealing(1, 0.00071, [0.01, 0.001], 0.93)