from multiprocessing.resource_sharer import stop
from tracemalloc import start
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

def progress(step, init_time, evaluations, fitness):
    match config['stop_cond']:
        case "time":
            return 1 - (time.perf_counter() - init_time)/config['max_time']
        case "steps":
            return 1 - (step - config['max_steps'])/config['max_steps']
        case "eval":
            return 1 - (evaluations - config['max_evaluations'])/config['max_evaluations']
        case "fit":
            return 1 - (fitness - config['fitness_target'])/config['fitness_target']
    return 1

def stop_condition(step, init_time, evaluations, fitness):
    match config['stop_cond']:
        case "time":
            return time.perf_counter() - init_time >= config['max_time']
        case "steps":
            return step >= config['max_steps']
        case "eval":
            return evaluations >= config['max_evaluations']
        case "fit":
            return fitness >= config['fitness_target']
    return True

def random_search():
    if config['verbose']:
        fit_history = []

    best_img = PixelImage()
    best_fit = fitness(best_img)

    for rand_type in ['random', 'black', 'white']:
        img = PixelImage(img_init=rand_type)
        img_fit = fitness(img)
        if  img_fit > best_fit:
            best_img = img
            best_fit = img_fit

    start_time = time.perf_counter()
    step = 0
    while not stop_condition(step, start_time, step+3, best_fit):
        # process GUI events and reset screen
        if config['display']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    best_img.save_to_image(step)
                    exit(0)
            src.fill('#000000')
        
        # Generate new random image
        img = PixelImage(img_init='random')
        
        # Compare with the best image and replace if it's better
        img_fit = fitness(img)
        if  img_fit > best_fit:
            best_img = img
            best_fit = img_fit
        
        # display the current image
        if config['display']:
            img.render()
            pygame.display.update()
        
        # show info about the fitness
        if config['verbose']:
            if step%500 == 0:
                print(f"{step}: {best_fit}")
            fit_history.append(best_fit)
        
        step += 1
    
    if config['verbose']:
        plt.plot(fit_history)
        plt.show()
        np.savetxt("data_Rand.csv", np.array(fit_history).reshape([1,len(fit_history)]), delimiter=",")
    
    best_img.save_to_image()


def oneplusone(mut_str_range = [0.01, 0.001]):
    # choose the method used
    if config['method'] == 'triangles':
        current_img = ShapeList(init_n=1)
    elif config['method'] == 'pixels':
        current_img = PixelImage()
    else:
        current_img = PixelImage()

    if config['verbose']:
        fit_history = []

    start_time = time.perf_counter()
    step = 0
    eval_count = 0
    best_fit = fitness(current_img)
    while not stop_condition(step, start_time, step, best_fit):
        # process GUI events and reset screen
        if config['display']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    current_img.save_to_image(step)
                    exit(0)
            src.fill('#000000')
        
        # update current image
        current_img.update(step)

        # mutate
        mut_str = lerp(progress(step, start_time, step, best_fit), mut_str_range[1], mut_str_range[0])
        aux = current_img.clone()
        aux.mutate(mut_str)

        # new fitness
        fit_aux = fitness(aux)
        eval_count += 1

        # decide whether to accept the new image or not
        if best_fit < fit_aux:
            current_img = aux
            best_fit = fit_aux
        
        # display the current image
        if config['display']:
            aux.render()
            pygame.display.update()
        
        # show info about the fitness
        if config['verbose']:
            if step%500 == 0:
                print(f"{step}: {best_fit}")
            fit_history.append(best_fit)
        
        step += 1
    
    if config['verbose']:
        plt.plot(fit_history)
        plt.show()
        np.savetxt("data_ES.csv", np.array(fit_history).reshape([1,len(fit_history)]), delimiter=",")
    
    current_img.save_to_image()


def step_generation(generation, gen_n):
    fits = [fitness(i) for i in generation]

    for i in range(gen_n):
        idx = fits.index(min(fits))
        fits.pop(idx)
        generation.pop(idx)
    
    best_fit = max(fits)
    idx = fits.index(best_fit)
    generation = [generation[idx]] + generation[:idx] + generation[idx+1:]

    return (generation, best_fit)

def genetic_alg(popul_size=100, mut_prob=0.1, mut_str_range=[0.01, 0.001]):
    current_gen = []
    if config['method'] == 'triangles':
        current_img = ShapeList(init_n=1)
    elif config['method'] == 'pixels':
        current_gen  = [PixelImage(img_init='random') for i in range(popul_size//3)]
        current_gen += [PixelImage(img_init='black') for i in range(popul_size//3)]
        current_gen += [PixelImage(img_init='white') for i in range(popul_size - (2*popul_size)//3)]
    else:
        current_gen = [PixelImage() for i in range(popul_size)]
        for i in current_gen:
            i.mutate_noise(1)
            i.mutate(1)
    
    if config['verbose']:
        fit_history = []
    
    start_time = time.perf_counter()
    step = 0
    eval_count = 0
    best_fit = 0
    while not stop_condition(step, start_time, eval_count, best_fit):
        # process GUI events and reset screen
        if config['display']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    current_gen[0].save_to_image(step)
                    exit(0)
            src.fill('#000000')

        mut_str = lerp(progress(step, start_time, step, best_fit), mut_str_range[1], mut_str_range[0])


        next_gen = []
        while len(next_gen) <= popul_size:
            parent1 = random.choice(current_gen)
            parent2 = random.choice(current_gen)

            (new_ind1, new_ind2) = parent1.cross(parent2)

            if random.random() <= mut_prob:
                new_ind1.mutate(mut_str)
            
            if random.random() <= mut_prob:
                new_ind2.mutate(mut_str)
            
            next_gen += [new_ind1, new_ind2]
        
        eval_count += len(current_gen) + len(next_gen)
        current_gen, best_fit = step_generation(current_gen+next_gen, popul_size)
        

        # display the current image
        if config['display']:
            current_gen[0].render()
            pygame.display.update()
        
        # show info about the fitness
        if config['verbose']:
            if step%10 == 0:
                print(f"{step}: {best_fit}")
            fit_history.append(best_fit)
        step += 1
    
    if config['verbose']:
        plt.plot(fit_history)
        plt.show()
        np.savetxt("data_GA.csv", np.array(fit_history).reshape([1,len(fit_history)]), delimiter=",")
    
    current_gen[0].save_to_image()
    


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
                        current_img.save_to_image(n_steps)
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
        np.savetxt("data_SA.csv", fit_history.reshape([1,fit_history.size]), delimiter=",")
    
    current_img.save_to_image()

def particle_swarm(popul_size=300, inertia_r=[0.8, 0.1], cognitive_w=1.5, social_w=2.2):
    # Particle initialization
    particle_list  = [PixelImage(img_init='random') for i in range(popul_size//3)]
    particle_list += [PixelImage(img_init='black') for i in range(popul_size//3)]
    particle_list += [PixelImage(img_init='white') for i in range(popul_size - (2*popul_size)//3)]

    # Record of the fitness of each particle
    fitness_list = [fitness(i) for i in particle_list]

    # Record of best position of each particle
    particle_best = [i.data for i in particle_list]

    # Best fitness of each particle
    best_fitness_list = fitness_list.copy()
    
    # Assigning of speed of each particle
    img_shape = particle_list[0].data.shape
    particle_speed = [(2*255*np.random.random(img_shape) - 255) for i in particle_list]
    
    # Best overall particle
    best_fit = max(fitness_list)
    best_particle = particle_list[fitness_list.index(best_fit)].data

    if config['verbose']:
        fit_history = []

    start_time = time.perf_counter()
    step = 0
    eval_count = 0
    while not stop_condition(step, start_time, step, best_fit):
        #inertia = (1-(step/n_gen))*inertia_r[0] - (step/n_gen)*inertia_r[1]
        inertia = lerp(progress(step, start_time, step, best_fit), inertia_r[0], inertia_r[1])
        # process GUI events and reset screen
        if config['display']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    PixelImage(data=best_particle).save_to_image(step)
                    exit(0)
            src.fill('#000000')
        
        # Update each particle
        for i, particle in enumerate(particle_list):
            r_p = cognitive_w * np.random.random()
            r_g = social_w * np.random.random()
            best_diff = best_particle - particle.data
            self_diff = particle_best[i] - particle.data

            particle_speed[i] = inertia * particle_speed[i] + r_p * best_diff + r_g * self_diff

            particle.data += particle_speed[i]
            particle.data = np.clip(particle.data, 0, 255)

            fitness_list[i] = fitness(particle)
            eval_count += 1
            if best_fitness_list[i] < fitness_list[i]:
                best_fitness_list[i] = fitness_list[i]
                particle_best[i] = particle.data 
        
        best_particle = particle_best[np.argmax(best_fitness_list)]
        best_fit = max(fitness_list)

        # display the current image
        if config['display']:
            particle_list[fitness_list.index(best_fit)].render()
            pygame.display.update()
        
        # show info about the fitness
        if config['verbose']:
            if step%100 == 0:
                print(f"{step}: {best_fit}")
            fit_history.append(best_fit)
        
        step += 1
    
    if config['verbose']:
        plt.plot(fit_history)
        plt.show()
        np.savetxt("data_PSO.csv", np.array(fit_history).reshape([1,len(fit_history)]), delimiter=",")

    PixelImage(data=best_particle).save_to_image()
    

def DE_method(current_gen, current_img, fitnesses, F):
    ## DE/rand/1
    #F2 = 0
    #d = e = 0
    #a, b, c = random.sample([i for i in current_gen if i not in [current_img]], 3)

    ## DE/best/1
    #F2 = 0
    #d = e = 0
    #a = current_gen[fitnesses.index(max(fitnesses))]
    #b, c = random.sample([i for i in current_gen if i not in [current_img,a]], 2)

    ## DE/rand/2
    # F2 = F
    #a, b, c, d, e = random.sample([i for i in current_gen if i not in [current_img]], 5)

    ## DE/best/2
    #F2 = F
    #a = current_gen[fitnesses.index(max(fitnesses))]
    #b, c, d, e = random.sample([i for i in current_gen if i not in [current_img,a]], 4)

    ## DE/current-to-best/1
    #F2 = F
    #a = c = current_img
    #b = current_gen[fitnesses.index(max(fitnesses))] 
    #d, e = random.sample([i for i in current_gen if i not in [a,b]], 2)

    ## DE/current-to-rand/1
    #F2 = random.random()
    #a = e = current_gen[fitnesses.index(max(fitnesses))]
    #b, d, c = random.sample([i for i in current_gen if i != current_img and i != a], 3)

    ## DE/current-to-rand-p/1
    F2 = random.random()
    a = c = current_gen[fitnesses.index(max(fitnesses))]
    b, d, e = random.sample([i for i in current_gen if i != current_img and i != a], 3)

    return F2,a,b,c,d,e

def diff_evolution(popul_size=500, F=0.8, CR=0.8):
    if config['verbose']:
        fit_history = []

    current_gen = []
    if config['method'] == 'triangles':
        current_img = ShapeList(init_n=1)
    elif config['method'] == 'pixels':
        current_gen  = [PixelImage(img_init='random') for i in range(popul_size//3)]
        current_gen += [PixelImage(img_init='black') for i in range(popul_size//3)]
        current_gen += [PixelImage(img_init='white') for i in range(popul_size - (2*popul_size)//3)]
    else:
        current_gen = [PixelImage() for i in range(popul_size)]
        for i in current_gen:
            i.mutate_noise(1)
            i.mutate(1)
    
    fitnesses = [fitness(i) for i in current_gen]

    step = 0
    start_time = time.perf_counter()
    eval_count = len(fitnesses)
    best_fit = max(fitnesses)
    while not stop_condition(step, start_time, eval_count, best_fit):
        # process GUI events and reset screen
        if config['display']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    current_gen[0].save_to_image(step)
                    exit(0)
            src.fill('#000000')
        
        for i, current_img in enumerate(current_gen):
            
            F2,a,b,c,d,e = DE_method(current_gen, current_img, fitnesses, F)

            if config['method'] == 'pixels':
                v = a.data + F*np.clip(b.data-c.data, -255, 255) + F2*np.clip(d.data-e.data, -255, 255)
                v = np.clip(v, 0, 255)
                u = current_img.data.copy()
                decision = np.random.random(current_img.data.shape) <= CR
                u[decision] = v[decision]
                new_ind = PixelImage(data=u)
                new_fitness = fitness(new_ind)
                eval_count += 1
                if  new_fitness > fitnesses[i]:
                    current_gen[i] = new_ind
                    fitnesses[i] = new_fitness

        best_fit = max(fitnesses)

        # display the current image
        if config['display']:
            current_gen[fitnesses.index(best_fit)].render()
            pygame.display.update()
        
        # show info about the fitness
        if config['verbose']:
            if step%100 == 0:
                print(f"{step}: {best_fit }")
            fit_history.append(best_fit )
        
        step += 1
    
    if config['verbose']:
        plt.plot(fit_history)
        plt.show()
        np.savetxt("data_DE.csv", np.array(fit_history).reshape([1,len(fit_history)]), delimiter=",")
    
    current_gen[np.argmax(fitnesses)].save_to_image()

def main():
    alg = config['algorithm']

    if alg == "rand":
        random_search()
    elif alg == 'one_plus_one':
        oneplusone([0.01, 0.001])
    elif alg == 'genetic':
        genetic_alg(200, 0.1, [0.1, 0.04])
    elif alg == 'sim_annealing':
        simulated_annealing(1, 0.00071, [0.01, 0.001], 0.93)
    elif alg == 'PSO':
        particle_swarm(300, [0.8, 0.6], 2, 1.8)
    elif alg == 'DE':
        diff_evolution(300, 0.9, 0.8)
    else:
        print("ERROR: incorrect optimization method")
        exit(1)

if __name__ == '__main__':
    main()