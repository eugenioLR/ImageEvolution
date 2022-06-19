import pygame
import random
import time
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
from imgApprox import *

max_shapes = 30

class Shape:
    def __init__(self, color = [255,255,255,255], center = [0,0]):
        self.color = color
        self.center = center
    
    def mutate_center(self, strength):
        change = np.random.normal(0, strength)
        self.center[0,0] = clamp(self.center[0,0] + change, 0, 1)

        change = np.random.normal(0, strength)
        self.center[1,0] = clamp(self.center[1,0] + change, 0, 1)


    def mutate_color(self, strength):
        change = np.random.normal(0, strength*128)
        self.color[0] = clamp(self.color[0] + change, 0, 255)

        change = np.random.normal(0, strength*128)
        self.color[1] = clamp(self.color[1] + change, 0, 255)

        change = np.random.normal(0, strength*128)
        self.color[2] = clamp(self.color[2] + change, 0, 255)

        change = np.random.normal(0, strength*32)
        self.color[3] = clamp(self.color[1] + change, 0, 255)

    def mutate(self, strength):
        pass
    
    def render(self):
        pass

class Triangle(Shape):
    def __init__(self, color=None, center=None, points=None):
        if color is None:
            self.color = [random.random()*255,random.random()*255,random.random()*255,255]
        else:
            self.color = color
        
        if center is None:
            self.center = np.random.rand(2,1)
        else:
            self.center = center

        if points is None:
            self.points = np.concatenate([np.zeros([2,1]), (2*np.random.rand(2,2)-1)], axis=1)
        else:
            self.points = points
        
        self.normalize()
        

    def mutate_points(self, strength):
        for i in range(self.points.shape[0]):
            for j in range(self.points.shape[1]):
                change = np.random.normal(0, strength)
                self.points[i,j] = clamp(self.points[i,j] + change, -1, 1)

    def mutate(self, strength):
        decision = random.random()
        if decision >= 2/3:
            self.mutate_center(strength)
        elif decision >= 1/3:
            self.mutate_color(strength)
        else:
            self.mutate_points(strength)        
        self.normalize()
    
    def normalize(self):
        for i in range(self.points.shape[1]):
            self.points[0,i] = clamp(self.points[0,i] + self.center[0], 0, 1) - self.center[0]
            self.points[1,i] = clamp(self.points[1,i] + self.center[1], 0, 1) - self.center[1]
    
    def get_vertices(self):
        points = (self.points + self.center) * np.array([[sizeX, sizeY]]).T
        points = points.reshape((-1, 1, 2)).astype(np.int32)
        return points

    def render(self, canvas):
        points = (self.points + self.center) * np.array([[sizeX, sizeY]]).T
        points = points.reshape((-1, 1, 2)).astype(np.int32)
        canvas_new = cv2.fillPoly(np.copy(canvas), [points], color=self.color[:3], lineType=cv2.LINE_AA)
        return cv2.addWeighted(canvas, 1-self.color[3]/255, canvas_new, self.color[3]/255, 0)


class ShapeList(ImgApprox):
    def __init__(self, shapes = [], init_n = 1):
        self.shapes = shapes
        for i in range(init_n):
            self.shapes.append(Triangle())
        
        img_init = config['img_init']
        if img_init == 'black':
            self.__data = (np.zeros([sizeX, sizeY, 3])).astype(np.float64)
        elif img_init == 'white':
            self.__data = (np.ones([sizeX, sizeY, 3])*255).astype(np.float64)
        else:
            self.__data = (np.zeros([sizeX, sizeY, 3])).astype(np.float64)

        self.base = deepcopy(self.__data)
        self.rendered = False
    
    @property
    def data(self):
        if not self.rendered:
            self.__data = np.copy(self.base).astype(np.int8)
            for i in self.shapes:
                self.__data = i.render(self.__data)
            self.rendered = True
        return self.__data

    def add_shape(self, shape):
        self.rendered = False
        if len(self.shapes) < max_shapes:
            self.shapes.append(shape)

    def update(self, step):
        if step%300 == 400:
            self.add_shape(Triangle())

    def mutate(self, strength):
        strength *= 6
        self.rendered = False
        decision = random.random()
        if decision > 0.25:
            idx = random.randint(0, len(self.shapes)-1)
            self.shapes[idx].mutate(strength)
        elif decision > 0.1 or len(self.shapes) == 0:
            self.add_shape(Triangle())
        else:
            self.shapes.pop(random.randint(0, len(self.shapes)-1))
        
        

    def render(self):     
        texture = cv2.resize(self.data.astype(np.uint8), [w_height, w_width], interpolation = cv2.INTER_NEAREST)
        pygame.surfarray.blit_array(src, texture)
        pygame.display.flip()
    
    def img_array(self):
        return self.data


if __name__ == '__main__':
    s = ShapeList()
    s.add_shape(Triangle())
    s.add_shape(Triangle())
    s.add_shape(Triangle())
    s.render()
    time.sleep(1)
    s.mutate(0.5)
    s.render()
    
    time.sleep(1)
    s.mutate(0.5)
    s.render()
    #t = Triangle()
    #t.render(None)
