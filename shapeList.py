import pygame
import random
import time
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
from imgApprox import *

max_shapes = 40

class Shape:
    def __init__(self, color = '#00000000', center = [0,0]):
        self.color = color
        self.center = center
    
    def mutate_center(self, strength):
        change = np.random.normal(0, strength*sizeX*10)
        self.center[0,0] = clamp(self.center[0,0] + change, 0, sizeX)

        change = np.random.normal(0, strength*sizeY*10)
        self.center[1,0] = clamp(self.center[1,0] + change, 0, sizeY)


    def mutate_color(self, strength):
        change = np.random.normal(0, strength*255)
        self.color[0] = clamp(int(self.color[0] + change), 0, 255)

        change = np.random.normal(0, strength*255)
        self.color[1] = clamp(int(self.color[1] + change), 0, 255)

        change = np.random.normal(0, strength*255)
        self.color[2] = clamp(int(self.color[2] + change), 0, 255)

        change = np.random.normal(0, strength*255)
        self.color[3] = clamp(int(self.color[3] + change), 0, 255)

    def mutate(self, strength):
        pass
    
    def render(self):
        pass

class Triangle(Shape):
    def __init__(self, color=None, center=None, points=None):
        if color is None:
            self.color = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255),random.randint(10, 255)]
        else:
            self.color = color
        
        if center is None:
            self.center = np.random.rand(2,1) * np.array([[sizeX, sizeY]]).T
        else:
            self.center = center

        if points is None:
            self.points = np.concatenate([np.zeros([2,1]), (2*np.random.rand(2,2)-1) * np.array([[sizeX, sizeY]]).T], axis=1)
        else:
            self.points = points
        
        self.normalize()
        

    def mutate_points(self, strength):
        for i in range(self.points.shape[0]):
            for j in range(self.points.shape[1]):
                change = np.random.normal(0, strength)
                self.points[i,j] = clamp(self.points[i,j] + change, -sizeX, sizeX)

    def mutate(self, strength):
        strength*=2
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
            self.points[0,i] = clamp(self.points[0,i] + self.center[0], 0, sizeX) - self.center[0]
            self.points[1,i] = clamp(self.points[1,i] + self.center[1], 0, sizeY) - self.center[1]

    def render(self, canvas):
        #http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
        #pygame.draw.polygon(src, self.color, (self.points + self.center).T)
        points = (self.points + self.center).reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=self.color, thickness=1)
        cv2.fillPoly(canvas, [points], color=self.color)


class ShapeList(ImgApprox):
    def __init__(self, shapes = [], init_n = 1):
        self.shapes = shapes
        for i in range(init_n):
            self.shapes.append(Triangle())
        
        img_init = config['img_init']
        if img_init == 'black':
            self.data = (np.zeros([sizeX, sizeY, 3])).astype(np.float64)
        elif img_init == 'white':
            self.data = (np.ones([sizeX, sizeY, 3])*255).astype(np.float64)
        else:
            self.data = (np.zeros([sizeX, sizeY, 3])).astype(np.float64)

        self.base = deepcopy(self.data)
    
    def add_shape(self, shape):
        if len(self.shapes) < max_shapes:
            self.shapes.append(shape)

    def update(self, step):
        if step%3000:
            self.add_shape(Triangle())

    def mutate(self, strength):
        for i in self.shapes:
            i.mutate(strength)

    def render(self):
        self.data = deepcopy(self.base)
        for i in self.shapes:
            i.render(self.data)
        
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
