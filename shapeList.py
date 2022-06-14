import pygame
import random
import time
import numpy as np
from copy import deepcopy
from PIL import Image
from imgApprox import ImgApprox
from imgApprox import src, clamp, lerp, sizeX, sizeY


max_shapes = 40

class Shape:
    def __init__(self, color = '#00000000', center = [0,0]):
        self.color = color
        self.center = center
    
    def mutate_center(self, strength):
        change = np.random.normal(0, strength*780)
        self.center[0,0] = clamp(self.center[0,0] + change, 0, sizeX)

        change = np.random.normal(0, strength*580)
        self.center[1,0] = clamp(self.center[1,0] + change, 0, sizeY)


    def mutate_color(self, strength):
        r = int(self.color[1:3], 16)
        g = int(self.color[3:5], 16)
        b = int(self.color[5:7], 16)
        b = int(self.color[7:9], 16)

        change = np.random.normal(0, strength*100)
        r = clamp(int(r + change), 0, 255)

        change = np.random.normal(0, strength*100)
        g = clamp(int(g + change), 0, 255)

        change = np.random.normal(0, strength*100)
        b = clamp(int(b + change), 0, 255)

        change = np.random.normal(0, strength*100)
        alpha = clamp(int(b + change), 0, 255)

        self.color = f"#{r:02x}{g:02x}{b:02x}{alpha:02x}"

    def mutate(self, strength):
        pass
    
    def render(self):
        pass

class Triangle(Shape):
    def __init__(self, color=None, center=None, points=None):
        if color is None:
            self.color = f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}FF"
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
                change = np.random.normal(0, strength*400)
                self.points[i,j] = clamp(self.points[i,j] + change, -sizeX, sizeX)

    def mutate(self, strength):
        self.mutate_center(strength)
        self.mutate_color(strength)
        self.mutate_points(strength)
        self.normalize()
    
    def normalize(self):
        for i in range(self.points.shape[1]):
            self.points[0,i] = clamp(self.points[0,i] + self.center[0], 0, sizeX) - self.center[0]
            self.points[1,i] = clamp(self.points[1,i] + self.center[1], 0, sizeY) - self.center[1]

    def render(self):
        #http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
        pygame.draw.polygon(src, self.color, (self.points + self.center).T)

class ShapeList(ImgApprox):
    def __init__(self, shapes = [], init_n = 1):
        self.shapes = shapes
        for i in range(init_n):
            print("hello", len(self.shapes))
            self.shapes.append(Triangle())
    
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
        for i in self.shapes:
            i.render()
    
    def img_array(self):
        return pygame.surfarray.pixels3d(pygame.display.get_surface())