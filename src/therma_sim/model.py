#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd

class ThermaSim(mesa.Model):
    '''
    A model class to mange the kangaroorat, rattlesnake predator-prey interactions
    '''
    def __init__(self, width=50, height=50,
                 initialize_agents_dictionary,):
        # Initialize width and height attributes
        self.width=width
        self.height=height
        # Population Parameters
        self.initial_population = initial_population
        self.running = True


        # Intialize mesa grid class (Check Mesa website for different types ofr grids)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        
        # Schedular 
        # Random activation, random by type Simultanious, staged
        self.schedule = mesa.time.RandomActivationByType(self)

    def run_model(self, step_count=1000):
        for i in range(step_count):
            model.step()

if __name__ ==  "__main__":