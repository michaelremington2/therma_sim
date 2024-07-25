#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import landscape

class ThermaSim(mesa.Model):
    '''
    A model class to mange the kangaroorat, rattlesnake predator-prey interactions
    '''
    def __init__(self, initial_agents_dictionary,
                 thermal_profile_csv_fp, width=50, height=50,
                 torus=False, moore=False):
        # Initialize  attributes
        self.width = width
        self.height = height
        self.initial_agents_dictionary = initial_agents_dictionary
        self.thermal_profile_csv_fp = thermal_profile_csv_fp
        # Population Parameters
        self.running = True


        # Intialize mesa grid class (Check Mesa website for different types ofr grids)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=torus)
        
        # Schedular 
        # Random activation, random by type Simultanious, staged
        self.schedule = mesa.time.RandomActivationByType(self)

    def make_landscape(self):
        self.landscape = landscape.Landscape(thermal_profile_csv_fp, width=width, height=False, torus=False)

    def run_model(self, step_count=1000):
        for i in range(step_count):
            model.step()

if __name__ ==  "__main__":
    pass