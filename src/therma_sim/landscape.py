#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd

class Landscape(mesa.space.MultiGrid):
    '''
    Landscape class used in the model class to make the virtual landscape agents use
    Args:
        width (int): Width of the landscape.
        height (int): Height of the landscape.
        torus (bool): If true, the edges of the grid wrap around.
    
    '''
    def __init__(self, model, thermal_profile_csv_fp, width: int, height: int, torus: bool):
        super().__init__(width, height, torus)
        self.model = model
        self.thermal_profile = pd.read_csv(thermal_profile_csv_fp)
        self.grid = self.make_grid()

        # make Property Layers
        self.burrow_temp = mesa.space.PropertyLayer("Burrow_Temp", self.width, self.height, default_value=0)
        self.open_temp = mesa.space.PropertyLayer("Open_Temp", self.width, self.height, default_value=0)
        self.shrub_temp = mesa.space.PropertyLayer("Shrub_Temp", self.width, self.height, default_value=0)

        # Populate the elevation layer with random values

    def make_grid(self):
        # Intialize mesa grid class (Check Mesa website for different types ofr grids)
        return mesa.space.MultiGrid(self.width, self.height, torus=self.torus)

    def make_property_layer(self, property_name):
        # Assuming this creates and initializes a property attribute
        for cell in self.coord_iter():
            x, y = cell[1], cell[2]
            self.grid[x][y][property_name] = None

    def get_property_attribute(self, property_name, pos):
        x, y = pos
        return self.grid[x][y][property_name]
    
    def set_property_attribute(self, property_name, pos, property_value):
        x, y = pos
        self.grid[x][y][property_name] = property_value


