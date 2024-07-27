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
    def __init__(self, thermal_profile_csv_fp, width: int, height: int, torus: bool):
        super().__init__(width, height, torus)
        self.thermal_profile = pd.read_csv(thermal_profile_csv_fp)
        self.grid = self.make_grid()

        # make Property Attributes
        self.make_property_attribute('Burrow_Temp')
        self.make_property_attribute('Open_Temp')
        self.make_property_attribute('Shrub_Temp')


    def make_grid(self):
        # Intialize mesa grid class (Check Mesa website for different types ofr grids)
        return mesa.space.MultiGrid(self.width, self.height, torus=self.torus)

    def make_property_layer(self, property_name):
        return self.grid.add_property(property_name)
    
    def get_property_attribute(self, property_name, pos):
        return self.grid.get_property(property_name, pos)
    
    def set_property_attribute(self, property_name)
        for x in range(width):
            for y in range(height):
                self.grid.set_property(property_name, (x, y), np.random.randint(0, 100))


