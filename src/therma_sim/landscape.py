#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd

class Landscape(mesa.space.MultiGrid):
    def __init__(self, model, thermal_profile_csv_fp, width: int, height: int, torus: bool):
        super().__init__(width, height, torus)
        self.model = model
        self.thermal_profile = pd.read_csv(thermal_profile_csv_fp, header=0)
        self.grid = self.make_grid()

        # make Property Layers
        self.burrow_temp = mesa.space.PropertyLayer("Burrow_Temp", self.width, self.height, default_value=0.0)
        self.open_temp = mesa.space.PropertyLayer("Open_Temp", self.width, self.height, default_value=0.0)
        self.shrub_temp = mesa.space.PropertyLayer("Shrub_Temp", self.width, self.height, default_value=0.0)
        self.microhabitat_profile = mesa.space.PropertyLayer("Microhabitat_Profile", self.width, self.height, default_value=0.0)

    def make_grid(self):
        '''
        Initialize mesa grid class (Check Mesa website for different types of grids)
        '''
        return mesa.space.MultiGrid(self.width, self.height, self.torus)

    def get_property_attribute(self, property_name, pos):
        '''
        Helper function that returns landscape property values
        '''
        x, y = pos
        return self.grid[x][y][property_name]
    
    def set_property_attribute(self, property_name, pos, property_value):
        '''
        Helper function for setting property attributes in the landscape
        '''
        x, y = pos
        # Set the property value in the PropertyLayer
        if property_name == "Shrub_Temp":
            self.shrub_temp.set_cell((x, y), property_value)
        elif property_name == "Open_Temp":
            self.open_temp.set_cell((x, y), property_value)
        elif property_name == "Burrow_Temp":
            self.burrow_temp.set_cell((x, y), property_value)
        elif property_name == "Microhabitat_Profile":
            self.microhabitat_profile.set_cell((x, y), property_value)

    def set_landscape_temperatures(self, step_id):
        '''
        Helper function for setting and resetting the thermal temperatures in the landscape
        '''
        shrub_emp_mean = self.thermal_profile['Shrub_mean_Temperature'].iloc[step_id]
        shrub_emp_std = self.thermal_profile['Shrub_stddev_Temperature'].iloc[step_id]

        open_emp_mean = self.thermal_profile['Open_mean_Temperature'].iloc[step_id]
        open_emp_std = self.thermal_profile['Open_stddev_Temperature'].iloc[step_id]
        burrow_emp_mean = self.thermal_profile['Burrow_mean_Temperature'].iloc[step_id]
        burrow_emp_std = self.thermal_profile['Burrow_stddev_Temperature'].iloc[step_id]

        for cell in self.coord_iter():
            pos = cell[1]
            print(pos)
            # shrub
            shrub_temp = np.random.normal(shrub_emp_mean, shrub_emp_std, 1)[0]
            self.set_property_attribute('Shrub_Temp', pos, shrub_temp)
            # Open
            open_temp = np.random.normal(open_emp_mean, open_emp_std, 1)[0]
            self.set_property_attribute('Open_Temp', pos, open_temp)
            # Burrow
            burrow_temp = np.random.normal(burrow_emp_mean, burrow_emp_std, 1)[0]
            self.set_property_attribute('Burrow_Temp', pos, burrow_temp)


