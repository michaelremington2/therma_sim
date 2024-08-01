#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd

class Landscape(mesa.space.MultiGrid):
    def __init__(self, model, thermal_profile_csv_fp: str, width: int, height: int, torus: bool):
        super().__init__(width, height, torus)
        self.model = model
        self.thermal_profile = pd.read_csv(thermal_profile_csv_fp, header=0)

        # make Property Layers
        self.burrow_temp = mesa.space.PropertyLayer("Burrow_Temp", self.width, self.height, default_value=0.0)
        self.add_property_layer(self.burrow_temp)
        self.open_temp = mesa.space.PropertyLayer("Open_Temp", self.width, self.height, default_value=0.0)
        self.add_property_layer(self.open_temp)
        self.shrub_temp = mesa.space.PropertyLayer("Shrub_Temp", self.width, self.height, default_value=0.0)
        self.add_property_layer(self.shrub_temp)
        self.microhabitat_profile = mesa.space.PropertyLayer("Microhabitat_Profile", self.width, self.height, default_value=0.0)
        self.add_property_layer(self.microhabitat_profile)

    def get_property_layer(self, property_name):
        '''
        Helper function for querying environemntal property layers
        '''
        if property_name == "Shrub_Temp":
            return self.shrub_temp
        elif property_name == "Open_Temp":
            return self.open_temp
        elif property_name == "Burrow_Temp":
            return self.burrow_temp
        elif property_name == "Microhabitat_Profile":
            return self.microhabitat_profile
        else:
            print(f"Unknown layer: {property_name}")
            return

    def get_property_attribute(self, property_name, pos):
        '''
        Helper function that returns landscape property values
        '''
        x, y = pos
        layer = self.get_property_layer(property_name=property_name)
        return layer.data[x, y]

    def set_property_attribute(self, property_name, pos, property_value):
        '''
        Helper function for setting property attributes in the landscape
        '''
        x, y = pos
        layer = self.get_property_layer(property_name=property_name)
        layer.set_cell((x, y), property_value)

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
            # Shrub
            shrub_temp = np.random.normal(shrub_emp_mean, shrub_emp_std, 1)[0]
            self.set_property_attribute('Shrub_Temp', pos, shrub_temp)
            # Open
            open_temp = np.random.normal(open_emp_mean, open_emp_std, 1)[0]
            self.set_property_attribute('Open_Temp', pos, open_temp)
            # Burrow
            burrow_temp = np.random.normal(burrow_emp_mean, burrow_emp_std, 1)[0]
            self.set_property_attribute('Burrow_Temp', pos, burrow_temp)

    def print_property_layer(self, property_name):
        '''
        Helper function for printing details of various property layers 
        '''
        layer = self.get_property_layer(property_name=property_name)
        print(f"\nValues for {property_name}:")
        for y in range(self.height):
            for x in range(self.width):
                print(f"{layer.data[x, y]:.2f}", end=" ")
            print()

    def visualize_property_layer(self, property_name):
        '''
        Function to visualize a property layer as a heat map
        '''
        layer = self.get_property_layer(property_name=property_name)
        data = layer.data

        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Temperature')
        plt.title(f'Heat Map of {property_name}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()
        plt.show()

