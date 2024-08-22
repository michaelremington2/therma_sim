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
        self.open_microhabitat = mesa.space.PropertyLayer("Open_Microhabitat", self.width, self.height, default_value=0.0)
        self.add_property_layer(self.open_microhabitat)
        self.shrub_microhabitat = mesa.space.PropertyLayer("Shrub_Microhabitat", self.width, self.height, default_value=0.0)
        self.add_property_layer(self.shrub_microhabitat)
        # set microhabitat profile
        self.set_microhabitat_profile()

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
        elif property_name == "Open_Microhabitat":
            return self.open_microhabitat
        elif property_name == "Shrub_Microhabitat":
            return self.shrub_microhabitat
        else:
            raise ValueError(f"Unknown layer: {property_name}")

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
        layer = self.get_property_layer(property_name=property_name)
        layer.set_cell(pos, property_value)

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
        #print('mean ', shrub_emp_mean)
        #print('std ', shrub_emp_std)
        # Property values arent setting right or might be rounding weirdly

        for cell in self.coord_iter():
            pos = cell[1]
            # Shrub
            shrub_temp = np.random.normal(shrub_emp_mean, shrub_emp_std, 1)[0]
            #print('sim', shrub_temp)
            self.set_property_attribute('Shrub_Temp', pos, shrub_temp)
            # Open
            open_temp = np.random.normal(open_emp_mean, open_emp_std, 1)[0]
            self.set_property_attribute('Open_Temp', pos, open_temp)
            # Burrow
            burrow_temp = np.random.normal(burrow_emp_mean, burrow_emp_std, 1)[0]
            self.set_property_attribute('Burrow_Temp', pos, burrow_temp)

    def set_microhabitat_profile(self):
        '''
        Helper function used in the init function to set the microhabitat percentages of the cell
        '''
        for cell in self.coord_iter():
            pos = cell[1]
            open_percent = np.random.uniform(0,1)
            shrub_percent = 1-open_percent
            self.set_property_attribute("Open_Microhabitat", pos, open_percent)
            self.set_property_attribute("Shrub_Microhabitat", pos, shrub_percent)

    def get_mh_availability_dict(self, pos):
        '''
        Helper function to return the microhabitat availability dictionary by position. returns a dictionary like this
            availability = {
                'Shrub': 0.8,
                'Open': 0.2,
                'Burrow': 1.0
                }
        Eventually might add in more burrow dynamics but for now assuming there is always availability for burrow
        '''
        shrub_per = self.get_property_attribute(property_name='Shrub_Microhabitat', pos=pos)
        open_per = self.get_property_attribute(property_name='Open_Microhabitat', pos=pos)
        availability = {
            'Shrub': shrub_per,
            'Open': open_per,
            'Burrow': 1
        }
        return availability


    def print_property_layer(self, property_name):
        '''
        Helper function for printing details of various property layers 
        '''
        layer = self.get_property_layer(property_name=property_name)
        print(f"\nValues for {property_name}:")
        print(self.properties[property_name].data)
        # for y in range(self.height):
        #     for x in range(self.width):
        #         print(f"{layer.data[x, y]}", end=" ")
        #     print()

    def visualize_property_layer(self, property_name):
        '''
        Utility Function to visualize a property layer as a heat map
        '''
        layer = self.get_property_layer(property_name=property_name)
        data = layer.data

        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Temperature')
        plt.title(f'Heat Map of {property_name} at time step {self.model.step_id}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()
        plt.show()

    def check_landscape(self):
        '''
        Utility function for checking that agents are being placed and moved aroundd the landscape
        '''
        for cell in self.coord_iter():
            print(f"Cell id {cell[1]} contents:")
            print(self.get_cell_list_contents(cell[1]))


