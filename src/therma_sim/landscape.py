#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
#import networkx as nx
import pandas as pd
import agents


import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import polars as pl
import agents

class Spatially_Implicit_Landscape(object):
    def __init__(self, model, site_name:str, width: int, height: int, thermal_profile_csv_fp: str):
        self.model = model
        self.hectare_to_meter = 10000
        self.width_hectare = width
        self.height_hectare = height
        self.landscape_size = self.width_hectare * self.height_hectare
        self.microhabitats = ['Burrow', 'Open']
        self.thermal_profile = pl.read_csv(thermal_profile_csv_fp)
        self.site_name = site_name
        self._burrow_temperature = None
        self._open_temperature = None
    
    @property
    def burrow_temperature(self):
        return self._burrow_temperature
    
    @burrow_temperature.setter
    def burrow_temperature(self, value):
        self._burrow_temperature = value
    
    @property
    def open_temperature(self):
        return self._open_temperature
    
    @open_temperature.setter
    def open_temperature(self, value):
        self._open_temperature = value
    
    def meters_to_hectares(self, val):
        return float(val / self.hectare_to_meter)
    
    def set_landscape_temperatures(self, step_id):
        open_emp_mean = self.thermal_profile.select("Open_mean_Temperature").row(step_id)[0]
        burrow_emp_mean = self.thermal_profile.select("Burrow_mean_Temperature").row(step_id)[0]
        self.open_temperature = open_emp_mean
        self.burrow_temperature = burrow_emp_mean

    def count_steps_in_one_year(self) -> int:
        self.thermal_profile = self.thermal_profile.with_columns(
            pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
        first_day = self.thermal_profile["datetime"].min()
        one_year_later = first_day + pl.duration(days=365)
        steps_count = self.thermal_profile.filter(
            (pl.col("datetime") >= first_day) & (pl.col("datetime") < one_year_later)
        ).height
        return steps_count


class Continous_Landscape(mesa.space.ContinuousSpace):
    def __init__(self, model, thermal_profile_csv_fp: str, width: int, height: int, torus: bool, _test=False):
        super().__init__(width, height, torus)
        self.model = model
        ## Change spaticial dementions, prolly meters
        self.hectare_to_meter = 10000
        self.width_hectare = self.meters_to_hectares(val=self.width)
        self.height_hectare = self.meters_to_hectares(val=self.height)
        self.microhabitats = ['Burrow', 'Open']
        self.thermal_profile = pd.read_csv(thermal_profile_csv_fp, header=0)
        self._burrow_temperature = None
        self._open_temperature = None

    @property
    def burrow_temperature(self):
        return self._burrow_temperature

    @burrow_temperature.setter
    def burrow_temperature(self, value):
        self._burrow_temperature = value

    @property
    def open_temperature(self):
        return self._open_temperature

    @open_temperature.setter
    def open_temperature(self, value):
        self._open_temperature = value

    def meters_to_hectares(self, val):
        '''Val is any value in meters you need to conver to hectares'''
        return float(val / self.hectare_to_meter)
    
    def get_random_point(self):
        x = np.random.uniform(0, self.width)
        y = np.random.uniform(0, self.height)
        point = (x,y)
        return point
    
    def give_birth(self, species_name, pos, agent_id):
        '''
        Helper function - Adds new agents to the landscape
        '''
        if species_name=='KangarooRat':
            kr_params = self.model.get_krat_params(config=self.model.config)
            krat = agents.KangarooRat(unique_id = agent_id, 
                                        model = self.model,
                                        initial_pos = pos,
                                        krat_config=kr_params)
            self.place_agent(krat, pos)
            self.model.schedule.add(krat)
        elif species_name=='Rattlesnake':
            rs_params = self.model.get_rattlesnake_params(config=self.model.config)
            snake = agents.Rattlesnake(unique_id = agent_id, 
                                        model = self.model,
                                        initial_pos = pos,
                                        snake_config = rs_params )
            self.place_agent(snake, pos)
            self.model.schedule.add(snake)
        else:
            raise ValueError(f'Class for species: {species_name} DNE')
    
    def initialize_populations(self, initial_agent_dictionary):
        '''
        Helper function in the landscape class used to intialize populations.
        Populations sizes should be a range of individuals per hectare
        '''
        agent_id = 0
        for x_hect in range(0,self.width, self.hectare_to_meter):
            for y_hect in range(0,self.height, self.hectare_to_meter):
                for species, initial_population_size_range in initial_agent_dictionary.items():
                    start, stop = initial_population_size_range.start, initial_population_size_range.stop
                    initial_pop_size = round(np.random.uniform(start, stop))
                    print(f'species {species}\{initial_population_size_range}, initial_pop {initial_pop_size}, x_hect {x_hect}, y_hect {y_hect}')
                    for i in range(initial_pop_size):
                        x = np.random.uniform(x_hect, x_hect + self.hectare_to_meter)
                        y = np.random.uniform(y_hect, y_hect + self.hectare_to_meter)
                        pos = (x,y)
                        self.give_birth(species_name=species, pos=pos, agent_id=agent_id)
                        agent_id +=1
                        self.model.next_agent_id = agent_id+1
                        #print(pos,agent_id)
                                    
    def get_mh_availability_dict(self, pos):
        '''
        Helper function to return the microhabitat availability dictionary by position. returns a dictionary like this
            availability = {
                'Shrub': 0.8,
                'Open': 0.2,
                'Burrow': 1.0
                }
        right now, assuming individuals always have access to any microhabitat
        '''
        availability = {
            'Burrow':1,
            'Open': 1,
        }
        return availability 

    def set_landscape_temperatures(self, step_id, spatial_heterogeonus=False):
        '''
        Helper function for setting and resetting the thermal temperatures in the landscape
        '''
        open_emp_mean = self.thermal_profile['Open_mean_Temperature'].iloc[step_id]
        open_emp_std = self.thermal_profile['Open_stddev_Temperature'].iloc[step_id]
        burrow_emp_mean = self.thermal_profile['Burrow_mean_Temperature'].iloc[step_id]
        burrow_emp_std = self.thermal_profile['Burrow_stddev_Temperature'].iloc[step_id]
        self.open_temperature = open_emp_mean
        self.burrow_temperature = burrow_emp_mean



class Discrete_Landscape(mesa.space.MultiGrid):
    def __init__(self, model, thermal_profile_csv_fp: str, width: int, height: int, torus: bool):
        super().__init__(width, height, torus)
        self.model = model
        self.thermal_profile = pd.read_csv(thermal_profile_csv_fp, header=0)
        self.microhabitats = ['Burrow', 'Open', 'Shrub']

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

    def set_landscape_temperatures(self, step_id, spatial_heterogeonus=False):
        '''
        Helper function for setting and resetting the thermal temperatures in the landscape
        '''
        ## Come up with a method w
        ### Maybe switch to 2 state model or 3 state model
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

    def initialize_populations(self, initial_agent_dictionary):
        rs_params = self.model.get_rattlesnake_params(config=self.config)
        kr_params = self.model.get_krat_params(config=self.config)
        agent_id = 0
        for species, initial_population_size in initial_agent_dictionary.items():
            for x in range(self.width):
                for y in range(self.height):
                    for i in range(int(initial_population_size)):
                        pos = (x,y)
                        #print(pos,agent_id)
                        if species=='KangarooRat':
                            # Create agent
                            krat = agents.KangarooRat(unique_id = agent_id, 
                                                        model = self,
                                                        initial_pos = pos,
                                                        krat_config=kr_params)
                            # place agent
                            self.place_agent(krat, pos)
                            self.model.schedule.add(krat)
                            agent_id += 1
                        elif species=='Rattlesnake':
                            # Create agent
                            snake = agents.Rattlesnake(unique_id = agent_id, 
                                                        model = self,
                                                        initial_pos = pos,
                                                        snake_config = rs_params)
                            # place agent
                            self.place_agent(snake, pos)
                            self.model.schedule.add(snake)
                            agent_id += 1
                        else:
                            raise ValueError(f'Class for species: {species} DNE')


