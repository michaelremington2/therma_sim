#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import warnings
import logging
import json
import landscape
import agents 
import interaction
import uuid

warnings.filterwarnings("ignore")

class ThermaSim(mesa.Model):
    '''
    A model class to mange the kangaroorat, rattlesnake predator-prey interactions
    '''
    def __init__(self, config, seed=None, _test=False):
        self.config = config
        self.initial_agents_dictionary = self.get_initial_population_params(config=self.config)
        self.step_id = 0
        self.running = True
        self.seed = seed
        self._hour = None
        self._day = None
        self._month = None
        self._year = None
        self.next_agent_id = 0
        if seed is not None:
            np.random.seed(self.seed)
        
        # Schedular 
        # Random activation, random by type Simultanious, staged
        self.schedule = mesa.time.RandomActivationByType(self)

        ## Make Initial Landscape
        self.landscape = self.make_landscape(model=self)
        self.steps_per_year = self.landscape.count_steps_in_one_year()
        self.interaction_map = self.make_interaction_module(model=self)
        ## Intialize agents
        self.initialize_populations(initial_agent_dictionary=self.initial_agents_dictionary)

        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                'Step_ID': lambda m: m.step_id,
                'Hour': lambda m: m.hour, 
                'Day': lambda m: m.day,
                'Month': lambda m: m.month,
                'Year': lambda m: m.year,
                "Rattlesnakes": lambda m: m.schedule.get_type_count(agents.Rattlesnake),
                "Krats": lambda m: m.schedule.get_type_count(agents.KangarooRat),
            },
            agenttype_reporters={
                agents.Rattlesnake: {
                    "Time_Step": lambda a: a.model.step_id,
                    "Agent_ID": lambda a: a.unique_id,
                    "Behavior": lambda a: a.current_behavior,
                    "Microhabitat": lambda a: a.current_microhabitat,
                    "Body_Temperature": lambda a: a.body_temperature,
                    "Metabolic_State": lambda a: a.metabolism.metabolic_state,
                },
                # agents.KangarooRat: {
                #     "Time_Step": lambda a: a.model.step_id,
                #     "Agent_ID": lambda a: a.unique_id,
                #     "Active_Hours": lambda a: a.active_hours,
                #     # Add more KangarooRat-specific reporters as needed
                # }
            }
        )


    @property
    def hour(self):
        return self._hour

    @hour.setter
    def hour(self, value):
        self._hour = value

    @property
    def day(self):
        return self._day

    @day.setter
    def day(self, value):
        self._day = value

    @property
    def month(self):
        return self._month

    @month.setter
    def month(self, value):
        self._month = value

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, value):
        self._year = value

    @property
    def rattlesnake_pop_size(self):
        """Returns the count of Rattlesnake agents in the model."""
        return self.schedule.get_type_count(agents.Rattlesnake)

    @property
    def krats_pop_size(self):
        """Returns the count of KangarooRat agents in the model."""
        return self.schedule.get_type_count(agents.KangarooRat)
    
    @property
    def active_krats_count(self):
        active_krats = [krat for krat in self.schedule.agents_by_type[agents.KangarooRat].values() if krat.active]
        return len(active_krats)

    def get_landscape_params(self, config):
        return config['Landscape_Parameters']

    def get_rattlesnake_params(self, config):
        return config['Rattlesnake_Parameters']
    
    def get_krat_params(self, config):
        return config['KangarooRat_Parameters']
    
    def get_interaction_params(self, config):
        return config['Interaction_Parameters']
    
    def get_initial_population_params(self, config):
        return config['Initial_Population_Sizes']
    
    def get_interaction_map(self, config):
        return config['Interaction_Map']

    def make_landscape(self, model):
        '''
        Helper function for intializing the landscape class
        '''
        ls_params = self.get_landscape_params(config = self.config)
        return landscape.Spatially_Implicit_Landscape(model = model,
                                                      width = ls_params['Width'],
                                                      height = ls_params['Height'],
                                                      thermal_profile_csv_fp = ls_params['Thermal_Database_fp'])

    def make_interaction_module(self, model):
        '''
        Retired: Helper function for making an interaction model between a predator and prey     
        '''
        interaction_map = self.get_interaction_map(config=self.config)
        return interaction.Interaction_Map(model = self, interaction_map=interaction_map)

    def logistic_population_density_function(self, global_population, total_area, carrying_capacity, growth_rate, threshold_density):
        """
        Computes local density based on global population using logistic scaling.

        Parameters:
        - global_population (int or array): Total population size.
        - total_area (int): Total hectares.
        - carrying_capacity (float): Maximum possible density per hectare.
        - growth_rate (float): Sensitivity of density changes.
        - threshold_density (float): Density at which changes have the largest effect.

        Returns:
        - local_density (float or array): Adjusted local density based on feedback.
        """
        avg_density = global_population / total_area
        adjusted_density = carrying_capacity / (1 + np.exp(-growth_rate * (avg_density - threshold_density)))
        return adjusted_density
    
    def calc_local_population_density(self, population_size, middle_range, max_density):
        # Define parameters
        total_area = self.landscape.landscape_size
        carrying_capacity = max_density  # max density per hectare
        growth_rate = 1  # scaling sensitivity
        threshold_density = middle_range  # middle of the range
        global_population = population_size 
        # Compute local densities
        new_local_density = self.logistic_population_density_function(global_population, total_area, carrying_capacity, growth_rate, threshold_density)
        return new_local_density
        
    ## Intialize populations and births
    def give_birth(self, species_name, agent_id, pos=None, parent=None):
        """
        Helper function - Adds new agents to the landscape
        """
        species_map = {
            "KangarooRat": (agents.KangarooRat, self.get_krat_params),
            "Rattlesnake": (agents.Rattlesnake, self.get_rattlesnake_params)
        }

        if species_name not in species_map:
            raise ValueError(f"Class for species: {species_name} does not exist")

        agent_class, param_func = species_map[species_name]
        agent_params = param_func(config=self.config)

        agent = agent_class(unique_id=agent_id, model=self, config=agent_params)
        if pos is not None:
            self.place_agent(agent, pos)
        self.schedule.add(agent)

    def initialize_populations(self, initial_agent_dictionary, spatially_explicit=False):
        '''
        Helper function in the landscape class used to intialize populations.
        Populations sizes should be a range of individuals per hectare
        '''
        agent_id = 0
        for hect in range(self.landscape.landscape_size):
            for species, initial_population_size_range in initial_agent_dictionary.items():
                start, stop = initial_population_size_range.start, initial_population_size_range.stop
                initial_pop_size = round(np.random.uniform(start, stop))
                # Need to scale population size by hectare
                for i in range(initial_pop_size):
                    if spatially_explicit:
                        # Code for future models. Not working right now.
                        x_hect = 0 
                        y_hect = 0
                        x = np.random.uniform(x_hect, x_hect + self.hectare_to_meter)
                        y = np.random.uniform(y_hect, y_hect + self.hectare_to_meter)
                        pos = (x,y)
                        self.give_birth(species_name=species, pos=pos, agent_id=agent_id)
                    else:
                        self.give_birth(species_name=species, agent_id=agent_id)
                    agent_id +=1
                    self.next_agent_id = agent_id+1
                #print(pos,agent_id)
                                    

    def randomize_snakes(self):
        '''
        helper function for self.step()

        puts snakes in a list and shuffles them
        '''
        snake_shuffle = list(self.schedule.agents_by_type[agents.Rattlesnake].values())
        #print(f'Snakes: {len(snake_shuffle)}')
        self.random.shuffle(snake_shuffle)
        return snake_shuffle
    
    def randomize_active_snakes(self):
        '''
        Helper function for self.step()

        Puts active snakes in a list and shuffles them
        '''
        # Filter only active snakes
        active_snakes = [snake for snake in self.schedule.agents_by_type[agents.Rattlesnake].values() if snake.active]
        
        # Shuffle the list of active snakes
        self.random.shuffle(active_snakes)
    
        return active_snakes 
    
    def randomize_krats(self):
        '''
        helper function for self.step()

        puts snakes in a list and shuffles them
        '''
        krat_shuffle = list(self.schedule.agents_by_type[agents.KangarooRat].values())
        #print(f'Krats: {len(krat_shuffle)}')
        self.random.shuffle(krat_shuffle)
        return krat_shuffle
    
    def randomize_active_krats(self):
        '''
        Helper function for self.step()

        Puts active Kangaroo Rats in a list and shuffles them
        '''
        # Filter only active KangarooRats
        active_krats = [krat for krat in self.schedule.agents_by_type[agents.KangarooRat].values() if krat.active]
        
        # Shuffle the list of active KangarooRats
        self.random.shuffle(active_krats)
    
        return active_krats
    
    def get_active_krat(self):
        active_krats = self.randomize_active_krats()
        if not active_krats:  
            return None  
        return np.random.choice(active_krats)
    
    def remove_dead_agents(self):
        '''
        Helper function: Create a list of agents to remove because they are dead
        '''
        dead_snakes = [snake for snake in self.schedule.agents_by_type[agents.Rattlesnake].values() if not snake.alive]
        for snake in dead_snakes:
            self.schedule.remove(snake)

        dead_krats = [krat for krat in self.schedule.agents_by_type[agents.KangarooRat].values() if not krat.alive]
        for krat in dead_krats:
            self.schedule.remove(krat)

    def too_many_agents_check(self):
        total_agents = len(self.schedule.agents_by_type[agents.KangarooRat].values()) + len(self.schedule.agents_by_type[agents.Rattlesnake].values())
        if total_agents > 50000:
           raise RuntimeError(f"Too many agents in the simulation: {total_agents}")

    def step(self):
        '''
        Main model step function used to run one step of the model.
        '''
        self.hour = self.landscape.thermal_profile['hour'].iloc[self.step_id]
        self.day = self.landscape.thermal_profile['day'].iloc[self.step_id]
        self.month = self.landscape.thermal_profile['month'].iloc[self.step_id]
        self.year = self.landscape.thermal_profile['year'].iloc[self.step_id]
        self.datacollector.collect(self)
        self.landscape.set_landscape_temperatures(step_id=self.step_id)
        # Snakes
        snake_shuffle = self.randomize_snakes()
        for snake in snake_shuffle:
            snake.step()
        snake_shuffle = self.randomize_snakes()
        # Krats
        krat_shuffle = self.randomize_krats()
        for krat in krat_shuffle:
            krat.step()
        krat_shuffle = self.randomize_krats()
        self.remove_dead_agents()
        self.step_id += 1  # Increment the step counter
        self.schedule.step()

    def run_model(self, step_count=None):
        max_steps = len(self.landscape.thermal_profile)-1
        if step_count is None:
            step_count = max_steps
        elif max_steps <= step_count:
            print(f'Step argument exceeds length of data. Using {max_steps} instead.')
            step_count=max_steps

        for i in range(step_count):
            self.step()
            

if __name__ ==  "__main__":
    pass