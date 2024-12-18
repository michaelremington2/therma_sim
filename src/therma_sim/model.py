#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import landscape
import agents 
import warnings
import logging
import json
from interaction import Interaction_Dynamics

warnings.filterwarnings("ignore")

class ThermaSim(mesa.Model):
    '''
    A model class to mange the kangaroorat, rattlesnake predator-prey interactions
    '''
    def __init__(self, config, seed=None):
        self.config = config
        self.initial_agents_dictionary = self.get_initial_population_params(config=self.config)
        self.step_id = 0
        self.running = True
        self.seed = seed
        self._time_of_day = None
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
        self.kr_rs_interaction_module = self.make_interaction_module(model=self)
        ## Intialize agents
        self.landscape.initialize_populations(initial_agent_dictionary=self.initial_agents_dictionary)
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
    def time_of_day(self):
        return self._time_of_day

    @time_of_day.setter
    def time_of_day(self, value):
        self._time_of_day = value

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

    def make_landscape(self, model):
        '''
        Helper function for intializing the landscape class
        '''
        ls_params = self.get_landscape_params(config = self.config)
        return landscape.Continous_Landscape(model = model, 
                                            thermal_profile_csv_fp = ls_params['Thermal_Database_fp'],
                                            width=ls_params['Width'],
                                            height=ls_params['Height'],
                                            torus=ls_params['torus'])

    def make_interaction_module(self, model):
        '''
        Helper function for making an interaction model between a predator and prey     
        '''
        interaction_params = self.get_interaction_params(config=self.config)
        
        # Loop through each interaction type and its parameters
        for pred_prey, params in interaction_params.items():
            # Unpack the interaction parameters
            interaction_distance = params['Interaction_Distance']
            prey_cals_per_gram = params['Prey_Cals_per_gram']
            digestion_efficiency = params['Digestion_Efficency']
            
            # Assume that the predator and prey names are derived from the interaction type
            predator_name, prey_name = pred_prey.split('_')
            
            # Return or store the Interaction_Dynamics object
            return Interaction_Dynamics(
                model=model,                  
                predator_name=predator_name, 
                prey_name=prey_name, 
                interaction_distance=interaction_distance, 
                calories_per_gram=prey_cals_per_gram, 
                digestion_efficiency=digestion_efficiency
        )
                
    def useful_check_landscape_functions(self, property_layer_name):
        self.landscape.visualize_property_layer('Open_Microhabitat')
        self.landscape.print_property_layer('Shrub_Microhabitat')
        self.landscape.check_landscape()

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
        self.time_of_day = self.landscape.thermal_profile['hour'].iloc[self.step_id]
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
        self.kr_rs_interaction_module.interaction_module()
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