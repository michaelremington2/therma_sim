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
        if seed is not None:
            np.random.seed(self.seed)
        
        # Schedular 
        # Random activation, random by type Simultanious, staged
        self.schedule = mesa.time.RandomActivationByType(self)

        ## Make Initial Landscape
        self.landscape = self.make_landscape(model=self)
        self.kr_rs_interaction_module = self.make_interaction_module(model=self)
        ## Intialize agents
        self.initialize_populations(initial_agent_dictionary=self.initial_agents_dictionary)
        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                'Step_ID': lambda m: m.step_id,
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
        return landscape.Landscape(model = model, 
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
    
    def initialize_populations(self, initial_agent_dictionary):
        rs_params = self.get_rattlesnake_params(config=self.config)
        kr_params = self.get_krat_params(config=self.config)
        agent_id = 0
        for species, initial_population_size in initial_agent_dictionary.items():
            for x in range(self.landscape.width):
                for y in range(self.landscape.height):
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
                            self.landscape.place_agent(krat, pos)
                            self.schedule.add(krat)
                            agent_id += 1
                        elif species=='Rattlesnake':
                            # Create agent
                            snake = agents.Rattlesnake(unique_id = agent_id, 
                                                        model = self,
                                                        initial_pos = pos,
                                                        snake_config = rs_params)
                            # place agent
                            self.landscape.place_agent(snake, pos)
                            self.schedule.add(snake)
                            agent_id += 1
                        else:
                            raise ValueError(f'Class for species: {species} DNE')
                
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
        # Create a list of agents to remove
        dead_snakes = [snake for snake in self.schedule.agents_by_type[agents.Rattlesnake].values() if not snake.alive]
        for snake in dead_snakes:
            self.schedule.remove(snake)

        dead_krats = [krat for krat in self.schedule.agents_by_type[agents.KangarooRat].values() if not krat.alive]
        for krat in dead_krats:
            self.schedule.remove(krat)


    def step(self):
        '''
        Main model step function used to run one step of the model.
        '''
        self.time_of_day = self.landscape.thermal_profile['hour'].iloc[self.step_id]
        self.datacollector.collect(self)
        #print(f'Hour: {self.time_of_day}')
        self.landscape.set_landscape_temperatures(step_id=self.step_id)
        #self.schedule.step()
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
        if step_count==None:
            step_count = len(self.landscape.thermal_profile)

        for i in range(step_count):
            self.step()
            

if __name__ ==  "__main__":
    pass