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
warnings.filterwarnings("ignore")

class ThermaSim(mesa.Model):
    '''
    A model class to mange the kangaroorat, rattlesnake predator-prey interactions
    '''
    def __init__(self, initial_agents_dictionary,
                 thermal_profile_csv_fp, interaction_dist=0.068,
                 delta_t=70, width=50, height=50,
                 torus=False, moore=False, seed=None):

        self.initial_agents_dictionary = initial_agents_dictionary
        self.thermal_profile = pd.read_csv(thermal_profile_csv_fp, header=0)
        self.interaction_dist = interaction_dist
        self.delta_t = delta_t
        self.moore = moore
        # 
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
        self.landscape = self.make_landscape(model=self, thermal_profile_csv_fp = thermal_profile_csv_fp, width=width, height=height, torus=torus)
        ## Intialize agents
        self.initialize_populations(initial_agent_dictionary=self.initial_agents_dictionary)
        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters = {"Rattlesnakes": lambda m: m.schedule.get_type_count(agents.Rattlesnake),
                               "Krats": lambda m: m.schedule.get_type_count(agents.KangarooRat),},
            agent_reporters = {"Behavior": "current_behavior",
                               'Microhabitat': "current_microhabitat",
                               'Body_Temperature': "Body_Temp"})

    @property
    def time_of_day(self):
        return self._time_of_day

    @time_of_day.setter
    def time_of_day(self, value):
        self._time_of_day = value

    def make_landscape(self, model, thermal_profile_csv_fp, width, height, torus):
        '''
        Helper function for intializing the landscape class
        '''
        return landscape.Landscape(model = model, thermal_profile_csv_fp = thermal_profile_csv_fp, width=width, height=height, torus=torus)   
    
    def initialize_populations(self, initial_agent_dictionary):
        #### I set this up wrong fix later. Need to make it per hectare
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
                                                        moore = self.moore)
                            # place agent
                            self.landscape.place_agent(krat, pos)
                            self.schedule.add(krat)
                            agent_id += 1
                        elif species=='Rattlesnake':
                            # Create agent
                            snake = agents.Rattlesnake(unique_id = agent_id, 
                                                        model = self,
                                                        initial_pos = pos,
                                                        moore = self.moore)
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
        print(f'Snakes: {len(snake_shuffle)}')
        self.random.shuffle(snake_shuffle)
        return snake_shuffle
    
    def randomize_active_snakes(self):
        '''
        Helper function for self.step()

        Puts active snakes in a list and shuffles them
        '''
        # Filter only active KangarooRats
        active_snakes = [snake for snake in self.schedule.agents_by_type[agents.Rattlesnake].values() if snake.active]
        
        # Shuffle the list of active KangarooRats
        self.random.shuffle(active_snakes )
    
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
    
    def check_for_interaction(self, snake_point, krat_point, interaction_dist):
        '''
        Helper function in the interaction model to test if a snake agent interacts with a krat agent
        '''
        x1, y1 = snake_point
        x2, y2 = krat_point
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist <= interaction_dist:
            print('Strike!')
            print(f'Distance: {dist}')
    
    def interaction_model(self, interaction_dist):
        '''
        Main interaction model between agents. The model simulates point locations within a hectare then checks if a snake and a krat agent 
        are within striking distance of eachother if they are active.
        '''
        active_krats = self.randomize_active_krats()
        active_snakes = self.randomize_active_snakes()
        for snake in active_snakes:
            for krat in active_krats:
                snake_point = snake.point
                krat_point = krat.point
                self.check_for_interaction(snake_point=snake_point, krat_point=krat_point,successful_strike_dist=interaction_dist)
                # Parameter is from https://www.nature.com/articles/srep40412/tables/1


    def step(self):
        '''
        Main model step function used to run one step of the model.
        '''
        self.time_of_day = self.thermal_profile['hour'].iloc[self.step_id]
        #print(f'Hour: {self.time_of_day}')
        self.landscape.set_landscape_temperatures(step_id=self.step_id)
        #self.schedule.step()
        # Snakes
        snake_shuffle = self.randomize_snakes()
        for snake in snake_shuffle:
            availability = self.landscape.get_mh_availability_dict(pos=snake.pos)
            snake.step(availability_dict = availability)
            #agent.eat()
            #agent.maybe_die()
        snake_shuffle = self.randomize_snakes()
        # Krats
        krat_shuffle = self.randomize_krats()
        for krat in krat_shuffle:
            krat.step(hour = self.time_of_day)
        krat_shuffle = self.randomize_krats()
        self.interaction_model(interaction_dist=self.interaction_dist)
        self.datacollector.collect(self)
        self.step_id += 1  # Increment the step counter

    def run_model(self, step_count=1000):
        for i in range(step_count):
            self.step()
            

if __name__ ==  "__main__":
    pass