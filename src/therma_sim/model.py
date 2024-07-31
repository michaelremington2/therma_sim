#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import landscape

class ThermaSim(mesa.Model):
    '''
    A model class to mange the kangaroorat, rattlesnake predator-prey interactions
    '''
    def __init__(self, initial_agents_dictionary,
                 thermal_profile_csv_fp, width=50, height=50,
                 torus=False, moore=False):

        self.initial_agents_dictionary = initial_agents_dictionary
        self.thermal_profile_csv_fp = thermal_profile_csv_fp
        # 
        self.step_id = 0
        self.running = True
        
        # Schedular 
        # Random activation, random by type Simultanious, staged
        self.schedule = mesa.time.RandomActivationByType(self)

        ## Make Initial Landscape
        self.landscape = self.make_landscape(model=self, thermal_profile_csv_fp = thermal_profile_csv_fp, width=width, height=height, torus=torus)

    def make_landscape(self, model, thermal_profile_csv_fp, width, height, torus):
        '''
        Helper function for intializing the landscape class
        '''
        return landscape.Landscape(model = model, thermal_profile_csv_fp = thermal_profile_csv_fp, width=width, height=height, torus=torus)   
    
    def step(self):
        '''
        Main model step function used to run one step of the model.
        '''
        self.landscape.set_landscape_temperatures(step_id=self.step_id)
        self.landscape.print_property_layer('Burrow_Temp')
        self.schedule.step()
        self.step_id += 1  # Increment the step counter

    def run_model(self, step_count=1000):
        for i in range(step_count):
            self.step()
            

if __name__ ==  "__main__":
    pass