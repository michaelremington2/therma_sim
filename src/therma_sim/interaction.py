#!/usr/bin/python
import TPC

class Interaction_Dynamics(object):
    '''
    This is a static class that dictates the rules of interactions between a predator and prey.
    p
    '''
    def __init__(self,                  
                 predator_name: str, 
                 prey_name: str, 
                 interaction_distance: float, 
                 calories_per_gram: float, 
                 digestion_efficiency: float):
        self.predator_name = predator_name
        self.prey_name = prey_name
        self.interaction_distance = interaction_distance
        self.calories_per_gram = calories_per_gram
        self.digestion_efficiency = digestion_efficiency
