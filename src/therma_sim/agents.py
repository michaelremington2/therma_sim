#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd

class Kangaroo_Rat(mesa.Agent):
    '''
    Agent Class for kangaroo rat agents.
      A kangaroo rat agent is one that is at the bottom of the trophic level and only gains energy through foraging from the 
    seed patch class.
    '''
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        pass

class Rattlesnake(mesa.Agent):
    '''
    Agent Class for rattlesnake predator agents.
        Rattlsnakes are sit and wait predators that forage on kangaroo rat agents
    '''
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        pass