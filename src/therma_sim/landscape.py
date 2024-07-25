#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd

class Landscape(mesa.space.MultiGrid):
    '''
    Landscape class used in the model class to make the virtual landscape agents use
    Args:
        width (int): Width of the landscape.
        height (int): Height of the landscape.
        torus (bool): If true, the edges of the grid wrap around.
    
    '''
    def __init__(self, thermal_profile_csv, width: int, height: int, torus: bool):
        super().__init__(width, height, torus)
        self.thermal_profile =pd
