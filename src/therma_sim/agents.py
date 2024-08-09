#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd


class Rattlesnake(mesa.Agent):
    '''
    Agent Class for rattlesnake predator agents.
        Rattlsnakes are sit and wait predators that forage on kangaroo rat agents
    '''
    def __init__(self, unique_id, model, pos, moore=False):
        super().__init__(unique_id, model)
        self.pos = pos
        self.moore = moore

        # Behavioral profile
        self.behaviors = ['Rest', 'Thermoregulate', 'Forage']
        self.behavior_weights = self.random_make_behavioral_preference_weights(_test=True)
        self.utility_scores = self.generate_static_utility_vector()
        self._current_behavior = None

        # Agent is actively foraging
        self.active = True

    @property
    def current_behavior(self):
        return self._current_behavior

    @current_behavior.setter
    def current_behavior(self, value):
        self._current_behavior = value

    def random_make_behavioral_preference_weights(self, _test=False):
        '''
        Creates the vector of behavior preferences at random (uniform distribution).
        Args:
            - None
        Funcions method is used in:
            - Init
        '''
        if _test:
            rest_weight = 0.5
            forage_weight = 0.25
            thermoregulate_weight = 0.25
        else:
            rest_weight = np.random.uniform(0.4, 0.6)
            forage_weight = np.random.uniform(0.2, 0.4)
            thermoregulate_weight = 1 - rest_weight - thermoregulate_weight
        assert math.isclose(sum([rest_weight, forage_weight, thermoregulate_weight]), 1, rel_tol=1e-9)
        return [rest_weight, forage_weight, thermoregulate_weight]
    
    def generate_static_utility_vector(self):
        '''
        This function returns a static dictionary of unadjusted expected utility scores.
        Funcions method is used in:
            - Init
        '''
        utility_scores = {
            'Shrub': {
                'Rest': 0,
                'Forage': 4,
                'Thermoregulate': 4,
            },
            'Open': {
                'Rest': 0,
                'Forage': 4,
                'Thermoregulate': 4,
            },
            'Burrow': {
                'Rest': 5,
                'Forage': 0,
                'Thermoregulate': 2
            }
        }
        return utility_scores
    
    def calculate_overall_utility_additive(self, utility_scores, availability, weights):
        '''
        Helper function for calculating the additive utility scores adjusted for microhabitat availability in the landscape
        '''
        overall_utility = {}
        for habitat in utility_scores:
            habitat_utilities = []
            for behavior in utility_scores[habitat]:
                habitat_utilities.append(utility_scores[habitat][behavior] * weights[behavior])
            overall_utility[habitat] = sum(habitat_utilities) * availability[habitat]
        return overall_utility
    
    def simulate_decision(self, overall_utility, behaviors, utility_scores):
        '''
        Behavior function for simulating a snakes behavioral decision of what based on utility 
        Args:
            - overal_utility: utility scores associated with the microhabitat adjusted for landscape availability
            - behaviors: List of available behaviors to choose from
            - utility_scores: un adjusted utility scores associated with the behavior
        Returns:
            - behavior: (str) label of the behavior the organism chooses
            - microhabitat: (str) the microhabitat that the 
        '''
        # Choose microhabitat based on overall utility
        habitat_probs = np.array(list(overall_utility.values()))
        habitat_probs /= np.sum(habitat_probs)
        microhabitat = np.random.choice(list(overall_utility.keys()), p=habitat_probs)
        
        # Choose behavior within the selected microhabitat
        behavior_utilities = [utility_scores[microhabitat][behavior] for behavior in behaviors]
        behavior_probs = np.array(behavior_utilities) / np.sum(behavior_utilities)
        behavior = np.random.choice(behaviors, p=behavior_probs)
        # Potentially think about nesting microhabitat in behavior rather than behavior in microhabitat to see if it makes a difference
        return microhabitat, behavior
    
    def is_starved(self):
        pass

    def step(self):
        pass

class KangarooRat(mesa.Agent):
    '''
    Agent Class for kangaroo rat agents.
      A kangaroo rat agent is one that is at the bottom of the trophic level and only gains energy through foraging from the 
    seed patch class.
    '''
    def __init__(self, unique_id, model, pos, moore=False):
        super().__init__(unique_id, model)
        self.pos = pos
        self.moore = moore
        

        # Agent is actively foraging
        self.active = True

    def random_make_behavioral_preference_weights(self):
        '''
        Creates the vector of behavior preferences at random (uniform distribution).
        Args:
            - None
        Funcions method is used in:
            - Init
        '''
        rest_weight = np.random.uniform(0.4, 0.6)
        forage_weight = np.random.uniform(0.2, 0.4)
        thermoregulate_weight = 1 - rest_weight - thermoregulate_weight
        return [rest_weight, forage_weight, thermoregulate_weight]

    def step(self):
        pass



class SeedPatch(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        pass