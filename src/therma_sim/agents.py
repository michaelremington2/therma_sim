#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd


# Rattlesnake temperature model
# https://journals.biologists.com/jeb/article/223/14/jeb223859/224523/The-effects-of-temperature-on-the-defensive

class Rattlesnake(mesa.Agent):
    '''
    Agent Class for rattlesnake predator agents.
        Rattlsnakes are sit and wait predators that forage on kangaroo rat agents
    '''
    def __init__(self, unique_id, model, initial_pos, initial_body_temperature=25, k=0.01,t_pref_min=18, t_pref_max=32, moore=False):
        super().__init__(unique_id, model)
        self.pos = initial_pos
        self.moore = moore

        # Behavioral profile
        self.behaviors = ['Rest', 'Thermoregulate', 'Forage']
        self.behavior_weights = self.random_make_behavioral_preference_weights(_test=True)
        self.utility_scores = self.generate_static_utility_vector_b1mh2()
        self._current_behavior = ''
        self.behavior_history = []
        # Microhabitat
        self._current_microhabitat = ''
        self.microhabitat_history = []
        # Temperature
        self._body_temperature = initial_body_temperature
        self.k = k
        self.t_pref_min = t_pref_min
        self.t_pref_max = t_pref_max
        self.body_temp_history = []

        # Agent is actively foraging
        self.active = True

    @property
    def current_behavior(self):
        return self._current_behavior

    @current_behavior.setter
    def current_behavior(self, value):
        self._current_behavior = value

    @property
    def current_microhabitat(self):
        return self._current_microhabitat

    @current_microhabitat.setter
    def current_microhabitat(self, value):
        self._current_microhabitat = value

    @property
    def body_temperature(self):
        return self._body_temperature

    @body_temperature.setter
    def body_temperature(self, value):
        self._body_temperature = value

    def cooling_eq_k(self, k, t_body, t_env, delta_t):
        return t_env+(t_body-t_env)*math.exp(-k*delta_t) # add time back in if it doesnt work
    
    def get_t_env(self, current_microhabitat):
        if current_microhabitat=='Burrow':
            t_env = self.model.landscape.get_property_attribute(property_name='Burrow_Temp', pos=self.pos)
        elif current_microhabitat=='Open':
            t_env = self.model.landscape.get_property_attribute(property_name='Open_Temp', pos=self.pos)
        elif current_microhabitat=='Shrub':
            t_env = self.model.landscape.get_property_attribute(property_name='Shrub_Temp', pos=self.pos)
        else:
            raise ValueError('Microhabitat Property Value cant be found')
        return t_env
    
    def update_body_temp(self, t_env, delta_t):
        old_body_temp = self.body_temperature
        self.body_temperature = self.cooling_eq_k(k=self.k, t_body=self.body_temperature, t_env=t_env, delta_t=delta_t)
        return


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
        weights = {'Rest': rest_weight,
                   'Forage': forage_weight,
                   'Thermoregulate': thermoregulate_weight}
        return weights
    
    ########################################################################################################
    #
    #### Choose microhabitat and behavior
    #
    ########################################################################################################
    
    def generate_static_utility_vector_mh1b2(self):
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
    
    def calculate_overall_utility_additive_mh1b2(self, utility_scores, mh_availability, behavior_preferences):
        '''
        Helper function for calculating the additive utility scores adjusted for microhabitat availability in the landscape
        Args:
            - utility_scores: un adjusted utility scores associated with the behavior dictionary
            - availability of habitat dictionary. An example would be
            availability = {
                'Shrub': 0.8,
                'Open': 0.2,
                'Burrow': 1.0
                }
        '''
        overall_utility = {}
        for habitat in utility_scores:
            habitat_utilities = []
            for behavior in utility_scores[habitat]:
                habitat_utilities.append(utility_scores[habitat][behavior] * behavior_preferences[behavior])
            overall_utility[habitat] = sum(habitat_utilities) * mh_availability[habitat]
        return overall_utility
    
    def simulate_decision_mh1b2(self, overall_utility, behaviors, utility_scores):
        '''
        Behavior function for simulating a snakes behavioral decision of what based on utility 
        Args:
            - overal_utility: utility scores associated with the microhabitat adjusted for landscape availability
            - behaviors: List of available behaviors to choose from
            - utility_scores: un adjusted utility scores associated with the behavior dictionary
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
        self.current_behavior=behavior
        self.current_microhabitat=microhabitat
        return

    def generate_static_utility_vector_b1mh2(self):
        '''
        This function returns a static dictionary of unadjusted expected utility scores.
        '''
        utility_scores = {
            'Rest': {
                'Shrub': 0,
                'Open': 0,
                'Burrow': 5,
            },
            'Forage': {
                'Shrub': 4,
                'Open': 4,
                'Burrow': 0,
            },
            'Thermoregulate': {
                'Shrub': 4,
                'Open': 4,
                'Burrow': 2,
            }
        }
        return utility_scores

    def calculate_overall_utility_additive_b1mh2(self, utility_scores, mh_availability, behavior_preferences):
        '''
        Helper function for calculating the additive utility scores adjusted for behavior availability in the landscape.
        '''
        overall_utility = {}
        for behavior in utility_scores:
            behavior_utilities = []
            for habitat in utility_scores[behavior]:
                behavior_utilities.append(utility_scores[behavior][habitat] * behavior_preferences[behavior])
            overall_utility[behavior] = sum(behavior_utilities) * mh_availability[habitat]
        return overall_utility

    def simulate_decision_b1mh2(self, overall_utility, microhabitats, utility_scores):
        '''
        Behavior function for simulating a snake's behavioral decision based on utility.
        Args:
            - overall_utility: utility scores associated with the behavior adjusted for landscape availability
            - microhabitats: List of available microhabitats to choose from
            - utility_scores: unadjusted utility scores associated with the microhabitat dictionary
        Returns:
            - behavior: (str) label of the behavior the organism chooses
            - microhabitat: (str) the microhabitat that the organism chooses
        '''
        # Choose behavior based on overall utility
        behavior_probs = np.array(list(overall_utility.values()))
        behavior_probs /= np.sum(behavior_probs)
        behavior = np.random.choice(list(overall_utility.keys()), p=behavior_probs)

        # Choose microhabitat within the selected behavior
        microhabitat_utilities = [utility_scores[behavior][microhabitat] for microhabitat in microhabitats]
        microhabitat_probs = np.array(microhabitat_utilities) / np.sum(microhabitat_utilities)
        microhabitat = np.random.choice(microhabitats, p=microhabitat_probs)

        self.current_behavior = behavior
        self.current_microhabitat = microhabitat
        return 
    
    def log_choice(self, microhabitat, behavior, body_temp):
        '''
        Helper function for generating a list of the history of microhabitat and behavior
        '''
        self.behavior_history.append(behavior)
        self.microhabitat_history.append(microhabitat)
        self.body_temp_history.append(body_temp)

    def log_choice(self, microhabitat, behavior, body_temp):
        '''
        Helper function for generating a list of the history of microhabitat and behavior,
        ensuring the history lists are only 10 elements long.
        '''
        self.behavior_history.append(behavior)
        self.microhabitat_history.append(microhabitat)
        self.body_temp_history.append(body_temp)

        # Ensure each list is only 10 elements long
        if len(self.behavior_history) > 10:
            self.behavior_history.pop(0)
        if len(self.microhabitat_history) > 10:
            self.microhabitat_history.pop(0)
        if len(self.body_temp_history) > 10:
            self.body_temp_history.pop(0)
            
    ###########################################################
    #
    #### Main Functions
    #
    ###########################################################
    
    def is_starved(self):
        pass

    def move(self):
        pass

    def step(self, availability_dict):
        self.move()
        overall_utility = self.calculate_overall_utility_additive_b1mh2(utility_scores = self.utility_scores, mh_availability = availability_dict, behavior_preferences=self.behavior_weights)
        self.simulate_decision_b1mh2(microhabitats = self.model.landscape.microhabitats, utility_scores=self.utility_scores, overall_utility=overall_utility)
        t_env = self.get_t_env(current_microhabitat = self.current_microhabitat)
        self.update_body_temp(t_env, delta_t=self.model.delta_t)
        self.log_choice(behavior=self.current_behavior, microhabitat=self.current_microhabitat, body_temp=self.body_temperature)
        #self.print_history()


class KangarooRat(mesa.Agent):
    '''
    Agent Class for kangaroo rat agents.
      A kangaroo rat agent is one that is at the bottom of the trophic level and only gains energy through foraging from the 
    seed patch class.
    '''
    def __init__(self, unique_id, model, initial_pos, moore=False):
        super().__init__(unique_id, model)
        self.initial_pos = initial_pos
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
    
    def move(self):
        pass

    def step(self):
        pass



# class SeedPatch(mesa.Agent):
#     def __init__(self, unique_id, model):
#         super().__init__(unique_id, model)

#     def step(self):
#         pass