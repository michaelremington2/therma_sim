#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import metabolism


# Rattlesnake temperature model
# https://journals.biologists.com/jeb/article/223/14/jeb223859/224523/The-effects-of-temperature-on-the-defensive

class Rattlesnake(mesa.Agent):
    '''
    Agent Class for rattlesnake predator agents.
        Rattlsnakes are sit and wait predators that forage on kangaroo rat agents
    '''
    def __init__(self, unique_id, model, initial_pos,  snake_config):
        super().__init__(unique_id, model)
        self.pos = initial_pos
        self.snake_config = snake_config
        self.metabolism = metabolism.EctothermMetabolism(X1_mass=self.snake_config['X1_mass'],
                                                         X2_temp=self.snake_config['X2_temp'],
                                                         X3_const=self.snake_config['X3_const'])
        self.mass = self.set_mass(body_size_range=self.snake_config['Body_sizes'])
        self.moore = self.snake_config['moore']

        # Behavioral profile
        self.behaviors = ['Rest', 'Thermoregulate', 'Forage']
        self.behavior_weights = self.random_make_behavioral_preference_weights(_test=True)
        self.utility_scores = self.generate_static_utility_vector_b1mh2()
        self._current_behavior = ''
        self.behavior_history = []
        self.activity_coefficients = {'Rest':1,
                                      'Thermoregulate':1,
                                      'Forage':1.5}

        # Microhabitat
        self._current_microhabitat = ''
        self.microhabitat_history = []

        # Temperature
        self.delta_t = self.snake_config['delta_t']
        self._body_temperature = self.snake_config['Initial_Body_Temperature']
        self.k = self.snake_config['k']
        self.t_pref_min = self.snake_config['t_pref_min']
        self.t_pref_max = self.snake_config['t_pref_max']
        self.t_opt = self.snake_config['t_opt']
        self.strike_performance_opt = self.snake_config['strike_performance_opt']
        self.body_temp_history = []

        # Agent logisic checks
        self._point = None
        self._active = False
        self._alive = True

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

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    @property
    def alive(self):
        return self._alive

    @alive.setter
    def alive(self, value):
        self._alive = value
        if value==False:
            self.active=False

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, value):
        self._point = value

    def set_mass(self, body_size_range):
        mass = np.random.uniform(body_size_range)
        return mass

    def generate_random_point(self):
        hectare_size = 100
        x = np.random.uniform(0, hectare_size)
        y = np.random.uniform(0, hectare_size)
        self.point = (x, y)
    
    def activate_snake(self):
        if self.current_microhabitat != 'Burrow':
            self.active = True
        else:
            self.active = False

    def get_activity_coefficent(self):
        return self.activity_coeffiecents[self.current_behavior]

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
            thermoregulate_weight = 1 - rest_weight - forage_weight
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
                adjusted_utility = utility_scores[behavior][habitat] * behavior_preferences[behavior] * mh_availability[habitat]
                behavior_utilities.append(adjusted_utility)
            overall_utility[behavior] = sum(behavior_utilities)
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
        # Normalize overall utility to get valid probabilities for behavior selection
        behavior_probs = np.array(list(overall_utility.values()), dtype=float)
        
        # Handle the case where all utilities are zero (assign equal probabilities)
        if np.sum(behavior_probs) == 0:
            behavior_probs = np.ones_like(behavior_probs) / len(behavior_probs)
        else:
            behavior_probs /= np.sum(behavior_probs)
        
        # Choose behavior based on overall utility probabilities
        behavior = np.random.choice(list(overall_utility.keys()), p=behavior_probs)

        # Choose microhabitat within the selected behavior
        microhabitat_utilities = [utility_scores[behavior][microhabitat] for microhabitat in microhabitats]
        
        # Ensure the microhabitat probabilities are floating-point for division
        microhabitat_probs = np.array(microhabitat_utilities, dtype=float)
        
        # Handle the case where all microhabitat utilities are zero (assign equal probabilities)
        if np.sum(microhabitat_probs) == 0:
            microhabitat_probs = np.ones_like(microhabitat_probs) / len(microhabitat_probs)
        else:
            microhabitat_probs /= np.sum(microhabitat_probs)
        
        # Choose microhabitat based on normalized probabilities
        microhabitat = np.random.choice(microhabitats, p=microhabitat_probs)

        # Update the object's current state
        self.current_behavior = behavior
        self.current_microhabitat = microhabitat
        
        return
    
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

    def print_history(self):
        print(self.behavior_history)
        print(self.microhabitat_history)
            
    ###########################################################
    #
    #### Main Functions
    #
    ###########################################################
    
    def is_starved(self):
        pass

    def move(self):
        pass

    def cals_spent(self):
        smr = self.metabolism.smr_eq(mass=self.mass, temperature=self.body_temperature)
        activity_coefficient = self.get_activity_coefficent()
        self.metabolism.hourly_energy_expendeture(smr=smr, activity_coefficient=activity_coefficient) 

    def step(self, availability_dict):
        self.activate_snake()
        
        self.move()
        self.generate_random_point()
        overall_utility = self.calculate_overall_utility_additive_b1mh2(utility_scores = self.utility_scores, mh_availability = availability_dict, behavior_preferences=self.behavior_weights)
        self.simulate_decision_b1mh2(microhabitats = self.model.landscape.microhabitats, utility_scores=self.utility_scores, overall_utility=overall_utility)
        t_env = self.get_t_env(current_microhabitat = self.current_microhabitat)
        self.update_body_temp(t_env, delta_t=self.delta_t)
        self.log_choice(behavior=self.current_behavior, microhabitat=self.current_microhabitat, body_temp=self.body_temperature)
        self.print_history()


class KangarooRat(mesa.Agent):
    '''
    Agent Class for kangaroo rat agents.
      A kangaroo rat agent is one that is at the bottom of the trophic level and only gains energy through foraging from the 
    seed patch class.
    '''
    def __init__(self, unique_id, model, initial_pos, krat_config):
        super().__init__(unique_id, model)
        self.initial_pos = initial_pos
        self.krat_config = krat_config
        self.active_hours = self.krat_config['active_hours']
        self.mass = self.set_mass(body_size_range=self.krat_config['Body_sizes'])
        self.moore = self.krat_config['moore']
        

        # Agent is actively foraging
        self._point = None
        self._active = False
        self._alive = True

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    @property
    def alive(self):
        return self._alive

    @alive.setter
    def alive(self, value):
        self._alive = value
        if value==False:
            self.active=False

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, value):
        if isinstance(value, tuple) and len(value) == 2:
            self._point = value
        else:
            raise ValueError("Point must be a tuple with two elements (x, y)")

    def generate_random_point(self):
        hectare_size = 100
        x = np.random.uniform(0, hectare_size)
        y = np.random.uniform(0, hectare_size)
        self.point = (x, y)

    def set_mass(self, body_size_range):
        mass = np.random.uniform(body_size_range)
        return mass
    
    def activate_krat(self, hour):
        if hour in self.active_hours:
            self.active = True
        else:
            self.active = False
    
    def move(self):
        pass

    def step(self, hour):
        self.activate_krat(hour=hour)
        if self.active:
            self.generate_random_point()



# class SeedPatch(mesa.Agent):
#     def __init__(self, unique_id, model):
#         super().__init__(unique_id, model)

#     def step(self):
#         pass