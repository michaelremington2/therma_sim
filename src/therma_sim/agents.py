#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import metabolism
import utility_functions as uf


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
        self.metabolism = metabolism.EctothermMetabolism(initial_metabolic_state=self.snake_config['initial_calories'],
                                                         X1_mass=self.snake_config['X1_mass'],
                                                         X2_temp=self.snake_config['X2_temp'],
                                                         X3_const=self.snake_config['X3_const'])
        self.mass = self.set_mass(body_size_range=self.snake_config['Body_sizes'])
        self.moore = self.snake_config['moore']

        # Behavioral profile
        self.behaviors = ['Rest', 'Thermoregulate', 'Forage']
        self.utility_module = uf.Utiility(snake=self)
        self.behavior_weights = self.random_make_behavioral_preference_weights(_test=True)
        self.utility_scores = self.utility_module.generate_static_utility_vector_b1mh2()
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
        mass = np.random.uniform(min(body_size_range), max(body_size_range))
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
        return self.activity_coefficients[self.current_behavior]

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
    
    def is_starved(self):
        '''
        Internal state function to switch the state of the agent from alive to dead when their energy drops below 0.
        '''
        if self.metabolism.metabolic_state<=0:
            self.alive = False

    def move(self):
        pass

    def step(self, availability_dict):
        self.is_starved()
        self.activate_snake()
        self.move()
        self.generate_random_point()
        overall_utility = self.utility_module.calculate_overall_utility_additive_b1mh2(utility_scores = self.utility_scores, mh_availability = availability_dict, behavior_preferences=self.behavior_weights)
        self.current_behavior, self.current_microhabitat = self.utility_module.simulate_decision_b1mh2(microhabitats = self.model.landscape.microhabitats, utility_scores=self.utility_scores, overall_utility=overall_utility)
        t_env = self.get_t_env(current_microhabitat = self.current_microhabitat)
        self.metabolism.cals_lost(mass=self.mass, temperature=self.body_temperature, activity_coeffcient=self.activity_coefficients[self.current_behavior])
        self.update_body_temp(t_env, delta_t=self.delta_t)
        self.log_choice(behavior=self.current_behavior, microhabitat=self.current_microhabitat, body_temp=self.body_temperature)
        #print(f'Metabolic State {self.metabolism.metabolic_state}')
        #self.print_history()


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
        mass = np.random.uniform(min(body_size_range), max(body_size_range))
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