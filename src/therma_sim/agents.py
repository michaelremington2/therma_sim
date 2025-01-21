#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
from . import metabolism
from . import behavior 
from . import birth


# Rattlesnake temperature model
# https://journals.biologists.com/jeb/article/223/14/jeb223859/224523/The-effects-of-temperature-on-the-defensive

class Rattlesnake(mesa.Agent):
    '''
    Agent Class for rattlesnake predator agents.
        Rattlsnakes are sit and wait predators that forage on kangaroo rat agents
    '''
    def __init__(self, unique_id, model, initial_pos, snake_config=None):
        super().__init__(unique_id, model)
        self.pos = initial_pos
        self.snake_config = snake_config
        self.sex = np.random.choice(['Male', 'Female'], 1)[0]
        if snake_config is not None:
            self.metabolism = metabolism.EctothermMetabolism(initial_metabolic_state=self.snake_config['initial_calories'],
                                                            X1_mass=self.snake_config['X1_mass'],
                                                            X2_temp=self.snake_config['X2_temp'],
                                                            X3_const=self.snake_config['X3_const'])
            self.mass = self.set_mass(body_size_range=self.snake_config['Body_sizes'])
            self.moore = self.snake_config['moore']
            self.brumation_months = self.snake_config['brumination_months']
            self.background_death_probability = self.snake_config['background_death_probability']
            # Temperature
            self.delta_t = self.snake_config['delta_t']
            self._body_temperature = self.snake_config['Initial_Body_Temperature']
            self.k = self.snake_config['k']
            self.t_pref_min = self.snake_config['t_pref_min']
            self.t_pref_max = self.snake_config['t_pref_max']
            self.t_opt = self.snake_config['t_opt']
            self.strike_performance_opt = self.snake_config['strike_performance_opt']
            self.max_thermal_accuracy = 5 #Replace this with an input value later
            # Birth Module
            self.birth_module = self.initiate_birth_module(birth_config=self.snake_config['birth_module'])
        else:
            # Initialize attributes to None or defaults
            self.metabolism = None
            self.mass = None
            self.moore = False
            self.brumation_months = []
            self.background_death_probability = 0.0
            self.delta_t = None
            self._body_temperature = None
            self.k = None
            self.t_pref_min = None
            self.t_pref_max = None
            self.t_opt = None
            self.strike_performance_opt = None
            self.birth_module = None

        # Behavioral profile
        self.behaviors = ['Rest', 'Thermoregulate', 'Forage']
        self.behavior_module = behavior.EctothermBehavior(snake=self)
        self._current_behavior = ''
        self.behavior_history = []
        self.activity_coefficients = {'Rest':1,
                                      'Thermoregulate':1,
                                      'Forage':1.5}

        # Microhabitat
        self._current_microhabitat = ''
        self.microhabitat_history = []
        self.body_temp_history = []

        # Agent logisic checks
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
        if self.model.month in self.brumation_months:
            self._active = False
        else:
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
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    def initiate_birth_module(self, birth_config):
        '''
        Helper function for setting up bith module for organisms
        '''
        months_since_last_litter = np.random.choice(range(0,12))
        return birth.Birth_Module(model=self.model, agent=self,
                frequency=birth_config["frequency"], mean_litter_size=birth_config["mean_litter_size"], std_litter_size=birth_config["std_litter_size"],
                upper_bound_litter_size=birth_config["upper_bound_litter_size"], lower_bound_litter_size=birth_config["upper_bound_litter_size"],
                litters_per_year=birth_config["litters_per_year"], months_since_last_litter=months_since_last_litter,
                partuition_months=birth_config["partuition_months"])

    def set_mass(self, body_size_range):
        mass = np.random.uniform(min(body_size_range), max(body_size_range))
        return mass

    def generate_random_pos(self):
        hectare_size = 100
        x = np.random.uniform(0, hectare_size)
        y = np.random.uniform(0, hectare_size)
        self.pos = (x, y)
    
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
            t_env = self.model.landscape.burrow_temperature
        elif current_microhabitat=='Open':
            t_env = self.model.landscape.open_temperature
        # elif current_microhabitat=='Shrub':
        #     t_env = self.model.landscape.get_property_attribute(property_name='Shrub_Temp', pos=self.pos)
        else:
            raise ValueError('Microhabitat Property Value cant be found')
        return t_env
    
    def update_body_temp(self, t_env, delta_t):
        old_body_temp = self.body_temperature
        self.body_temperature = self.cooling_eq_k(k=self.k, t_body=self.body_temperature, t_env=t_env, delta_t=delta_t)
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
    
    def is_starved(self):
        '''
        Internal state function to switch the state of the agent from alive to dead when their energy drops below 0.
        '''
        if self.metabolism.metabolic_state<=0:
            self.alive = False

    def random_death(self):
        '''
        Helper function - represents a background death rate from other preditors, disease, vicious cars, etc
        '''
        random_val = np.random.random()
        if random_val <= self.background_death_probability:
            self.alive = False

    def move(self):
        pass

    def step(self):
        self.random_death()
        self.is_starved()
        self.activate_snake()
        self.birth_module.step()
        self.move()
        #self.generate_random_pos()
        availability = self.model.landscape.get_mh_availability_dict(pos=self.pos)
        overall_utility = self.utility_module.calculate_overall_utility_additive_b1mh2(utility_scores = self.utility_scores, mh_availability = availability, behavior_preferences=self.behavior_weights)
        self.current_behavior = self.behavior_module.choose_behavior()
        self.current_microhabitat = None # Pick up hear tomorrow
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
    def __init__(self, unique_id, model, initial_pos, krat_config=None):
        super().__init__(unique_id, model)
        self.pos = initial_pos
        self.krat_config = krat_config
        self.sex = np.random.choice(['Male', 'Female'], 1)[0]
        if self.krat_config is not None:
            self.active_hours = self.krat_config['active_hours']
            self.mass = self.set_mass(body_size_range=self.krat_config['Body_sizes'])
            self.moore = self.krat_config['moore']
            self.background_death_probability = self.krat_config['background_death_probability']
            self.birth_module = self.initiate_birth_module(birth_config=self.krat_config['birth_module'])
        else:
            self.active_hours = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
            self.mass = 10
            self.moore = True
            self.background_death_probability = 0.0
            self.birth_module = None
        # Agent is actively foraging
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
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    def generate_random_pos(self):
        hectare_size = 100
        x = np.random.uniform(0, hectare_size)
        y = np.random.uniform(0, hectare_size)
        self.pos = (x, y)

    def set_mass(self, body_size_range):
        mass = np.random.uniform(min(body_size_range), max(body_size_range))
        return mass
    
    def activate_krat(self, hour):
        if hour in self.active_hours:
            self.active = True
        else:
            self.active = False

    def initiate_birth_module(self, birth_config):
        '''
        Helper function for setting up bith module for organisms
        '''
        months_since_last_litter = np.random.choice(range(0,3))
        return birth.Birth_Module(model=self.model, agent=self,
                frequency=birth_config["frequency"], mean_litter_size=birth_config["mean_litter_size"], std_litter_size=birth_config["std_litter_size"],
                upper_bound_litter_size=birth_config["upper_bound_litter_size"], lower_bound_litter_size=birth_config["upper_bound_litter_size"],
                litters_per_year=birth_config["litters_per_year"], months_since_last_litter=months_since_last_litter,
                partuition_months=birth_config["partuition_months"])
    
    def random_death(self):
        '''
        Helper function - represents a background death rate from other preditors, disease, vicious cars, etc
        '''
        random_val = np.random.random()
        if random_val <= self.background_death_probability:
            self.alive = False

    def von_mises_move(self, current_pos):
        direction = np.random.vonmises()
        
    
    def move(self):
        pass

    def step(self):
        hour = self.model.landscape.thermal_profile['hour'].iloc[self.model.step_id]
        self.random_death()
        self.activate_krat(hour=hour)
        self.birth_module.step()
        if self.active:
            self.generate_random_pos()



class SeedPatch(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        pass