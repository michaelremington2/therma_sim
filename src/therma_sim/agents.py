#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import metabolism
import behavior 
import birth_death


# Rattlesnake temperature model
# https://journals.biologists.com/jeb/article/223/14/jeb223859/224523/The-effects-of-temperature-on-the-defensive

class Rattlesnake(mesa.Agent):
    '''
    Agent Class for rattlesnake predator agents.
        Rattlsnakes are sit and wait predators that forage on kangaroo rat agents
    '''
    def __init__(self, unique_id, model, hourly_survival_probability = 1, age=0, initial_pop=False, initial_pos=None, config=None):
        super().__init__(unique_id, model)
        self.initial_pop = initial_pop
        self.pos = initial_pos
        self.snake_config = config
        self.sex = np.random.choice(['Male', 'Female'], 1)[0]
        self.age = age
        if config is not None:
            self.metabolism = metabolism.EctothermMetabolism(org=self,
                                                             model=self.model,
                                                             initial_metabolic_state=self.snake_config['initial_calories'],
                                                             max_meals = self.snake_config['max_meals'],
                                                             X1_mass=self.snake_config['X1_mass'],
                                                             X2_temp=self.snake_config['X2_temp'],
                                                             X3_const=self.snake_config['X3_const'])
            self.mass = self.set_mass(body_size_range=self.snake_config['Body_sizes'])
            self.max_age_steps = self.snake_config['max_age']*self.model.steps_per_year
            self.moore = self.snake_config['moore']
            self.brumation_months = self.snake_config['brumination_months']
            self.hourly_survival_probability = hourly_survival_probability
            # Temperature
            self.delta_t = self.snake_config['delta_t']
            self._body_temperature = self.snake_config['Initial_Body_Temperature']
            self.k = self.snake_config['k']
            self.t_pref_min = self.snake_config['t_pref_min']
            self.t_pref_max = self.snake_config['t_pref_max']
            self.t_opt = self.snake_config['t_opt']
            self.strike_performance_opt = self.snake_config['strike_performance_opt']
            self.max_thermal_accuracy = self.snake_config['max_thermal_accuracy'] #Replace this with an input value later
            # Birth Module
            self.reproductive_age_steps = self.set_reproductive_age_steps(reproductive_age_years = self.snake_config['reproductive_age_years'])
            self.birth_death_module = self.initiate_birth_death_module(birth_config=self.snake_config['birth_death_module'], initial_pop=self.initial_pop)
        else:
            # Initialize attributes to None or defaults
            self.metabolism = None
            self.mass = None
            self.moore = False
            self.brumation_months = []
            self.hourly_survival_probability = 1
            self.delta_t = None
            self._body_temperature = None
            self.k = None
            self.t_pref_min = None
            self.t_pref_max = None
            self.t_opt = None
            self.strike_performance_opt = None
            self.birth_death_module = None
            self.max_age = 20

        # Behavioral profile
        self.behaviors = ['Rest', 'Thermoregulate', 'Forage']
        self.behavior_module = behavior.EctothermBehavior(snake=self)
        self._current_behavior = ''
        self.behavior_history = []
        self.activity_coefficients = {'Rest':1,
                                      'Thermoregulate':2,
                                      'Forage':2}

        # Microhabitat
        self._current_microhabitat = ''
        self.microhabitat_history = []
        self.body_temp_history = []

        # Agent logisic checks
        self._active = False
        self._alive = True


    @property
    def species_name(self):
        """Returns the class name as a string."""
        return self.__class__.__name__

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
        elif self.alive==False:
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

    def set_reproductive_age_steps(self, reproductive_age_years):
        """
        Sets the reproductive age in simulation steps if not already set.

        Args:
            reproductive_age_years (float): Age at which reproduction begins, in years.

        Returns:
            int: The reproductive age in simulation steps.
        """
        existing_value = self.model.get_static_variable(species=self.species_name, variable_name='reproductive_age_steps')

        if existing_value is None:
            reproductive_age_steps = reproductive_age_years * self.model.steps_per_year
            self.model.set_static_variable(species=self.species_name, variable_name='reproductive_age_steps', value=reproductive_age_steps)
            return reproductive_age_steps  
        return existing_value


    def initiate_birth_death_module(self, birth_config, initial_pop):
        '''
        Helper function for setting up bith module for organisms
        '''
        return birth_death.Birth_Death_Module(model=self.model, agent=self,
                mean_litter_size=birth_config["mean_litter_size"], std_litter_size=birth_config["std_litter_size"],
                upper_bound_litter_size=birth_config["upper_bound_litter_size"], lower_bound_litter_size=birth_config["lower_bound_litter_size"],
                litters_per_year=birth_config["litters_per_year"],
                birth_hazard_rate=birth_config["birth_hazard_rate"], death_hazard_rate=birth_config["death_hazard_rate"],
                initial_pop=initial_pop)
    
    def report_data(self):
        """
        Extracts rattlesnake-specific data into a list for CSV logging.
        """
        return [
            self.model.step_id,
            self.model.hour,
            self.model.day,
            self.model.month,
            self.model.year,
            self.unique_id,
            self.active,
            self.alive,
            self.current_behavior,
            self.current_microhabitat,
            self.body_temperature,
            self.metabolism.metabolic_state,
            self.behavior_module.handling_time,
            self.behavior_module.attack_rate,
            self.behavior_module.prey_density,
            self.behavior_module.prey_consumed
        ]

    def set_mass(self, body_size_range):
        mass = np.random.uniform(min(body_size_range), max(body_size_range))
        return mass

    def generate_random_pos(self):
        hectare_size = 100
        x = np.random.uniform(0, hectare_size)
        y = np.random.uniform(0, hectare_size)
        self.pos = (x, y)
    
    def activate_snake(self):
        if self.current_behavior in ['Thermoregulate', 'Forage']:
            self.active = True
        else:
            self.active = False

    def get_activity_coefficent(self):
        return self.activity_coefficients[self.current_behavior]

    def cooling_eq_k(self, k, t_body, t_env, delta_t):
        exp_decay = self.model.get_static_variable(species=self.species_name, variable_name='exp_decay')
        if exp_decay is None:
            exp_decay = math.exp(-k*delta_t)
            self.model.set_static_variable(species=self.species_name, variable_name='exp_decay', value=exp_decay)
        return t_env+(t_body-t_env)*exp_decay
    
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
    
    def update_body_temp(self, t_env):
        old_body_temp = self.body_temperature
        self.body_temperature = self.cooling_eq_k(k=self.k, t_body=self.body_temperature, t_env=t_env, delta_t=self.delta_t)
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

    def random_death(self):
        '''
        Helper function - represents a background death rate from other preditors, disease, vicious cars, etc
        '''
        self.alive = np.random.choice([True, False], p=[self.hourly_survival_probability, 1 - self.hourly_survival_probability])

    # def check_reproductive_status(self):
    #     """
    #     This function checks if a rattlesnake is reproductive based on age and sex.
    #     It should be called inside `agent_checks()`.
    #     """
    #     if self.sex == 'Female' and self.age >= self.reproductive_age_steps:
    #         self._reproductive_agent = True  
    #     else:
    #         self._reproductive_agent = False 
    
    def is_starved(self):
        '''
        Internal state function to switch the state of the agent from alive to dead when their energy drops below 0.
        '''
        if self.metabolism.metabolic_state<=0:
            self.alive = False

    def move(self):
        pass

    def agent_checks(self):
        '''
        Helper function run in the step function to run all functions t hat are binary checks of the individual to manage its state of being active, alive, or giving birth.
        '''
        self.is_starved()
        self.random_death()
        self.activate_snake()

    def simulate_decision(self):
        '''
        Function to facilitate agents picking a behavior and microhabitat
        '''
        self.behavior_module.step()
        t_env = self.get_t_env(current_microhabitat = self.current_microhabitat)
        self.update_body_temp(t_env)
        self.metabolism.cals_lost(mass=self.mass, temperature=self.body_temperature, activity_coefficient=self.activity_coefficients[self.current_behavior])

    def step(self):
        self.agent_checks()
        self.simulate_decision()
        self.birth_death_module.step()
        self.age += 1
        #self.log_choice(behavior=self.current_behavior, microhabitat=self.current_microhabitat, body_temp=self.body_temperature)
        #print(f'Metabolic State {self.metabolism.metabolic_state}')
        #self.print_history()


class KangarooRat(mesa.Agent):
    '''
    Agent Class for kangaroo rat agents.
      A kangaroo rat agent is one that is at the bottom of the trophic level and only gains energy through foraging from the 
    seed patch class.
    '''
    def __init__(self, unique_id, model, age=0, initial_pop=False, initial_pos=None, config=None):
        super().__init__(unique_id, model)
        self.initial_pop = initial_pop
        self.pos = initial_pos
        self.krat_config = config
        self.sex = np.random.choice(['Male', 'Female'], 1)[0]
        self.age = age
        if self.krat_config is not None:
            self.active_hours = self.krat_config['active_hours']
            self.mass = self.set_mass(body_size_range=self.krat_config['Body_sizes'])
            self.max_age_steps = self.krat_config['max_age']*self.model.steps_per_year
            self.moore = self.krat_config['moore']
            self.hourly_survival_probability = self.bernouli_trial_hourly(annual_probability=self.krat_config['annual_survival_probability'])
            self.reproductive_age_steps = int(self.krat_config['reproductive_age_years']*self.model.steps_per_year)
            self.birth_death_module = self.initiate_birth_death_module(birth_config=self.krat_config['birth_death_module'], initial_pop=self.initial_pop)
        else:
            self.active_hours = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
            self.mass = 10
            self.max_age = 6
            self.moore = True
            self.hourly_survival_probability = 1
            self.birth_death_module = None
        # Agent is actively foraging
        self._active = False
        self._alive = True

    @property
    def species_name(self):
        """Returns the class name as a string."""
        return self.__class__.__name__

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

    
    @property
    def reproductive_agent(self):
        return self._reproductive_agent

    @reproductive_agent.setter
    def reproductive_agent(self, value):
        self._reproductive_agent = value  # Ensure manual setting works

    def report_data(self):
        """
        Extracts kangaroo rat-specific data into a list for CSV logging.
        """
        return [
            self.model.step_id,
            self.model.hour,
            self.model.day,
            self.model.month,
            self.model.year,
            self.unique_id,
            self.alive,
            self.active
        ]

    def check_reproductive_status(self):
        """
        This function checks if a rattlesnake is reproductive based on age and sex.
        It should be called inside `agent_checks()`.
        """
        if self.sex == 'Female' and self.age_steps >= self.reproductive_age_steps:
            self._reproductive_agent = True  
        else:
            self._reproductive_agent = False  

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

    def bernouli_trial_hourly(self, annual_probability):
        '''
        Used to calculate hourly probability of survival
        '''
        P_H = annual_probability ** (1 / self.model.steps_per_year)
        return P_H

    def random_death(self):
        '''
        Helper function - represents a background death rate from other preditors, disease, vicious cars, etc
        '''
        self.alive = np.random.choice([True, False], p=[self.hourly_survival_probability, 1 - self.hourly_survival_probability])

    def initiate_birth_death_module(self, birth_config, initial_pop):
        '''
        Helper function for setting up bith module for organisms
        '''
        return birth_death.Birth_Death_Module(model=self.model, agent=self,
                mean_litter_size=birth_config["mean_litter_size"], std_litter_size=birth_config["std_litter_size"],
                upper_bound_litter_size=birth_config["upper_bound_litter_size"], lower_bound_litter_size=birth_config["lower_bound_litter_size"],
                litters_per_year=birth_config["litters_per_year"],
                birth_hazard_rate=birth_config["birth_hazard_rate"], death_hazard_rate=birth_config["death_hazard_rate"],
                initial_pop=initial_pop)

    def von_mises_move(self, current_pos):
        direction = np.random.vonmises()
        
    def move(self):
        pass

    def step(self):
        self.random_death()
        self.activate_krat(hour=self.model.hour)
        self.birth_death_module.step()
        self.age += 1



class SeedPatch(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        pass