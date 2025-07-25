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
import json



# Rattlesnake temperature model
# https://journals.biologists.com/jeb/article/223/14/jeb223859/224523/The-effects-of-temperature-on-the-defensive



class Rattlesnake(mesa.Agent):
    '''
    Agent Class for rattlesnake predator agents.
        Rattlsnakes are sit and wait predators that forage on kangaroo rat agents
    '''
    def __init__(self, unique_id, model, mass,  hourly_survival_probability = 1, age=0, initial_pop=False, initial_pos=None, config=None,report_agent_data=False):
        super().__init__(unique_id, model)
        self.initial_pop = initial_pop
        self.pos = initial_pos
        self.snake_config = config
        self.sex = np.random.choice(['Male', 'Female'], 1)[0]
        self.age = age
        self.mass = mass
        self.report_agent_data = report_agent_data
        if config is not None:
            self.metabolism = metabolism.EctothermMetabolism(org=self,
                                                             model=self.model,
                                                             initial_metabolic_state=self.snake_config['initial_calories'],
                                                             max_meals = self.snake_config['max_meals'],
                                                             X1_mass=self.snake_config['X1_mass'],
                                                             X2_temp=self.snake_config['X2_temp'],
                                                             X3_const=self.snake_config['X3_const'])
            self.max_age_steps = self.snake_config['max_age']*self.model.steps_per_year
            self.moore = self.snake_config['moore']
            self.active_hours = self.snake_config['active_hours']
            self.brumation_period = self.get_brumination_period(file_path = self.snake_config['brumination']['file_path'])
            self.brumation_temp = self.snake_config['brumination']['temperature']
            self.hourly_survival_probability = hourly_survival_probability
            # Temperature
            self.delta_t = self.snake_config['delta_t']
            self._body_temperature = self.snake_config['Initial_Body_Temperature']
            self.k = self.snake_config['k']
            self.t_pref_min = self.snake_config['t_pref_min']
            self.t_pref_max = self.snake_config['t_pref_max']
            self.t_opt = self.snake_config['t_opt']
            self.ct_min = self.snake_config['voluntary_ct']['min_temp']
            self.ct_max = self.snake_config['voluntary_ct']['max_temp']
            self.ct_max_steps = self.snake_config['voluntary_ct']['max_steps']
            self.searching_behavior = self.snake_config['searching_behavior']
            self.utility_temperature = self.snake_config['behavioral_utility_temperature']
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
            self.brumation_period = []
            self.hourly_survival_probability = 1
            self.delta_t = None
            self._body_temperature = None
            self.k = None
            self.t_pref_min = None
            self.t_pref_max = None
            self.t_opt = None
            self.ct_min = None
            self.ct_max = None
            self.strike_performance_opt = None
            self.birth_death_module = None
            self.max_age = 20

        # Behavioral profile
        self.emergent_behaviors = ['Rest', 'Thermoregulate', 'Forage']
        self.search_counter = 0
        self.behavior_module = behavior.EctothermBehavior(snake=self)
        self._current_behavior = ''
        self._t_env = 0
        self.activity_coefficients = self.snake_config['behavior_activity_coefficients']

        # Microhabitat
        self._current_microhabitat = 'Burrow'
        # self.microhabitat_history = []
        # self.body_temp_history = []

        # Agent logisic checks
        self._active = False
        self._alive = True
        self._cause_of_death  = None


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
        if self.is_bruminating_today():
            self._body_temperature = self.brumation_temp
        else:
            self._body_temperature = value

    @property
    def t_env(self):
        return self._t_env

    @t_env.setter
    def t_env(self, value):
        self._t_env = value

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        # Force inactivity if agent is dead or in brumation
        if not self.alive:
            self._active = False
        elif self.current_microhabitat=='Burrow':
            self._active = False
        elif self.current_behavior == 'Rest':
            self._active = False
        elif self.current_behavior == 'Thermoregulate':
            self._active = True
        elif self.current_behavior == 'Forage':
            self._active = True
        elif self.is_bruminating_today():
            self._active = False
        else:
            self._active = bool(value)  # Ensures it's explicitly True/False

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
    def cause_of_death(self):
        """Returns the cause of death, if any."""
        return self._cause_of_death

    @cause_of_death.setter
    def cause_of_death(self, value):
        """Sets the cause of death only if the agent is dead."""
        if not self.alive:
            self._cause_of_death = value
        else:
            raise ValueError("Cannot set cause of death while agent is still alive.")
        
    def get_brumination_period(self, file_path):
        '''
        Function to read in the brumation period from a JSON file
        and convert date strings into (month, day) tuples.
        '''
        with open(file_path, 'r') as f:
            data = json.load(f)  # Expects format: {"Canada": ["01-01", "01-02", ...]}
        if len(data) != 1:
            raise ValueError("JSON must contain exactly one site entry.")
        site_name = list(data.keys())[0]
        date_strs = data[site_name]
        if site_name != self.model.landscape.site_name:
            raise ValueError(
                f"Site name in JSON file '{site_name}' does not match model site name '{self.model.landscape.site_name}'."
            )
        return [
            (int(date.split('-')[0]), int(date.split('-')[1]))
            for date in date_strs
        ]

    def is_bruminating_today(self):
        # print((self.model.month, self.model.day) in self.brumination_period)
        # print(f'Brumination Period: {self.brumination_period}')
        # print(f'Current Date: {(self.model.month, self.model.day)}')
        return (self.model.month, self.model.day) in self.brumation_period


    def set_reproductive_age_steps(self, reproductive_age_years):
        """
        Sets the reproductive age in simulation steps if not already set.

        Args:
            reproductive_age_years (float): Age at which reproduction begins, in years.

        Returns:
            int: The reproductive age in simulation steps.
        """
        reproductive_age_steps = reproductive_age_years * self.model.steps_per_year
        return reproductive_age_steps 


    def initiate_birth_death_module(self, birth_config, initial_pop):
        '''
        Helper function for setting up bith module for organisms
        '''
        return birth_death.Birth_Death_Module(model=self.model, agent=self,
                mean_litter_size=birth_config["mean_litter_size"], std_litter_size=birth_config["std_litter_size"],
                upper_bound_litter_size=birth_config["upper_bound_litter_size"], lower_bound_litter_size=birth_config["lower_bound_litter_size"],
                max_litters=birth_config["max_litters"],
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
            self.model.landscape.site_name,
            self.unique_id,
            self.active,
            self.alive,
            self.current_behavior,
            self.current_microhabitat,
            self.body_temperature,
            self.t_env,
            self.mass,
            self.metabolism.metabolic_state,
            self.behavior_module.handling_time,
            self.behavior_module.attack_rate,
            self.behavior_module.prey_density,
            self.behavior_module.prey_encountered, 
            self.behavior_module.prey_consumed
        ]

    # def set_mass(self, body_size_config):
    #     dist = body_size_config.get("distribution", "uniform")
    #     mean = body_size_config.get("mean")
    #     std = body_size_config.get("std")
    #     min_val = body_size_config.get("min")
    #     max_val = body_size_config.get("max")

    #     if dist == "normal":
    #         if None in (mean, std, min_val, max_val):
    #             raise ValueError("Normal distribution requires mean, std, min, and max.")
            
    #         # Convert to standard normal bounds
    #         a, b = (min_val - mean) / std, (max_val - mean) / std
    #         mass = truncnorm.rvs(a, b, loc=mean, scale=std)

    #     elif dist == "uniform":
    #         if None in (min_val, max_val):
    #             raise ValueError("Uniform distribution requires min and max.")
    #         mass = np.random.uniform(min_val, max_val)

    #     elif dist == "static":
    #         if None in (min_val, max_val):
    #             raise ValueError("Static distribution requires min and max.")
    #         mass = np.random.choice(np.arange(min_val, max_val + 1))

    #     else:
    #         raise ValueError(f"Unsupported distribution: {dist}")

    #    return mass


    def generate_random_pos(self):
        hectare_size = 100
        x = np.random.uniform(0, hectare_size)
        y = np.random.uniform(0, hectare_size)
        self.pos = (x, y)
    
    def activate_snake(self):
        if self.current_behavior in ['Thermoregulate', 'Forage', 'Search']:
            self.active = True
        else:
            self.active = False

    def get_activity_coefficent(self):
        return self.activity_coefficients[self.current_behavior]

    def cooling_eq_k(self, k, t_body, t_env, delta_t):
        exp_decay = math.exp(-k*delta_t)
        return t_env+(t_body-t_env)*exp_decay
    
    def get_t_env(self, current_microhabitat):
        if current_microhabitat=='Burrow':
            t_env = self.model.landscape.burrow_temperature
        elif current_microhabitat=='Open':
            t_env = self.model.landscape.open_temperature
        elif current_microhabitat=='Winter_Burrow':
            t_env = self.brumation_temp
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
        stay_alive = np.random.choice([True, False], p=[self.hourly_survival_probability, 1 - self.hourly_survival_probability])
        if stay_alive == False:
            self.alive = False
            self.cause_of_death = 'Random'
            self.model.logger.log_data(file_name = self.model.output_folder+"BirthDeath.csv",
                            data=self.birth_death_module.report_data(event_type='Death'))
            #self.model.remove_agent(self)
    
    def is_starved(self):
        '''
        Internal state function to switch the state of the agent from alive to dead when their energy drops below 0.
        '''
        if self.metabolism.metabolic_state<=0:
            self.alive = False
            self.cause_of_death = 'Starved'
            self.model.logger.log_data(file_name = self.model.output_folder+"BirthDeath.csv",
                                       data=self.birth_death_module.report_data(event_type='Death'))
            #self.model.remove_agent(self)

    def move(self):
        pass

    def collect_data(self):
        if self.report_agent_data:
            data = self.report_data()
            self.model.logger.log_data(file_name = self.model.output_folder+"Rattlesnake.csv", data=data)
            data = None

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
        self.t_env = self.get_t_env(current_microhabitat = self.current_microhabitat)
        self.update_body_temp(self.t_env)
        self.metabolism.cals_lost(mass=self.mass, temperature=self.body_temperature, activity_coefficient=self.activity_coefficients[self.current_behavior])

    def step(self):
        self.agent_checks()
        self.simulate_decision()
        self.birth_death_module.step()
        self.age += 1
        self.collect_data()
        #self.log_choice(behavior=self.current_behavior, microhabitat=self.current_microhabitat, body_temp=self.body_temperature)
        #print(f'Metabolic State {self.metabolism.metabolic_state}')
        #self.print_history()


class KangarooRat(mesa.Agent):
    '''
    Agent Class for kangaroo rat agents.
      A kangaroo rat agent is one that is at the bottom of the trophic level and only gains energy through foraging from the 
    seed patch class.
    '''
    def __init__(self, unique_id, model, mass, hourly_survival_probability,  age=0, initial_pop=False, initial_pos=None, config=None,report_agent_data=False):
        super().__init__(unique_id, model)
        self.initial_pop = initial_pop
        self.pos = initial_pos
        self.krat_config = config
        self.sex = np.random.choice(['Male', 'Female'], 1)[0]
        self.age = age
        self.mass = mass
        self.hourly_survival_probability = hourly_survival_probability
        if self.krat_config is not None:
            self.active_hours = self.krat_config['active_hours']
            self.energy_budget = self.krat_config["energy_budget"]
            self.max_age_steps = self.krat_config['max_age']*self.model.steps_per_year
            self.moore = self.krat_config['moore']
            self.reproductive_age_steps = int(self.krat_config['reproductive_age_years']*self.model.steps_per_year)
            self.birth_death_module = self.initiate_birth_death_module(birth_config=self.krat_config['birth_death_module'], initial_pop=self.initial_pop)
        else:
            self.active_hours = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
            self.energy_budget = len(self.active_hours)
            self.mass = 60
            self.max_age = 6
            self.moore = True
            self.hourly_survival_probability = 1
            self.birth_death_module = None
        # Agent is actively foraging
        self._active = False
        self._alive = True
        self._cause_of_death = None

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

    @property
    def cause_of_death(self):
        """Returns the cause of death, if any."""
        return self._cause_of_death

    @cause_of_death.setter
    def cause_of_death(self, value):
        """Sets the cause of death only if the agent is dead."""
        if not self.alive:
            self._cause_of_death = value
        else:
            self._cause_of_death = None
            

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
  
    def activate_krat(self, hour):
        activity_budget = np.random.choice(self.active_hours, self.energy_budget, replace=False)
        if hour in activity_budget:
            self.active = True
        else:
            self.active = False

    def random_death(self):
        '''
        Helper function - represents a background death rate from other preditors, disease, vicious cars, etc
        '''
        stay_alive = np.random.choice([True, False], p=[self.hourly_survival_probability, 1 - self.hourly_survival_probability])
        if stay_alive == False:
            self.alive = False
            self.cause_of_death = 'Random'
            self.model.logger.log_data(file_name = self.model.output_folder+"BirthDeath.csv",
                            data=self.birth_death_module.report_data(event_type='Death'))
            #self.model.remove_agent(self)
    

    def initiate_birth_death_module(self, birth_config, initial_pop):
        '''
        Helper function for setting up bith module for organisms
        '''
        return birth_death.Birth_Death_Module(model=self.model, 
                                              agent=self,
                                              mean_litter_size=birth_config["mean_litter_size"], 
                                              std_litter_size=birth_config["std_litter_size"],
                                              upper_bound_litter_size=birth_config["upper_bound_litter_size"], 
                                              lower_bound_litter_size=birth_config["lower_bound_litter_size"],
                                              max_litters=birth_config["max_litters"],
                                              birth_hazard_rate=birth_config["birth_hazard_rate"], 
                                              death_hazard_rate=birth_config["death_hazard_rate"],
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