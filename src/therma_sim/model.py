#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import warnings
import logging
import json
import landscape
import agents 
import interaction
import utility_softmax_lookup as usl
import uuid
import time
import data_logger as dl
from numba import njit

warnings.filterwarnings("ignore")

def get_range(range_dict):
    """
    Converts a dictionary representation of a range into a Python `range` object.

    Example Input:
        {"start": 60, "stop": 71, "step": 1}
    Returns:
        range(60, 71, 1)
    """
    return range(range_dict["start"], range_dict["stop"], range_dict.get("step", 1))


class ThermaSim(mesa.Model):
    '''
    A model class to mange the kangaroorat, rattlesnake predator-prey interactions
    '''
    def __init__(self, config, seed=42, _test=False, output_folder=None,sim_id=None):
        self.running = True
        self.config = config
        self.initial_agents_dictionary = self.get_initial_population_params()
        self.step_id = 0
        self.seed = seed
        self.sim_id = sim_id

        self._hour = None
        self._day = None
        self._month = None
        self._year = None
        self.next_agent_id = 0
        if seed is not None:
            np.random.seed(self.seed)
        self.output_folder = output_folder or ''
        
        
        # Schedular 
        # Random activation, random by type Simultanious, staged
        self.schedule = mesa.time.RandomActivationByType(self)

        ## Make Initial Landscape
        self.landscape = self.make_landscape(model=self)
        self.steps_per_year = self.landscape.count_steps_in_one_year()
        self.steps_per_month = self.steps_per_year / 12
        self.interaction_map = self.make_interaction_module(model=self)
        self.initiate_species_map()
        self.softmax_lookup_table = usl.SoftmaxLookupTable()
        ## Intialize agents
        self.initialize_populations(initial_agent_dictionary=self.initial_agents_dictionary)
        # Data Collector
        self.make_data_loggers()


    @property
    def hour(self):
        return self._hour

    @hour.setter
    def hour(self, value):
        self._hour = value

    @property
    def day(self):
        return self._day

    @day.setter
    def day(self, value):
        self._day = value

    @property
    def month(self):
        return self._month

    @month.setter
    def month(self, value):
        self._month = value

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, value):
        self._year = value

    @property
    def rattlesnake_pop_size(self):
        """Returns the count of Rattlesnake agents in the model."""
        return self.schedule.get_type_count(agents.Rattlesnake)

    @property
    def krats_pop_size(self):
        """Returns the count of KangarooRat agents in the model."""
        return self.schedule.get_type_count(agents.KangarooRat)
    
    @property
    def active_krats_count(self):
        active_krats = [krat for krat in self.schedule.agents_by_type[agents.KangarooRat].values() if krat.active and krat.alive]
        return len(active_krats)

    def get_landscape_params(self, config):
        return config['Landscape_Parameters']

    def get_rattlesnake_params(self):
        params = self.config['Rattlesnake_Parameters']
        
        # Convert stored dictionaries into Python range objects
        if isinstance(params["Body_sizes"], dict):
            params["Body_sizes"] = get_range(params["Body_sizes"])
        if isinstance(params["initial_calories"], dict):
            params["initial_calories"] = get_range(params["initial_calories"])

        return params
    
    def get_interaction_params(self, config):
        return config['Interaction_Parameters']
    
    def get_initial_population_params(self):
        pop_params = self.config['Initial_Population_Sizes']
        
        # Convert stored dictionaries into Python range objects
        for species in pop_params:
            if isinstance(pop_params[species], dict):
                pop_params[species] = get_range(pop_params[species])

        return pop_params
    
    def get_interaction_map(self):
        interaction_map = self.config['Interaction_Map']

        # Convert `expected_prey_body_size` from midpoint of range
        for predator_prey, interaction_data in interaction_map.items():
            if isinstance(interaction_data["expected_prey_body_size"], dict):
                prey_body_size_range = get_range(interaction_data["expected_prey_body_size"])
                interaction_data["expected_prey_body_size"] = (prey_body_size_range.start + prey_body_size_range.stop - 1) / 2

        return interaction_map

    @staticmethod
    @njit
    def bernouli_trial_hourly(annual_probability, steps_per_year):
        '''
        Used to calculate hourly probability of survival
        '''
        P_H = annual_probability ** (1 / steps_per_year)
        return P_H
    
    def make_data_loggers(self):
        '''
        Initiate logger_data_bases
        '''
        rattlesnake_columns = [
            "Time_Step","Hour", "Day", "Month", "Year", "Agent_id", "Active", "Behavior", "Microhabitat",
            "Body_Temperature", "Metabolic_State", "Handling_Time",
            "Attack_Rate", "Prey_Density", "Prey_Consumed"
        ]
        kangaroo_rat_columns = [
            "Time_Step", "Hour", "Day", "Month", "Year","Agent_id", "Active"
        ]
        model_columns = [
            "Time_Step", "Hour", "Day", "Month", "Year",
            "Rattlesnakes", "Krats", 'seed', 'sim_id'
        ]
        birth_death_columns = [
        "Time_Step", "Agent_id","Species", "Age", "Sex", "Birth_Counter",
        "Death_Counter", "Alive", "Event_Type", "Litter_Size"
        ]
        self.logger = dl.DataLogger()
        self.logger.make_data_reporter(file_name=self.output_folder+"Rattlesnake.csv", column_names = rattlesnake_columns)
        self.logger.make_data_reporter(file_name=self.output_folder+"KangarooRat.csv", column_names=kangaroo_rat_columns)
        self.logger.make_data_reporter(file_name=self.output_folder+"Model.csv", column_names=model_columns)
        self.logger.make_data_reporter(file_name=self.output_folder+"BirthDeath.csv", column_names=birth_death_columns)

    def report_data(self):
        """
        Extracts model-level data into a list for CSV logging.
        Includes seed and sim-ID only on the first step.
        """
        data = [
            self.step_id,
            self.hour,
            self.day,
            self.month,
            self.year,
            self.rattlesnake_pop_size,  # Number of rattlesnakes
            self.krats_pop_size         # Number of kangaroo rats
        ]
        
        # Only include seed and sim-ID on the first step
        if self.step_id == 0:
            return data + [self.seed, self.sim_id]
        else:
            return data + [None, None]  # Use None to maintain column alignment



    def make_landscape(self, model):
        '''
        Helper function for intializing the landscape class
        '''
        ls_params = self.get_landscape_params(config = self.config)
        return landscape.Spatially_Implicit_Landscape(model = model,
                                                      width = ls_params['Width'],
                                                      height = ls_params['Height'],
                                                      thermal_profile_csv_fp = ls_params['Thermal_Database_fp'])

    def make_interaction_module(self, model):
        '''
        Retired: Helper function for making an interaction model between a predator and prey     
        '''
        interaction_map = self.get_interaction_map()
        return interaction.Interaction_Map(model = self, interaction_map=interaction_map)

    @staticmethod
    @njit
    def logistic_population_density_function(global_population, total_area, carrying_capacity, growth_rate, threshold_density):
        """
        Computes local density based on global population using logistic scaling.

        Parameters:
        - global_population (int or array): Total population size.
        - total_area (int): Total hectares.
        - carrying_capacity (float): Maximum possible density per hectare.
        - growth_rate (float): Sensitivity of density changes.
        - threshold_density (float): Density at which changes have the largest effect.

        Returns:
        - local_density (float or array): Adjusted local density based on feedback.
        """
        avg_density = global_population / total_area
        adjusted_density = carrying_capacity / (1 + np.exp(-growth_rate * (avg_density - threshold_density)))
        return adjusted_density
    
    def calc_local_population_density(self, population_size, middle_range, max_density):
        # Define parameters
        total_area = self.landscape.landscape_size
        expected_pop_size = population_size/total_area #Average individuals per hectare
        carrying_capacity = min(max_density,population_size) 
        growth_rate = 1  # scaling sensitivity
        threshold_density = min(middle_range,expected_pop_size)  # middle of the range
        global_population = population_size 
        # Compute local densities
        new_local_density = ThermaSim.logistic_population_density_function(global_population, total_area, carrying_capacity, growth_rate, threshold_density)
        return new_local_density
    
    def get_rattlesnake_params(self):
        params = self.config['Rattlesnake_Parameters']
        if isinstance(params["Body_sizes"], dict):
            params["Body_sizes"] = get_range(params["Body_sizes"])
        if isinstance(params["initial_calories"], dict):
            params["initial_calories"] = get_range(params["initial_calories"])
        return params
    
    def get_krat_params(self):
        params = self.config['KangarooRat_Parameters']
        if isinstance(params["Body_sizes"], dict):
            params["Body_sizes"] = get_range(params["Body_sizes"])
        return params

    def initiate_species_map(self):
        """
        Initializes a species map with class references, input parameters, 
        and precomputed static variables, including hourly survival probabilities.

        The hourly survival rate is computed from the annual survival rate using 
        a Bernoulli process.
        """
        self.species_map = {
            "KangarooRat": {
                "class_name": agents.KangarooRat,
                "input_parameters": self.get_krat_params,  # Precomputed parameters
                "static_variables": {}
            },
            "Rattlesnake": {
                "class_name": agents.Rattlesnake,
                "input_parameters": self.get_rattlesnake_params,  # Precomputed parameters
                "static_variables": {}
            }
        }
        for species, values in self.species_map.items():
            params = values["input_parameters"]()
            if "annual_survival_probability" in params:
                hourly_survival_probability = ThermaSim.bernouli_trial_hourly(
                    annual_probability=params["annual_survival_probability"],
                    steps_per_year=self.steps_per_year
                )
            else:
                raise ValueError(f"Missing 'annual_survival_probability' for {species}")
            # Store precomputed values
            values["static_variables"]["hourly_survival_probability"] = hourly_survival_probability


    def set_static_variable(self, species, variable_name, value, overwrite=False):
        """
        Sets a static variable for a given species in the species map.

        Args:
            species (str): The name of the species (e.g., "Rattlesnake", "KangarooRat").
            variable_name (str): The name of the static variable.
            value (any): The value to assign to the static variable.
            overwrite (bool): If False (default), prevents overwriting an existing variable.

        Raises:
            ValueError: If the species does not exist in the species map.
        """
        if species not in self.species_map:
            raise ValueError(f"Species '{species}' not found in species map.")

        static_vars = self.species_map[species]["static_variables"]

        # Only set the variable if it doesn't exist OR if overwrite=True
        if overwrite or variable_name not in static_vars:
            static_vars[variable_name] = value

    def get_static_variable(self, species, variable_name, default=None):
        """
        Retrieves a static variable for a given species in the species map.

        Args:
            species (str): The name of the species (e.g., "Rattlesnake", "KangarooRat").
            variable_name (str): The name of the static variable to retrieve.
            default (any, optional): A default value to return if the variable does not exist. 
                                    Defaults to None.

        Returns:
            any: The value of the static variable if found, else the default value.

        Raises:
            ValueError: If the species does not exist in the species map.
        """
        if species not in self.species_map:
            raise ValueError(f"Species '{species}' not found in species map.")

        return self.species_map[species]["static_variables"].get(variable_name, default)

        
    ## Intialize populations and births
    def give_birth(self, species_name, agent_id, pos=None, parent=None, initial_pop=False):
        """
        Helper function - Adds new agents to the landscape
        """
        if species_name not in self.species_map:
            raise ValueError(f"Class for species: {species_name} does not exist")

        agent_info = self.species_map[species_name]
        agent_params = agent_info["input_parameters"]()
        agent_class = agent_info['class_name']
        if initial_pop:
            max_age = agent_params['max_age']
            rand_age = int(np.random.uniform(0,max_age*self.steps_per_year))
            agent = agent_class(unique_id=agent_id, model=self, config=agent_params, age = rand_age, initial_pop=initial_pop)
        else:
            agent = agent_class(unique_id=agent_id, model=self, age=0, config=agent_params)

        if pos is not None:
            self.place_agent(agent, pos)
        self.schedule.add(agent)

    def initialize_populations(self, initial_agent_dictionary, spatially_explicit=False):
        """
        Initializes the model's agent populations.
        Population sizes should be a range of individuals per hectare.
        """
        agent_id = 0
        for hect in range(self.landscape.landscape_size):
            for species, initial_population_size_range in initial_agent_dictionary.items():
                # Ensure `initial_population_size_range` is a range object
                if isinstance(initial_population_size_range, dict):
                    initial_population_size_range = get_range(initial_population_size_range)

                start, stop = initial_population_size_range.start, initial_population_size_range.stop
                initial_pop_size = round(np.random.uniform(start, stop))

                # Scale population size by hectare
                for _ in range(initial_pop_size):
                    if spatially_explicit:
                        pos = (np.random.uniform(0, self.landscape.width), np.random.uniform(0, self.landscape.height))
                        self.give_birth(species_name=species, pos=pos, initial_pop=True, agent_id=agent_id)
                    else:
                        self.give_birth(species_name=species, agent_id=agent_id, initial_pop=True)
                    
                    agent_id += 1
                    self.next_agent_id = agent_id + 1
                    #print(pos,agent_id)
                                    

    def randomize_snakes(self):
        '''
        helper function for self.step()

        puts snakes in a list and shuffles them
        '''
        snake_shuffle = list(self.schedule.agents_by_type[agents.Rattlesnake].values())
        #print(f'Snakes: {len(snake_shuffle)}')
        self.random.shuffle(snake_shuffle)
        return snake_shuffle
    
    def randomize_active_snakes(self):
        '''
        Helper function for self.step()

        Puts active snakes in a list and shuffles them
        '''
        # Filter only active snakes
        active_snakes = [snake for snake in self.schedule.agents_by_type[agents.Rattlesnake].values() if snake.active]
        
        # Shuffle the list of active snakes
        self.random.shuffle(active_snakes)
    
        return active_snakes 
    
    def randomize_krats(self):
        '''
        helper function for self.step()

        puts snakes in a list and shuffles them
        '''
        krat_shuffle = list(self.schedule.agents_by_type[agents.KangarooRat].values())
        #print(f'Krats: {len(krat_shuffle)}')
        self.random.shuffle(krat_shuffle)
        return krat_shuffle
    
    def randomize_active_krats(self):
        '''
        Helper function for self.step()

        Puts active Kangaroo Rats in a list and shuffles them
        '''
        # Filter only active KangarooRats
        active_krats = [krat for krat in self.schedule.agents_by_type[agents.KangarooRat].values() if krat.active]
        
        # Shuffle the list of active KangarooRats
        self.random.shuffle(active_krats)
    
        return active_krats
    
    def get_active_krat(self):
        active_krats = self.randomize_active_krats()
        if not active_krats:  
            return None  
        return np.random.choice(active_krats)
    
    def remove_agent(self, agent):
        self.schedule.remove(agent)

    def remove_dead_agents(self):
        """Removes dead agents efficiently."""
        for agent in list(self.schedule.agents_by_type[agents.Rattlesnake].values()) + \
                      list(self.schedule.agents_by_type[agents.KangarooRat].values()):
            if not agent.alive:
                self.remove_agent(agent)

    def end_sim_early_check(self):
        snakes = self.rattlesnake_pop_size
        krats = self.krats_pop_size
        total_agents = snakes + krats
        if total_agents > 20000:
           self.running=False
        elif krats==0:
            self.running=False
        elif snakes==0:
            self.running=False
        elif total_agents>21000:
            raise RuntimeError('Simulation didnt stop')

    def step(self):
        '''
        Main model step function used to run one step of the model.
        '''
        self.end_sim_early_check()
        self.hour = self.landscape.thermal_profile.select("hour").row(self.step_id)[0]

        self.day = self.landscape.thermal_profile.select('day').row(self.step_id)[0]
        self.month = self.landscape.thermal_profile.select('month').row(self.step_id)[0]
        self.year = self.landscape.thermal_profile.select('year').row(self.step_id)[0]
        self.landscape.set_landscape_temperatures(step_id=self.step_id)
        self.logger.log_data(file_name = self.output_folder+"Model.csv", data=self.report_data())
        # Snakes
        snake_shuffle = self.randomize_snakes()
        for snake in snake_shuffle:
            self.logger.log_data(file_name = self.output_folder+"Rattlesnake.csv", data=snake.report_data())
            snake.step()
        snake_shuffle = self.randomize_snakes()
        # Krats
        krat_shuffle = self.randomize_krats()
        for krat in krat_shuffle:
            #self.logger.log_data(file_name = self.output_folder+"KangarooRat.csv", data=krat.report_data())
            krat.step()
        krat_shuffle = self.randomize_krats()
        
        self.remove_dead_agents()
        self.step_id += 1  # Increment the step counter
        self.schedule.step()


    def run_model(self, step_count=None):
        max_steps = len(self.landscape.thermal_profile)-1
        if step_count is None:
            step_count = max_steps
        elif max_steps <= step_count:
            print(f'Step argument exceeds length of data. Using {max_steps} instead.')
            step_count=max_steps

        for i in range(step_count):
            start_time = time.time()
            self.step()
            end_time= time.time()
            execution_time = end_time - start_time
            print(f'Step {self.step_id}, snakes {self.rattlesnake_pop_size}, krats {self.krats_pop_size}, time_to_run_step {execution_time}')

            

if __name__ ==  "__main__":
    pass