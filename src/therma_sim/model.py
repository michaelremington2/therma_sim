#!/usr/bin/python
import mesa
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import warnings
import logging
import yaml
import landscape
import agents 
import interaction
import utility_softmax_lookup as usl
import uuid
import time
import data_logger as dl
from numba import njit
import os
from scipy.stats import truncnorm
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
    def __init__(self, config, seed=42, snake_sample_frequency=None, _test=False, output_folder=None,sim_id=None, print_progress=False):
        super().__init__()
        self.running = True
        self.config = config
        self.initial_agents_dictionary = self.get_initial_population_parameters(config=self.config)
        self.step_id = 0
        self.seed = seed
        self.sim_id = sim_id
        self.print_progress = print_progress

        self._hour = None
        self._day = None
        self._month = None
        self._year = None
        self.next_agent_id = 0
        self._initial_mean_densities = {}  # backing dict for initial densities
        if seed is not None:
            np.random.seed(self.seed)
        self.snake_sample_frequency = snake_sample_frequency
        if self.snake_sample_frequency is not None:
            self.sampled_snake_ids = set()
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            self.output_folder = output_folder
        else:
            self.output_folder = ''
                
        # Schedular 
        # Random activation, random by type Simultanious, staged
        self.schedule = mesa.time.RandomActivationByType(self)

        ## Make Initial Landscape
        self.landscape = self.make_landscape(model=self)
        self.steps_per_year = self.landscape.count_steps_in_one_year()
        self.steps_per_month = self.steps_per_year / 12
        self.interaction_map = self.make_interaction_module(model=self)
        #self.initiate_species_map()
        self.softmax_lookup_table = usl.SoftmaxLookupTable()
        ## Intialize agents
        self.make_initial_population()
        # Data Collector
        self.make_data_loggers()

    ################################################
    ### Properties
    ################################################

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
    def initial_mean_densities(self):
        return {k: round(v, 2) for k, v in self._initial_mean_densities.items()}
    
    @property
    def rattlesnake_mean_density(self):
        """Returns realized mean density for rattlesnakes (individuals per hectare)."""
        return self.rattlesnake_pop_size / self.landscape.landscape_size

    @property
    def krat_mean_density(self):
        """Returns realized mean density for kangaroo rats (individuals per hectare)."""
        return self.krats_pop_size / self.landscape.landscape_size

    
    @property
    def active_krats_count(self):
        return sum(1 for krat in self.schedule.agents_by_type[agents.KangarooRat].values()
                if krat.active and krat.alive)
    
    @property
    def active_snakes_count(self):
        return sum(1 for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()
                if snake.active and snake.alive)
    # Behavioral profile
    @property
    def count_foraging(self):
        """Counts the number of Rattlesnakes that are foraging."""
        return sum(1 for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()
                if snake.current_behavior == "Forage")
    
    @property
    def count_thermoregulate(self):
        """Counts the number of Rattlesnakes that are thermoregulating."""
        return sum(1 for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()
                if snake.current_behavior == "Thermoregulate")
    
    @property
    def count_rest(self):
        """Counts the number of Rattlesnakes that are resting."""
        return sum(1 for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()
                if snake.current_behavior == "Rest")
    
    @property
    def count_search(self):
        """Counts the number of Rattlesnakes that are Searching for prey items."""
        return sum(1 for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()
                if snake.current_behavior == "Search")
    
    @property
    def count_brumation(self):
        """Counts the number of Rattlesnakes in brumation."""
        return sum(1 for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()
                if snake.current_behavior == "Brumation")
    
    @property
    def snakes_in_burrow(self):
        """Counts the number of Rattlesnakes in brumation."""
        return sum(1 for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()
                if snake.current_microhabitat == "Burrow")
    
    @property
    def snakes_in_open(self):
        """Counts the number of Rattlesnakes in brumation."""
        return sum(1 for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()
                if snake.current_microhabitat == "Open")

    @property
    def mean_thermal_quality(self):
        return np.mean([np.abs(snake.t_env - snake.t_opt) for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()])

    @property
    def mean_thermal_accuracy(self):
        return np.mean([np.abs(snake.body_temperature - snake.t_opt) for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()])
    
    @property
    def mean_metabolic_state(self):
        return np.mean([snake.metabolism.metabolic_state for snake in self.schedule.agents_by_type[agents.Rattlesnake].values()])
    
    @property
    def count_interactions(self):
        """Counts the number of Rattlesnakes that interacted with prey."""
        return sum(snake.behavior_module.prey_encountered for snake in self.schedule.agents_by_type[agents.Rattlesnake].values())
    
    @property
    def count_successful_interactions(self):
        """Counts the number of Rattlesnakes that consumed a prey."""
        return sum(snake.behavior_module.prey_consumed for snake in self.schedule.agents_by_type[agents.Rattlesnake].values())

    ###################################################
    ### Methods
    ###################################################

    def get_landscape_params(self, config):
        return config['Landscape_Parameters']

    def get_rattlesnake_params(self):
        params = self.config['Rattlesnake_Parameters']
        return params
    
    def get_interaction_params(self, config):
        return config['Interaction_Parameters']
    
    def get_initial_population_parameters(self,config):
        return config['Initial_Population_Sizes']
    
    def get_population_densities(self, species):
        """Retrieve the min and max population density values for a given species."""
        params = self.initial_agents_dictionary.get(species, {})
        density_params = params.get("Density", {})

        min_density = density_params.get("start")
        max_density = density_params.get("stop")

        return min_density, max_density
    
    def get_initial_population_size(self, species):
        """Retrieve the initial population size for a given species, if available."""
        params = self.initial_agents_dictionary.get(species, {})
        return params.get("Initial_Population")
    
    def get_population_carrying_capacity(self, species):
        """Retrieve the carrying capacity for a given species, if available."""
        params = self.initial_agents_dictionary.get(species, {})
        return params.get("Carrying_Capacity", None)
    
    def make_initial_population(self):
        total_area = self.landscape.landscape_size  # Get the total area in hectares
        for species, params in self.initial_agents_dictionary.items():
            if params.get('Initial_Population'):
                initial_pop_size = int(params.get('Initial_Population'))
                self.initialize_populations_input(species=species,
                                                  initial_population_size=initial_pop_size)
            else:
                density_params = params.get("Density")
                if not density_params:
                    raise ValueError(
                        f"No 'Initial_Population' or 'Density' provided for species: {species}. "
                        "Please define at least one in your config."
                    )
                min_density = density_params.get("start")
                max_density = density_params.get("stop")
                initial_pop_size = self.initialize_populations_density(
                    species=species,
                    min_density=min_density,
                    max_density=max_density
                )
                self.initial_agents_dictionary[species]["Initial_Population"] = initial_pop_size


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
            "Time_Step","Hour", "Day", "Month", "Year", "Site_Name", "Agent_id", "Active","Alive",
            "Behavior", "Microhabitat",
            "Body_Temperature", 'T_Env', "Mass", "Metabolic_State", 
            "Handling_Time", "Attack_Rate",
            "Prey_Density", "Prey_Encountered", "Prey_Consumed"
        ]
        kangaroo_rat_columns = [
            "Time_Step", "Hour", "Day", "Month", "Year","Agent_id","Alive", "Active"
        ]
        model_columns = [
            "Time_Step", "Hour", "Day", "Month", "Year", "Site_Name",
            "Rattlesnakes", "Krats", "Rattlesnakes_Density", "Krats_Density", 'Rattlesnakes_Active', 'Krats_Active',
            'Foraging', 'Thermoregulating', 'Resting', 'Searching', 'Brumating',
            'Snakes_in_Burrow', 'Snakes_in_Open',
            'mean_thermal_quality', 'mean_thermal_accuracy', 'mean_metabolic_state',
            'count_interactions', 'count_successful_interactions',
            'seed', 'sim_id'
        ]
        birth_death_columns = [
            "Time_Step","Hour", "Day", "Month", "Year", "Site_Name",
            "Agent_id","Species", "Age", "Sex", "Mass", "Birth_Counter",
            "Death_Counter", "Alive", "Event_Type", "Cause_Of_Death", "Litter_Size",
            "Body_Temperature", 'ct_min', 'ct_max'
        ]
        self.logger = dl.DataLogger()
        self.logger.make_data_reporter(file_name=self.output_folder+"Rattlesnake.csv", column_names = rattlesnake_columns)
        self.logger.make_data_reporter(file_name=self.output_folder+"KangarooRat.csv", column_names=kangaroo_rat_columns)
        self.logger.make_data_reporter(file_name=self.output_folder+"Model.csv", column_names=model_columns)
        self.logger.make_data_reporter(file_name=self.output_folder+"BirthDeath.csv", column_names=birth_death_columns)

    def report_data(self):
        data = [
            self.step_id,
            self.hour,
            self.day,
            self.month,
            self.year,
            self.landscape.site_name,
            self.rattlesnake_pop_size,
            self.krats_pop_size,
            round(self.rattlesnake_mean_density, 2),
            round(self.krat_mean_density, 2),
            self.active_snakes_count,
            self.active_krats_count,
            self.count_foraging,
            self.count_thermoregulate,
            self.count_rest,
            self.count_search,
            self.count_brumation,
            self.snakes_in_burrow,
            self.snakes_in_open,
            self.mean_thermal_quality,
            self.mean_thermal_accuracy,
            self.mean_metabolic_state,
            self.count_interactions,
            self.count_successful_interactions,
            self.seed,
            self.sim_id
        ]
        return data
        
    def make_landscape(self, model):
        '''
        Helper function for intializing the landscape class
        '''
        ls_params = self.get_landscape_params(config = self.config)
        return landscape.Spatially_Implicit_Landscape(model = model,
                                                      site_name = ls_params['site_name'],
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
    
    def get_krat_params(self):
        params = self.config['KangarooRat_Parameters']
        return params
    
    def set_mass(self, body_sizes_config):
        distribution = body_sizes_config['distribution']
        if distribution == 'uniform':
            mass = np.random.choice(range(body_sizes_config['start'], body_sizes_config['stop'], body_sizes_config['step']))
        elif distribution == 'normal':
            mean = body_sizes_config['mean']
            std = body_sizes_config['std']
            a, b = (body_sizes_config['min'] - mean) / std, (body_sizes_config['max'] - mean) / std
            mass = truncnorm.rvs(a, b, loc=mean, scale=std)
        else:
            raise ValueError('No mass distribution given')
        return mass

    def give_birth(self, species_name, pos=None, parent=None, initial_pop=False):
        """
        Adds new agents to the simulation with random body size and optional position.
        """
        if species_name == "KangarooRat":
            agent_class = agents.KangarooRat
            params = self.get_krat_params()
        elif species_name == "Rattlesnake":
            agent_class = agents.Rattlesnake
            params = self.get_rattlesnake_params()
        else:
            raise ValueError(f"Unknown species: {species_name}")
        krat_carrying_capacity = self.get_population_carrying_capacity(species_name)

        if not initial_pop and krat_carrying_capacity is not None and self.krats_pop_size >= krat_carrying_capacity and species_name == "KangarooRat":
            return
        # Set range variables
        else:
            if isinstance(params.get("body_size_config"), dict):
                mass = self.set_mass(params["body_size_config"])
            else:
                mass = params["body_size_config"]

            if "annual_survival_probability" in params:
                hourly_survival_probaility = ThermaSim.bernouli_trial_hourly(
                    annual_probability=params["annual_survival_probability"],
                    steps_per_year=self.steps_per_year
                )
            age = int(np.random.uniform(0, params['max_age'] * self.steps_per_year)) if initial_pop else 0
            if self.snake_sample_frequency is not None and species_name == "Rattlesnake" and len(self.sampled_snake_ids) < self.snake_sample_frequency and age==0:
                report_agent_data = True
            else:
                report_agent_data = False
            agent = agent_class(
                unique_id=self.next_id(),
                model=self,
                age=age,
                mass=mass,
                hourly_survival_probability=hourly_survival_probaility,
                config=params,
                initial_pop=initial_pop,
                report_agent_data=report_agent_data
            )
            if report_agent_data:
                self.sampled_snake_ids.add(agent.unique_id)
            self.schedule.add(agent)


    def initialize_populations_density(self, species, min_density, max_density, spatially_explicit=False):
        total_pop_size = 0
        per_hectare_densities = []

        for hect in range(self.landscape.landscape_size):
            density = round(np.random.uniform(min_density, max_density))
            for _ in range(density):
                if spatially_explicit:
                    pos = (np.random.uniform(0, self.landscape.width), np.random.uniform(0, self.landscape.height))
                    self.give_birth(species_name=species, pos=pos, initial_pop=True)
                else:
                    self.give_birth(species_name=species, initial_pop=True)
            total_pop_size += density
            per_hectare_densities.append(density)

        mean_density = np.mean(per_hectare_densities)
        self._initial_mean_densities[species] = mean_density
        print(f"No initial Population size for {species}, calculated {total_pop_size} from densities and landscape size. Mean density was {mean_density:.2f}.")
        return total_pop_size

    def initialize_populations_input(self, species, initial_population_size, spatially_explicit=False):
        """
        Initializes the model's agent populations.
        Population sizes can be explicitly provided or calculated based on density.
        """
        for _ in range(initial_population_size):
            if spatially_explicit:
                pos = (np.random.uniform(0, self.landscape.width), np.random.uniform(0, self.landscape.height))
                self.give_birth(species_name=species, pos=pos, initial_pop=True)
            else:
                self.give_birth(species_name=species, initial_pop=True)

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
                if agent.species_name == "Rattlesnake" and agent.unique_id in self.sampled_snake_ids:
                    self.sampled_snake_ids.remove(agent.unique_id)
                    #print('Removed dead snake from sampled ids:', agent.unique_id)
                self.remove_agent(agent)

    def end_sim_early_check(self):
        snakes = self.rattlesnake_pop_size
        krats = self.krats_pop_size
        total_agents = snakes + krats
        if krats==0:
            self.running=False
            raise RuntimeError('Simulation ended early, no Kangaroo Rats left')
        elif snakes==0:
            self.running=False
            raise RuntimeError('Simulation ended early, no snakes left')
        elif total_agents>30000:
            raise RuntimeError('Simulation didnt stop')

    def step(self):
        '''
        Main model step function used to run one step of the model.
        '''
        self.hour = self.landscape.thermal_profile.select("hour").row(self.step_id)[0]
        self.day = self.landscape.thermal_profile.select('day').row(self.step_id)[0]
        self.month = self.landscape.thermal_profile.select('month').row(self.step_id)[0]
        self.year = self.landscape.thermal_profile.select('year').row(self.step_id)[0]
        self.landscape.set_landscape_temperatures(step_id=self.step_id)
        self.logger.log_data(file_name = self.output_folder+"Model.csv", data=self.report_data())
        self.schedule.step()
        self.remove_dead_agents()
        self.step_id += 1  # Increment the step counter
        self.end_sim_early_check()


    def run_model(self, step_count=None):
        max_steps = len(self.landscape.thermal_profile)-1
        print(f'Site: {self.landscape.site_name}, Simulation ID: {self.sim_id}, Seed: {self.seed}, sampling_snakes {self.snake_sample_frequency}, Steps: {max_steps}')
        if step_count is None:
            step_count = max_steps
        elif max_steps <= step_count:
            print(f'Step argument exceeds length of data. Using {max_steps} instead.')
            step_count=max_steps

        for i in range(step_count):
            start_time = time.time()
            self.step()
            if self.running is False:
                print(f'Simulation {self.sim_id} ended at step {self.step_id} with {self.rattlesnake_pop_size} snakes and {self.krats_pop_size} krats.')
                break
            end_time= time.time()
            execution_time = end_time - start_time
            if self.print_progress:
                print(f'Step {self.step_id},hour {self.hour}, date {self.month}/{self.day}/{self.year} - snakes {self.rattlesnake_pop_size} active {self.active_snakes_count}, krats {self.krats_pop_size} active {self.active_krats_count}, time_to_run_step {round(execution_time,2)}, sss {len(self.sampled_snake_ids)}')

            

if __name__ ==  "__main__":
    pass