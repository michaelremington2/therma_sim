#!/usr/bin/python
import numpy as np
from scipy.stats import truncnorm

class Birth_Death_Module(object):
    '''
    Class for handling birth and death events using an exponential waiting time counter.
    '''
    def __init__(self, model, agent,
                litters_per_year: float, mean_litter_size: float, std_litter_size: float,
                upper_bound_litter_size: int, lower_bound_litter_size: int,
                birth_hazard_rate: float, death_hazard_rate: float):
        self.model = model
        self.agent = agent

        # Birth parameters
        self.mean_litter_size = mean_litter_size
        self.std_litter_size = std_litter_size
        self.upper_bound_litter_size = upper_bound_litter_size
        self.lower_bound_litter_size = lower_bound_litter_size
        self.litters_per_year = litters_per_year

        # Death parameters
        self.death_hazard_rate = death_hazard_rate

        # Compute hazard rates
        self.hazard_rate_birth = birth_hazard_rate

        # Initialize countdown timers (sample from exponential distribution)
        self.birth_counter = self.sample_exponential_wait_time(self.hazard_rate_birth, self.model.steps_per_year)
        self.death_counter = self.sample_exponential_wait_time(self.death_hazard_rate, self.model.steps_per_year)

        ## Hidden variables for litter size distribution
        self.a = (self.lower_bound_litter_size - self.mean_litter_size) / self.std_litter_size
        self.b = (self.upper_bound_litter_size - self.mean_litter_size) / self.std_litter_size

    
    def sample_exponential_wait_time(self, hazard_rate, steps_per_year):
        """
        Samples a waiting time from an exponential distribution and converts it into discrete time steps.
        
        Parameters:
        - hazard_rate (float): The hazard rate (event rate per year).
        - steps_per_year (int): The number of simulation steps per year.
        
        Returns:
        - int: The number of time steps until the next event, or np.inf if hazard_rate is zero.
        """
        #print(f"hazard rate {hazard_rate}")
        if hazard_rate < 0:
            raise ValueError("Hazard Rate is less than zero")
        return int(np.random.exponential(scale=1 / hazard_rate) * steps_per_year)


    def litter_size(self):
        '''
        Helper function for calculating litter size using a truncated normal distribution.
        '''
        return int(truncnorm.rvs(self.a, self.b, loc=self.mean_litter_size, scale=self.std_litter_size, size=1)[0])

    def step(self):
        """
        Decrement counters at each timestep and trigger events when they reach zero.
        Optimized: If the agent will die before reproducing, we only decrement the death counter.
        """
        # If the agent is expected to die before it reproduces, ignore birth updates
        if self.death_counter <= self.birth_counter:
            self.death_counter -= 1
            if self.death_counter <= 0:
                self.agent.alive = False
                return  # Stop processing if the agent dies
            return  # Skip birth updates

        # Process birth event if the agent is still alive
        if self.agent.sex == 'Female' and self.birth_counter <= 0:
            litter_size = self.litter_size()
            species = self.agent.species_name
            for _ in range(litter_size):
                self.model.give_birth(species_name=species, agent_id=self.model.next_agent_id)
                self.model.next_agent_id += 1 
            self.birth_counter = self.sample_exponential_wait_time(self.hazard_rate_birth, self.model.steps_per_year)
        self.birth_counter -= 1
        self.death_counter -= 1


