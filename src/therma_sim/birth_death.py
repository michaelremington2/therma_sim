#!/usr/bin/python
import numpy as np
from scipy.stats import truncnorm

class Birth_Death_Module(object):
    '''
    Class for handling birth and death events using an exponential waiting time counter.
    '''
    def __init__(self, model, agent,
                max_litters: float, mean_litter_size: float, std_litter_size: float,
                upper_bound_litter_size: int, lower_bound_litter_size: int,
                birth_hazard_rate: float, death_hazard_rate: float, initial_pop=False):
        self.model = model
        self.agent = agent
        self.death_hazard_rate = death_hazard_rate
        self.death_counter = self.bounded_exponential_wait_time(hazard_rate=self.death_hazard_rate, 
                                                                steps_per_year=self.model.steps_per_year,
                                                                min_steps = 0,
                                                                max_steps = self.agent.max_age_steps)
        # Birth parameters
        self.birth_counter = np.inf
        if self.agent.sex=='Female':
            self.mean_litter_size = mean_litter_size
            self.std_litter_size = std_litter_size
            self.upper_bound_litter_size = upper_bound_litter_size
            self.lower_bound_litter_size = lower_bound_litter_size
            self.max_litters = max_litters
            # Compute hazard rates
            self.hazard_rate_birth = birth_hazard_rate
            ## Hidden variables for litter size distribution
            self.a = (self.lower_bound_litter_size - self.mean_litter_size) / self.std_litter_size
            self.b = (self.upper_bound_litter_size - self.mean_litter_size) / self.std_litter_size
            if self.agent.age > self.agent.reproductive_age_steps:
                self.birth_counter = self.bounded_exponential_wait_time(
                    hazard_rate=self.hazard_rate_birth, 
                    steps_per_year=self.model.steps_per_year,
                    min_steps=0
                )
            else:
                self.birth_counter = self.bounded_exponential_wait_time(
                    hazard_rate=self.hazard_rate_birth, 
                    steps_per_year=self.model.steps_per_year,
                    min_steps=self.agent.reproductive_age_steps
                )
        else:
            self.birth_counter = np.inf

        self._ct_out_of_bounds_tcounter = 0

    @property
    def ct_out_of_bounds_tcounter(self):
        """
        Property to get the current value of the critical temperature out of bounds counter.
        """
        if self.agent.is_bruminating_today():
            return 0
        else:
            return self._ct_out_of_bounds_tcounter
        
    @ct_out_of_bounds_tcounter.setter
    def ct_out_of_bounds_tcounter(self, value):
        """
        Property to set the value of the critical temperature out of bounds counter.
        """
        if self.agent.is_bruminating_today:
            self._ct_out_of_bounds_tcounter = 0
        else:
            self._ct_out_of_bounds_tcounter = value
        # Ensure the counter is not negative
        if self._ct_out_of_bounds_tcounter < 0:
            self._ct_out_of_bounds_tcounter = 0
        # Ensure the counter does not exceed the maximum value
        if self._ct_out_of_bounds_tcounter > self.agent.ct_max_steps:
            self._ct_out_of_bounds_tcounter = self.agent.ct_max_steps
    
    def bounded_exponential_wait_time(self, hazard_rate, steps_per_year, min_steps, max_steps=None):
        """
        Samples a waiting time from an exponential distribution but ensures it falls within a reasonable range.

        Parameters:
        - hazard_rate (float): Hazard rate per year.
        - steps_per_year (int): Number of simulation steps per year.
        - min_steps (int): Minimum number of steps before an event.
        - max_steps (int): Maximum number of steps before an event.

        Returns:
        - int: The number of time steps until the next event.
        """
        if hazard_rate <= 0:
            return np.inf  # No event occurs if hazard rate is zero or negative

        # Sample from exponential and shift the minimum before applying bounds
        raw_wait_time = np.random.exponential(scale=1 / hazard_rate) * steps_per_year
        shifted_wait_time = min_steps + raw_wait_time  # Ensures a minimum wait time
        if max_steps:
            return int(np.clip(shifted_wait_time, min_steps, max_steps))
        else:
            return int(shifted_wait_time)


    def litter_size(self):
        '''
        Helper function for calculating litter size using a truncated normal distribution.
        '''
        return int(truncnorm.rvs(self.a, self.b, loc=self.mean_litter_size, scale=self.std_litter_size, size=1)[0])
    
    def report_data(self, event_type, litter_size=0):
        """
        Extracts model-level data into a list for CSV logging.
        """
        if hasattr(self.agent, 'body_temperature'):
            bd = self.agent.body_temperature
            ct_min = self.agent.ct_min
            ct_max = self.agent.ct_max
        else:
            bd = None
            ct_min = None
            ct_max = None

        return [
            self.model.step_id,
            self.agent.unique_id,
            self.agent.species_name, 
            self.agent.age,
            self.agent.sex,
            self.agent.mass, 
            self.birth_counter,
            self.death_counter,
            self.agent.alive,
            event_type,
            self.agent.cause_of_death, 
            litter_size,
            bd,
            ct_min,
            ct_max
        ]
    
    def ct_death_probability_calculator(self, initialProb=0):
        """
        Calculates death probability using y = mx + b form.
        
        Parameters:
        - t (int): Current time step
        - tmax (int): Time step at which death probability reaches 1
        - initialProb (float): Death probability at t = 0
        
        Returns:
        - float: Probability of death at time t
        """
        if self.ct_out_of_bounds_tcounter <= 0:
            return initialProb
        elif self.ct_out_of_bounds_tcounter >= self.agent.ct_max_steps:
            return 1.0
        else:
            m = (1.0 - initialProb) / self.agent.ct_max_steps
            return m * self.ct_out_of_bounds_tcounter + initialProb
    
    def critical_death_sub_model(self):
        '''
        Function for checking if the agent is within its critical temperature range
        Checks a out out of bounds counter and the probability of death increases linearly.
        '''
        if self.agent.body_temperature < self.agent.ct_min:
            self.ct_out_of_bounds_tcounter += 1
            # Calculate Death Probability
            death_prob = self.ct_death_probability_calculator(initialProb=0.0)
            # Check if the agent dies
            if np.random.rand() < death_prob:
                self.thermal_critical_death()
                return
        elif self.agent.body_temperature > self.agent.ct_max:
            self.ct_out_of_bounds_tcounter += 1
            death_prob = self.ct_death_probability_calculator(initialProb=0.0)
            # Check if the agent dies
            if np.random.rand() < death_prob:
                self.thermal_critical_death()
                return
        else:
            self.ct_out_of_bounds_tcounter = 0
            # Reset the counter if within bounds
 
    
    def thermal_critical_death(self):
        """
        Function for murdering snakes that dear leave their critcal thermal bredth
        """
         
        if self.agent.body_temperature < self.agent.ct_min:
            self.agent.alive = False
            self.agent.cause_of_death = 'Frozen'
            self.model.logger.log_data(file_name = self.model.output_folder+"BirthDeath.csv", data=self.report_data(event_type='Death'))
            self.model.remove_agent(self.agent)
            return
        elif self.agent.body_temperature > self.agent.ct_max:
            self.agent.alive = False
            self.agent.cause_of_death = 'Cooked'
            self.model.logger.log_data(file_name = self.model.output_folder+"BirthDeath.csv", data=self.report_data(event_type='Death'))
            self.model.remove_agent(self.agent)
            return
            

    def step(self):
        """
        Decrement counters at each timestep and trigger events when they reach zero.
        Optimized: If the agent will die before reproducing, we only decrement the death counter.
        """
        # Thermal Crtical
        if hasattr(self.agent, 'body_temperature'):
            self.critical_death_sub_model()
        # If the agent is expected to die before it reproduces, ignore birth updates
        if self.death_counter <= 0:
            self.agent.alive = False
            self.agent.cause_of_death = 'old_age'
            self.model.logger.log_data(file_name = self.model.output_folder+"BirthDeath.csv", data=self.report_data(event_type='Death'))
            self.model.remove_agent(self.agent)
            return  # Stop processing if the agent dies

        # If birth happens before death, process birth
        if self.agent.sex == 'Female' and self.birth_counter <= 0 and self.max_litters>0:
            litter_size = self.litter_size()
            species = self.agent.species_name
            self.model.logger.log_data(file_name = self.model.output_folder+"BirthDeath.csv", data=self.report_data(event_type='Birth', litter_size=litter_size))
            for _ in range(litter_size):
                self.model.give_birth(species_name=species)
            # Reset Birth Counter
            self.birth_counter = self.bounded_exponential_wait_time(
                hazard_rate=self.hazard_rate_birth, 
                steps_per_year=self.model.steps_per_year,
                min_steps=self.agent.reproductive_age_steps
            )
            self.max_litters-=1
        self.birth_counter -= 1
        self.death_counter -= 1
        


