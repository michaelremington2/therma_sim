#!/usr/bin/python
import numpy as np
from scipy.stats import truncnorm

class Birth_Module(object):
    '''
    Class for handling the frequency, timing, and quantity of births per agent
    '''
    def __init__(self, model, agent,
                frequency: str, mean_litter_size: float, std_litter_size: float,
                upper_bound_litter_size: int, lower_bound_litter_size: int,
                litters_per_year: float, months_since_last_litter: int,
                partuition_months: list):
        self.model = model
        self.frequency = frequency
        self.agent = agent
        self.mean_litter_size = mean_litter_size
        self.std_litter_size = std_litter_size
        self.upper_bound_litter_size = upper_bound_litter_size
        self.lower_bound_litter_size = lower_bound_litter_size
        self.partuition_months = partuition_months
        self.litters_per_year = litters_per_year
        self.litters_left = self.init_litters_left()
        self.months_since_last_litter = months_since_last_litter
        self.time_to_birth_unit_conversion()
        self.start_month = self.model.month
        self.current_month = self.start_month

        ## Hidden variables
        self.a, self.b = (self.lower_bound_litter_size - self.mean_litter_size) / self.std_litter_size, (self.upper_bound_litter_size - self.mean_litter_size) / self.std_litter_size

    def init_litters_left(self):
        '''
        init function for the property litters left in year. 
        This is designed to handle birth frequencies where there may be multiple in a year or births happen every other year or third year.
        '''
        if isinstance(self.litters_per_year, range):
            return np.random.uniform(self.litters_per_year.start, self.litters_per_year.stop)
        elif self.litters_per_year < 1:
            # Biennal or trinial reproduction
            if np.random.random() < self.litters_per_year:
                return 1
            else:
                return 0
        else:
            return self.litters_per_year
    
    def litter_size(self):
        '''
        Helper function for calculationg litter size of organism
        '''
        return int(truncnorm.rvs(self.a, self.b, loc=self.mean_litter_size, scale=self.std_litter_size, size=1)[0])
    
    def check_birth_event(self):
        """Check if the agent can give birth in the current month."""
        sex_check = self.agent.sex == 'Female'
        breeding_season_check = self.model.month in self.partuition_months
        birth_frequency_check = self.months_since_last_litter >= self.birth_frequency_int
        litters_per_year_check = self.litters_left >0
        if sex_check and breeding_season_check and birth_frequency_check and litters_per_year_check:
            return True
        else:
            return False

    def time_to_birth_unit_conversion(self):
        '''
        Helper Function for setting the units for the frequency of reproduction to months
        '''
        if self.frequency =='biennial':
            self.birth_frequency_int = 24
        elif self.frequency =='yearly':
            self.birth_frequency_int = 12
        elif self.frequency == 'monthly':
            self.birth_frequency_int = 1
        else:
            raise ValueError(f'Birth frequency {self.frequency} has not been developed yet.')
        return

    def increment_time(self):
        """Increment time and update the counter for reproductive readiness."""
        if self.current_month != self.model.month:
            self.months_since_last_litter += 1
            self.current_month = self.model.month

    def step(self):
        if self.check_birth_event():
            pos = self.model.landscape.get_random_point()
            agent_id = self.model.next_agent_id
            self.model.next_agent_id += 1
            species = self.agent.__class__.__name__
            litter_size = self.litter_size()
            print(f'{species} Just gave birth to {litter_size} on month-year {self.model.month}-{self.model.year}')
            for i in range(litter_size):
                self.model.landscape.give_birth(species_name=species, pos=pos, agent_id=agent_id)
            self.litters_left -= 1
            self.months_since_last_litter = 0
        self.increment_time()
 

