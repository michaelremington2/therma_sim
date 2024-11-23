#!/usr/bin/python
import numpy as np
from scipy.stats import truncnorm

class birth_module(object):
    def __init__(self, model, agent,
                frequency: str, mean_litter_size: float, std_litter_size: float,
                upper_bound_litter_size: int, lower_bound_litter_size: int,
                litters_per_year: float, time_since_last_litter: int,
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
        self.time_since_last_litter = time_since_last_litter
        self.a, self.b = (self.lower_bound_litter_size - self.mean_litter_size) / self.std_litter_size, (self.upper_bound_litter_size - self.mean_litter_size) / self.std_litter_size

    def litter_size(self):
        '''
        Helper function for designating if a birth event will occur for an individual
        '''
        # if self.agent.sex == 'Female' and self.model.month in self.partuition_months:
        #     offspring = 
        # else:
        #     offspring = 0
        return truncnorm.rvs(self.a, self.b, loc=self.mean_litter_size, scale=self.std_litter_size, size=1)[0]
    
    def check_birth_event(self):
        """Check if the agent can give birth in the current month."""
        if self.agent.sex == 'Female' and self.model.current_month in self.partuition_months:
            if self.frequency == 'biannual' and self.model.current_month % 6 == 0:
                return True
            elif self.frequency == 'monthly':
                return True
        return False

    def give_birth(self):
        """Perform a birth event if conditions are met."""
        if self.check_birth_event():
            num_offspring = self.litter_size()
            offspring = []
            for _ in range(num_offspring):
                # Create a new agent (this depends on your model's agent initialization)
                new_agent = self.model.create_agent(species=self.agent.species)
                offspring.append(new_agent)
            return offspring
        return []
