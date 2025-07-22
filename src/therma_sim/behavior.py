import numpy as np
#import ThermaNewt.sim_snake_tb as tn
from numba import njit
from scipy.special import softmax
from math import isclose

def sparsemax(z):
    """
    Sparsemax: projects input vector z onto probability simplex with possible sparsity.
    Assumes z is a 1D array scaled to reasonable range (e.g., [0,1]).
    """
    z = np.asarray(z, dtype=np.float64)

    # Sort z in descending order
    z_sorted = np.sort(z)[::-1]
    k = np.arange(1, len(z) + 1)  # 1-based indexing for algorithm
    
    # Determine k_max: largest k where threshold holds
    z_cumsum = np.cumsum(z_sorted)
    condition = 1 + k * z_sorted > z_cumsum
    if not np.any(condition):
        # fallback: uniform distribution if no support found
        return np.ones_like(z) / len(z)
    k_z = k[condition][-1]
    
    # Compute threshold tau
    tau_z = (z_cumsum[k_z - 1] - 1) / k_z

    # Compute projection onto simplex
    p = np.maximum(z - tau_z, 0)
    p /= p.sum()  # ensure probabilities sum to 1 for numerical stability
    return p

class EctothermBehavior(object):
    def __init__(self, snake):
        self.snake = snake
        self.model = self.snake.model
        # self.thermoregulation_module = tn.ThermalSimulator(flip_logic='preferred',
        #                                                    t_pref_min=self.snake.t_pref_min,
        #                                                    t_pref_max=self.snake.t_pref_max,
        #                                                    t_pref_opt=self.snake.t_opt)
        self._log_prey_density = 0  
        self._log_attack_rate = 0  
        self._log_handling_time = 0 
        self._log_prey_encountered = 0 
        self._log_prey_consumed = 0
    
    @property
    def prey_density(self):
        return self._log_prey_density

    @prey_density.setter
    def prey_density(self, value):
        self._log_prey_density = value

    @property
    def attack_rate(self):
        return self._log_attack_rate

    @attack_rate.setter
    def attack_rate(self, value):
        self._log_attack_rate = value

    @property
    def handling_time(self):
        return self._log_handling_time

    @handling_time.setter
    def handling_time(self, value):
        self._log_handling_time = value

    @property
    def prey_encountered(self):
        return self._log_prey_encountered

    @prey_encountered.setter
    def prey_encountered(self, value):
        self._log_prey_encountered = value

    @property
    def prey_consumed(self):
        return self._log_prey_consumed

    @prey_consumed.setter
    def prey_consumed(self, value):
        self._log_prey_consumed = value

    def reset_log_metrics(self):
        self.handling_time = 0
        self.attack_rate = 0
        self.prey_density = 0
        self.prey_encountered = 0
        self.prey_consumed = 0

    def thermal_accuracy_calculator(self):
        '''Calculate thermal accuracy'''
        return np.abs(float(self.snake.t_opt) - float(self.snake.body_temperature))
    
    def get_metabolic_state_variables(self):
        return self.snake.metabolism.metabolic_state, self.snake.metabolism.max_metabolic_state
    
    @staticmethod
    @njit
    def scale_value(value, max_value):
        '''Numba-optimized function to normalize values between 0 and 1'''
        x = value / max_value
        return min(x, 1.0)

    @staticmethod
    @njit
    def holling_type_2(prey_density, attack_rate, handling_time, strike_success=1):
        """
        Computes the Holling Type II functional response.

        Parameters:
        - prey_density (float): Prey density per hectare
        - attack_rate (float): Area searched per predator per time unit
        - handling_time (float): Handling time per prey item
        - strike_success (float): Probability of a successful strike.
            Default of 1: Use this argument to calculate number of encounters
            less than 1: Function calculates successful prey items caught.

        Returns:
        - Expected number of prey consumed per predator per time unit
        """
        if strike_success>1:
            raise(ValueError("Strike success is a probability that cant exceed 1"))
        return ((strike_success * attack_rate) * prey_density) / (1 + (strike_success * attack_rate) * handling_time * prey_density)

    def forage(self):
        '''Foraging behavior logic with optimized functional response calculations'''
        self.snake.current_microhabitat = 'Open'
        self.snake.current_behavior = 'Forage'
        self.snake.active = True

        predator_label = self.snake.species_name
        prey_label = self.model.interaction_map.get_prey_for_predator(predator_label=predator_label)[0]
        self.prey_density = self.model.active_krats_count / self.model.landscape.landscape_size

        # Attack rate
        attack_range = self.model.interaction_map.get_attack_rate_range(predator=predator_label, prey=prey_label)
        if attack_range['min']!=attack_range['max']:
            attack_rate = np.random.uniform(attack_range['min'], attack_range['max'])
        else:
            attack_rate = attack_range['min']
        self.attack_rate = attack_rate

        # Handling time
        handling_time_range = self.model.interaction_map.get_handling_time_range(predator=predator_label, prey=prey_label)
        if handling_time_range['min']!=handling_time_range['max']:
            handling_time = np.random.uniform(handling_time_range['min'], handling_time_range['max'])
        else:
            handling_time = handling_time_range['min']
        self.handling_time = handling_time

        # switched to using holling 2 function for just consumption rate`
        prey_encountered = self.holling_type_2(self.prey_density,  attack_rate, handling_time, strike_success=self.snake.strike_performance_opt)
        self.prey_encountered = prey_encountered
        prey_consumed = int(np.random.poisson(prey_encountered)) 
        if prey_consumed> 0 and self.model.active_krats_count >= prey_encountered:
            prey = self.model.get_active_krat()
            cal_per_gram = self.model.interaction_map.get_calories_per_gram(predator=predator_label, prey=prey.species_name)
            digestion_efficiency = self.model.interaction_map.get_digestion_efficiency(predator=predator_label, prey=prey.species_name)
            self.snake.metabolism.cals_gained(prey.mass, cal_per_gram, digestion_efficiency)
            prey.alive = False
            prey.cause_of_death = 'predation'
            self.model.logger.log_data(file_name = self.model.output_folder+"BirthDeath.csv",
                                        data=prey.birth_death_module.report_data(event_type='Death'))
            if self.snake.searching_behavior:
                self.snake.search_counter = handling_time
            self.model.remove_agent(prey)
        self.prey_consumed = prey_consumed

    def rest(self):
        '''Resting behavior'''
        self.snake.current_microhabitat = 'Burrow'
        self.snake.current_behavior = 'Rest'
        self.snake.active = False

    def search(self):
        '''looking for a prey item that has been hit behavior'''
        self.snake.current_microhabitat = 'Open'
        self.snake.current_behavior = 'Search'
        self.snake.active = True
        self.snake.search_counter -= 1

    def bruminate(self):
        '''overwintering behavior'''
        self.snake.current_microhabitat = 'Winter_Burrow'
        self.snake.current_behavior = 'Brumation'
        self.snake.body_temperature = self.snake.brumation_temp
        self.snake.active = False

    ## Thermoregulation calculation
    def calc_prob_preferred_topt(self, t_body, t_pref_opt, t_pref_max, t_pref_min):
        if t_body >= t_pref_opt:
            prob_flip = ((t_body - t_pref_opt) / (t_pref_max - t_pref_opt))
        elif t_body < t_pref_opt:
            prob_flip = ((t_pref_opt - t_body) / (t_pref_opt - t_pref_min))
        else:
            raise ValueError("Something is messed up")
        if prob_flip > 1:
            prob_flip = 1
        return prob_flip

    def best_habitat_t_opt(self, t_body,burrow_temp, open_temp):
        if t_body > self.snake.t_opt and burrow_temp < open_temp:
            flip_direction = 'Burrow'
        elif t_body < self.snake.t_opt and burrow_temp > open_temp:
            flip_direction = 'Burrow'
        else:
            flip_direction = 'Open'
        return flip_direction
    
    def preferred_topt(self, t_body, burrow_temp, open_temp):
        """Determines if the snake should switch microhabitats based on preferred temperatures."""
        
        # Calculate the probability of flipping microhabitats based on the snake's body temperature.
        prob_flip = self.calc_prob_preferred_topt(
            t_body=t_body,
            t_pref_opt=self.snake.t_opt,
            t_pref_max=self.snake.t_pref_max, 
            t_pref_min=self.snake.t_pref_min
        )  

        # If the body temperature is nearly optimal OR the two microhabitat temperatures are nearly equal,
        # retain the current state (or randomly choose if not set).
        if isclose(t_body, self.snake.t_opt, abs_tol=0.01) or isclose(burrow_temp, open_temp, abs_tol=0.01):
            return self.snake.current_microhabitat

        # Decide to flip microhabitats based on a random draw and the calculated probability.
        if np.random.random() <= prob_flip:
            bu = self.best_habitat_t_opt(t_body = t_body, burrow_temp=burrow_temp, open_temp=open_temp)
            return bu
        else: 
            return self.snake.current_microhabitat


    def thermoregulate(self):
        '''Thermoregulation behavior based on preferred sub module'''
        self.snake.current_behavior = 'Thermoregulate'
        self.snake.active = True
        mh = self.preferred_topt(
            t_body=self.snake.body_temperature,
            burrow_temp=self.snake.model.landscape.burrow_temperature,
            open_temp=self.snake.model.landscape.open_temperature
        )
        self.snake.current_microhabitat = mh
        
    # Behavioral Algorithm
    def set_utilities(self):
        '''Calculate utilities for behavior selection'''
        if self.model.hour in self.snake.active_hours:
            db = self.thermal_accuracy_calculator()
            metabolic_state, max_metabolic_state = self.get_metabolic_state_variables()
            thermoregulate_utility = self.scale_value(db, self.snake.max_thermal_accuracy)
            rest_utility = self.scale_value(metabolic_state, max_metabolic_state)
            forage_utility = 1 - rest_utility
        else:
            rest_utility = 1
            thermoregulate_utility = 0
            forage_utility = 0
        return np.array([rest_utility, thermoregulate_utility, forage_utility])
    
    # Changing from softmax to sparsemax for behavioral weights
    # def set_behavioral_weights(self,utl_temperature=1.0):
    #     utilities = self.set_utilities()
    #     if np.allclose(utilities, 0):
    #         return np.ones_like(utilities) / len(utilities)  # Avoid divide-by-zero
    #     masked_utilities = np.where(utilities == 0, -np.inf, utilities)
    #     return softmax(masked_utilities / utl_temperature)

        # def choose_behavior(self):
    #     behavior_probabilities = self.set_behavioral_weights(utl_temperature=self.snake.utility_temperature)
    #     return np.random.choice(self.snake.emergent_behaviors, p=behavior_probabilities)

    def set_behavioral_weights(self):
        '''Calculate sparsemax-based probabilities for behavior selection'''
        utilities = self.set_utilities()
        if np.allclose(utilities, 0):
            return np.ones_like(utilities) / len(utilities)  # Avoid divide-by-zero
        return sparsemax(utilities)
    
    def choose_behavior(self):
        '''Choose a behavior stochastically from sparsemax probabilities'''
        behavior_probabilities = self.set_behavioral_weights()
        return np.random.choice(self.snake.emergent_behaviors, p=behavior_probabilities)

    def step(self):
        '''Handles picking and executing behavior functions'''
        self.reset_log_metrics()
        if self.snake.is_bruminating_today():
            self.bruminate()
        elif self.snake.search_counter > 0:
            self.search()
        elif self.snake.birth_death_module.ct_out_of_bounds_tcounter>0:
            self.thermoregulate()
        else:
            behavior = self.choose_behavior()
            behavior_actions = {
                'Rest': self.rest,
                'Thermoregulate': self.thermoregulate,
                'Forage': self.forage,
            }
            behavior_actions.get(behavior, lambda: ValueError(f"Unknown behavior: {behavior}"))()
