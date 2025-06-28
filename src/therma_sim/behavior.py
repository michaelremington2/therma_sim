import numpy as np
import ThermaNewt.sim_snake_tb as tn
from numba import njit
from scipy.special import softmax

class EctothermBehavior(object):
    def __init__(self, snake):
        self.snake = snake
        self.model = self.snake.model
        self.thermoregulation_module = tn.ThermalSimulator(flip_logic='preferred',
                                                           t_pref_min=self.snake.t_pref_min,
                                                           t_pref_max=self.snake.t_pref_max,
                                                           t_pref_opt=self.snake.t_opt)
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

        # prey_min_density, prey_max_density = self.model.get_population_densities(species=prey_label)
        # active_prey_population_size = 
        # prey_density = self.model.calc_local_population_density(
        #     population_size=active_prey_population_size,
        #     middle_range=prey_min_density,
        #     max_density=prey_max_density
        # )
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

        # switched to using holling 2 function for just encounter rate 03/16`
        expected_prey_encountered = self.holling_type_2(self.prey_density,  attack_rate, handling_time)
        prey_encountered = int(np.random.poisson(expected_prey_encountered))
        self.prey_encountered = prey_encountered
        prey_consumed = 0 
        if prey_encountered> 0 and self.model.active_krats_count >= prey_encountered:
#            for _ in range(prey_encountered): removing this with the assumption only one kangaroo rat encounter per time
            strike_check = np.random.rand()
            if strike_check < self.snake.strike_performance_opt:
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
                prey_consumed +=1
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

    def thermoregulate(self):
        '''Thermoregulation behavior using ThermaNewt'''
        self.snake.current_behavior = 'Thermoregulate'
        self.snake.active = True
        mh = self.thermoregulation_module.do_i_flip(
            t_body=self.snake.body_temperature,
            burrow_temp=self.snake.model.landscape.burrow_temperature,
            open_temp=self.snake.model.landscape.open_temperature
        )
        if mh == 'In':
            self.snake.current_microhabitat = 'Burrow'
        elif mh == 'Out':
            self.snake.current_microhabitat = 'Open'
        else:
            raise ValueError(f"Microhabitat: {mh} has not been programmed into the system")
        
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
    
    def set_behavioral_weights(self,utl_temperature=1.0):
        utilities = self.set_utilities()
        if np.allclose(utilities, 0):
            return np.ones_like(utilities) / len(utilities)  # Avoid divide-by-zero
        masked_utilities = np.where(utilities == 0, -np.inf, utilities)
        return softmax(masked_utilities / utl_temperature)
        
    def choose_behavior(self):
        behavior_probabilities = self.set_behavioral_weights(utl_temperature=self.snake.utility_temperature)
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
