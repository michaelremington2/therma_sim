import numpy as np
import ThermaNewt.sim_snake_tb as tn
from numba import njit

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
    def prey_consumed(self):
        return self._log_prey_consumed

    @prey_consumed.setter
    def prey_consumed(self, value):
        self._log_prey_consumed = value

    def reset_log_metrics(self):
        self.handling_time = 0
        self.attack_rate = 0
        self.prey_density = 0
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
    def holling_type_2(prey_density, strike_success, attack_rate, handling_time):
        """
        Computes the Holling Type II functional response.

        Parameters:
        - prey_density (float): Prey density per hectare
        - strike_success (float): Probability of a successful strike
        - attack_rate (float): Area searched per predator per time unit
        - handling_time (float): Handling time per prey item

        Returns:
        - Expected number of prey consumed per predator per time unit
        """
        return ((strike_success * attack_rate) * prey_density) / (1 + (strike_success * attack_rate) * handling_time * prey_density)
    
    def set_utilities(self):
        '''Calculate utilities for behavior selection'''
        db = self.thermal_accuracy_calculator()
        metabolic_state, max_metabolic_state = self.get_metabolic_state_variables()
        thermoregulate_utility = self.scale_value(db, self.snake.max_thermal_accuracy)
        rest_utility = self.scale_value(metabolic_state, max_metabolic_state)
        forage_utility = 1 - rest_utility
        return np.array([rest_utility, thermoregulate_utility, forage_utility])
    
    def set_behavioral_weights(self):
        utilities = self.set_utilities()
        return self.model.softmax_lookup_table.get_probabilities(utilities)
    
    def choose_behavior(self):
        behavior_probabilities = self.set_behavioral_weights()
        return np.random.choice(self.snake.behaviors, p=behavior_probabilities)

    def forage(self):
        '''Foraging behavior logic with optimized functional response calculations'''
        self.snake.current_microhabitat = 'Open'
        self.snake.current_behavior = 'Forage'

        predator_label = self.snake.species_name
        prey_label = self.model.interaction_map.get_prey_for_predator(predator_label=predator_label)[0]

        prey_density_range = self.model.initial_agents_dictionary[prey_label]
        active_prey_population_size = self.model.active_krats_count
        prey_density = self.model.calc_local_population_density(
            population_size=active_prey_population_size,
            middle_range=prey_density_range.start,
            max_density=prey_density_range.stop
        )
        self.prey_density = prey_density

        # Attack rate
        attack_range = self.model.interaction_map.get_attack_rate_range(predator=predator_label, prey=prey_label)
        attack_rate = np.random.uniform(attack_range['min'], attack_range['max'])
        self.attack_rate = attack_rate

        # Handling time
        handling_time_range = self.model.interaction_map.get_handling_time_range(predator=predator_label, prey=prey_label)
        handling_time = np.random.uniform(handling_time_range['min'], handling_time_range['max'])
        self.handling_time = handling_time

        # Compute expected prey consumption using optimized `holling_type_2()`
        expected_prey_consumed = self.holling_type_2(prey_density, self.snake.strike_performance_opt, attack_rate, handling_time)
        prey_consumed = int(np.random.poisson(expected_prey_consumed))
        self.prey_consumed = prey_consumed

        if prey_consumed > 0 and active_prey_population_size > 0:
            for _ in range(prey_consumed):
                prey = self.model.get_active_krat()
                cal_per_gram = self.model.interaction_map.get_calories_per_gram(predator=predator_label, prey=prey.species_name)
                digestion_efficiency = self.model.interaction_map.get_digestion_efficiency(predator=predator_label, prey=prey.species_name)

                # Optimize `cals_gained()` call
                self.snake.metabolism.cals_gained(prey.mass, cal_per_gram, digestion_efficiency)
                prey.alive = False

    def rest(self):
        '''Resting behavior'''
        self.snake.current_microhabitat = 'Burrow'
        self.snake.current_behavior = 'Rest'

    def thermoregulate(self):
        '''Thermoregulation behavior using ThermaNewt'''
        self.snake.current_behavior = 'Thermoregulate'
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

    def step(self):
        '''Handles picking and executing behavior functions'''
        self.reset_log_metrics()
        behavior = self.choose_behavior()
        behavior_actions = {
            'Rest': self.rest,
            'Thermoregulate': self.thermoregulate,
            'Forage': self.forage,
        }
        behavior_actions.get(behavior, lambda: ValueError(f"Unknown behavior: {behavior}"))()
