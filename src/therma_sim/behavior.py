#!/usr/bin/python
import numpy as np
import ThermaNewt.sim_snake_tb as tn

class EctothermBehavior(object):
    def __init__(self, snake):
        self.snake = snake
        self.model = self.snake.model
        self.thermoregulation_module = tn.ThermalSimulator(flip_logic='preferred',
                                                           t_pref_min=self.snake.t_pref_min,
                                                           t_pref_max=self.snake.t_pref_max ,
                                                           t_pref_opt=self.snake.t_opt)
        self._log_prey_density = 0  
        self._log_attack_rate = 0  
        self._log_handling_time = 0  
        self._log_prey_consumed = 0
    
    @property
    def prey_density(self):
        """Tracks prey density for bookkeeping purposes."""
        return self._log_prey_density

    @prey_density.setter
    def prey_density(self, value):
        self._log_prey_density = value

    @property
    def attack_rate(self):
        """Tracks attack rate for bookkeeping purposes."""
        return self._log_attack_rate

    @attack_rate.setter
    def attack_rate(self, value):
        self._log_attack_rate = value

    @property
    def handling_time(self):
        """Tracks handling time for bookkeeping purposes."""
        return self._log_handling_time

    @handling_time.setter
    def handling_time(self, value):
        self._log_handling_time = value

    @property
    def prey_consumed(self):
        """Tracks number of prey consumed for bookkeeping purposes."""
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
        '''
        Helper function for calculating thermal accuracy 
        '''
        db = np.abs(float(self.snake.t_opt) - float(self.snake.body_temperature))
        return db
    
    def get_metabolic_state_variables(self):
        return self.snake.metabolism.metabolic_state, self.snake.metabolism.max_metabolic_state
    
    def scale_value(self, value, max_value):
        x = value/max_value
        if x>1:
            x=1
        return x
    
    #def do_i_flip(self, )
    
    def holling_type_2(self, prey_density, strike_success, attack_rate, handling_time):
        """
        Computes the Holling Type II functional response.

        Parameters:
        prey_density (float or array): Prey density (prey per hectare or meter)
        strike_success (float): probability of a successful strike
        attack_rate (float): Attack rate (area searched per predator per time unit)
        handling_time (float): Handling time (time per prey item)

        Returns:
        float or array: Number of prey consumed per predator per unit time
        """
        return ((strike_success*attack_rate) * prey_density) / (1 + (strike_success*attack_rate) * handling_time * prey_density)
    
    def set_utilities(self):
        db = self.thermal_accuracy_calculator()
        metabolic_state, max_metabolic_state = self.get_metabolic_state_variables()
        thermoregulate_utility = self.scale_value(value=db, max_value=self.snake.max_thermal_accuracy)
        rest_utility = self.scale_value(value=metabolic_state, max_value=max_metabolic_state)
        forage_utility = 1 - rest_utility
        utility_vector = np.array([rest_utility, thermoregulate_utility, forage_utility])
        return utility_vector
    
    def set_behavioral_weights(self):
        utilities = self.set_utilities()
        behavioral_weights = self.model.softmax_lookup_table.get_probabilities(utilities)
        return behavioral_weights
    
    def choose_behavior(self):
        behavior_probabilities = self.set_behavioral_weights()
        behavior = np.random.choice(self.snake.behaviors,p=behavior_probabilities)
        return behavior

    def move(self):
        pass
    
    def forage(self):
        '''
        Function to set the state of the individual as resting.
        Resting entails:
         -Inactive
         -In Open
         -AMR instead of SMR
        '''
        self.snake.current_microhabitat = 'Open'
        self.snake.current_behavior = 'Forage'
        #prey density
        predator_label = self.snake.species_name
        prey_label = self.model.interaction_map.get_prey_for_predator(predator_label = predator_label)
        prey_label = prey_label[0] #will make more sophisticated eventually!
        prey_density_range = self.model.initial_agents_dictionary[prey_label]
        active_prey_population_size = self.model.active_krats_count
        prey_density = self.model.calc_local_population_density(population_size = active_prey_population_size, middle_range=prey_density_range.start, max_density=prey_density_range.stop)
        self.prey_density = prey_density
        # attack rate
        attack_range = self.model.interaction_map.get_attack_rate_range(predator=predator_label, prey=prey_label)
        attack_rate = np.random.uniform(attack_range['min'], attack_range['max'])
        self.attack_rate = attack_rate
        # Handling time
        handling_time_range = self.model.interaction_map.get_handling_time_range(predator=predator_label, prey=prey_label)
        handling_time = np.random.uniform(handling_time_range['min'], handling_time_range['max'])
        self.handling_time = handling_time
        #Expected_prey
        expected_prey_consumed = self.holling_type_2(prey_density=prey_density,
                                                     strike_success=self.snake.strike_performance_opt,
                                                     attack_rate=attack_rate, 
                                                     handling_time=handling_time)
        prey_consumed = np.random.poisson(expected_prey_consumed)
        self.prey_consumed = prey_consumed
        if prey_consumed > 0 and active_prey_population_size>0:
                for i in range(prey_consumed):
                    # Get krat
                    prey = self.model.get_active_krat()
                    # Get Interaction Parameters
                    cal_per_gram = self.model.interaction_map.get_calories_per_gram(predator=self.snake.species_name,
                                                                                    prey = prey.species_name)
                    pecent_digestion_cals = self.model.interaction_map.get_digestion_efficiency(predator=self.snake.species_name,
                                                                                                prey = prey.species_name)
                    # Eat Krat
                    self.snake.metabolism.cals_gained(prey_mass=prey.mass,
                                                    cal_per_gram_conversion=cal_per_gram,
                                                    percent_digestion_cals=pecent_digestion_cals)
                    prey.alive = False
                    self.model.remove_agent(prey) 
        # Select a krat at random from the active list
        # Get its body size
        return 

    def rest(self):
        '''
        Function to set the state of the individual as resting.
        Resting entails:
         -Inactive
         -In burrow
         -SMR instead of AMR
        '''
        self.snake.current_microhabitat = 'Burrow'
        self.snake.current_behavior = 'Rest'


    def thermoregulate(self):
        '''
        Behavior function for agents thermoregulating using the package ThermaNewt.
        This function dictates which microhabitat the individual would sample 
        '''
        self.snake.current_behavior = 'Thermoregulate'
        mh = self.thermoregulation_module.do_i_flip(t_body=self.snake.body_temperature,
                                                    burrow_temp=self.snake.model.landscape.burrow_temperature,
                                                    open_temp=self.snake.model.landscape.open_temperature)
        if mh == 'In':
            self.snake.current_microhabitat = 'Burrow'
        elif mh == 'Out':
            self.snake.current_microhabitat = 'Open'
        else:
            print(mh)
            raise ValueError(f"Microhabitat: {mh} has not been programmed into the system")

    def step(self):
        """
        Handles the logistics of picking and executing behavior functions.

        Available behaviors:
        - Rest
        - Thermoregulate
        - Forage
        """
        self.reset_log_metrics()
        behavior = self.choose_behavior()
        behavior_actions = {
            'Rest': self.rest,
            'Thermoregulate': self.thermoregulate,
            'Forage': self.forage,
        }
        action = behavior_actions.get(behavior)
        if action:
            action()
        else:
            raise ValueError(f"Unknown behavior: {behavior}")
        



