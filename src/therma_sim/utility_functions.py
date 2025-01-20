#!/usr/bin/python
import numpy as np
from scipy.special import softmax

class EctothermBehavior(object):
    def __init__(self, snake, _statitc_utility_test=False):
        self.snake = snake

    def thermal_accuracy_calculator(self):
        '''
        Helper function for calculating thermal accuracy 
        '''
        if self.snake.body_temperature < self.snake.t_pref_min:
            db = np.abs(float(self.snake.t_pref_min) - float(self.snake.body_temperature))
            state = 'cold'
        elif self.snake.body_temperature > self.snake.t_pref_max:
            db = np.abs(float(self.snake.t_pref_max) - float(self.snake.body_temperature))
            state = 'hot'
        else:
            db = 0.0
            state = 'neutral'
        return db, state
    
    def get_metabolic_state_variables(self):
        return self.snake.metabolism.metabolic_state, self.snake.metabolism.max_metabolic_state
    
    def scale_value(self, value,max_value):
        x = value/max_value
        if x>1:
            x=1
        return x
    
    def set_utilities(self):
        db, thermal_state = self.thermal_accuracy_calculator()
        metabolic_state, max_metabolic_state = self.get_metabolic_state_variables()
        thermoregulate_utility = self.scale_value(value=db, max_value=self.snake.max_thermal_accuracy)
        rest_utility = self.scale_value(value=metabolic_state, max_value=max_metabolic_state)
        forage_utility = 1 - rest_utility
        utility_vector = np.array([rest_utility, thermoregulate_utility, forage_utility])
        return utility_vector
    
    def set_behavioral_weights(self):
        from scipy.special import softmax
        utilities = self.set_utilities()
        masked_utility = np.where(utilities == 0, -np.inf, utilities)
        behavioral_weights = softmax(masked_utility)
        return behavioral_weights
    
    def choose_behavior(self):
        behavior_probabilities = self.set_behavioral_weights()
        behavior = np.random.choice(self.snake.behaviors,p=behavior_probabilities)
        return behavior
        

