#!/usr/bin/python
import numpy as np

class Utiility(object):
    def __init__(self, snake, _statitc_utility_test=False):
        self.snake = snake
        self._statitc_utility_test = _statitc_utility_test

    ########################################################################################################
    #
    #### Utility Functions
    #
    ########################################################################################################

    def thermo_accuracy_calc(self, t_pref_min, t_pref_max, t_body):
        if t_body < t_pref_min:
            db = np.abs(float(t_pref_min) - float(t_body))
            state = 'cold'
        elif t_body > t_pref_max:
            db = np.abs(float(t_pref_max) - float(t_body))
            state = 'hot'
        else:
            db = 0.0
            state = 'neutral'
        return db, state


    ########################################################################################################
    #
    #### Choose microhabitat then behavior
    #
    ########################################################################################################
    
    def generate_static_utility_vector_mh1b2(self):
        '''
        This function returns a static dictionary of unadjusted expected utility scores.
        Funcions method is used in:
            - Init
        '''
        utility_scores = {
            'Shrub': {'Rest': 0,'Forage': 4,'Thermoregulate': 4,},
            'Open': {'Rest': 0, 'Forage': 4, 'Thermoregulate': 4,},
            'Burrow': {'Rest': 5,'Forage': 0,'Thermoregulate': 2}
        }
        return utility_scores
    
    def calculate_overall_utility_additive_mh1b2(self, utility_scores, mh_availability, behavior_preferences):
        '''
        Helper function for calculating the additive utility scores adjusted for microhabitat availability in the landscape
        Args:
            - utility_scores: un adjusted utility scores associated with the behavior dictionary
            - availability of habitat dictionary. An example would be
            availability = {
                'Shrub': 0.8,
                'Open': 0.2,
                'Burrow': 1.0
                }
        '''
        overall_utility = {}
        for habitat in utility_scores:
            habitat_utilities = []
            for behavior in utility_scores[habitat]:
                habitat_utilities.append(utility_scores[habitat][behavior] * behavior_preferences[behavior])
            overall_utility[habitat] = sum(habitat_utilities) * mh_availability[habitat]
        return overall_utility
    
    def simulate_decision_mh1b2(self, overall_utility, behaviors, utility_scores):
        '''
        Behavior function for simulating a snakes behavioral decision of what based on utility 
        Args:
            - overal_utility: utility scores associated with the microhabitat adjusted for landscape availability
            - behaviors: List of available behaviors to choose from
            - utility_scores: un adjusted utility scores associated with the behavior dictionary
        Returns:
            - behavior: (str) label of the behavior the organism chooses
            - microhabitat: (str) the microhabitat that the 
        '''
        # Choose microhabitat based on overall utility
        habitat_probs = np.array(list(overall_utility.values()))
        habitat_probs /= np.sum(habitat_probs)
        microhabitat = np.random.choice(list(overall_utility.keys()), p=habitat_probs)
        
        # Choose behavior within the selected microhabitat
        behavior_utilities = [utility_scores[microhabitat][behavior] for behavior in behaviors]
        behavior_probs = np.array(behavior_utilities) / np.sum(behavior_utilities)
        behavior = np.random.choice(behaviors, p=behavior_probs)
        # Potentially think about nesting microhabitat in behavior rather than behavior in microhabitat to see if it makes a difference
        # self.current_behavior=behavior
        # self.current_microhabitat=microhabitat
        return behavior, microhabitat

    def generate_static_utility_vector_b1mh2(self):
        '''
        This function returns a static dictionary of unadjusted expected utility scores.
        '''
        utility_scores = {
            'Rest': {'Shrub': 0,'Open': 0,'Burrow': 5,},
            'Forage': {'Shrub': 4,'Open': 4,'Burrow': 0,},
            'Thermoregulate': {'Shrub': 4,'Open': 4,'Burrow': 2,}
        }
        return utility_scores

    def calculate_overall_utility_additive_b1mh2(self, utility_scores, mh_availability, behavior_preferences):
        '''
        Helper function for calculating the additive utility scores adjusted for behavior availability in the landscape.
        '''
        overall_utility = {}
        for behavior in utility_scores:
            behavior_utilities = []
            for habitat in utility_scores[behavior]:
                adjusted_utility = utility_scores[behavior][habitat] * behavior_preferences[behavior] * mh_availability[habitat]
                behavior_utilities.append(adjusted_utility)
            overall_utility[behavior] = sum(behavior_utilities)
        return overall_utility

    def simulate_decision_b1mh2(self, overall_utility, microhabitats, utility_scores):
        '''
        Behavior function for simulating a snake's behavioral decision based on utility.
        Args:
            - overall_utility: utility scores associated with the behavior adjusted for landscape availability
            - microhabitats: List of available microhabitats to choose from
            - utility_scores: unadjusted utility scores associated with the microhabitat dictionary
        Returns:
            - behavior: (str) label of the behavior the organism chooses
            - microhabitat: (str) the microhabitat that the organism chooses
        '''
        # Normalize overall utility to get valid probabilities for behavior selection
        behavior_probs = np.array(list(overall_utility.values()), dtype=float)
        
        # Handle the case where all utilities are zero (assign equal probabilities)
        if np.sum(behavior_probs) == 0:
            behavior_probs = np.ones_like(behavior_probs) / len(behavior_probs)
        else:
            behavior_probs /= np.sum(behavior_probs)
        
        # Choose behavior based on overall utility probabilities
        behavior = np.random.choice(list(overall_utility.keys()), p=behavior_probs)

        # Choose microhabitat within the selected behavior
        microhabitat_utilities = [utility_scores[behavior][microhabitat] for microhabitat in microhabitats]
        
        # Ensure the microhabitat probabilities are floating-point for division
        microhabitat_probs = np.array(microhabitat_utilities, dtype=float)
        
        # Handle the case where all microhabitat utilities are zero (assign equal probabilities)
        if np.sum(microhabitat_probs) == 0:
            microhabitat_probs = np.ones_like(microhabitat_probs) / len(microhabitat_probs)
        else:
            microhabitat_probs /= np.sum(microhabitat_probs)
        
        # Choose microhabitat based on normalized probabilities
        microhabitat = np.random.choice(microhabitats, p=microhabitat_probs)

        # Update the object's current state
        # self.current_behavior = 
        # self.current_microhabitat = 
        
        return behavior, microhabitat