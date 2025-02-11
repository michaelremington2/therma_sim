#!/usr/bin/python
import TPC
import math
import numpy as np
import agents

class Interaction_Map(object):
    """
    Class for handling interactions between predator and prey 
    and managing the interaction map that holds parameters for these relationships.
    
    Args:
        model: The simulation model instance (optional, for integration with the broader model).
        interaction_map (dict): A dictionary mapping (predator, prey) tuples to their interaction parameters.
    """

    def __init__(self, model, interaction_map):
        self.model = model  
        self.interaction_map = interaction_map 
    
    def get_interaction_parameters(self, predator, prey):
        """
        Retrieve all interaction parameters for a given predator-prey pair.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            dict: Dictionary of interaction parameters if found, else None.
        """
        return self.interaction_map.get((predator, prey), None)
    
    def get_prey_for_predator(self, predator_label):
        """
        Extract all prey species associated with a given predator label.
        
        Parameters:
            predator_label (str): The predator species name to filter interactions.
            
        Returns:
            list: A list of unique prey species associated with the given predator.
        """
        return [prey for (predator, prey) in self.interaction_map.keys() if predator == predator_label]

    
    def get_interaction_distance(self, predator, prey):
        """
        Get the interaction (strike) distance between a predator and prey.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            float: Interaction distance if found, else None.
        """
        params = self.get_interaction_parameters(predator, prey)
        return params.get("interaction_distance") if params else None

    def get_calories_per_gram(self, predator, prey):
        """
        Get the calorie value per gram of prey consumed by a predator.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            float: Calories per gram if found, else None.
        """
        params = self.get_interaction_parameters(predator, prey)
        return params.get("calories_per_gram") if params else None
    
    def get_digestion_efficiency(self, predator, prey):
        """
        Get the digestion efficiency of the predator for a given prey.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            float: Digestion efficiency (0-1) if found, else None.
        """
        params = self.get_interaction_parameters(predator, prey)
        return params.get("digestion_efficiency") if params else None

    def get_max_meals(self, predator, prey):
        """
        Get the maximum number of meals a predator can eat from a given prey.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            int: Max meals if found, else None.
        """
        params = self.get_interaction_parameters(predator, prey)
        return params.get("max_meals") if params else None

    def get_strike_success_rate(self, predator, prey):
        """
        Get the strike success rate of a predator attacking a specific prey.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            float: Strike success probability (0-1) if found, else None.
        """
        params = self.get_interaction_parameters(predator, prey)
        return params.get("strike_success_rate") if params else None
    
    def get_expected_prey_body_size(self, predator, prey):
        """
        Get the strike success rate of a predator attacking a specific prey.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            float: Strike success probability (0-1) if found, else None.
        """
        params = self.get_interaction_parameters(predator, prey)
        return params.get("expected_prey_body_size") if params else None
    
    def get_handling_time_range(self, predator, prey):
        """
        Get the handling time range of a predator consuming a specific prey.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            range: range object to be passed into a distribution, else None.
        """
        params = self.get_interaction_parameters(predator, prey)
        return params.get("handling_time_range") if params else None
    
    def get_attack_rate_range(self, predator, prey):
        """
        Get the handling time range of a predator consuming a specific prey.

        Args:
            predator (str): The predator species name.
            prey (str): The prey species name.

        Returns:
            range: range object to be passed into a distribution, else None.
        """
        params = self.get_interaction_parameters(predator, prey)
        return params.get("attack_rate_range") if params else None




## Recode this to be a general predator and prey agent class
# Script Retired (For now)

class Interaction_Dynamics(object):
    '''
    Retired (For Now) - This is a static class that dictates the rules of interactions between a predator and prey.
    Args:
        predator_name - string to match the class
    '''
    def __init__(self,
                 model,                  
                 predator_name: object, 
                 prey_name: object, 
                 interaction_distance: float, 
                 calories_per_gram: float, 
                 digestion_efficiency: float,
                 max_meals: int):
        super().__init__()
        self.model = model
        self.predator_name = predator_name
        self.prey_name = prey_name
        self.interaction_distance = interaction_distance
        self.max_meals = max_meals
        self.prey_mass = 70  # grams
        self._calories_per_gram = calories_per_gram
        self._digestion_efficiency = digestion_efficiency
        self.prey_meal = self.calories_per_gram * self.prey_mass * self.digestion_efficiency
        self.predator_max_metabolic_state = self.prey_meal*max_meals

    @property
    def calories_per_gram(self):
        return self._calories_per_gram

    @calories_per_gram.setter
    def calories_per_gram(self, value):
        if value < 0:
            raise ValueError("Calories per gram must be non-negative")
        self._calories_per_gram = value

    @property
    def digestion_efficiency(self):
        return self._digestion_efficiency

    @digestion_efficiency.setter
    def digestion_efficiency(self, value):
        if not (0 <= value <= 1):
            raise ValueError("Digestion efficiency must be between 0 and 1")
        self._digestion_efficiency = value

                    
    def check_for_interaction_retired(self, snake_point, krat_point):
        '''
        Helper function in the interaction model to test if a snake agent interacts with a krat agent
        '''
        x1, y1 = snake_point
        x2, y2 = krat_point
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist <= self.interaction_distance:
            # print('Strike!')
            # print(f'Distance: {dist}')
            return True
            # distance = maybe collect distance eventually
        else:
            return False
    
    def spatial_interaction_module(self, snake, _test=False):
        '''
        Main interaction model between agents. The model simulates point locations within a hectare then checks if a snake and a krat agent 
        are within striking distance of eachother if they are active.
        '''
        center_pos = snake.pos
        neighbors = self.model.landscape.get_neighbors(center_pos, radius=self.interaction_distance, include_center=False)
        potential_prey = [agent for agent in neighbors if isinstance(agent, agents.KangarooRat) and agent.active]
        if len(potential_prey)>0:
            interaction = True
            krat = np.random.choice(potential_prey)
            self.strike_module(krat=krat, snake=snake)
        else:
            interaction = False
        if _test:
            return interaction
        else:
            return

    def spatial_strike_module(self, krat, snake):
        '''
        Module for simulating a strike between a krat and a snake.
        Args:
            -krat object
            -snake object
        '''
        strike_probability = self.strike_tpc_ss(body_temp=snake.body_temperature,
                                                t_pref_min=snake.t_pref_min,
                                                t_pref_max=snake.t_pref_max, 
                                                t_opt= snake.t_pref_min, 
                                                performance_opt=snake.strike_performance_opt)
        random_value = np.random.random()
        if random_value <= strike_probability:
            print('Successful Strike!')
            # Strike occurs
            prey_mass = krat.mass
            snake.metabolism.cals_gained(prey_mass=prey_mass,
                                           cal_per_gram_conversion=self.calories_per_gram ,
                                           percent_digestion_cals=self.digestion_efficiency)
            krat.alive = False       

    def strike_tpc_ss(self, body_temp, t_pref_min, t_pref_max, t_opt, performance_opt):
        """
        Calculate strike performance using the Sharpe-Schoolfield equation for thermal performance curves.

        Parameters:
        body_temp (float): The current body temperature of the rattlesnake.
        t_pref_min (float): The minimum preferred temperature (lower critical temperature for performance).
        t_pref_max (float): The maximum preferred temperature (upper critical temperature for performance).
        t_opt (float): The optimal temperature where the snake's performance is maximized.
        performance_opt (float): The maximum possible performance at the optimal temperature (R_ref in the Sharpe-Schoolfield equation).

        Returns:
        float: The strike performance at the given body temperature based on the Sharpe-Schoolfield equation. range(0,1)
        """

        # R_ref is the performance at the optimal temperature, which represents the maximum potential strike performance.
        R_ref = performance_opt

        # E_A is the activation energy of the process, which influences how performance increases with temperature. 
        # Here, it's set to 1 for simplicity, but this could be adjusted based on empirical data.
        E_A = 1
        
        # E_L is the deactivation energy at lower temperatures (how quickly performance drops below the lower critical temperature).
        # Set to -5 to reflect stronger performance suppression below the critical threshold.
        E_L = -5
        
        # T_L is the lower critical temperature, which is the minimum temperature where performance starts to decline significantly.
        T_L = t_pref_min
        
        # E_H is the deactivation energy at higher temperatures (how quickly performance drops above the upper critical temperature).
        # Set to 5 to reflect stronger performance suppression above the critical threshold.
        E_H = 5  # Holding this constant for now.

        # T_H is the upper critical temperature, which is the maximum temperature where performance starts to decline significantly.
        T_H = t_pref_max

        # T_ref is the reference temperature (optimal temperature), where the performance is maximized (at R_ref).
        T_ref = t_opt

        # Return the calculated strike performance using the Sharpe-Schoolfield equation from the TPC module.
        return TPC.sharpe_schoolfield(
            T=body_temp,        # Current body temperature
            R_ref=R_ref,        # Max performance at optimal temperature
            E_A=E_A,            # Activation energy for increasing performance with temperature
            E_L=E_L,            # Deactivation energy at low temperatures
            T_L=T_L,            # Lower critical temperature
            E_H=E_H,            # Deactivation energy at high temperatures
            T_H=T_H,            # Upper critical temperature
            T_ref=T_ref         # Optimal temperature
        )
