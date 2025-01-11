#!/usr/bin/python
from . import TPC
import math
import numpy as np
from . import agents
import sys
print(sys.path)

class Interaction_Dynamics(object):
    '''
    This is a static class that dictates the rules of interactions between a predator and prey.
    Args:
        predator_name - string to match the class
    '''
    def __init__(self,
                 model,                  
                 predator_name: str, 
                 prey_name: str, 
                 interaction_distance: float, 
                 calories_per_gram: float, 
                 digestion_efficiency: float):
        self.model = model
        self.predator_name = predator_name
        self.prey_name = prey_name
        self.interaction_distance = interaction_distance
        self.calories_per_gram = calories_per_gram
        self.digestion_efficiency = digestion_efficiency

                    
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
    
    def interaction_module(self, snake, _test):
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

    def strike_module(self, krat, snake):
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
