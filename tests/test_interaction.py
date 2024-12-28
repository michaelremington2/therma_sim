import unittest
import numpy as np
from therma_sim.interaction import Interaction_Dynamics
from therma_sim.agents import KangarooRat, Rattlesnake
from therma_sim.model import ThermaSim


class TestInteractionDynamics(unittest.TestCase):
    def setUp(self):
        # Mocking model and agents
        self.ni_config = {
            'Landscape_Parameters': {'Thermal_Database_fp':'Data/Texas-Marathon_data.csv',
                                    'Width':3,
                                    'Height':3,
                                    'torus': False
                                    },
            'Initial_Population_Sizes': {'KangarooRat': 1,
                                         'Rattlesnake': 1},
            'Rattlesnake_Parameters':{'Body_sizes':snake_body_sizes,
                                    'Initial_Body_Temperature': initial_body_temperature,
                                    'initial_calories': range(300,600),
                                    'k': k,
                                    't_pref_min': t_pref_min,
                                    't_pref_max': t_pref_max,
                                    't_opt': t_opt,
                                    'strike_performance_opt': strike_performance_opt,
                                    'delta_t': delta_t,
                                    'X1_mass':0.930,
                                    'X2_temp': 0.044,
                                    'X3_const': -2.58,
                                    'background_death_probability':0.000001,
                                    'brumination_months': [10, 11, 12, 1, 2, 3, 4],
                                    'birth_module': {
                                                    "frequency": "biennial",
                                                    "mean_litter_size": 4.6,
                                                    "std_litter_size": 0.31,
                                                    "upper_bound_litter_size": 8,
                                                    "lower_bound_litter_size": 2,
                                                    "litters_per_year": 0.5,
                                                    "partuition_months": [9, 10]
                                                    },
                                    'moore': moore},
            'KangarooRat_Parameters':{'Body_sizes':krat_body_sizes,
                                    'active_hours':krat_active_hours,
                                    'background_death_probability':0.0000001,
                                    'birth_module': {
                                                    "frequency": "monthly",
                                                    "mean_litter_size": 3.5,
                                                    "std_litter_size": 1,
                                                    "upper_bound_litter_size": 6,
                                                    "lower_bound_litter_size": 1,
                                                    "litters_per_year": range(1,2),
                                                    "partuition_months": [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
                                                    },
                                    'moore': moore},
            'Interaction_Parameters':{'Rattlesnake_KangarooRat':{'Interaction_Distance':interaction_distance,
                                                                'Prey_Cals_per_gram': krat_cals_per_gram,
                                                                'Digestion_Efficency':digestion_efficency,}
                                    }
        }   
        self.model = ThermaSim(config=input_dictionary,seed=42)
        model.run_model() #step_count=step_count

        self.snake = MagicMock()
        self.snake.pos = (0, 0)
        self.snake.body_temperature = 30
        self.snake.t_pref_min = 20
        self.snake.t_pref_max = 40
        self.snake.t_opt = 30
        self.snake.strike_performance_opt = 1
        self.snake.metabolism = MagicMock()

        self.krat = MagicMock()
        self.krat.mass = 10
        self.krat.alive = True
        self.krat.active = True

        # Interaction_Dynamics instance
        self.interaction_dynamics = Interaction_Dynamics(
            model=self.mock_model,
            predator_name="Rattlesnake",
            prey_name="KangarooRat",
            interaction_distance=10,
            calories_per_gram=4.0,
            digestion_efficiency=0.8
        )

    def test_check_for_interaction_retired_true(self):
        # Test when prey is within interaction distance
        result = self.interaction_dynamics.check_for_interaction_retired((0, 0), (0, 5))
        self.assertTrue(result)

    def test_check_for_interaction_retired_false(self):
        # Test when prey is outside interaction distance
        result = self.interaction_dynamics.check_for_interaction_retired((0, 0), (15, 15))
        self.assertFalse(result)

    def test_interaction_module_with_prey(self):
        # Mock neighbors to include active KangarooRat
        self.mock_model.landscape.get_neighbors.return_value = [self.krat]
        self.mock_model.randomize_active_snakes.return_value = [self.snake]

        # Call the interaction module
        self.interaction_dynamics.interaction_module()

        # Ensure strike module is called
        self.assertTrue(self.krat.alive is False)
        self.snake.metabolism.cals_gained.assert_called_with(
            prey_mass=10,
            cal_per_gram_conversion=4.0,
            percent_digestion_cals=0.8
        )

    def test_interaction_module_no_prey(self):
        # Mock neighbors to include no active KangarooRat
        self.mock_model.landscape.get_neighbors.return_value = []
        self.mock_model.randomize_active_snakes.return_value = [self.snake]

        # Call the interaction module
        self.interaction_dynamics.interaction_module()

        # Ensure strike module is not called
        self.snake.metabolism.cals_gained.assert_not_called()

    def test_strike_module_successful(self):
        # Test successful strike with a random value <= strike probability
        np.random.random = MagicMock(return_value=0.1)
        self.interaction_dynamics.strike_module(self.krat, self.snake)

        self.assertFalse(self.krat.alive)
        self.snake.metabolism.cals_gained.assert_called_with(
            prey_mass=10,
            cal_per_gram_conversion=4.0,
            percent_digestion_cals=0.8
        )

    def test_strike_module_unsuccessful(self):
        # Test unsuccessful strike with a random value > strike probability
        np.random.random = MagicMock(return_value=0.9)
        self.interaction_dynamics.strike_module(self.krat, self.snake)

        self.assertTrue(self.krat.alive)
        self.snake.metabolism.cals_gained.assert_not_called()

    def test_strike_tpc_ss(self):
        # Mock the TPC.sharpe_schoolfield function
        TPC.sharpe_schoolfield = MagicMock(return_value=0.75)
        result = self.interaction_dynamics.strike_tpc_ss(
            body_temp=30,
            t_pref_min=20,
            t_pref_max=40,
            t_opt=30,
            performance_opt=1
        )

        self.assertEqual(result, 0.75)
        TPC.sharpe_schoolfield.assert_called_with(
            T=30,
            R_ref=1,
            E_A=1,
            E_L=-5,
            T_L=20,
            E_H=5,
            T_H=40,
            T_ref=30
        )

if __name__ == "__main__":
    unittest.main()
