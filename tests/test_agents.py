import unittest
import numpy as np
from unittest.mock import MagicMock
from therma_sim.agents import Rattlesnake, KangarooRat
import empty_landscape as eu

class TestRattlesnake(unittest.TestCase):
    def setUp(self):
        
        self.mock_model = eu.ContinuousLandscapeModel(width=5, height=5)
        self.mock_model.month = 5
        snake_config = {
            'initial_calories': 1000,
            'X1_mass': 1.0,
            'X2_temp': 1.0,
            'X3_const': 1.0,
            'Body_sizes': [500, 1000],
            'moore': True,
            'brumination_months': [11, 12, 1],
            'background_death_probability': 0.01,
            'delta_t': 1.0,
            'Initial_Body_Temperature': 25.0,
            'k': 0.1,
            't_pref_min': 20.0,
            't_pref_max': 30.0,
            't_opt': 25.0,
            'strike_performance_opt': 0.9,
            'birth_module': {
                'frequency': 'yearly',
                'mean_litter_size': 10,
                'std_litter_size': 2,
                'upper_bound_litter_size': 15,
                'lower_bound_litter_size': 5,
                'litters_per_year': 2,
                'partuition_months': [5, 6, 7]
            }
        }
        self.snake = Rattlesnake(unique_id=1, model=self.mock_model, initial_pos=(50, 50), snake_config=snake_config)

    def test_set_mass(self):
        mass = self.snake.set_mass(body_size_range=[500, 1000])
        self.assertGreaterEqual(mass, 500)
        self.assertLessEqual(mass, 1000)

    def test_update_body_temp(self):
        t_env = 20.0
        old_temp = self.snake.body_temperature
        self.snake.update_body_temp(t_env=t_env, delta_t=1.0)
        self.assertNotEqual(self.snake.body_temperature, old_temp)

    def test_is_starved(self):
        self.snake.metabolism.metabolic_state = -1
        self.snake.is_starved()
        self.assertFalse(self.snake.alive)

    def test_random_death(self):
        np.random.seed(0)  
        self.snake.random_death()
        self.assertTrue(self.snake.alive) 

    def test_log_choice(self):
        self.snake.log_choice(microhabitat="Open", behavior="Forage", body_temp=25.0)
        self.assertEqual(self.snake.microhabitat_history[-1], "Open")
        self.assertEqual(self.snake.behavior_history[-1], "Forage")
        self.assertEqual(self.snake.body_temp_history[-1], 25.0)


class TestKangarooRat(unittest.TestCase):
    def setUp(self):
        self.mock_model = eu.ContinuousLandscapeModel(width=5, height=5)
        self.mock_model.month = 5
        krat_config = {
            'active_hours': [0, 1, 2, 3, 20, 21, 22, 23],
            'Body_sizes': [10, 15],
            'moore': True,
            'background_death_probability': 0.01,
            'birth_module': {
                'frequency': 'monthly',
                'mean_litter_size': 5,
                'std_litter_size': 1,
                'upper_bound_litter_size': 7,
                'lower_bound_litter_size': 3,
                'litters_per_year': 3,
                'partuition_months': [3, 6, 9]
            }
        }
        self.krat = KangarooRat(unique_id=2, model=self.mock_model, initial_pos=(20, 20), krat_config=krat_config)

    def test_activate_krat(self):
        self.krat.activate_krat(hour=2)
        self.assertTrue(self.krat.active)
        self.krat.activate_krat(hour=10)
        self.assertFalse(self.krat.active)

    def test_random_death(self):
        np.random.seed(1)  # Set seed for reproducibility
        self.krat.random_death()
        self.assertTrue(self.krat.alive)  # Should not die with low probability

    def test_generate_random_pos(self):
        self.krat.generate_random_pos()
        x, y = self.krat.pos
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 100)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y, 100)

    def test_set_mass(self):
        mass = self.krat.set_mass(body_size_range=[10, 15])
        self.assertGreaterEqual(mass, 10)
        self.assertLessEqual(mass, 15)


if __name__ == "__main__":
    unittest.main()
