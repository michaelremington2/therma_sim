import unittest
import numpy as np
from therma_sim.interaction import Interaction_Dynamics
from therma_sim.agents import KangarooRat, Rattlesnake
from therma_sim.model import ThermaSim
import empty_landscape as eu
from unittest.mock import MagicMock, create_autospec

class TestInteractionDynamics(unittest.TestCase):
    def setUp(self):
        # Mocking model and agents
        self.model = eu.ContinuousLandscapeModel(width=5, height=5)

        self.snake = Rattlesnake(unique_id=1, model=self.model, initial_pos=(0, 0))
        self.snake.active = True
        self.snake.body_temperature = 30
        self.snake.t_pref_min = 20
        self.snake.t_pref_max = 40
        self.snake.t_opt = 30
        self.snake.strike_performance_opt = 1
        self.snake.metabolism = MagicMock()
        self.snake.alive = True
        self.snake.active = True

        self.krat = KangarooRat(unique_id=1, model=self.model, initial_pos=(0.1, 0.1))
        self.krat.mass = 10
        self.krat.alive = True
        self.krat.active = True

        # Add snake
        self.model.add_agents([self.snake])

        # Interaction_Dynamics instance
        self.interaction_dynamics = Interaction_Dynamics(
            model=self.model,
            predator_name="Rattlesnake",
            prey_name="KangarooRat",
            interaction_distance=1,
            calories_per_gram=4.0,
            digestion_efficiency=0.8
        )

    # def test_check_for_interaction_retired(self):
    #     """Test the retired check_for_interaction method."""
    #     result = self.interaction_dynamics.check_for_interaction_retired(
    #         snake_point=(0, 0), krat_point=(0.5, 0.5)
    #     )
    #     self.assertTrue(result)

    #     result = self.interaction_dynamics.check_for_interaction_retired(
    #         snake_point=(0, 0), krat_point=(2, 2)
    #     )
    #     self.assertFalse(result)

    def test_interaction_module_no_prey(self):
        """Test interaction module when no prey is nearby."""
        self.interaction_dynamics.interaction_module(self.snake)
        self.model.step()
        self.assertTrue(self.krat.alive, "Krat should be alive")

    def test_interaction_module_with_prey(self):
        """Test interaction module when prey is nearby."""
        self.model.add_agents([self.krat])
        self.interaction_dynamics.interaction_module(self.snake)
        self.model.step()
        self.assertFalse(self.krat.alive)

    def test_strike_probability_success(self):
        """Test strike probability with successful strike."""
        np.random.random = MagicMock(return_value=0.0001)
        strike_probability = self.interaction_dynamics.strike_tpc_ss(
            body_temp=self.snake.body_temperature,
            t_pref_min=self.snake.t_pref_min,
            t_pref_max=self.snake.t_pref_max,
            t_opt=self.snake.t_opt,
            performance_opt=self.snake.strike_performance_opt
        )
        self.assertGreaterEqual(strike_probability, 0)
        self.assertLessEqual(strike_probability, 1)

    # def test_strike_probability_failure(self):
    #     """Test strike probability with unsuccessful strike."""
    #     np.random.random = MagicMock(return_value=1) # Large value to indicate it wont happen
    #     self.interaction_dynamics.strike_module(krat=self.krat, snake=self.snake)
    #     self.snake.metabolism.cals_gained.assert_not_called()
    #     self.assertTrue(self.krat.alive)


if __name__ == "__main__":
    unittest.main()
