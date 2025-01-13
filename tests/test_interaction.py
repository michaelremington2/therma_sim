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
        print(f"Snake initialized with pos: {self.snake.pos}")
        
        self.snake.active = True
        self.snake.body_temperature = 30
        self.snake.t_pref_min = 20
        self.snake.t_pref_max = 40
        self.snake.t_opt = 30
        self.snake.strike_performance_opt = 1

            # Mock metabolism
        self.snake.metabolism = MagicMock()
        self.snake.metabolism.metabolic_state = 100  # Set an initial metabolic state
        self.snake.birth_module = MagicMock()
        self.snake.birth_module.step =  MagicMock()
        self.snake.alive = True
        self.snake.active = True
        self.model.landscape.get_mh_availability_dict = MagicMock(return_value={'Burrow':1,'Open': 1,})
        # self.snake.utility_module.calculate_overall_utility_additive_b1mh2.return_value = {'behavior1': 0.5, 'behavior2': 0.8}
        # self.snake.utility_module.simulate_decision_b1mh2.return_value = ('behavior2', 'mh1')
        self.snake.get_t_env = MagicMock(return_value=25)
        self.snake.metabolism.cals_lost = MagicMock()
        self.snake.update_body_temp = MagicMock()

        # Initialize KangarooRat
        self.krat = KangarooRat(unique_id=2, model=self.model, initial_pos=(0.1, 0.1))
        self.krat.birth_module = MagicMock()
        self.krat.birth_module.step =  MagicMock()
        self.krat.alive = True
        self.krat.active = True
        print(f"KangarooRat initialized with pos: {self.krat.pos}")

        # Add snake
        self.model.add_agents([self.snake])

        # Interaction_Dynamics instance
        self.interaction_dynamics = Interaction_Dynamics(
            model=self.model,
            predator_name="Rattlesnake",
            prey_name="KangarooRat",
            interaction_distance=5,
            calories_per_gram=4.0,
            digestion_efficiency=0.8
        )

    def test_interaction_module_no_prey(self):
        """Test interaction module when no prey is nearby."""
        self.interaction_dynamics.interaction_module(self.snake)
        self.model.step()
        self.assertTrue(self.krat.alive, "Krat should be alive")

    def test_interaction_module_with_prey(self):
        """Test interaction module when prey is nearby."""
        self.model.add_agents([self.krat])
        interaction = self.interaction_dynamics.interaction_module(self.snake, _test=True)
        self.model.step()
        self.assertTrue(interaction)
        self.assertFalse(self.krat.alive)

    def test_strike_probability_success(self):
        """Test strike probability with successful strike."""
        
        strike_probability = self.interaction_dynamics.strike_tpc_ss(
            body_temp=self.snake.body_temperature,
            t_pref_min=self.snake.t_pref_min,
            t_pref_max=self.snake.t_pref_max,
            t_opt=self.snake.t_opt,
            performance_opt=self.snake.strike_performance_opt
        )
        print(strike_probability)
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
