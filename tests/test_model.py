from unittest.mock import MagicMock
import pytest
from therma_sim.model import ThermaSim
import therma_sim.agents as ag

import warnings

# Ignore Mesa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mesa")
warnings.filterwarnings("ignore", category=FutureWarning, module="mesa")


@pytest.fixture
def test_model():
    """Fixture for creating an instance of EctothermMetabolism."""
    mock_config = {
        "Landscape_Parameters": {
            "Thermal_Database_fp": "Data/thermal_db.csv",
            "Width": 10000,
            "Height": 10000,
            "torus": False,
        },
    'Rattlesnake_Parameters':{'Body_sizes':range(60,70),
                              'Initial_Body_Temperature': 25,
                              'initial_calories': range(300,600),
                              'k': 0.01,
                              't_pref_min': 32,
                              't_pref_max': 18,
                              't_opt': 28,
                              'strike_performance_opt': 0.3,
                              'delta_t': 70,
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
                              'moore': True},
    'KangarooRat_Parameters':{'Body_sizes':range(10,20),
                              'active_hours':list(range(0,25)),
                              'background_death_probability':0.0000001,
                              "moore": True,
                              'birth_module': {
                                            "frequency": "monthly",
                                            "mean_litter_size": 3.5,
                                            "std_litter_size": 1,
                                            "upper_bound_litter_size": 6,
                                            "lower_bound_litter_size": 1,
                                            "litters_per_year": range(1,2),
                                            "partuition_months": [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
                                            },},
        "Interaction_Parameters": {
            "Rattlesnake_KangarooRat": {
                "Interaction_Distance": 10,
                "Prey_Cals_per_gram": 5,
                "Digestion_Efficency": 0.8,
            }
        },
        "Initial_Population_Sizes": {"Rattlesnake": range(5, 10), "KangarooRat": range(20, 30)},
    }
    
    model = ThermaSim(config=mock_config)
    return model

def test_thermasim_initialization(test_model):
    assert test_model.landscape is not None
    assert test_model.schedule is not None
    assert test_model.kr_rs_interaction_module is not None
    assert test_model.datacollector is not None
    assert test_model.step_id == 0
    assert test_model.running is True

def test_initialize_populations(test_model):
    snake_count = len(test_model.schedule._agents_by_type[ag.Rattlesnake])
    krat_count = len(test_model.schedule._agents_by_type[ag.KangarooRat])
    assert snake_count >= 5
    assert snake_count <= 10
    assert krat_count >= 20
    assert krat_count <= 30

if __name__=='__main__':
    pass