import pytest
from unittest.mock import MagicMock
import pandas as pd
from therma_sim.landscape import Continous_Landscape, Discrete_Landscape

@pytest.fixture
def thermal_profile_csv():
    return 'Data/thermal_db.csv'

def test_continuous_landscape_init(thermal_profile_csv):
    # Mock the model object
    mock_model = MagicMock()
    cl = Continous_Landscape(model=mock_model, thermal_profile_csv_fp=str(thermal_profile_csv),
                              width=10000, height=10000, torus=False)
    
    # Assert hectare conversion works
    assert cl.width_hectare == 1
    assert cl.height_hectare == 1
    
    # Assert that the thermal profile is loaded correctly
    assert isinstance(cl.thermal_profile, pd.DataFrame)
    assert not cl.thermal_profile.empty

def test_set_landscape_temperatures_continuous(thermal_profile_csv):
    mock_model = MagicMock()
    cl = Continous_Landscape(model=mock_model, thermal_profile_csv_fp=str(thermal_profile_csv),
                              width=100, height=100, torus=False)
    
    # Mock the thermal profile
    cl.thermal_profile = pd.DataFrame({
        "Open_mean_Temperature": [30],
        "Open_stddev_Temperature": [1.5],
        "Burrow_mean_Temperature": [20],
        "Burrow_stddev_Temperature": [1]
    })
    
    # Call the method and assert temperatures are set correctly
    cl.set_landscape_temperatures(step_id=0)
    assert cl.burrow_temperature == 20
    assert cl.open_temperature == 30


