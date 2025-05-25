import yaml
import os
from pathlib import Path
import shutil

BASE_YAML = "test.yaml"
BASE_JOB = "test.job"
EXPERIMENTS_DIR = "experiments"
software_path = 'run_model.py'

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def create_experiment(study_site, parameter_key, parameter_value, yaml_updates, job_replacements):
    # Setup directory
    exp_dir = Path(EXPERIMENTS_DIR) / study_site / f"{parameter_key}_{parameter_value}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load and update YAML
    with open(BASE_YAML, "r") as f:
        config = yaml.safe_load(f)

    config = recursive_update(config, yaml_updates)

    yaml_out = exp_dir / "config.yaml"
    with open(yaml_out, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Replace job file template
    with open(BASE_JOB, "r") as f:
        job_template = f.read()

    for key, val in job_replacements.items():
        job_template = job_template.replace(f"{{{{{key}}}}}", str(val))

    job_out = exp_dir / "run.job"
    with open(job_out, "w") as f:
        f.write(job_template)

    print(f"Created experiment in {exp_dir}")

# === Example experiment ===
sites = {"Canada": "canada_16yr_climate_sim.csv",
         'Nebraska': 'nebraska_16yr_climate_sim.csv',
         'Texas': 'texas_16yr_climate_sim.csv'}
climate_variables = {"Current": ('Open_mean_Temperature', 'Burrow_mean_Temperature'),
                     "One_degree": ("open_plus_1", "burrow_plus_1"),
                     "Two_degree": ("open_plus_2", "burrow_plus_2"),
                     "Three_degree": ("open_plus_3", "burrow_plus_3"),}

param_key = "max_meals"
param_value = 5
exp_params_and_values = {
    {   "Landscape_Parameters": {
        "site_name": site,
        "Thermal_Database_fp": sites[site],
        "landscape_size": [(15,15), (25,25), (35, 35)],},
        "Rattlesnake_Parameters": {
        "max_meals": [2, 3, 4, 5],
        "max_thermal_accuracy": [3, 4, 5],
        "max_energy": [1000, 2000, 3000],
        "strike_performance_opt": [0.05, 0.1, 0.22, 0.3],
        'annual_survival_probability': [0.6, 0.7, 0.8, 0.9],
        'behavioral_utility_temperature': [1, 2, 3, 4],
        'behavior_activity_coefficients': [{'Rest':1,
                                            'Thermoregulate':2,
                                            'Forage':2,
                                            'Search':2,
                                            'Brumation':1}, 
                                            {'Rest':1,
                                            'Thermoregulate':1.5,
                                            'Forage':1.5,
                                            'Search':1.5,
                                            'Brumation':1}]
        },
        "KangarooRat_Parameters": {
            "annual_survival_probability": [0.6, 0.7, 0.8],
            "energy_budget": [6, 7, 8]
        },
        "Interaction_Map":{
            'handling_time_range': [1, 3, 5],
            'attack_rate_range': [0.001, 0.0015, 0.002],
        }

}}
# Need to set a simulation id and seed
for site in sites:
    yaml_mods = {
        "Landscape_Parameters": {"site_name": site},
        "Rattlesnake_Parameters": {param_key: param_value}
    }

    job_vars = {
        "JOB_NAME": f"{site}_{param_key}_{param_value}",
        "CONFIG_PATH": f"{EXPERIMENTS_DIR}/{site}/{param_key}_{param_value}/config.yaml",
        "PYTHON_SCRIPT": "/path/to/main_simulation.py",
        "OUTPUT_DIR": f"{EXPERIMENTS_DIR}/{site}/{param_key}_{param_value}"
    }

    create_experiment(site, param_key, param_value, yaml_mods, job_vars)
