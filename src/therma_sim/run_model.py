from model import ThermaSim
import agents
import time

### Population sizes are set as individuals per hectFalseare

def halfway_in_range(r):
    return (r.start + r.stop - 1) / 2

## Landscape Parameters
thermal_data_profile_fp = 'Data/Texas-Marathon_data.csv'
torus = False
moore = False
width=32# 32 hectares to 320000 meters
height=32 # 32 hectares to 320000 meters

# Rattlesnake Parameters
snake_body_sizes = range(370,790+1)
delta_t = 60
initial_body_temperature=25
k=0.01
t_pref_min=18
t_pref_max=32
t_opt = 28 
strike_performance_opt = 0.21
max_meals = 6

# KangarooRat Parameters
krat_body_sizes = range(60, 70+1) #https://www.depts.ttu.edu/nsrl/mammals-of-texas-online-edition/Accounts_Rodentia/Dipodomys_ordii.php
krat_active_hours = [20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6]
# pred-prey parameters
interaction_distance = 0.068 # Parameter is from https://www.nature.com/articles/srep40412/tables/1
krat_cals_per_gram = 1.38 # Technically for ground squirrels. Metric is from Crowell 2021
digestion_efficency = 0.8 # Assumed rate of calorie assimilation. Metric is from crowell 2021
performance_opt = 0.5 #From Grace, Rulon paper 0.21

initial_population_sizes = {'KangarooRat': range(3,14),
                            'Rattlesnake': range(0,1)} # Individuals per hectare

interaction_map = {
    ("Rattlesnake", "KangarooRat"): {
        "interaction_distance": interaction_distance,
        "calories_per_gram": krat_cals_per_gram,
        "digestion_efficiency": digestion_efficency,
        "expected_prey_body_size": halfway_in_range(r=krat_body_sizes),
        "handling_time_range": {"min": 15/60, "max": 180/60}, # scaled to a hour
        "attack_rate_range": {"min": 0/10, "max": 3/10} # 1 to 5 strikes a night assuming there are 10 hour (time steps in a night)
    },}


#Predator_Prey     
input_dictionary = {
    'Landscape_Parameters': {'Thermal_Database_fp':thermal_data_profile_fp,
                             'Width':width,
                             'Height':height,
                             'torus':torus
                             },
    'Initial_Population_Sizes': initial_population_sizes,
    'Rattlesnake_Parameters':{'Body_sizes':snake_body_sizes,
                              'Initial_Body_Temperature': initial_body_temperature,
                              'initial_calories': range(300,600),
                              'max_meals':max_meals,
                              'reproductive_age_years':2,
                              'max_age':20,
                              'max_thermal_accuracy': 5, 
                              'k': k,
                              't_pref_min': t_pref_min,
                              't_pref_max': t_pref_max,
                              't_opt': t_opt,
                              'strike_performance_opt': strike_performance_opt,
                              'delta_t': delta_t,
                              'X1_mass':0.930,
                              'X2_temp': 0.044,
                              'X3_const': -2.58,
                              'annual_survival_probability':0.9, 
                              'brumination_months': [10, 11, 12, 1, 2, 3, 4],
                              'birth_death_module': {
                                            "mean_litter_size": 4.6,
                                            "std_litter_size": 0.31,
                                            "upper_bound_litter_size": 8,
                                            "lower_bound_litter_size": 2,
                                            "litters_per_year": 0.5,
                                            "birth_hazard_rate":1/2, # give birth every other year
                                            "death_hazard_rate":1/18, # die once ever 18 years
                                            },
                              'moore': moore},
    'KangarooRat_Parameters':{'Body_sizes':krat_body_sizes,
                              'active_hours':krat_active_hours,
                              'annual_survival_probability':0.55,
                              'reproductive_age_years':1/4,
                              'max_age':6,
                              'birth_death_module': {
                                            "mean_litter_size": 3.5,
                                            "std_litter_size": 1,
                                            "upper_bound_litter_size": 6,
                                            "lower_bound_litter_size": 1,
                                            "litters_per_year": 1,
                                            "birth_hazard_rate":1, # one litter a year
                                            "death_hazard_rate":1/5, # once every 5 years (1/5)
                                            },
                              'moore': moore},
    'Interaction_Map':interaction_map
                            
}



def main():
    step_count = 1000
    start_time = time.time()
    model = ThermaSim(config=input_dictionary,seed=42)
    model.run_model() #step_count=step_count
    model_data = model.datacollector.get_model_vars_dataframe()
    rattlesnake_data = model.datacollector.get_agenttype_vars_dataframe(agents.Rattlesnake)

    model_data.to_csv("Output_Data/model_data.csv", index=False)
    rattlesnake_data.to_csv("Output_Data/agent_data.csv", index=False)
    run_time = time.time() - start_time
    print(f"Model run completed in {run_time:.2f} seconds.")

    # print(model_data)
    # print(agent_data)



if __name__ ==  "__main__":
    main()

    

    
