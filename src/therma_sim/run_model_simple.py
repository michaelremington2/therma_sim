# from model import ThermaSim
# import agents
# import time

# ### Population sizes are set as individuals per hectFalseare
# co{
#     "Landscape_Parameters": {
#         "Thermal_Database_fp": "Data/thermal_db.csv",
#         "Width": 32,
#         "Height": 32,
#         "torus": false,
#         "moore": false
#     },
#     "Initial_Population_Sizes": {
#         "KangarooRat": {"start": 3, "stop": 14, "step": 1},
#         "Rattlesnake": {"start": 0, "stop": 1, "step": 1}
#     },
#     "Rattlesnake_Parameters": {
#         "Body_sizes": {"start": 370, "stop": 791, "step": 1},
#         "Initial_Body_Temperature": 25,
#         "initial_calories": {"start": 300, "stop": 601, "step": 1},
#         "max_meals": 6,
#         "reproductive_age_years": 2,
#         "max_age": 20,
#         "max_thermal_accuracy": 5,
#         "k": 0.01,
#         "t_pref_min": 18,
#         "t_pref_max": 32,
#         "t_opt": 28,
#         "strike_performance_opt": 0.21,
#         "delta_t": 60,
#         "X1_mass": 0.930,
#         "X2_temp": 0.044,
#         "X3_const": -2.58,
#         "annual_survival_probability": 0.9,
#         "brumination_months": [10, 11, 12, 1, 2, 3, 4],
#         "birth_death_module": {
#             "mean_litter_size": 4.6,
#             "std_litter_size": 0.31,
#             "upper_bound_litter_size": 8,
#             "lower_bound_litter_size": 2,
#             "litters_per_year": 1,
#             "birth_hazard_rate": 0.5,
#             "death_hazard_rate": 0.0667
#         },
#         "moore": false
#     },
#     "KangarooRat_Parameters": {
#         "Body_sizes": {"start": 60, "stop": 71, "step": 1},
#         "active_hours": [20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6],
#         "annual_survival_probability": 0.5,
#         "reproductive_age_years": 1,
#         "max_age": 6,
#         "birth_death_module": {
#             "mean_litter_size": 3.5,
#             "std_litter_size": 1,
#             "upper_bound_litter_size": 6,
#             "lower_bound_litter_size": 1,
#             "litters_per_year": 1,
#             "birth_hazard_rate": 1,
#             "death_hazard_rate": 0.2
#         },
#         "moore": false
#     },
#     "Interaction_Map": {
#         "Rattlesnake-KangarooRat": {
#             "interaction_distance": 0.068,
#             "calories_per_gram": 1.38,
#             "digestion_efficiency": 0.8,
#             "expected_prey_body_size": 65,
#             "handling_time_range": {
#                 "min": 0.25,
#                 "max": 3.0
#             },
#             "attack_rate_range": {
#                 "min": 0,
#                 "max": 0.3
#             }
#         }
#     }
# }




# def main():
#     step_count = 1000
#     start_time = time.time()
#     model = ThermaSim(config=input_dictionary,seed=42)
#     model.run_model() #step_count=step_count
#     model_data = model.datacollector.get_model_vars_dataframe()
#     rattlesnake_data = model.datacollector.get_agenttype_vars_dataframe(agents.Rattlesnake)

#     model_data.to_csv("Output_Data/model_data.csv", index=False)
#     rattlesnake_data.to_csv("Output_Data/agent_data.csv", index=False)
#     run_time = time.time() - start_time
#     print(f"Model run completed in {run_time:.2f} seconds.")

#     # print(model_data)
#     # print(agent_data)



# if __name__ ==  "__main__":
#     main()

    

    
