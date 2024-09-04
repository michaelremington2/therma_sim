from model import ThermaSim

### Population sizes are set as individuals per hectFalseare


thermal_data_profile_fp = 'Data/Texas-Marathon_data.csv'
torus = False
moore = False
width=2
height=2
popualation_initial_conditions = {'KangarooRat': 10,
                                  'Rattlesnake': 10}

def main():
    model = ThermaSim(width=width, height=height, 
                      initial_agents_dictionary=popualation_initial_conditions,
                      thermal_profile_csv_fp=thermal_data_profile_fp,torus=torus, moore=moore,seed=42)
    model.run_model(step_count=10)
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    print(model_data)
    print(agent_data)



if __name__ ==  "__main__":
    main()

    

    
