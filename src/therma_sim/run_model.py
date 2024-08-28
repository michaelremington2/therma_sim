from model import ThermaSim

### Population sizes are set as individuals per hectFalseare


thermal_data_profile_fp = 'Data/Texas-Marathon_data.csv'
torus = False
moore = False
width=2
height=2
popualation_initial_conditions = {'KangarooRat': 2,
                                  'Rattlesnake': 2}


if __name__ ==  "__main__":
    model = ThermaSim(width=width, height=height, 
                      initial_agents_dictionary=popualation_initial_conditions,
                      thermal_profile_csv_fp=thermal_data_profile_fp,torus=torus, moore=moore,seed=42)
    model.run_model(step_count=10)
    

    
