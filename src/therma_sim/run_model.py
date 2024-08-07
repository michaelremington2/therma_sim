from model import ThermaSim

### Population sizes are set as individuals per hectFalseare


thermal_data_profile_fp = 'Data/Canada_data.csv'
torus = False
moore = False
width=3
height=3
popualation_initial_conditions = {'KangarooRat': 0,
                                  'Rattlesnake': 0}


if __name__ ==  "__main__":
    model = ThermaSim(width=width, height=height, 
                      initial_agents_dictionary=popualation_initial_conditions,
                      thermal_profile_csv_fp=thermal_data_profile_fp,torus=torus, moore=moore)
    model.run_model(step_count=2)
    
