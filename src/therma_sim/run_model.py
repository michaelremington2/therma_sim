from model import ThermaSim

### Population sizes are set as individuals per hectFalseare
popualation_initial_conditions = {'Kangaroo_rat': 100,
                                  'Snake': 50}

thermal_data_profile_fp = 'Data/Canada_data.csv'

if __name__ ==  "__main__":
    model = ThermaSim(initial_agents_dictionary=popualation_initial_conditions,
                      thermal_profile_csv_fp=thermal_data_profile_fp)
