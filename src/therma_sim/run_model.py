from model import ThermaSim
import TPC

### Population sizes are set as individuals per hectFalseare


thermal_data_profile_fp = 'Data/Texas-Marathon_data.csv'
torus = False
moore = False
width=1
height=1

# pred-prey parameters
interaction_distance = 0.068 # Parameter is from https://www.nature.com/articles/srep40412/tables/1
krat_cals_per_gram = 1.38 # Technically for ground squirrels. Metric is from Crowell 2021
digestion_efficency = 0.8 # Assumed rate of calorie assimilation. Metric is from crowell 2021

popualation_initial_conditions = {'KangarooRat': 1,
                                  'Rattlesnake': 1}

def strike_tpc(body_temp):
    R_ref = 0.21
    E_A = 1
    E_L = -5
    T_L = 15
    E_H = 5
    T_H = 32
    T_ref = 28
    return TPC.sharpe_schoolfield(T = body_temp, R_ref=R_ref, E_A=E_A, E_L=E_L, T_L=T_L, E_H=E_H, T_H=T_H, T_ref=T_ref)
     
input_dictionary = {}



def main():
    step_count = 10
    model = ThermaSim(width=width, height=height, 
                      initial_agents_dictionary=popualation_initial_conditions,
                      thermal_profile_csv_fp=thermal_data_profile_fp,torus=torus, moore=moore,seed=42)
    model.run_model(step_count=step_count)
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    print(model_data)
    print(agent_data)



if __name__ ==  "__main__":
    help(TPC.sharpe_schoolfield)
    #main()

    

    
