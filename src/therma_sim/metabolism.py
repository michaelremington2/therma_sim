#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

class EctothermMetabolism(object):
    def __init__(self):
        self.mlo2_to_joules = 19.874
        self.joules_to_cals = 2.39e-4


    def smr_eq(self, mass, temperature, X1_mass, X2_temp, X3_const):
        '''This returns VO2 which is a proxy for SMR. 
            mass - the individuals mass in grams
            temperature - body temperture of the individual in celsius'''
        log_smr = (X1_mass * np.log10(mass)) + (X2_temp * temperature) + X3_const
        smr = 10**log_smr
        return smr
    
    def hourly_energy_expendeture(self, smr, activity_coeffcient):
        '''
        Our model for hourly energy expendature.
        Inputs:
            SMR - log(V02) which is in ml o^2 / hour
            Activity_coeffcient - multiplier to convert smr to amr. smr is the resting rate so a multiplier of 1 returns smr. A multiplier of >1 represents activity.
        Outputs:
            cals_burnt_per_hour - number of calories burnt by an individual.
        '''
        hee = smr*activity_coeffcient
        joules_per_hour = hee*self.mlo2_to_joules
        cals_burnt_per_hour = joules_per_hour*self.joules_to_cals
        return cals_burnt_per_hour
    
    def energy_intake(self, prey_mass, cal_per_gram_conversion, percent_digestion_cals):
        '''
        Amount of calories yielded from a prey agent.
        Inputs:
            prey_mass - mass of prey agent in grams
            cal_per_gram_conversion - conversion rate to get prey grams to calories
            percent_digestion_cals - percentage of calories lost to digestion.
        '''
        return float(prey_mass*cal_per_gram_conversion*percent_digestion_cals)
    

    
    def energy_expendeture_graph(self):
        '''
        Helper function for visualizing the calories burnt in the energy expendeture model.
        Equation and coefficents are from Dorcus 2004.
        '''
        # Generate values for body mass and temperature
        mass_values = np.linspace(800, 5000, 100)  # Mass range of 800g to 5000g
        temperature_values = np.linspace(5, 40, 100)  # Temperature range from 5°C to 35°C

        # Create a grid of mass and temperature values for plotting
        mass_grid, temp_grid = np.meshgrid(mass_values, temperature_values)

        # Calculate SMR for each combination of mass and temperature
        X1_mass = 0.930
        X2_temp = 0.044
        X3_const = -2.58
        smr_values = self.smr_eq(mass_grid, temp_grid, X1_mass=X1_mass, X2_temp=X2_temp, X3_const=X3_const)
        energy_expenditure_resting = self.hourly_energy_expendeture(smr=smr_values, activity_coeffcient=1)
        energy_expenditure_foraging = self.hourly_energy_expendeture(smr=smr_values, activity_coeffcient=1.5)

        # Plotting the SMR as a function of mass and temperature
        vmin = min(energy_expenditure_resting.min(), energy_expenditure_foraging.min())
        vmax = max(energy_expenditure_resting.max(), energy_expenditure_foraging.max())

        # Create a figure and two subplots arranged in 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figsize as needed

        # First heatmap (resting energy expenditure)
        contour1 = ax1.contourf(mass_grid, temp_grid, energy_expenditure_resting, cmap='viridis', levels=20, vmin=vmin, vmax=vmax)
        fig.colorbar(contour1, ax=ax1, label='Resting')
        ax1.set_title('Resting Energy Expenditure')
        ax1.set_xlabel('Body Mass (g)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.grid(True)

        # Second heatmap (active energy expenditure)
        contour2 = ax2.contourf(mass_grid, temp_grid, energy_expenditure_foraging, cmap='viridis', levels=20, vmin=vmin, vmax=vmax)
        fig.colorbar(contour2, ax=ax2, label='Foraging')
        ax2.set_title('Active Energy Expenditure')
        ax2.set_xlabel('Body Mass (g)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.grid(True)

        # Display the two heatmaps side by side
        plt.tight_layout()  # Adjust the layout so plots don't overlap
        plt.show()

if __name__ ==  "__main__":
    met = EctothermMetabolism()
    met.energy_expendeture_graph()