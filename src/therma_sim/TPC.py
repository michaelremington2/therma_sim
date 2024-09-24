#!/usr/bin/python
import numpy as np

def sharpe_schoolfield(self, T, R_ref, E_A, E_L, T_L, E_H, T_H, T_ref):
    '''
    Inputs:
        T:
            This is the current temperature (in degrees Celsius) at which you're evaluating the performance or rate. 
            The model converts this to Kelvin within the function by adding 273.15.
        R_ref:
            This is the reference rate or the performance of the organism at the reference temperature T_ref. 
            This can be the maximum rate of a physiological process under optimal conditions. It sets the scaling of the rate curve.
        E_A:
            Activation energy in electron volts (eV). 
            This represents the energy required to "activate" the physiological process at moderate temperatures. 
            It determines how sensitive the rate is to temperature changes within a certain optimal range (before any inactivation effects occur).
        E_L:
            Low-temperature inactivation energy in electron volts (eV). 
            This parameter describes how quickly the rate declines at temperatures lower than the optimal range. 
            In this case, it models inactivation of the rate at low temperatures.
        T_L:
            Low-temperature threshold (in degrees Celsius) for inactivation. 
            Below this threshold, the rate starts to decline due to low-temperature stress or inactivation. 
            This value is converted to Kelvin in the function for use in the exponential term.
        E_H:
            High-temperature inactivation energy in electron volts (eV). 
            This parameter describes how the rate declines at temperatures higher than the optimal range. 
            It models inactivation at high temperatures, where enzymes or physiological systems may become unstable.
        T_H:
            High-temperature threshold (in degrees Celsius) for inactivation. 
            Above this threshold, the rate declines due to high-temperature stress or inactivation. 
            This value is also converted to Kelvin in the function.
        T_ref:
            Reference temperature (in degrees Celsius) at which R_ref is measured. 
            This temperature is used as a standard to anchor the model and is converted to Kelvin inside the function.
    '''
    k = 8.617e-5  # Boltzmann constant in eV/K
    T_K = T + 273.15  # Convert Celsius to Kelvin
    T_ref_K = T_ref + 273.15
    T_L_K = T_L + 273.15
    T_H_K = T_H + 273.15
    
    return (R_ref * np.exp((E_A / k) * (1 / T_ref_K - 1 / T_K))) / \
        (1 + np.exp((E_L / k) * (1 / T_L_K - 1 / T_K)) + np.exp((E_H / k) * (1 / T_H_K - 1 / T_K)))

# Gaussian-based thermal performance curve
def gaussian_tpc(T, P_max, T_opt, w):
    return P_max * np.exp(-((T - T_opt) / w) ** 2)

# Quadratic thermal performance curve
def quadratic_tpc(T, a, T_opt, P_max):
    return a * (T - T_opt) ** 2 + P_max