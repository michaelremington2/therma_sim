#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from numpy.linalg import eig

# Define the Lotka-Volterra model as a system of ODEs
def lotka_volterra_model(y, t, r, a, b, d):
    N, P = y  # N: prey population, P: predator population
    dNdt = r * N - a * N * P
    dPdt = b * a * N * P - d * P
    return [dNdt, dPdt]

# Simulate population with given parameters for a time period
def simulate_population(time, r, a, b, d, N0, P0):
    y0 = [N0, P0]
    sol = odeint(lotka_volterra_model, y0, time, args=(r, a, b, d))
    return sol[:, 0], sol[:, 1]  # Return prey and predator populations

# Define a cost function for optimization
def cost_function(params, time, observed_prey, observed_predator):
    r, a, b, d, N0, P0 = params
    simulated_prey, simulated_predator = simulate_population(time, r, a, b, d, N0, P0)
    # Calculate the sum of squared errors
    error = np.sum((observed_prey - simulated_prey) ** 2 + (observed_predator - simulated_predator) ** 2)
    return error

# Example observed data (replace these arrays with your actual data)
time_steps = np.linspace(0, 10, 100)  # Example time steps
observed_prey = np.sin(time_steps) * 20 + 40  # Replace with actual prey data
observed_predator = np.cos(time_steps) * 5 + 10  # Replace with actual predator data

# Bounds for the parameters (adjust if needed)
bounds = [(0.1, 1), (0.01, 0.1), (0.05, 0.2), (0.1, 0.5), (20, 60), (5, 15)]

# Define an empty list to store convergence data
convergence_data = []

# Callback function to store parameters at each iteration
def track_convergence(params, convergence=None):
    convergence_data.append(params)

# Run differential evolution with tracking
result = differential_evolution(
    cost_function,
    bounds,
    args=(time_steps, observed_prey, observed_predator),
    callback=track_convergence,
    maxiter=100,
    tol=1e-6
)

# Extract optimized parameters
optimized_params = result.x
print("Optimized Parameters:", optimized_params)

# Convert convergence data to a numpy array for easier plotting
convergence_data = np.array(convergence_data)

# Plot convergence for each parameter
param_names = ['r', 'a', 'b', 'd', 'N0', 'P0']
plt.figure(figsize=(12, 8))
for i, param_name in enumerate(param_names):
    plt.plot(convergence_data[:, i], label=f'Parameter: {param_name}')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Parameter Convergence during Optimization')
plt.legend()
plt.grid()
plt.show()

r, a, b, d, N0, P0 = result.x

# Define the equilibrium point (non-trivial equilibrium)
N_eq = d / (b * a)
P_eq = r / a

print("Equilibrium Point:", (N_eq, P_eq))

# Jacobian matrix for the Lotka-Volterra system
def jacobian(N, P, r, a, b, d):
    J = np.array([
        [r - a * P, -a * N],
        [b * a * P, b * a * N - d]
    ])
    return J

# Calculate the Jacobian matrix at the equilibrium point
J_eq = jacobian(N_eq, P_eq, r, a, b, d)
print("Jacobian Matrix at Equilibrium Point:\n", J_eq)

# Calculate the eigenvalues of the Jacobian matrix
eigenvalues = eig(J_eq)[0]
print("Eigenvalues of the Jacobian Matrix:", eigenvalues)

# Analyze stability based on eigenvalues
if all(np.real(eigenvalues) < 0):
    stability = "Stable (Attractor)"
elif all(np.real(eigenvalues) > 0):
    stability = "Unstable (Repeller)"
else:
    stability = "Saddle Point or Oscillatory"

print("Stability of the Equilibrium Point:", stability)

# Define equilibrium point
N_eq = d / (b * a)
P_eq = r / a

# Generate a grid of initial conditions for trajectories
initial_conditions = [
    [N_eq * 0.8, P_eq * 0.8],
    [N_eq * 1.2, P_eq * 1.2],
    [N_eq * 1.0, P_eq * 1.5],
    [N_eq * 1.5, P_eq * 1.0],
    [N_eq * 0.5, P_eq * 0.5]
]

# Time steps for the simulation
time_steps = np.linspace(0, 15, 500)

# Plot phase space with trajectories
plt.figure(figsize=(10, 8))

# Plot each trajectory from different initial conditions
for initial in initial_conditions:
    solution = odeint(lotka_volterra_model, initial, time_steps, args=(r, a, b, d))
    plt.plot(solution[:, 0], solution[:, 1], label=f"IC: {initial}")

# Plot equilibrium point
plt.plot(N_eq, P_eq, 'ro', label="Equilibrium Point")

# Add nullclines
N_nullcline = np.linspace(0, 2 * N_eq, 100)
P_nullcline = np.linspace(0, 2 * P_eq, 100)
plt.plot(N_nullcline, r / a * np.ones_like(N_nullcline), 'b--', label="Prey Nullcline (dN/dt=0)")
plt.plot(d / (b * a) * np.ones_like(P_nullcline), P_nullcline, 'g--', label="Predator Nullcline (dP/dt=0)")

# Labels and title
plt.xlabel("Prey Population (N)")
plt.ylabel("Predator Population (P)")
plt.title("Phase Plot of Lotka-Volterra Model with Trajectories")
plt.legend()
plt.grid()
plt.show()
##############################################################
# Create a grid of prey and predator values for the phase plot
##############################################################
# Create a grid of prey and predator values for the phase plot
N = np.linspace(0, 2 * N_eq, 20)
P = np.linspace(0, 2 * P_eq, 20)
N_grid, P_grid = np.meshgrid(N, P)

# Calculate the vector field for each point in the grid
dNdt, dPdt = lotka_volterra_model([N_grid, P_grid], 0, r, a, b, d)

# Normalize the vectors
magnitude = np.sqrt(dNdt**2 + dPdt**2)
# Avoid division by zero in normalization
dNdt_normalized = np.where(magnitude == 0, 0, dNdt / magnitude)
dPdt_normalized = np.where(magnitude == 0, 0, dPdt / magnitude)

# Plot the phase plot with normalized vector field
plt.figure(figsize=(10, 8))
plt.quiver(N_grid, P_grid, dNdt_normalized, dPdt_normalized, color='gray', alpha=0.6)
plt.xlabel('Prey Population (N)')
plt.ylabel('Predator Population (P)')
plt.title('Phase Plot of Predator-Prey Dynamics with Normalized Arrows')

# Plot trajectories for different initial conditions
time = np.linspace(0, 20, 1000)  # Time span for the simulation
initial_conditions = [
    (N_eq * 0.8, P_eq * 0.8), (N_eq * 1.2, P_eq * 1.2), (N_eq, P_eq)
]

for N0, P0 in initial_conditions:
    trajectory = odeint(lotka_volterra_model, [N0, P0], time, args=(r, a, b, d))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Init: N0={N0:.1f}, P0={P0:.1f}')

# Mark equilibrium point
plt.plot(N_eq, P_eq, 'ro', label='Equilibrium Point')
plt.legend()
plt.grid()
plt.show()

