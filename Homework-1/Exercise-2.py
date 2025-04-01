import numpy as np

rng = np.random.Generator(np.random.MT19937(
    np.random.SeedSequence(1234)
))

# Simulation parameters
num_simulations = 1000000  # Number of trials (large number for better accuracy)
mu = 1.0                 # Mean for the exponential distribution
low_uniform = 0.0        # Lower bound for the uniform distribution
high_uniform = 5.0       # Upper bound for the uniform distribution
probability_mean = 0

# Generate random variates
exponential_RV = rng.exponential(scale=mu, size=num_simulations)
uniform_RV = rng.uniform(low=low_uniform, high=high_uniform, size=num_simulations)

# Calculate the probability
probability_mean = np.mean(exponential_RV > uniform_RV)

# Print the result
print(f"\nSimulation parameters:")
print(f"Number of simulations: {num_simulations}")
print(f"Exponential mean: {mu}")
print(f"Uniform interval: [{low_uniform}, {high_uniform}]")
print(f"\nEstimated probability (Exponential > Uniform): {probability_mean:.4f}\n")