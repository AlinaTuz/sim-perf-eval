import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Random generator
rng = np.random.Generator(np.random.MT19937(np.random.SeedSequence(1234)))

# Define constants
M = 9
xmin, xmax = -3, 3
A = 8.8480182
z = norm.ppf(0.975)  # 95% CI z-value

# Unnormalized target density
def g(x):
    return x**2 * np.sin(np.pi * x)**2

# Rejection sampling function
def rejection_sampling(num_samples):
    samples = []
    while len(samples) < num_samples:
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(0, M)
        if y < g(x):
            samples.append(x)
    return np.array(samples)

# Analytical confidence interval for quantiles
def quantile_ci(data, q):
    n = len(data)
    q_hat = np.quantile(data, q)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    se_q = np.sqrt(q * (1 - q) / n) / (iqr / 1.34)
    return (q_hat - z * se_q, q_hat + z * se_q)

# Bootstrap confidence intervals
def bootstrap_cis(data, boot_iters=10000):
    n = len(data)
    boot_means, boot_medians, boot_q90s = [], [], []

    for _ in range(boot_iters):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
        boot_medians.append(np.median(boot_sample))
        boot_q90s.append(np.quantile(boot_sample, 0.9))
    
    return (
        (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)),
        (np.percentile(boot_medians, 2.5), np.percentile(boot_medians, 97.5)),
        (np.percentile(boot_q90s, 2.5), np.percentile(boot_q90s, 97.5))
    )

# Check coverage for confidence intervals
def coverage_probability(samples, true_mean=0):
    datasets = np.array_split(samples, 100)
    contained = 0

    for data_subset in datasets:
        n = len(data_subset)
        sample_mean = np.mean(data_subset)
        sample_std = np.std(data_subset, ddof=1)
        std_err = sample_std / np.sqrt(n)
        ci = (sample_mean - z * std_err, sample_mean + z * std_err)
        if ci[0] <= true_mean <= ci[1]:
            contained += 1
    return contained


# 1: Sampling from the Target Distribution
num_samples_task1 = 100000
samples_task1 = rejection_sampling(num_samples_task1)

print(f"Task 1: Generated {len(samples_task1)} samples.")


# 2: Empirical and Theoretical PDF Comparison
x_vals = np.linspace(xmin, xmax, 1000)
theoretical_pdf = g(x_vals) / A

plt.figure(figsize=(10, 6))
plt.hist(samples_task1, bins=100, density=True, alpha=0.6, label='Empirical PDF (Histogram)')
plt.plot(x_vals, theoretical_pdf, 'r-', lw=2, label='Theoretical PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Task 2: Rejection Sampling - Empirical vs. Theoretical PDF')
plt.legend()
plt.grid(True)
plt.show()


# 3: Confidence Intervals for Mean, Median, and 0.9-Quantile
num_samples_task3 = 20000
samples_task3 = rejection_sampling(num_samples_task3)

# Use first 200 samples
data = samples_task3[:200]
n = len(data)

# Analytical confidence intervals
mean = np.mean(data)
std_err = np.std(data, ddof=1) / np.sqrt(n)
ci_mean_analytical = (mean - z * std_err, mean + z * std_err)
ci_median_analytical = quantile_ci(data, 0.5)
ci_q90_analytical = quantile_ci(data, 0.9)

# Bootstrap confidence intervals
ci_mean_boot, ci_median_boot, ci_q90_boot = bootstrap_cis(data)

# Print Results
print("\nTask 3: Confidence Intervals for Mean, Median, and 0.9-Quantile")
print("Analytical Confidence Intervals:")
print(f"Mean:         ({ci_mean_analytical[0]:.4f}, {ci_mean_analytical[1]:.4f})")
print(f"Median:       ({ci_median_analytical[0]:.4f}, {ci_median_analytical[1]:.4f})")
print(f"0.9-Quantile: ({ci_q90_analytical[0]:.4f}, {ci_q90_analytical[1]:.4f})\n")

print("Bootstrap Confidence Intervals:")
print(f"Mean:         ({ci_mean_boot[0]:.4f}, {ci_mean_boot[1]:.4f})")
print(f"Median:       ({ci_median_boot[0]:.4f}, {ci_median_boot[1]:.4f})")
print(f"0.9-Quantile: ({ci_q90_boot[0]:.4f}, {ci_q90_boot[1]:.4f})")


# 4: Coverage Probability of the Confidence Intervals
contained = coverage_probability(samples_task3, true_mean=0)

print("\nTask 4: Coverage Probability of the Confidence Intervals")
print(f"Number of confidence intervals containing the true mean: {contained} out of 100")

