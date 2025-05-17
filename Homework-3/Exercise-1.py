import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Random generator
rng = np.random.Generator(np.random.MT19937(np.random.SeedSequence(1234)))

# Load the data
data = pd.read_csv('./Homework-3/data_ex1_wt.csv', header=None, names=['Time', 'Measurement'])
x = data['Time'].values
y = data['Measurement'].values

# Least Squares from scratch
def fit_polynomial_least_squares(x, y, degree):
    # Build design (Vandermonde) matrix (highest power first)
    X = np.vander(x, degree + 1, increasing=False)  
    # Solve normal equations: (XᵀX)c = Xᵀy
    XtX = X.T @ X
    Xty = X.T @ y
    coeffs = np.linalg.solve(XtX, Xty)
    return coeffs

# Expectation-Maximization algorithm
def expectation_maximization(data, nb_dist, nb_iter):
    # Initial assumption
    dists = []
    for d in range(nb_dist):
        dists.append({'mean': (nb_dist // 2) - d, 'var': 1, 'prob': 1/3})

    # Iterative computation of the parameters
    for i in range(nb_iter):
        print(i)

        likelihoods = []
        assignments = [[] for _ in range(nb_dist)]

        for x in range(len(data)):

            # Computation of likelihoods
            probs_by_dist = []
            for d in range(nb_dist):
                probs_by_dist.append(dists[d]['prob'] / (np.sqrt(2 * np.pi * dists[d]['var']))
                                     * np.exp(-((data[x] - dists[d]['mean'])**2) / (2 * dists[d]['var'])))   
            likelihoods.append([probs_by_dist[k] / sum(probs_by_dist) for k in range(nb_dist)])

            # Assignment of a Gaussian distribution for each point
            r = rng.uniform()
            d = 0
            cdf = likelihoods[x][d]
            while d < nb_dist and r > cdf:
                d += 1
                cdf += likelihoods[x][d]
            assignments[d].append(data[x])

        # Re-estimation of parameters
        for d in range(nb_dist):
            dists[d]['mean'] = np.mean(assignments[d])
            dists[d]['var'] = np.var(assignments[d])
            dists[d]['prob'] = len(assignments[d]) / len(data)

    return dists, assignments

# Loop through polynomial degrees from 1 to 6 for fitting
rss_values = []
plt.figure(figsize=(10, 8)) 
degrees = range(1, 7)
for i, deg in enumerate(degrees, 1):
    # Fit the polynomial and get coefficients
    coeffs = fit_polynomial_least_squares(x, y, deg)
    # Create the design matrix for the polynomial degree
    X = np.vander(x, deg + 1, increasing=False)
    # Calculate the fitted y values using the polynomial coefficients
    y_fit = X @ coeffs
    # Calculate residuals (difference between actual and predicted values)
    residuals = y - y_fit
    # Compute the sum of squared residuals (RSS)
    rss = np.sum(residuals ** 2)
    rss_values.append(rss)
    
    # Plot the data and the polynomial fit for each degree
    plt.subplot(3, 2, i)
    plt.scatter(x, y, s=5, alpha=0.4, label='Original Data')
    plt.plot(x, y_fit, color='red', linewidth=1.2, label=f'{deg}-degree Trend')
    plt.title(f'Degree {deg}', fontsize=10)
    plt.xlabel('Time (s)', fontsize=8)
    plt.ylabel('Measurement', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True)
    plt.legend(fontsize=6)
plt.subplots_adjust(top=0.92)
plt.suptitle("Polynomial Fits (Degrees 1 to 6)", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94]) 
plt.show()

# Plot the RSS for polynomial degrees
plt.figure(figsize=(6, 4))
plt.plot(range(1, 7), rss_values, marker='o', linestyle='-')
plt.title('Residual Sum of Squares (RSS) vs Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('RSS')
plt.grid(True)
plt.xticks(range(1, 7))
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# Fit 5th-degree polynomial
degree = 5
coeffs = fit_polynomial_least_squares(x, y, degree)

# Evaluate trend
trend_values = np.polyval(coeffs, x)
detrended = y - trend_values

# Plotting
plt.figure(figsize=(15, 4))

# Original with trend
plt.subplot(1, 3, 1)
plt.scatter(x, y, s=10, alpha=0.6, label='Original Data')
plt.plot(x, trend_values, color='red', label=f'{degree}-degree Trend')
plt.title('Original Data with Polynomial Trend')
plt.xlabel('Time (s)')
plt.ylabel('Measurement')
plt.legend()
plt.grid(True)

# Detrended
plt.subplot(1, 3, 2)
plt.scatter(x, detrended, s=10, color='green', alpha=0.6)
plt.axhline(0, color='gray', linestyle='--')
plt.title('Detrended Data')
plt.xlabel('Time (s)')
plt.ylabel('Residual')
plt.grid(True)

# Histogram
plt.subplot(1, 3, 3)
plt.hist(detrended, bins=20, color='purple', edgecolor='black')
plt.title('Histogram of Detrended Data')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

# Expectation / Maximization
dists, assignments = expectation_maximization(detrended, 3, 20)

for k in range(3):
    print(f"Gaussian distribution #{k+1}:\t Mean: {dists[k]['mean']}\t Variance: {dists[k]['var']}")

plt.plot()
plt.hist(detrended, bins=20, density=True, color='lightgray', edgecolor='black')

colors = ['blue', 'red', 'green']

x = np.linspace(min(detrended), max(detrended), 100)

for k in range(3):
    plt.plot(x, dists[k]['prob'] * norm.pdf(x, dists[k]['mean'], np.sqrt(dists[k]['var'])), color=colors[k], linewidth=2)

plt.title('Normalized Histogram of the 3 Gaussian Distributions\nObtained with Expectation-Maximization Algorithm')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

# Facultative:
# BIC computation function
def compute_bic(log_likelihood, num_params, num_data_points):
    return -2 * log_likelihood + num_params * np.log(num_data_points)

# Log-likelihood function for GMM
def log_likelihood_gmm(data, dists):
    log_likelihood = 0
    for x in data:
        prob_sum = 0
        for d in dists:
            mean = d['mean']
            var = d['var']
            prob = d['prob']
            prob_sum += prob / (np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
        log_likelihood += np.log(prob_sum)
    return log_likelihood

# Try different numbers of distributions
bic_scores = []
max_components = 6
for k in range(1, max_components + 1):
    dists_k, _ = expectation_maximization(detrended, k, 30)
    log_likelihood = log_likelihood_gmm(detrended, dists_k)
    # Each Gaussian has 3 parameters: mean, variance, and probability
    num_params = 3 * k - 1 
    bic = compute_bic(log_likelihood, num_params, len(detrended))
    bic_scores.append(bic)
    print(f'k = {k}, BIC = {bic:.2f}')

# Determine best k
best_k = np.argmin(bic_scores) + 1
print(f'\nOptimal number of Gaussian distributions according to BIC: {best_k}')

# Optional: plot BIC values
plt.figure(figsize=(6, 4))
plt.plot(range(1, max_components + 1), bic_scores, marker='o', linestyle='-')
plt.title('BIC vs Number of Gaussian Components')
plt.xlabel('Number of Gaussian Components')
plt.ylabel('BIC')
plt.grid(True)
plt.tight_layout()
plt.show()