import numpy as np

# Selection of a random Gaussian distribution

def select_gauss_random():
    # We first generate a random number u on [0,1]

    u = np.random.uniform()

    # We then select a Gaussian distribution
    # according to the value of u

    if u <= 0.15:
        return np.random.normal(-2, np.sqrt(2))
    elif u <= 0.4:
        return np.random.normal(4, 1)
    elif u <= 0.75:
        return np.random.normal(10, np.sqrt(3))
    else:
        return np.random.normal(15, np.sqrt(2))

# Computation of the mean

def expectation(results):
    return np.sum(results) / len(results)

# Computation of the variance

def variance(results):
    e = expectation(results)
    return np.sum([(e - r)**2 for r in results]) / len(results)

if __name__ == '__main__':
    # Testing the proposed distribution

    print("=== Test of the proposed distribution")
    
    results = [select_gauss_random() for _ in range(1000000)]

    expectations = [-2, 4, 10, 15]
    variances = [2, 1, 3 , 2]
    probs = [0.15, 0.25, 0.35, 0.25]

    theoretical_e = sum(
        probs[i] * expectations[i]
        for i in range(len(probs))
    )
    
    theoretical_v = sum(
        probs[i] * variances[i]                                 # E[Var(X|Y)]
        + probs[i] * (expectations[i] - theoretical_e) ** 2     # Var(E[X|Y])
        for i in range(len(probs))
    )

    e = expectation(results)
    v = variance(results)

    print("Theoretical expectation:", theoretical_e)
    print("--> Empirical expectation:", e)

    print("Theoretical variance:", theoretical_v)
    print("--> Empirical variance:", v)