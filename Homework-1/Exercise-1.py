from math import log, sqrt
import random

random.seed(123456789)

# Exponential random variable
# f(x) = l * exp(-l * x)

def exp_random(rate):
    return -log(1 - random.random()) / rate

# Gaussian distribution
# Based on the "more efficient process" presented on the slides

def gauss_random(mean, variance):
    y1 = exp_random(1)
    y2 = exp_random(1)

    while (y2 <= ((y1 - 1) ** 2 / 2)):
        y1 = exp_random(1)
        y2 = exp_random(1)

    if (random.random() <= 0.5):
        y1 = -y1

    return mean + y1 * sqrt(variance)
    

# Selection of a random Gaussian distribution

def select_gauss_random():
    # We first generate a random number u on [0,1]

    u = random.random()

    # We then select a Gaussian distribution
    # according to the value of u

    if u <= 0.15:
        mean = -2
        variance = 2
    elif u <= 0.4:
        mean = 4
        variance = 1
    elif u <= 0.75:
        mean = 10
        variance = 3
    else:
        mean = 15
        variance = 2

    return gauss_random(mean, variance)

# Computation of the mean

def expectation(results):
    return sum(results) / len(results)

# Computation of the variance

def variance(results):
    e = expectation(results)
    return sum([(e - r)**2 for r in results]) / len(results)

if __name__ == '__main__':
    # Testing the exponential RVG

    print("=== Test of the exponential random variable generator")

    rate = 2

    results = [exp_random(rate) for _ in range(1000000)]

    e = expectation(results)
    v = variance(results)

    theoretical_e = 1 / rate
    theoretical_v = 1 / (rate ** 2)

    print("Theoretical expectation:", theoretical_e)
    print("--> Empirical expectation:", e)

    print("Theoretical variance:", theoretical_v)
    print("--> Empirical variance:", v)

    # Testing the gaussian RVG

    print("=== Test of the gaussian random variable generator")

    theoretical_e = 50
    theoretical_v = 100

    results = [gauss_random(theoretical_e, theoretical_v) for _ in range(1000000)]

    e = expectation(results)
    v = variance(results)

    print("Theoretical expectation:", theoretical_e)
    print("--> Empirical expectation:", e)

    print("Theoretical variance:", theoretical_v)
    print("--> Empirical variance:", v)

    # Testing the proposed distribution

    print("=== Test of the proposed distribution")

    results = [select_gauss_random() for _ in range(1000000)]

    expectations = [-2, 4, 10, 15]
    variances = [2, 1, 3 , 2]
    probs = [0.15, 0.25, 0.35, 0.25]

    # Computation of the expectation

    theoretical_e = sum(
        probs[i] * expectations[i]
        for i in range(len(probs))
    )
    
    theoretical_v = sum(
        probs[i] * variances[i] # E[Var(X|Y)]
        + probs[i] * (expectations[i] - theoretical_e) ** 2 # Var(E[X|Y])
        for i in range(len(probs))
    )

    e = expectation(results)
    v = variance(results)

    print("Theoretical expectation:", theoretical_e)
    print("--> Empirical expectation:", e)

    print("Theoretical variance:", theoretical_v)
    print("--> Empirical variance:", v)