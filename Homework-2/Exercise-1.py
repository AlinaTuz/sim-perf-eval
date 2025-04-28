import numpy as np

def uniform_dist(low, high, x):
    if x < low:
        return 0
    
    if x > high:
        return 1
    
    return (x - low) / (high - low)

def exponential_dist(rate, x):
    if x < 0:
        return 0

    return 1 - np.exp(-rate * x)

def uniform_test_statistic(samples, k, low, high):
    n = len(samples)

    step = (high - low) / k

    test_samples = samples
    test_samples.sort()

    T = 0

    # In the interval

    for i in np.arange(low, high, step):
        samples_i = []

        while test_samples != [] and test_samples[0] < (i + step):
            sample = test_samples.pop(0)
            samples_i.append(sample)

        N = len(samples_i)
        p = uniform_dist(low, high, i + step) - uniform_dist(low, high, i)

        T += (N - n*p)**2 / (n*p)

    return T

def exponential_test_statistic(samples, k, rate):
    n = len(samples)

    low = 0
    high = max(samples)
    step = high / k

    test_samples = samples
    test_samples.sort()

    T = 0

    # In the interval

    for i in np.arange(low, high, step):
        samples_i = []

        while test_samples != [] and test_samples[0] < (i + step):
            sample = test_samples.pop(0)
            samples_i.append(sample)

        N = len(samples_i)
        p = exponential_dist(rate, i + step) - exponential_dist(rate, i)

        T += (N - n*p)**2 / (n*p)

    # Above high

    N = len(test_samples)
    p = 1 - exponential_dist(rate, high)

    T += (N - n*p)**2 / (n*p)

    return T


# Parameters

rate = 5
T = 2000
N = 10000

# Method 1: Drawing N arrival times uniformly at random in the interval [0,T]

iterations = 500

tests = []
p_values = []

for i in range(iterations):
    #print(i)

    rng = np.random.Generator(np.random.MT19937(
        np.random.SeedSequence(1000 + i)
    ))

    arrival_times = rng.uniform(low=0, high=T, size=N)

    arrival_times.sort()

    inter_arrival_times = [arrival_times[0]]
    inter_arrival_times.extend([arrival_times[i+1] - arrival_times[i] for i in range(N-1)])

    #print(sum(inter_arrival_times))

    # Chi-squared test: show that the inter-arrival times of method 1 are also exponentially distributed

    k = 10000

    test = exponential_test_statistic(inter_arrival_times, k, rate)
    tests.append(test)

    successes = 0
    draws = 10000

    for _ in range(draws):
        chisquare = rng.chisquare(k-2)

        if chisquare >= test:
            successes += 1

    p_value = successes / draws
    p_values.append(p_value)

print(f"Average test statistic T after {iterations} iterations: {np.mean(tests)}")
print(f"Average p-value after {iterations} iterations: {np.mean(p_values)}")

# Method 2: Drawing a set of N exponential inter-arrival times of average value 1/λ in the interval [0, T]

iterations = 500

tests = []
p_values = []

for i in range(iterations):
    #print(i)

    rng = np.random.Generator(np.random.MT19937(
        np.random.SeedSequence(1000 + i)
    ))

    inter_arrival_times = []

    for j in range(N):
        time = rng.exponential(scale=1/rate)

        while time < 0 or time > T:
            time = rng.exponential(scale=1/rate)

        inter_arrival_times.append(time)

    #print(sum(inter_arrival_times))

    arrival_times = [inter_arrival_times[0]]

    for k in range(1, len(inter_arrival_times)):
        arrival_times.append(arrival_times[k-1] + inter_arrival_times[k])
    
    # Chi-squared test: show that the inter-arrival times of method 1 are also exponentially distributed

    k = 10000

    test = uniform_test_statistic(arrival_times, k, 0, T)
    tests.append(test)

    successes = 0
    draws = 10000

    for _ in range(draws):
        chisquare = rng.chisquare(k-2)

        if chisquare >= test:
            successes += 1

    p_value = successes / draws
    p_values.append(p_value)

print(f"Average test statistic T after {iterations} iterations: {np.mean(tests)}")
print(f"Average p-value after {iterations} iterations: {np.mean(p_values)}")