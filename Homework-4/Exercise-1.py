import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Random generator

rng = np.random.Generator(np.random.MT19937(np.random.SeedSequence(1234)))

# Useful variables

lam = 2
mu = 4

max_time = 1000

# M/M/1 simulator

class MM1:
    # Initialization routine

    def __init__(self, lam=1, mu=2):
        self.lam = lam
        self.mu = mu
        self.rho = lam / mu

        # Initialization of clock
        self.clock = 0.0

        # Initialization of event queue
        self.events = []
        self.events.append((rng.exponential(1/lam), 'arrival'))

        # Initialization of state variables
        self.busy = 0 # The server is free
        self.packets = collections.deque()
        self.queue = 0

        # Initialization of statistical counters
        self.times = []
        self.nbs_packets = []

    # Report generator

    def _report_generator(self, display):
        # Compute estimates of interest

        avg_packets = self.nbs_packets[0] * self.times[0]

        for i in range(1, len(self.nbs_packets)):
            avg_packets += self.nbs_packets[i] * (self.times[i] - self.times[i-1])
        
        # Write report

        if display:
            plt.plot()
            plt.bar(x=self.times, height=self.nbs_packets, width=0.5, label='Instantaneous system utilisation')
            plt.axhline(y=self.rho/(1-self.rho), color="black", linestyle="--", label='Theoretical average')
            plt.title('Evolution of the number of packets in the system over time')
            plt.xlabel('Time')
            plt.ylabel('Number of packets in the system (server + queue)')
            plt.legend()
            plt.grid(True)
            plt.show()

        return avg_packets / max(self.times)

    # Simulation executive

    def run(self, max_time, display=False):
        # Reset
        self.clock = 0.0
        self.events.clear()
        self.busy = 0 # The server is free
        self.packets.clear()
        self.queue = 0
        self.times.clear()
        self.nbs_packets.clear()

        # Add first event
        self.events.append((rng.exponential(1/lam), 'arrival'))

        # Loop
        while self.clock < max_time:
            # Invoke timing routine
            time, event = self.events.pop(0)
            self.clock = time

            # Invoke event routine i
            if event == 'arrival':
                self._packet_arrival()
            elif event == 'departure':
                self._packet_departure()

        # Report generator
        return self._report_generator(display)

    # Event routines

    def _packet_arrival(self):
        # Schedule next arrival
        self.events.append((self.clock + rng.exponential(1/self.lam), 'arrival'))
        self.events.sort(key=lambda x: x[0])

        # Update statistical counters
        self.times.append(self.clock)
        self.nbs_packets.append(self.busy + self.queue)

        # Update system state
        self.packets.append(self.clock)

        if self.busy:
            self.queue += 1
        else:
            self.busy += 1

            # Schedule next departure
            self.events.append((self.clock + rng.exponential(1/self.mu), 'departure'))
            self.events.sort(key=lambda x: x[0])

    def _packet_departure(self):
        # Update statistical counters
        self.times.append(self.clock)
        self.nbs_packets.append(self.busy + self.queue)

        # Update system state
        if self.queue <= 0:
            self.busy -= 1
        else:
            self.queue -= 1

            # Schedule next departure
            self.events.append((self.clock + rng.exponential(1/self.mu), 'departure'))
            self.events.sort(key=lambda x: x[0])

        self.packets.popleft()

# Example on one run

sim = MM1(lam=lam, mu=mu)
sim.run(max_time=max_time, display=True)

# Computation of the average number of packets in the system

z = norm.ppf(0.975)  # 95% CI z-value

nb_runs = 1000
estimates_nb_packets = []

for _ in range(nb_runs):
    estimates_nb_packets.append(sim.run(max_time=max_time, display=False))

mean = np.mean(estimates_nb_packets)
std_err = np.std(estimates_nb_packets, ddof=1) / np.sqrt(nb_runs)
ci_mean_analytical = (mean - z * std_err, mean + z * std_err)

print(f"Average number of packets in the system: {mean}")
print(f"--> Confidence interval of 95 %: {ci_mean_analytical}")

rho = lam / mu

print(f"Theoretical average number of packets in the system:  {rho  / (1 - rho)}")