import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Random generator

rng = np.random.Generator(np.random.MT19937(np.random.SeedSequence(1234)))

# M/M/1 simulator

class MM1:
    clock = 0
    events = []

    busy = 0
    packets = []
    queue = 0

    times = []
    nbs_packets = []

    # Initialization routine

    def init_simulation(self, arrival_rate=1):
        # Initialization of clock
        self.clock = 0

        # Initialization of event queue
        self.events = [(rng.exponential(1/arrival_rate), 'arrival')]

        # Initialization of system state
        self.busy = 0 # The server is free
        self.packets = []
        self.queue = 0

        # Initialization of statistical counters
        self.times = []
        self.nbs_packets = []

    # Report generator

    def report_generator(self, arrival_rate, departure_rate, display):
        # Compute estimates of interest

        avg_packets = self.nbs_packets[0] * self.times[0]

        for i in range(1, len(self.nbs_packets)):
            avg_packets += self.nbs_packets[i] * (self.times[i] - self.times[i-1])
        
        # Write report
        rho = arrival_rate / departure_rate

        if display:
            plt.plot()
            plt.bar(x=self.times, height=self.nbs_packets, width=0.5, label='Instantaneous system utilisation')
            plt.axhline(y=rho/(1-rho), color="black", linestyle="--", label='Theoretical average')
            plt.title('Evolution of the number of packets in the system over time')
            plt.xlabel('Time')
            plt.ylabel('Number of packets in the system (server + queue)')
            plt.legend()
            plt.grid(True)
            plt.show()

        return avg_packets / max(self.times)

    # Simulation executive

    def run_simulation(self, max_time, arrival_rate=1, departure_rate=2, display=False):
        # Invoke initialization routine
        self.init_simulation(arrival_rate)

        # Loop
        while self.clock < max_time:
            # Invoke timing routine
            i = self.events.pop(0)
            self.clock = i[0]

            # Invoke event routine i
            if i[1] == 'arrival':
                self.packet_arrival(arrival_rate, departure_rate)
            elif i[1] == 'departure':
                self.packet_departure(arrival_rate, departure_rate)

        # Report generator
        return self.report_generator(arrival_rate, departure_rate, display)

    # Event routines

    def packet_arrival(self, arrival_rate, departure_rate):
        # Schedule next arrival
        self.events.append((self.clock + rng.exponential(1/arrival_rate), 'arrival'))
        self.events.sort()

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
            self.events.append((self.clock + rng.exponential(1/departure_rate), 'departure'))
            self.events.sort()

    def packet_departure(self, arrival_rate, departure_rate):
        # Update statistical counters
        self.times.append(self.clock)
        self.nbs_packets.append(self.busy + self.queue)

        # Update system state
        if self.queue <= 0:
            self.busy -= 1
        else:
            self.queue -= 1

            # Schedule next departure
            self.events.append((self.clock + rng.exponential(1/departure_rate), 'departure'))
            self.events.sort()

        self.packets.pop(0)


lam = 1
mu = 2

max_time = 1000

# Example on one run

sim = MM1()
avg_packets = sim.run_simulation(max_time=max_time, arrival_rate=lam, departure_rate=mu, display=True)

# Computation of the average number of packets in the system

z = norm.ppf(0.975)  # 95% CI z-value

nb_runs = 1000
avg_nb_packets = []

for _ in range(nb_runs):
    sim = MM1()
    avg_nb_packets.append(sim.run_simulation(max_time=max_time, arrival_rate=lam, departure_rate=mu, display=False))

mean = np.mean(avg_nb_packets)
std_err = np.std(avg_nb_packets, ddof=1) / np.sqrt(nb_runs)
ci_mean_analytical = (mean - z * std_err, mean + z * std_err)

print(f"Average number of packets in the system: {mean}")
print(f"--> Confidence interval of 95 %: {ci_mean_analytical}")

rho = lam / mu

print(f"Theoretical average number of packets in the system:  {rho  / (1 - rho)}")