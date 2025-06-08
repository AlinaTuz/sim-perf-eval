import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Random generator

rng = np.random.Generator(np.random.MT19937(np.random.SeedSequence(1234)))

# Poisson event generator

def poisson_events(lam, n):
    inter_arrival_times = rng.exponential(scale=1/lam, size=n)
    arrival_times = inter_arrival_times.cumsum()
    return arrival_times

# M/M/1 simulator

class MM1:
    clock = 0
    last_clock = 0
    events = []

    busy = 0
    packets = []
    nb_in_queue = 0

    times = []
    nbs_packets = []

    # Initialization routine

    def init_simulation(self, nb_packets, rate_arrivals=1, rate_departures=2):
        # Initialization of clock
        self.clock = 0
        self.last_clock = 0

        # Initialization of event queue
        arrivals = poisson_events(rate_arrivals, nb_packets)
        departures = poisson_events(rate_departures, nb_packets)

        # print(arrivals)
        # print(departures)

        for a in arrivals:
            self.events.append((a, 'arrival'))

        for d in departures:
            self.events.append((d, 'departure'))

        self.events.sort()

        # print(self.events)

        # Initialization of system state
        self.busy = 0 # The server is free
        self.packets = []
        self.nb_in_queue = 0

        # Initialization of statistical counters
        self.times = []
        self.nbs_packets = []

    # Simulation executive

    def run_simulation(self, nb_packets, rate_arrivals=1, rate_departures=2):
        # Invoke initialization routine
        self.init_simulation(nb_packets, rate_arrivals, rate_departures)

        # Loop
        while len(self.events) != 0:
            # Invoke timing routine
            i = self.events.pop(0)
            self.clock = i[0]

            # Invoke event routine i
            if i[1] == 'arrival':
                self.packet_arrival()
            elif i[1] == 'departure':
                self.packet_departure()

        # Report generator
        self.report_generator(nb_packets)

    # End routine & Report generator

    def report_generator(self, nb_packets):
        # Update statistical counters

        self.times.append(self.clock)
        self.nbs_packets.append(self.busy + self.nb_in_queue)

        print(self.times)
        print(self.nbs_packets)

        # Compute estimates of interest
        

        # Write report

        plt.bar(x=self.times, height=self.nbs_packets, width=0.1)
        plt.title('Histogram of Detrended Data')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
        pass

    # Event routines

    def packet_arrival(self):
        # Update statistical counters

        self.times.append(self.clock)
        self.nbs_packets.append(self.busy + self.nb_in_queue)

        # Update system state

        self.packets.append(self.clock)

        if self.busy:
            self.nb_in_queue += 1
        else:
            self.busy = 1

    def packet_departure(self):
        # Update statistical counters

        self.times.append(self.clock)
        self.nbs_packets.append(self.busy + self.nb_in_queue)

        # Update system state

        if self.busy: # There is at least one packet in service

            if self.nb_in_queue == 0:
                self.busy = 0
            else:
                self.nb_in_queue -= 1

            self.packets.pop(0)

sim = MM1()
sim.run_simulation(1000, 1, 2)