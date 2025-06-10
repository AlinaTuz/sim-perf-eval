import numpy as np
import collections
from scipy.stats import norm
import matplotlib.pyplot as plt

class MM1_Exercise2:
    """
    An M/M/1 queue simulator designed to estimate average packet traversal time
    using both a naive estimator and a control variate variance reduction technique.
    """

    def __init__(self, lam, mu):
        self.lam = lam
        self.mu = mu
        self.rho = lam / mu
        self.rng = np.random.default_rng()

        # State variables
        self.clock = 0.0
        self.server_busy = False
        self.queue = collections.deque()
        self.event_list = []

        # Data collection for analysis
        self.completed_packets_data = []

        # Packet tracking
        self.next_packet_id = 0
        self.packet_in_service = None
        self.arrival_info = {} # Stores {packet_id: queue_len_at_arrival}

    def _schedule_event(self, time, event_type, packet_id):
        """Adds an event to the event list, maintaining chronological order."""
        event = (time, event_type, packet_id)
        self.event_list.append(event)
        self.event_list.sort(key=lambda x: x[0])

    def _handle_arrival(self, arrival_time, packet_id):
        """Processes a packet arrival event."""
        # Record the number of packets in the queue when this packet arrived
        self.arrival_info[packet_id] = len(self.queue)

        # Schedule the next arrival
        next_arrival_time = arrival_time + self.rng.exponential(1 / self.lam)
        self._schedule_event(next_arrival_time, 'arrival', self.next_packet_id)
        self.next_packet_id += 1

        if not self.server_busy:
            # Server is free, packet enters service immediately
            self.server_busy = True
            self.packet_in_service = {'id': packet_id, 'arrival_time': arrival_time}
            departure_time = arrival_time + self.rng.exponential(1 / self.mu)
            self._schedule_event(departure_time, 'departure', packet_id)
        else:
            # Server is busy, packet enters the queue
            self.queue.append({'id': packet_id, 'arrival_time': arrival_time})

    def _handle_departure(self, departure_time, packet_id):
        """Processes a packet departure event."""
        # Ensure the departing packet is the one in service
        if self.packet_in_service and self.packet_in_service['id'] == packet_id:
            # Calculate traversal time
            arrival_time = self.packet_in_service['arrival_time']
            traversal_time = departure_time - arrival_time

            # Retrieve the queue length seen by this packet upon its arrival
            q_len_at_arrival = self.arrival_info.pop(packet_id)

            # Store data for analysis
            self.completed_packets_data.append((traversal_time, q_len_at_arrival))

            if self.queue:
                # Queue is not empty, start serving the next packet
                next_packet = self.queue.popleft()
                self.packet_in_service = next_packet
                next_departure_time = departure_time + self.rng.exponential(1 / self.mu)
                self._schedule_event(next_departure_time, 'departure', next_packet['id'])
            else:
                # Queue is empty, server becomes free
                self.server_busy = False
                self.packet_in_service = None

    def run(self, max_packets):
        """Runs the simulation until a maximum number of packets have been processed."""
        # Reset state for a new run
        self.clock = 0.0
        self.server_busy = False
        self.queue.clear()
        self.event_list = []
        self.completed_packets_data = []
        self.next_packet_id = 0
        self.packet_in_service = None
        self.arrival_info.clear()

        # Start with the first arrival
        first_arrival_time = self.rng.exponential(1 / self.lam)
        self._schedule_event(first_arrival_time, 'arrival', self.next_packet_id)
        self.next_packet_id += 1

        # Main simulation loop
        while len(self.completed_packets_data) < max_packets:
            if not self.event_list:
                break

            time, event_type, packet_id = self.event_list.pop(0)
            self.clock = time

            if event_type == 'arrival':
                self._handle_arrival(time, packet_id)
            elif event_type == 'departure':
                self._handle_departure(time, packet_id)

        # Analysis for this single run
        Y_vals = [data[0] for data in self.completed_packets_data] # Traversal times
        C_vals = [data[1] for data in self.completed_packets_data] # Queue lengths at arrival

        # Naive estimate
        mean_Y = np.mean(Y_vals)

        # Control variate estimate
        mean_C = np.mean(C_vals)
        cov_matrix = np.cov(Y_vals, C_vals)
        cov_YC = cov_matrix[0, 1]
        var_C = cov_matrix[1, 1]

        # Avoid division by zero if Var(C) is 0
        b_star = cov_YC / var_C if var_C > 0 else 0

        # Theoretical mean of control variate C (number in queue)
        # For M/M/1, the expected number of packets in the queue (Lq) is rho^2 / (1 - rho)
        # The number of packets seen upon arrival (E[N_q_arrival]) is also Lq in an M/M/1 due to PASTA property.
        E_C = (self.rho**2) / (1.0 - self.rho)
        
        controlled_estimate = mean_Y - b_star * (mean_C - E_C)

        return mean_Y, controlled_estimate

# Simulation Parameters
lam = 1      # Arrival rate
mu = 2       # Service rate
NB_REPLICATIONS = 100
PACKETS_PER_REP = 5000

# Run Multiple Replications
naive_estimates = []
controlled_estimates = []

simulator = MM1_Exercise2(lam=lam, mu=mu)

for i in range(NB_REPLICATIONS):
    print(f"Running replication {i+1}/{NB_REPLICATIONS}...")
    naive_est, controlled_est = simulator.run(max_packets=PACKETS_PER_REP)
    naive_estimates.append(naive_est)
    controlled_estimates.append(controlled_est)

print("\n--- Analysis Complete ---")

# Analyze Results
# Theoretical value
theoretical_mean_traversal_time = 1 / (mu - lam)

# Naive estimator statistics
mean_naive = np.mean(naive_estimates)
var_naive = np.var(naive_estimates, ddof=1)

# Control variate estimator statistics
mean_controlled = np.mean(controlled_estimates)
var_controlled = np.var(controlled_estimates, ddof=1)

# Variance reduction
variance_reduction_percentage = ((var_naive - var_controlled) / var_naive) * 100

# Print Report
print(f"\nParameters: lambda = {lam}, mu = {mu}, rho = {simulator.rho:.2f}")
print(f"Number of Replications: {NB_REPLICATIONS}")
print(f"Packets per Replication: {PACKETS_PER_REP}")
print("-" * 50)
print(f"Theoretical Average Traversal Time: {theoretical_mean_traversal_time:.4f}")
print("-" * 50)
print("Naive Estimator:")
print(f"   - Average Estimate: {mean_naive:.4f}")
print(f"   - Variance of Estimates: {var_naive:.6f}")
print("\nControl Variate Estimator:")
print(f"   - Average Estimate: {mean_controlled:.4f}")
print(f"   - Variance of Estimates: {var_controlled:.6f}")
print("-" * 50)
print(f"Variance Reduction Achieved: {variance_reduction_percentage:.2f}%")
print("-" * 50)

# Plotting
plt.figure(figsize=(12, 6))

# Plotting the distributions of the estimates
plt.subplot(1, 2, 1)
plt.hist(naive_estimates, bins=20, alpha=0.7, label='Naive Estimates', color='skyblue', edgecolor='black')
plt.axvline(theoretical_mean_traversal_time, color='red', linestyle='dashed', linewidth=2, label='Theoretical Mean')
plt.axvline(mean_naive, color='blue', linestyle='dotted', linewidth=2, label='Average Naive Estimate')
plt.title('Distribution of Naive Estimates')
plt.xlabel('Average Traversal Time')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.hist(controlled_estimates, bins=20, alpha=0.7, label='Control Variate Estimates', color='lightgreen', edgecolor='black')
plt.axvline(theoretical_mean_traversal_time, color='red', linestyle='dashed', linewidth=2, label='Theoretical Mean')
plt.axvline(mean_controlled, color='green', linestyle='dotted', linewidth=2, label='Average Control Variate Estimate')
plt.title('Distribution of Control Variate Estimates')
plt.xlabel('Average Traversal Time')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Comparison of Naive and Control Variate Estimators', y=1.02, fontsize=16)
plt.show()

plt.figure(figsize=(8, 6))
data_to_plot = [naive_estimates, controlled_estimates]
labels = ['Naive', 'Control Variate']
plt.boxplot(data_to_plot, labels=labels, patch_artist=True, medianprops={'color': 'black'})
plt.axhline(theoretical_mean_traversal_time, color='red', linestyle='dashed', linewidth=2, label='Theoretical Mean')
plt.title('Box Plot of Estimator Distributions')
plt.ylabel('Average Traversal Time')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()