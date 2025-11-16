import numpy as np
import collections

import torch
from scipy import stats


class OffloadEnvironment:
    def __init__(self, num_tasks, bit_range, num_BSs, time_slots_, es_capacities):
        # INPUT DATA
        self.n_tasks = num_tasks  # The number of mobile devices
        self.n_BSs = num_BSs  # The number of base station or edge server
        self.time_slots = time_slots_  # The number of time slot set
        self.state_dim = 2 + self.n_BSs  # The dimension of system state
        self.action_dim = num_BSs
        self.duration = 1  # The length of each time slot t. Unit: seconds
        self.ES_capacities = es_capacities  # GHz or Gigacycles/s
        np.random.seed(5)
        self.tran_rate_BSs = np.random.randint(400, 501, size=[self.n_BSs])  # Mbits/s
        # Set rhe computing density of each diffusion step in the range (100, 300) Cycles/Mbit
        np.random.seed(1)
        self.comp_density = np.random.uniform(0.1024, 0.3072, size=[self.n_tasks])  # Gigacycles/Mbit

        # Initialize the array to storage all the arrival tasks bits in the system
        self.tasks_bit = []
        self.min_bit = bit_range[0]  # Set the minimal bit of tasks
        self.max_bit = bit_range[1]  # Set the maximal bit of tasks

        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        # Initialize an array to storage the queue workload lengths before processing current task in all ESs
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        self.wait_delay = 0  # Initial waiting delay of each task

        # Initialize the overall space of latent action probability
        self.latent_action_prob_space = np.random.normal(size=[self.time_slots, self.n_tasks, self.action_dim])

    def reset_env(self, tasks_bit):
        self.tasks_bit = tasks_bit  # Initialize the whole tasks in the system environment
        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        # Initialize an array to storage the queue workload lengths before processing current task in all ESs
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        self.wait_delay = 0  # Initial waiting delay of each task

    # Perform task offloading to achieve:
    # (1) Service delays;
    # (2) The queue workload lengths of arrival tasks in all ES.
    def step(self, t, n, action):
        self.wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]
        # Calculate the total transmission and computing delay of task n
        tran_comp_delays = (self.tasks_bit[t][n] / self.tran_rate_BSs[action] +
                            self.comp_density[n] * self.tasks_bit[t][n] / self.ES_capacities[action])

        # calculate the total service delay of task n
        delay = tran_comp_delays + self.wait_delay
        reward = - delay  # Set the reward
        # Update the workload lengths of the processing queue at the selected ESs before processing next task
        self.proc_queue_bef[t][action] = self.proc_queue_bef[t][action] + self.comp_density[n] * self.tasks_bit[t][n]

        # Observe the next system state and potential action
        if n == len(self.tasks_bit[t]) - 1:
            next_state = np.hstack([self.tasks_bit[t + 1][0],
                                    self.comp_density[0] * self.tasks_bit[t+1][0],
                                    self.proc_queue_len[t + 1]])
            next_potential_action = self.latent_action_prob_space[t + 1][0]
        else:
            next_state = np.hstack([self.tasks_bit[t][n + 1],
                                    self.comp_density[n + 1] * self.tasks_bit[t][n+1],
                                    self.proc_queue_len[t]])
            next_potential_action = self.latent_action_prob_space[t][n + 1]

        return next_state, next_potential_action, reward, delay

    # Update the processing queue length of all ESs at the beginning of next time slot.
    def update_proc_queues(self, t):
        for b_ in range(self.n_BSs):
            self.proc_queue_len[t + 1][b_] = np.max(
                [self.proc_queue_len[t][b_] + self.proc_queue_bef[t][b_] - self.ES_capacities[b_] * self.duration, 0])
