from opt_environment import OffloadEnvironment
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Initial environment parameters
    NUM_BSs = 10  # The number of Base Stations （BSs）or Edge Servers (ESs)
    NUM_TASKS_max = 100  # The max number of tasks in each BS
    BIT_RANGE = [10, 40]  # The range [10, 40] (in Mbits) of task bits
    NUM_TIME_SLOTS = 100  # The number of time slot set
    ES_capacity_max = 50  # The maximal computing capacity in ESs.
    np.random.seed(2)
    ES_capacity = np.random.randint(10, ES_capacity_max + 1, size=[NUM_BSs])  # The computing capacity of ES
    episodes = 100

    env = OffloadEnvironment(NUM_TASKS_max, BIT_RANGE, NUM_BSs, NUM_TIME_SLOTS, ES_capacity)  # Generate environment

    # =============== Optimal Task Scheduling ===================
    average_delays = []  # average service delays
    for i_episode in range(episodes):
        # Generate the arrival tasks
        arrival_tasks = []
        for i in range(env.time_slots):
            task_dim = np.random.randint(1, env.n_tasks + 1)
            arrival_tasks.append(np.random.uniform(env.min_bit, env.max_bit, size=[task_dim]))

        env.reset(arrival_tasks)  # Reset environment
        episode_delays = []  # The all service delays of tasks in each episode
        for t in range(env.time_slots - 1):
            # ================ Online Task Scheduling ====================
            task_set_len = len(env.tasks_bit[t])
            for n in range(task_set_len):
                n_delay = env.step_(t, n)
                episode_delays.append(n_delay)  # Record the total service delay of task n
            env.update_proc_queues(t)  # Update the processing queue of all ESs

        average_delays.append(np.mean(episode_delays))  # Store the average delay of each episode
        print({'Episode': '%d' % (i_episode + 1), 'average delay': '%.4f' % average_delays[-1]})

    print('============  Finish all tasks offloading with opt method  ==========')

    # Plot the average delay varying episodes
    episodes_list = list(range(len(average_delays)))
    np.savetxt('../results/AveDelay_Opt_BS' + str(NUM_BSs) +
               '_tasks' + str(NUM_TASKS_max) +
               '_f' + str(ES_capacity_max) +
               '_episode' + str(episodes) + '.csv', average_delays, delimiter=',', fmt='%.4f')
    plt.figure(1)
    plt.plot(episodes_list, average_delays)
    plt.ylabel('Average service delay')
    plt.xlabel('Episode')
    plt.savefig('../results/AveDelay_Opt_BS' + str(NUM_BSs) +
                '_tasks' + str(NUM_TASKS_max) +
                '_f' + str(ES_capacity_max) +
                '_episode' + str(episodes) + '.png')
    plt.close()
