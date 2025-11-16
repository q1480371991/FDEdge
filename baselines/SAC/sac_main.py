from sac_environment import OffloadEnvironment
from sac_model import *
import matplotlib.pyplot as plt


def SACScheduling_algorithm():
    # Initial environment parameters
    NUM_BSs = 10  # The number of Base Stations （BSs）or Edge Servers (ESs). Default: 20
    NUM_TASKS_max = 100  # The max number of task in each BS. Default: 50
    BIT_RANGE = [10, 40]  # The range of task size. Default: [10, 40] Mbits
    NUM_TIME_SLOTS = 100  # The number of time slot set. Default: 100
    ES_capacity_max = 50  # The maximal computing capacity in ESs. Default: 50
    np.random.seed(2)
    ES_capacity = np.random.randint(10, ES_capacity_max + 1, size=[NUM_BSs])  # The computing capacity of ES

    # Initial DRL model parameters
    actor_lr = 1e-4  # The learn rate of the actor (i.e., diffusion) network
    critic_lr = 1e-3  # The learn rate of the critic network
    alpha = 0.01  # The temperature of action entropy regularization
    alpha_lr = 1e-3  # The learning rate of entropy
    episodes = 100  # The number of episodes in network training procedure
    hidden_dim = 128  # The hidden neurons of the DNs, CNs, and TNs
    gamma = 0.9  # The reward decay parameter
    tau = 0.005  # The weight parameter of soft updating operation
    train_buffer_size = 10000  # The capacity of experience pool in the memory
    batch_size = 64  # The batch size of sampling history tuple
    target_entropy = -1  # The target entropy parameter
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = OffloadEnvironment(NUM_TASKS_max, BIT_RANGE, NUM_BSs, NUM_TIME_SLOTS, ES_capacity)  # Generate environment
    agent = SAC(env.state_dim, hidden_dim, env.action_dim, actor_lr, critic_lr, alpha, alpha_lr,
                target_entropy, tau, gamma, device)  # Initialize a agent
    train_buffer = ReplayBuffer(train_buffer_size)  # Initialize the buffer size of experience pool

    # =============== SAC-based Online Task Scheduling and Offline Network Model Training in parallel ===============
    average_delays = []  # average service delays
    exe_count = 0  # Initial the execution count of task offloading
    train_count = 0  # Initial the execution count of network training
    for i_episode in range(episodes):
        # ======== Generate the arrival tasks of environment =========
        arrival_tasks = []
        for i in range(env.time_slots):
            task_dim = np.random.randint(1, env.n_tasks + 1)
            arrival_tasks.append(np.random.uniform(env.min_bit, env.max_bit, size=[task_dim]))

        env.reset_env(arrival_tasks)  # Reset environment
        episode_delays = []  # The all service delays of tasks in each episode
        for t in range(env.time_slots - 1):
            # ================ Online Task Scheduling ====================
            task_set_len = len(env.tasks_bit[t])
            for n in range(task_set_len):
                state = np.hstack([env.tasks_bit[t][n],
                                   env.comp_density[n] * env.tasks_bit[t][n],
                                   env.proc_queue_len[t]])  # Observe the system state
                action = agent.take_action(state)  # Generate the offloading decision using Actor
                next_state, reward, delay = env.step_(t, n, action)  # # Perform the processing of task n
                train_buffer.add(state, action, reward, next_state)  # Store history tuple
                episode_delays.append(delay)  # Record the total service delay of task n
                exe_count = exe_count + 1  # Update the execution count of task offloading
            env.update_proc_queues(t)  # Update the processing queue of all ESS

            # ============= Parallel Network Model Training in  ===============
            # 300 is the minimal size of history tuple in the experience pool
            # batch_size is set as the training interval of network model
            # if train_buffer.size() > 300 and exe_count % 10 == 0:
            if train_buffer.size() > 300:
                b_s, b_a, b_r, b_ns = train_buffer.sample(batch_size)  # Sampling a batch of Sample
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r}
                agent.update(transition_dict)  # Train network model and update its parameters
                train_count = train_count + 1
        average_delays.append(np.mean(episode_delays))  # Store the average delay of each episode
        print({'Episode': '%d' % (i_episode+1), 'average delay': '%.4f' % average_delays[-1]})

    print('============ Finish all tasks offloading and model training with sac method  ==========')

    # Plot the average delay varying episodes
    episodes_list = list(range(len(average_delays)))
    np.savetxt('../results/AveDelay_sac_BS' + str(NUM_BSs) +
               '_tasks' + str(NUM_TASKS_max) +
               '_f' + str(ES_capacity_max) +
               '_episode' + str(episodes) + '.csv', average_delays, delimiter=',', fmt='%.4f')
    plt.figure(1)
    plt.plot(episodes_list, average_delays)
    plt.ylabel('Average make-span')
    plt.xlabel('Episode')
    plt.savefig('../results/AveDelay_sac_BS' + str(NUM_BSs) +
                '_tasks' + str(NUM_TASKS_max) +
                '_f' + str(ES_capacity_max) +
                '_episode' + str(episodes) + '.png')
    plt.close()


if __name__ == '__main__':
    SACScheduling_algorithm()
