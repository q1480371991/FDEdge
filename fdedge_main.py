from edge_environment import OffloadEnvironment
from fdsac_model import *
import matplotlib.pyplot as plt


def FDScheduling_algorithm():
    # Initial environment parameters
    NUM_BSs = 10  # The number of Base Stations （BSs）or Edge Servers (ESs)
    NUM_TASKS_max = 100  # The max number of task arriving at the master node
    BIT_RANGE = [10, 40]  # The range [10, 40] (in Mbits) of task size.
    NUM_TIME_SLOTS = 100  # The number of time slot set
    ES_capacity_max = 50  # The maximal computing capacity in ESs.
    np.random.seed(2)
    ES_capacity = np.random.randint(10, ES_capacity_max + 1, size=[NUM_BSs])  # The computing capacities of all ESs

    # Initial DRL model parameters
    actor_lr = 1e-4  # The learn rate of the actor (i.e., diffusion) network
    critic_lr = 1e-3  # The learn rate of the critic network
    alpha = 0.05  # The temperature of action entropy regularization
    alpha_lr = 3e-4  # The learning rate of entropy
    episodes = 100  # The number of episodes in network training procedure
    denoising_steps = 5  # The denoising steps in the diffusion-based scheduling model
    hidden_dim = 128  # The hidden neurons of the DNs, CNs, and TNs
    gamma = 0.95  # The reward decay parameter
    tau = 0.005  # The weight parameter of soft updating operation
    train_buffer_size = 10000  # The capacity of experience pool in the memory
    batch_size = 64  # The batch size of sampling history tuple
    target_entropy = -1  # The target entropy parameter
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = OffloadEnvironment(NUM_TASKS_max, BIT_RANGE, NUM_BSs, NUM_TIME_SLOTS, ES_capacity)  # Generate environment
    agent = FDSAC(env.state_dim, hidden_dim, env.action_dim, actor_lr, critic_lr, alpha, alpha_lr,
                  target_entropy, tau, gamma, denoising_steps, device)  # Initialize a agent
    train_buffer = ReplayBuffer(train_buffer_size)  # Initialize the buffer size of experience pool

    # =============== FDSAC-based Online Task Scheduling with Parallel Network Model Training ===================
    average_delays = []  # average service delays
    for i_episode in range(episodes):
        # ======= Generate the arrival tasks of environment ===========
        arrival_tasks = []
        for i in range(env.time_slots):
            task_dim = np.random.randint(1, env.n_tasks + 1)
            arrival_tasks.append(np.random.uniform(env.min_bit, env.max_bit, size=[task_dim]))

        env.reset_env(arrival_tasks)  # Reset environment
        episode_delays = []  # The all service delays of tasks in each episode
        exe_count = 0  # Initial the execution count of task offloading
        for t in range(env.time_slots - 1):
            # ================ Online Task Scheduling ====================
            task_set_len = len(env.tasks_bit[t])
            for n in range(task_set_len):
                state = np.hstack([env.tasks_bit[t][n],
                                   env.comp_density[n] * env.tasks_bit[t][n],
                                   env.proc_queue_len[t]])  # Observe the system state
                latent_action_probs = env.latent_action_prob_space[t][n]  # Observe the latent action probability
                action, action_probs = agent.take_action(state, latent_action_probs)  # Generate the offloading decision using Actor
                env.latent_action_prob_space[t][n] = action_probs  # Update the latent action space
                next_state, next_latent_action, reward, delay = env.step(t, n, action)  # Perform the processing of task n
                train_buffer.add(state, action, latent_action_probs, reward, next_state, next_latent_action)  # Store history tuple
                episode_delays.append(delay)  # Record the total service delay of task n
                exe_count = exe_count + 1  # Update the execution count of task offloading
            env.update_proc_queues(t)  # Update the processing queue of all ESS

            # ============= Parallel Network Model Training ===============
            # 500 is the minimal size of history tuple in the experience pool
            # batch_size is set as the training interval of network model
            # if train_buffer.size() > 500 and exe_count % batch_size == 0:  # Network Model Training
            if train_buffer.size() > 500:  # Network Model Training
                b_s, b_a, b_p, b_r, b_ns, b_np = train_buffer.sample(batch_size)  # Sampling a batch of Sample
                transition_dict = {'states': b_s, 'actions': b_a, 'latent_action_probs': b_p,
                                   'rewards': b_r, 'next_states': b_ns, 'next_latent_action_probs': b_np}
                agent.update(transition_dict)  # Train network model and update its parameters

        average_delays.append(np.mean(episode_delays))  # Store the average delay of each episode
        print({'Episode': '%d' % (i_episode + 1), 'average delay': '%.4f' % average_delays[-1]})

    print('============ Finish all tasks offloading and model training with FDEdge method ==========')

    # Save and plot the average delay varying episodes
    episodes_list = list(range(len(average_delays)))
    np.savetxt('results/AveDelay_FDEdge_BS' + str(NUM_BSs) +
               '_tasks' + str(NUM_TASKS_max) +
               '_f' + str(ES_capacity_max) +
               '_steps' + str(denoising_steps) +
               '_episode' + str(episodes) + '.csv', average_delays, delimiter=',', fmt='%.4f')
    plt.figure(1)
    plt.plot(episodes_list, average_delays)
    plt.ylabel('Average service delay')
    plt.xlabel('Episode')
    plt.savefig('results/AveDelay_FDEdge_BS' + str(NUM_BSs) +
                '_tasks' + str(NUM_TASKS_max) +
                '_f' + str(ES_capacity_max) +
                '_steps' + str(denoising_steps) +
                '_episode' + str(episodes) + '.png')
    plt.close()


if __name__ == '__main__':
    FDScheduling_algorithm()
