from edge_environment import OffloadEnvironment
from fdsac_model import *
import matplotlib.pyplot as plt


def FDScheduling_algorithm():
    # ================== 1. 初始化环境参数 ==================
    NUM_BSs = 10  # 基站/边缘服务器（ES）的数量 B
    NUM_TASKS_max = 100  # 每个时隙最多有多少个任务到达（上限 N_max）
    BIT_RANGE = [10, 40]   # 每个任务的数据大小范围（单位 Mbits）
    NUM_TIME_SLOTS = 100  # 一个 episode 内的时间片数量 |T|
    ES_capacity_max = 50   # ES 最大计算能力上限，用来随机生成各 ES 的计算力
    np.random.seed(2) # 固定随机种子，保证可复现
    # 随机生成每个 ES 的计算能力（10 到 ES_capacity_max 之间的整数）
    ES_capacity = np.random.randint(10, ES_capacity_max + 1, size=[NUM_BSs])  # The computing capacities of all ESs

    # ================== 2. 初始化 DRL/FDEdge 模型参数 ==================
    actor_lr = 1e-4  # Actor（FDN 扩散网络）的学习率
    critic_lr = 1e-3  # Critic（两个 Q 网络）的学习率
    alpha = 0.05   # 策略熵正则的温度参数（越大越鼓励探索）
    alpha_lr = 3e-4   # 熵温度 alpha 的学习率
    episodes = 100   # 训练的 episode 数量
    denoising_steps = 5  # FDN 的去噪步数 I（论文里 diffusion steps）
    hidden_dim = 128  # Actor 和 Critic 等网络的隐藏层神经元数
    gamma = 0.95   # 折扣因子 γ（未来回报的折扣）
    tau = 0.005 # 软更新系数 τ（更新目标 Q 网络用）
    train_buffer_size = 10000  # 经验池容量（replay buffer size）
    batch_size = 64   # 每次从经验池采样的 batch 大小
    target_entropy = -1  # 目标熵（SAC 风格），用于自动调节 alpha
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ================== 3. 创建环境和智能体 ==================
    # 实例化协同边缘计算环境，内部会生成任务分布、传输速率、队列等
    env = OffloadEnvironment(NUM_TASKS_max, BIT_RANGE, NUM_BSs, NUM_TIME_SLOTS, ES_capacity)  # Generate environment
    # 创建 FDSAC 智能体：
    #   - env.state_dim：状态维度（任务大小 + 任务计算量 + 所有 ES 队列长度）
    #   - env.action_dim：动作维度 = ES 个数（把任务丢给哪个 ES）
    #   - 其他是超参数（学习率、γ、τ、扩散步数等）
    agent = FDSAC(env.state_dim, hidden_dim, env.action_dim, actor_lr, critic_lr, alpha, alpha_lr,
                  target_entropy, tau, gamma, denoising_steps, device)  # Initialize a agent
    # 经验回放池（用于存储 (s, a, latent_action_probs, r, s', next_latent_probs)）
    train_buffer = ReplayBuffer(train_buffer_size)  # Initialize the buffer size of experience pool

    # =============== 4. FDSAC 在线调度 + 并行模型训练 ===================
    # =============== FDSAC-based Online Task Scheduling with Parallel Network Model Training ===================
    average_delays = []  # average service delays
    for i_episode in range(episodes):
        # ===== 4.1 生成这一轮 episode 的到达任务序列 =====
        # ======= Generate the arrival tasks of environment ===========
        arrival_tasks = []
        for i in range(env.time_slots):
            # 对于每个时间片 t：随机这个时隙里到达多少个任务（1 ~ env.n_tasks）
            task_dim = np.random.randint(1, env.n_tasks + 1)
            # 再为每个任务随机生成 bit 大小，范围 [min_bit, max_bit]
            arrival_tasks.append(np.random.uniform(env.min_bit, env.max_bit, size=[task_dim]))
        # 把这一次 episode 的到达任务序列交给环境，重置各种状态、队列
        env.reset_env(arrival_tasks)  # Reset environment
        episode_delays = []  # 用来记录本 episode 中所有任务的服务时延
        exe_count = 0  # 计数：累积执行了多少次 offloading（用于触发训练）
        # 注意 range(env.time_slots - 1)：最后一个时隙多用于队列清空，不再产生新任务
        for t in range(env.time_slots - 1):
            # ================ Online Task Scheduling ====================
            # 当前时隙 t 实际到达的任务数量
            task_set_len = len(env.tasks_bit[t])
            # ------- 遍历当前时隙的每一个到达任务 n -------
            for n in range(task_set_len):
                # 4.2 观察当前状态 state：
                #   state = [任务大小 d_n,任务计算量 ρ_n * d_n,所有 ES 的队列长度 q_{t-1}]
                state = np.hstack([env.tasks_bit[t][n],
                                   env.comp_density[n] * env.tasks_bit[t][n],
                                   env.proc_queue_len[t]])  # Observe the system state
                # 4.3 读取“历史动作概率” latent_action_probs：
                #   这是 FDN 的初始输入 x_{t,n,I}，记录了之前类似任务被分配到各个 ES 的倾向
                latent_action_probs = env.latent_action_prob_space[t][n]  # Observe the latent action probability
                # 4.4 调用 Agent 的 Actor（FDN）进行一次扩散式决策：
                #   输入：state + latent_action_probs（历史概率）
                #   输出：action（选哪台 ES）以及当前时刻更新后的动作概率分布 action_probs
                action, action_probs = agent.take_action(state, latent_action_probs)  # Generate the offloading decision using Actor
                # 4.5 把最新的动作概率分布写回环境的 latent_action_prob_space：
                #   这样下一次看到同一位置的任务（或下一时隙、下个 episode）就有“记忆”
                env.latent_action_prob_space[t][n] = action_probs  # Update the latent action space
                # 4.6 环境执行这个调度动作
                #   - 更新对应 ES 的处理队列  计算传输+计算+等待时延  给出 reward（负的delay）和 delay 本身
                next_state, next_latent_action, reward, delay = env.step(t, n, action)  # Perform the processing of task n
                # 4.7 把这次交互的经历存进经验池：包含 状态、动作、历史动作分布、奖励、下一个状态、下一个历史动作分布
                train_buffer.add(state, action, latent_action_probs, reward, next_state, next_latent_action)  # Store history tuple
                # 4.8 记录这个任务的服务时延
                episode_delays.append(delay)  # Record the total service delay of task n
                # 4.9 累计 offloading 次数，用于控制训练频率（这里后来注释掉了间隔训练）
                exe_count = exe_count + 1  # Update the execution count of task offloading
            # 4.10 每处理完一整个时隙的任务后，更新所有 ES 的队列
            #   这个函数中会按 ES 计算能力 f_b，把上一个时隙残留的 workload 减掉一部分
            env.update_proc_queues(t)  # Update the processing queue of all ESS

            # ============= Parallel Network Model Training ===============
            # 500 is the minimal size of history tuple in the experience pool
            # batch_size is set as the training interval of network model
            # if train_buffer.size() > 500 and exe_count % batch_size == 0:  # Network Model Training
            # ============= 4.11 并行网络训练（在调度过程中穿插训练） =============
            # 原论文给了一个版本：当 buffer > 500 且 exe_count % batch_size == 0 时训练一次
            # 这里简化成：只要 buffer 里的数据超过 500，就每个时隙都训练
            # if train_buffer.size() > 500 and exe_count % batch_size == 0:
            if train_buffer.size() > 500:  # Network Model Training
                # 从经验池中随机采样 batch_size 条经验
                b_s, b_a, b_p, b_r, b_ns, b_np = train_buffer.sample(batch_size)  # Sampling a batch of Sample
                # 把采样结果打包成一个字典，方便传给 agent.update()
                transition_dict = {'states': b_s, # 当前状态 s
                                   'actions': b_a,# 当前动作 a
                                   'latent_action_probs': b_p,# 当前历史动作概率 x
                                   'rewards': b_r,# reward r
                                   'next_states': b_ns, # 下一个状态 s'
                                   'next_latent_action_probs': b_np# 下一个历史动作概率 x'
                                   }
                # 调用 FDSAC 的 update 方法：
                #   - 更新两套 Critic（Q 网络）
                #   - 更新 Actor（FDN 扩散网络）
                #   - 更新 熵温度 alpha
                #   - 软更新 target Critic
                agent.update(transition_dict)  # Train network model and update its parameters
        # 4.12 一个 episode 结束后，计算这一轮中所有任务的平均服务时延
        average_delays.append(np.mean(episode_delays))  # Store the average delay of each episode
        print({'Episode': '%d' % (i_episode + 1), 'average delay': '%.4f' % average_delays[-1]})

    print('============ Finish all tasks offloading and model training with FDEdge method ==========')

    # Save and plot the average delay varying episodes
    # ================== 5. 保存并画出平均时延曲线 ==================
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
