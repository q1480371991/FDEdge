from opt_environment import OffloadEnvironment
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ======================= 1. 初始化环境参数 ==========================
    # Initial environment parameters
    NUM_BSs = 10  # The number of Base Stations （BSs）or Edge Servers (ESs)  基站（BS）/边缘服务器（ES）数量
    NUM_TASKS_max = 100  # The max number of tasks in each BS   每个时间槽内最多任务数量（用于生成任务时的上界）
    BIT_RANGE = [10, 40]  # The range [10, 40] (in Mbits) of task bits   任务大小范围，单位：Mbits
    NUM_TIME_SLOTS = 100  # The number of time slot set  总时间槽数量
    ES_capacity_max = 50  # The maximal computing capacity in ESs.   单个 ES 的最大计算容量（Gigacycles/s 的上界）
    # 固定随机种子，保证可复现实验结果
    np.random.seed(2)
    # 随机生成每个 ES 的计算能力（整数），范围 [10, ES_capacity_max]
    ES_capacity = np.random.randint(10, ES_capacity_max + 1, size=[NUM_BSs])  # The computing capacity of ES
    episodes = 100# 仿真的总轮数（episode 数）
    # 创建 Offload 环境
    # 参数依次为：最大任务数、任务大小范围、BS 数量、时间槽数、每个 ES 的计算能力
    env = OffloadEnvironment(NUM_TASKS_max, BIT_RANGE, NUM_BSs, NUM_TIME_SLOTS, ES_capacity)  # Generate environment

    # ======================= 2. 进行“最优”任务调度仿真 ==========================
    # 这里的“最优”是指在每次决策时穷举所有 ES，选择当前时延最小的 ES，相当于贪心式的最短时延选择。
    # =============== Optimal Task Scheduling ===================
    average_delays = []  # average service delays 存储每个 episode 的平均服务时延
    for i_episode in range(episodes):
        # ---------- 2.1 生成一次 episode 内所有时间槽的任务到达 ----------
        # Generate the arrival tasks
        arrival_tasks = []
        for i in range(env.time_slots):
            # 随机生成当前时间槽的任务数量（至少 1 个，至多 env.n_tasks 个）
            task_dim = np.random.randint(1, env.n_tasks + 1)
            # 为该时间槽的每个任务生成一个任务大小（Mbits），均匀分布在 [min_bit, max_bit]
            arrival_tasks.append(np.random.uniform(env.min_bit, env.max_bit, size=[task_dim]))
            # 生成结果：arrival_tasks[i] 是一个长度为 task_dim 的一维数组
            #          第 j 个元素对应时间槽 i 的第 j 个任务的任务大小
        # 重置环境，把当前 episode 的任务序列传入
        env.reset(arrival_tasks)  # Reset environment
        # episode_delays：记录一个 episode 内所有任务的服务时延，用于后续计算平均值
        episode_delays = []  # The all service delays of tasks in each episode
        # ---------- 2.2 对每个时间槽进行在线任务调度 ----------
        # 这里只遍历到 env.time_slots - 1，因为 update_proc_queues 会写入 t+1 的状态
        for t in range(env.time_slots - 1):
            # ================ Online Task Scheduling ====================
            # 当前时间槽 t 的任务数量
            task_set_len = len(env.tasks_bit[t])
            # 对时间槽 t 中的每个任务依次进行决策
            for n in range(task_set_len):
                # 使用 env.step_：不区分来源 BS，仅根据任务大小和 ES 队列情况选择 ES
                n_delay = env.step_(t, n)
                # 将当前任务的服务时延记录下来
                episode_delays.append(n_delay)  # Record the total service delay of task n
            # 在处理完时间槽 t 中的所有任务后，更新所有 ES 队列，推进到时间槽 t+1
            env.update_proc_queues(t)  # Update the processing queue of all ESs
        # ---------- 2.3 计算并记录当前 episode 的平均服务时延 ----------
        average_delays.append(np.mean(episode_delays))  # Store the average delay of each episode
        print({'Episode': '%d' % (i_episode + 1), 'average delay': '%.4f' % average_delays[-1]})

    print('============  Finish all tasks offloading with opt method  ==========')
    # ======================= 3. 结果保存与画图 ==========================
    # Plot the average delay varying episodes
    episodes_list = list(range(len(average_delays)))
    np.savetxt('../../results/AveDelay_Opt_BS' + str(NUM_BSs) +
               '_tasks' + str(NUM_TASKS_max) +
               '_f' + str(ES_capacity_max) +
               '_episode' + str(episodes) + '.csv', average_delays, delimiter=',', fmt='%.4f')
    plt.figure(1)
    plt.plot(episodes_list, average_delays)
    plt.ylabel('Average service delay')
    plt.xlabel('Episode')
    plt.savefig('../../results/AveDelay_Opt_BS' + str(NUM_BSs) +
                '_tasks' + str(NUM_TASKS_max) +
                '_f' + str(ES_capacity_max) +
                '_episode' + str(episodes) + '.png')
    plt.close()
