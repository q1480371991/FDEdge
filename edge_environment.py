import numpy as np
import collections

import torch
from scipy import stats

# 实现了协同边缘计算环境，根据fdedge_main.py中的给定参数进行初始化

class OffloadEnvironment:
    """
       协同边缘计算系统的环境类。
       模拟多基站 / 多ES 的任务到达、排队、执行过程。
    """
    def __init__(self, num_tasks, bit_range, num_BSs, time_slots_, es_capacities):
        # INPUT DATA
        self.n_tasks = num_tasks  # The number of mobile devices device?  每个时间片最大任务数（论文里 N_max）
        self.n_BSs = num_BSs  # The number of base station or edge server  ES / 基站数量 B
        self.time_slots = time_slots_  # The number of time slot set  一个 episode 中时间片数量 |T|
        self.state_dim = 2 + self.n_BSs  # The dimension of system state 状态维度 = [任务大小, 任务计算量, 所有 ES 队列长度]
        self.action_dim = num_BSs #动作维度 = ES 数量（把任务丢到哪个 ES）
        self.duration = 1  # The length of each time slot t. Unit: seconds  每个 time slot 的长度 Δt = 1 秒
        self.ES_capacities = es_capacities  # GHz or Gigacycles/s 每个 ES 的计算能力 f_b，单位：Gigacycles/s
        np.random.seed(5)
        # ====== 传输速率（所有 ES 的上行速率） ======
        # 随机为每个 ES 生成 400~500 Mbits/s 的传输速率 v_{t,n,b}
        self.tran_rate_BSs = np.random.randint(400, 501, size=[self.n_BSs])  # Mbits/s
        # Set rhe computing density of each diffusion step in the range (100, 300) Cycles/Mbit
        # ====== 每个任务的计算密度 ρ_n ======
        # 注：这里 comp_density 是长度 = n_tasks 的向量，单位是 Gigacycles/Mbit
        np.random.seed(1)
        self.comp_density = np.random.uniform(0.1024, 0.3072, size=[self.n_tasks])  # Gigacycles/Mbit

        # Initialize the array to storage all the arrival tasks bits in the system
        # tasks_bit 之后会存每个时间片到达任务的 bit（列表，按 fdedge_main 传入）
        self.tasks_bit = []
        self.min_bit = bit_range[0]  # Set the minimal bit of tasks 任务 bit 大小下界
        self.max_bit = bit_range[1]  # Set the maximal bit of tasks 任务 bit 大小上界

        # Initialize the array to storage the queue workload lengths of all ESs
        # proc_queue_len[t][b]：在时隙 t 开始时，第 b 个 ES 的队列工作量（工作量单位：Gigacycles）
        # 注意大小是 time_slots + 1，因为需要存 t=0..time_slots
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        # Initialize an array to storage the queue workload lengths before processing current task in all ESs
        # proc_queue_bef[t][b]：在时隙 t 内“当前任务之前”已经累积的工作量（尚未被 ES 处理的）
        # 每次 step() 会往里加当前任务的计算量，用于等待时延计算
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        self.wait_delay = 0  # Initial waiting delay of each task 当前任务的等待时延初始化

        # Initialize the overall space of latent action probability
        # latent_action_prob_space[t][n]：时隙 t、第 n 个任务的“潜在动作概率向量”
        # 这里初始化为高斯随机数，稍后在 fdedge_main 里会被 Actor 输出覆盖
        self.latent_action_prob_space = np.random.normal(size=[self.time_slots, self.n_tasks, self.action_dim])

    def reset_env(self, tasks_bit):
        # 保存这次 episode 的任务 bit 序列，用于后续 step 中访问
        self.tasks_bit = tasks_bit  # Initialize the whole tasks in the system environment
        # Initialize the array to storage the queue workload lengths of all ESs
        # 重置各 ES 在每个时隙的队列工作量（全部清零）
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        # Initialize an array to storage the queue workload lengths before processing current task in all ESs
        # 重置“当前任务之前已累积的工作量”
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        # 重置等待时延
        self.wait_delay = 0  # Initial waiting delay of each task

    # Perform task offloading to achieve:
    # (1) Service delays;
    # (2) The queue workload lengths of arrival tasks in all ES.
    def step(self, t, n, action):
        """
                在 time slot t，对第 n 个任务执行 offloading 动作：
                :param t: 当前时隙索引
                :param n: 当前时隙中第 n 个任务
                :param action: 动作（选择的 ES 编号 b）
                :return: next_state, next_potential_action, reward, delay
        """
        # ===== 1. 计算等待时延 =====
        # proc_queue_len[t][action]：时隙 t 开始时该 ES 的排队工作量 q_{t-1,b}
        # proc_queue_bef[t][action]：时隙 t 内，在当前任务之前已经加入队列的工作量 q^{bef}_{t,n,b}
        # 等待时延 = (当前队列工作量之和) / 计算能力
        self.wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]
        # Calculate the total transmission and computing delay of task n
        # ===== 2. 计算传输 + 计算时延 =====
        # self.tasks_bit[t][n]：当前时隙t任务n的 bit 大小 d_n
        # self.tran_rate_BSs[action]：选中 ES 的上传速率 v_{t,n,b}
        # self.comp_density[n]：任务n的计算密度 ρ_n
        # self.ES_capacities[action]： ES 计算能力 f_b
        # tran_comp_delays = d_n / v_{t,n,b} + ρ_n d_n / f_b
        tran_comp_delays = (self.tasks_bit[t][n] / self.tran_rate_BSs[action] +
                            self.comp_density[n] * self.tasks_bit[t][n] / self.ES_capacities[action])
        # ===== 3. 总服务时延 + 奖励 =====
        # calculate the total service delay of task n
        delay = tran_comp_delays + self.wait_delay# 总时延 = 传输时延+计算时延+等待时延
        reward = - delay  # Set the reward  奖励设为负时延（时延越小越好）
        # Update the workload lengths of the processing queue at the selected ESs before processing next task
        # ===== 4. 更新队列：当前任务的工作量加到 proc_queue_bef 上 =====
        # 注意这里不是直接加到 proc_queue_len，而是加到 proc_queue_bef[t][action]。
        # 在同一个时隙 t 内，会有多个任务对同一个 ES 排队。
        # 这些工作量会累积在 proc_queue_bef[t]，在 update_proc_queues(t) 里统一更新到下一时隙的 proc_queue_len[t+1]。
        self.proc_queue_bef[t][action] = self.proc_queue_bef[t][action] + self.comp_density[n] * self.tasks_bit[t][n]

        # Observe the next system state and potential action
        # ===== 5. 构造下一个状态和对应的“潜在动作概率” =====
        #如果当前任务是该时隙最后一个任务 → 下一状态来自下一时隙 t+1 的第一个任务；否则 → 下一状态是同一时隙的下一个任务。
        if n == len(self.tasks_bit[t]) - 1:
            # 当前是该时隙最后一个任务：下一个状态 = t+1 的第一个任务
            next_state = np.hstack([self.tasks_bit[t + 1][0],
                                    self.comp_density[0] * self.tasks_bit[t+1][0],
                                    self.proc_queue_len[t + 1]])# 注意：这里用的是 t+1 时隙的队列长度
            next_potential_action = self.latent_action_prob_space[t + 1][0]
        else:
            # 同一时隙内，下一个任务 n+1
            next_state = np.hstack([self.tasks_bit[t][n + 1],
                                    self.comp_density[n + 1] * self.tasks_bit[t][n+1],
                                    self.proc_queue_len[t]])# 队列仍然是当前时隙 t 的状态
            next_potential_action = self.latent_action_prob_space[t][n + 1]

        return next_state, next_potential_action, reward, delay

    # Update the processing queue length of all ESs at the beginning of next time slot.
    def update_proc_queues(self, t):
        """
                在 time slot t 结束时，更新所有 ES 在 t+1 时刻的队列长度：
                q_{t,b} = max(q_{t-1,b} + q^{bef}_{t,b} - f_b * duration, 0)
        """
        for b_ in range(self.n_BSs):
            # self.proc_queue_len[t][b_]: t 时隙开始时队列（上一时隙留下的）
            # self.proc_queue_bef[t][b_]: t 时隙内所有任务新增的工作量总和
            # self.ES_capacities[b_] * self.duration: t 时隙内 ES 能处理掉的最大工作量
            self.proc_queue_len[t + 1][b_] = np.max(
                [self.proc_queue_len[t][b_] + self.proc_queue_bef[t][b_] - self.ES_capacities[b_] * self.duration, 0])
