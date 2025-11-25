import numpy as np
import collections
from scipy import stats


class OffloadEnvironment:
    def __init__(self, num_tasks, bit_range, num_BSs, time_slots_, es_capacities):
        # ======================= 环境基本配置参数 ==========================
        # num_tasks：每个时间槽内，单个基站（或系统）最多可能出现的任务数量上限
        # bit_range：任务大小范围 [min_bits, max_bits]，单位是 Mbits
        # num_BSs：基站/边缘服务器（BS/ES）的数量
        # time_slots_：总时间槽数量
        # es_capacities：每个 ES 的计算能力（单位：Gigacycles/s），长度为 num_BSs 的数组

        # --------- 输入数据与系统规模 ----------
        # INPUT DATA
        self.n_tasks = num_tasks  # The number of mobile devices    最大任务数（这里作为每个 time slot 中任务数量的上界）
        self.n_BSs = num_BSs  # The number of base station or edge server   基站/边缘服务器数量
        self.time_slots = time_slots_  # The number of time slot set 仿真的时间槽数
        self.duration = 1  # The length of each time slot t. Unit: seconds  每个时间槽的持续时间（秒），这里固定为 1s
        self.ES_capacities = es_capacities  # GHz or Gigacycles/s  每个 ES 的计算能力，单位：Gigacycles/s

        # --------- 下行/回传传输速率 ----------
        np.random.seed(5)
        # 每个 BS 对应的传输速率 Mbits/s（假设固定不随时间变化）
        # tran_rate_BSs[i] 表示把任务 offload 到第 i 个 BS 时使用的传输速率
        self.tran_rate_BSs = np.random.randint(400, 501, size=[self.n_BSs])  # Mbits/s

        # --------- 任务计算密度 ----------
        # 每个任务（或每个终端）的计算密度：每 1 比特需要的 CPU 周期数
        # 这里单位是 Gigacycles/step（相当于一个“扩散步骤”所需计算）
        # Set the computing density of each required diffusion step in the range (100, 300) # Cycles/step
        np.random.seed(1)
        self.comp_density = np.random.uniform(0.1024, 0.3072, size=[self.n_tasks])  # Gigacycles/step

        # ======================= 任务数据与队列状态 ==========================
        # tasks_bit[t][b][n]：在时间槽 t，由基站 b 接收到的第 n 个任务的任务大小（Mbits）
        # 或者在 step_ 版本中：tasks_bit[t][n]（不区分来源基站）
        # Initialize the array to storage all the arrival tasks bits in the system
        self.tasks_bit = []
        # 任务大小范围
        self.min_bit = bit_range[0]
        self.max_bit = bit_range[1]

        # proc_queue_len[t][b]：在时间槽 t 时，第 b 个 ES 队列中“尚未完成”的工作量（Gigacycles）
        # 注意：这里数组长度是 time_slots + 1，方便存储 t+1 时刻的状态
        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles

        # proc_queue_bef[t][b]：在时间槽 t 执行所有 step 之前，第 b 个 ES 在本时间槽内新到达任务的总工作量
        # 也可以理解为“当前时间槽内，尚未来得及服务完的新任务的累计计算量”
        # Initialize an array to storage the queue workload lengths before processing current task in all ESs
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles

    def reset(self, tasks_bit):
        """
                重置环境到初始状态，用给定的任务到达序列重新开始仿真。
                tasks_bit:
                    - 对应主程序中传入的 arrival_tasks
                    - 在 step_ 使用中：tasks_bit[t] 为时间槽 t 的任务大小列表，一维数组，长度为当前 time slot 的任务数
                    - 在 step 使用中：tasks_bit[t][b][n] 为时间槽 t、基站 b 的第 n 个任务大小
                """
        self.tasks_bit = tasks_bit
        # 重置所有 ES 的队列长度（历史累积工作量清零）
        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles
        # 重置所有 ES 在当前时间槽的新到达任务工作量（清零）
        # Initialize an array to storage the queue workload lengths before processing current task in all ESs
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])  # Gigacycles

    # ======================= 带“来源 BS”维度的 step ==========================
    # 用于有 tasks_bit[t][b][n] 这种 3 维结构时的决策。
    # (1) 计算服务时延（通信 + 计算 + 排队）
    # (2) 更新选中 ES 的队列工作量

    # Perform task offloading to achieve:
    # (1) Service delays;
    # (2) The queue workload lengths of arrival tasks in all ES.
    def step(self, t, b, n):
        """
                在时间槽 t，为来自基站 b 的第 n 个任务选择最优 offloading 目标 ES。
                t: 当前时间槽下标
                b: 当前任务所属的基站/BS 编号
                n: 当前任务在该 BS 中的索引

                return：min_service_delay: 该任务在最优 offload 选择下的总服务时延
        """
        opt_action = []# 最优动作（选中的 ES 下标）
        min_service_delay = 1000# 初始化为一个很大的值，用于后续求最小
        # 遍历所有 ES，穷举式选择带来最小时延的 ES
        for i in range(self.n_BSs):
            action = i# 当前候选的 ES 编号
            # ---------- 排队等待时延 ----------
            # (历史队列长度 + 当前时间槽累计要处理的新任务量) / ES 计算能力 = 需要的服务时间
            # queue_len: 之前时间槽遗留的工作量
            # queue_bef: 当前时间槽内在该 ES 已经分配的任务工作量（在处理当前任务前）
            wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]
            # ---------- 传输 + 计算时延 ----------
            if action == b:  # The transmission delay equals to 0 when task is processed at local BS b
                # 如果任务在本地 BS（b）对应的 ES 上处理，则假设不需要传输（传输延迟为 0）
                # 总时延 = 计算时延 = 任务所需计算量 / 该 ES 计算能力
                # Calculate the total transmission and computing delay of task n.
                tran_comp_delays = (self.comp_density[n] * self.tasks_bit[t][b][n] / self.ES_capacities[action])
            else:  # The transmission delay not equals to 0 when task is processed at other BS
                # 如果 offload 到其他 BS，则需要传输时间 + 计算时间
                # Calculate the total transmission and computing delay of task n
                tran_comp_delays = (self.tasks_bit[t][b][n] / self.tran_rate_BSs[action] +# 传输时间（大小 / 速率）
                                    self.comp_density[n] * self.tasks_bit[t][b][n] / self.ES_capacities[action]) # 计算时间
            # ---------- 总服务时延 ----------
            service_delay = tran_comp_delays + wait_delay  # Set the delay
            # ---------- 记录最小时延及对应 ES ----------
            if service_delay < min_service_delay:
                min_service_delay = service_delay
                opt_action = action

        # 选定最优 ES 之后，更新该 ES 在当前时间槽的“新增任务总工作量”
        # 这里不立刻减去可提供的服务能力，而是累积到本时间槽结束，再统一在 update_proc_queues 里结算。
        # Update the processing queue workload lengths at the selected ESs before processing next task
        self.proc_queue_bef[t][opt_action] = (self.proc_queue_bef[t][opt_action] +
                                              self.comp_density[n] * self.tasks_bit[t][b][n])

        return min_service_delay

    # ======================= 不区分“来源 BS”的 step_ ==========================
    # 与上面的 step 类似，但 tasks_bit 结构不同，只是二维：tasks_bit[t][n]。
    def step_(self, t, n):
        """
                在时间槽 t，为来自基站 b 的第 n 个任务选择最优 offloading 目标 ES。
                t: 当前时间槽下标
                b: 当前任务所属的基站/BS 编号
                n: 当前任务在该 BS 中的索引

                return：min_service_delay: 该任务在最优 offload 选择下的总服务时延
        """
        opt_action = []
        # 遍历所有 ES，选最小时延
        min_service_delay = 1000
        for i in range(self.n_BSs):
            action = i
            # ---------- 排队等待时延 ----------
            wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]
            # ---------- 传输 + 计算时延 ----------
            # 这里默认所有 offload 都要传输（模型中不区分“本地 ES”零传输延迟的情况）
            # Calculate the total transmission and computing delay of task n
            tran_comp_delays = (self.tasks_bit[t][n] / self.tran_rate_BSs[action] +
                                self.comp_density[n] * self.tasks_bit[t][n] / self.ES_capacities[action])
            # ---------- 总服务时延 ----------
            service_delay = tran_comp_delays + wait_delay  # Set the delay
            if service_delay < min_service_delay:
                min_service_delay = service_delay
                opt_action = action
        # 更新选中 ES 在当前时间槽的新增工作量
        # Update the processing queue workload lengths at the selected ESs before processing next task
        self.proc_queue_bef[t][opt_action] = (self.proc_queue_bef[t][opt_action] +
                                              self.comp_density[n] * self.tasks_bit[t][n])

        return min_service_delay

    # ======================= 时间推进：更新所有 ES 队列 ==========================
    # 在处理完时间槽 t 内的所有任务之后，推进到 t+1，并更新每个 ES 剩余的未完成工作量。
    # Update the processing queue length of all ESs at the beginning of next time slot.
    def update_proc_queues(self, t):
        """
                在时间槽 t 结束时，更新所有 ES 在时间槽 t+1 的队列长度。
                原理：
                - 当前时刻的未完成工作量：旧队列 + 本时间槽新分配的任务工作量 - 本时间槽可提供的服务量
                - 若结果为负，则说明可服务能力足够，队列清空，因此取 max(., 0)
                参数
                t: 当前时间槽下标（更新后会写入 t+1 的队列长度）
        """
        for b in range(self.n_BSs):
            # 本时间槽可提供的计算总量 = ES_capacities[b] * duration
            # 若 “旧队列 + 新到任务量 - 可提供量” > 0，则下一时间槽仍有积压；否则为 0
            self.proc_queue_len[t + 1][b] = np.max(
                [self.proc_queue_len[t][b] + self.proc_queue_bef[t][b] - self.ES_capacities[b] * self.duration, 0])
