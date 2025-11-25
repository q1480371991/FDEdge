from feedback_diffusion import * # 导入 Diffusion 类（FDN 扩散模型）
from helpers import SinusoidalPosEmb # 用于时间步的正弦位置编码
import numpy as np
import torch
import torch.nn as nn
import collections
import random
# 实现多层感知器（MLP）网络、Actor 网络、目标网络、Critic 网络以及网络训练过程

class ReplayBuffer:
    # 和普通 DRL 不同的一点是：
    # 这里每条经验里 多存了两个东西：
    # latent_action_probs：当前任务对应的“历史动作概率”向量（作为 FDN 输入）
    # next_latent_action_probs：下一状态对应的历史动作概率
    def __init__(self, capacity):
        # 采用双端队列，有上限长度
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, latent_action_probs, reward, next_state, next_latent_action_probs):
        # 往经验池里追加一条完整 transition
        self.buffer.append((state, action, latent_action_probs, reward, next_state, next_latent_action_probs))

    def sample(self, batch_size):
        # 随机采样 batch_size 条经验（off-policy）
        transitions = random.sample(self.buffer, batch_size)
        state, action, latent_action_probs, reward, next_state, next_latent_action_probs = zip(*transitions)
        # 返回时把 state / latent_action_probs / next_state 转成 np.array，方便后面转成 tensor
        return np.array(state), action, np.array(latent_action_probs), reward, np.array(next_state), np.array(
            next_latent_action_probs)
        # return np.array(state), action, latent_action_probs, reward, np.array(next_state), next_latent_action_probs

    def size(self):
        # 当前经验池里有多少条数据
        return len(self.buffer)

    def clear(self):
        # 清空经验池
        self.buffer.clear()


class MLP_PolicyNet(torch.nn.Module):
    # 这个 MLP 不是直接给 DRL 用来输出动作的，而是当作 Diffusion 模型里的噪声预测器：
    # 输入：当前 noisy 的动作向量 + 扩散时间步 + 当前状态
    # 输出：一个 action_dim 维向量，用来估计噪声 / 还原动作
    # Diffusion 把它包起来，变成“多步去噪后输出最终动作分布”。
    def __init__(self, state_dim, hidden_dim, action_dim, t_dim=16):
        super(MLP_PolicyNet, self).__init__()
        self.time_FCN = SinusoidalPosEmb(t_dim)# 用正弦位置编码把时间步 t 编成一个 t_dim 维向量
        #
        # self.time_FCN1 = SinusoidalPosEmb(t_dim)
        # self.time_FCN2 = nn.Linear(t_dim, t_dim * 2)
        # self.time_FCN3 = nn.Linear(t_dim * 2, t_dim)
        # self.time_FCN = nn.Sequential(
        #     SinusoidalPosEmb(t_dim),
        #     nn.Linear(t_dim, t_dim * 2),
        #     nn.ReLU,
        #     nn.Linear(t_dim * 2, t_dim)
        # )

        # 主网络：输入 = [当前动作向量 x_t, 时间编码 t_embed, 状态 s]
        self.fc1 = nn.Linear(state_dim + action_dim + t_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time_step, state):
        # x: 当前扩散链上的动作向量 x_t（比如 noisy 的概率）
        # time_step: 当前扩散步 t
        # state: 当前 RL 状态（任务大小+计算量+队列）
        t = self.time_FCN(time_step) # 把整数时间步编码成连续向量

        # t = self.time_FCN1(time_step)
        # t = F.relu(self.time_FCN2(t))
        # t = self.time_FCN3(t)

        # state 原本可能是 (batch_size, state_dim) 或更高维，这里 reshape 成 (batch_size, state_dim)
        state = state.reshape(state.size(0), -1)
        # 拼接 [当前动作 x, 时间编码 t, 状态 state]
        x = torch.cat([x, t, state], dim=1)

        # x = self.mid_layer(x)
        # return self.final_layer(x)

        # 三层全连接 + ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出对每个动作的“logits”，然后做 softmax 得到动作概率分布
        return F.softmax(self.fc3(x), dim=1)


class QValueNet(torch.nn.Module):
    """ Two hidden layers """
    """ 两层隐藏层的 Q 网络 """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 输入 x: state (batch_size, state_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出: 对每个动作的 Q 值 (batch_size, action_dim)
        return self.fc3(x)


class FDSAC:
    '''  处理离散动作的 SAC 算法（但 Actor 是 FDN 扩散模型） '''

    # Actor：Diffusion(FDN) + MLP_PolicyNet
    # Critic：SAC 标准的双 Q 网络 + 双 target Q
    # Alpha：采用自动调节的最大熵 RL（Soft Actor-Critic 的特点）
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha,
                 alpha_lr, target_entropy, tau, gamma, denoising_steps_, device):
        # Define the policy actor networks
        # ===== 1. Actor：Diffusion + MLP_PolicyNet =====
        self.actor = Diffusion(state_dim=state_dim,
                               action_dim=action_dim,
                               model=MLP_PolicyNet(state_dim, hidden_dim, action_dim),
                               beta_schedule='vp',# 使用 VP 型 β 调度
                               denoising_steps=denoising_steps_# 去噪步数 I
                               ).to(device)
        # ===== 2. Critic：两套 Q 网络（SAC 风格） =====
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # Define the critic1 (Q0) networks
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # Define the critic2 (Q1) networks
        # ===== 3. 对应的两个 target Q 网络 =====
        self.target_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # # Define the target1 networks
        self.target_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # Define the target2 networks

        # Define the initial parameters of the target 1 and 2 networks that are the same with Q0 and Q1 networks
        # 初始化 target 网络参数 = 当前 critic 的参数
        self.target_1.load_state_dict(self.critic_1.state_dict())
        self.target_2.load_state_dict(self.critic_2.state_dict())

        # Define the optimizers of three networks
        # ===== 4. 优化器 =====
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        # ===== 5. 熵温度 alpha 的 log 形式（SAC 经典做法） =====
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小，用来自动调节 alpha
        self.gamma = gamma# 折扣因子
        self.tau = tau# soft update 系数
        self.device = device
        self.history_loss = [] # 记录 critic_2 的 loss 演化情况（画图用）

    def take_action(self, state, latent_actions):#用当前策略选动作
        # 把 state / latent_actions 转成 batch_size = 1 的 tensor
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        latent_actions = torch.tensor(np.array([latent_actions]), dtype=torch.float).to(self.device)
        # 调用 Diffusion Actor：输入 (state, latent_action_probs)
        probs = self.actor(state, latent_actions) # 输出 shape: (1, action_dim) 的动作概率
        action_list = probs.tolist()# 转成 Python list 方便 argmax
        action = np.argmax(action_list[0]) # 选概率最大的动作（贪心）
        # 返回动作下标 + 当前动作概率分布（用于更新 env.latent_action_prob_space）
        return action, probs.detach().cpu().numpy()  # Get the action index and latent_ation_probs
        # action_dist = torch.distributions.Categorical(probs)  # Normalization
        # action = action_dist.sample()  # Get the index tensor of the maximal probability
        # return action.item(), probs.detach().cpu().numpy()  # Get the action index and latent_ation_probs

    # Calulcate the target Q values
    # Using the actor output, V output, and target V output with the input of reward and next state.
    # 计算 target Q 值，供 Critic 的 MSE loss 使用
    def calc_target_q(self, rewards, next_states, next_latent_actions):
        # 1) 用 Actor 在下一状态上算下一步动作分布 π(a'|s')
        next_probs = self.actor(next_states, next_latent_actions)
        next_log_probs = torch.log(next_probs + 1e-8)  # 1e-8 is used to ensure the definition sense of log function    避免 log(0)
        # 熵 H(π) = -∑ p log p
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)

        # 2) 用 target Q 网络评估下一状态各动作的 Q 值
        target1_q1_value = self.target_1(next_states)# Q1_target(s',·)
        target2_q2_value = self.target_2(next_states) # Q2_target(s',·)
        # 取两个 Q 的 min，做 Double Q，减少高估
        min_q_value = torch.sum(next_probs * torch.min(target1_q1_value, target2_q2_value), dim=1, keepdim=True)
        # 3) 计算 “soft” 的 state-value：V(s') = E_a[minQ] + α H(π)
        next_value = min_q_value + self.log_alpha.exp() * entropy
        # q_target = rewards + self.gamma * next_value * (1 - dones)
        # 4) 标准 TD 目标：Q_target = r + γ V(s')
        q_target = rewards + self.gamma * next_value
        return q_target

    # 标准 soft update 策略：让 target 网络慢慢跟随 online 网络，目标更平滑，训练更稳定。
    def soft_update(self, net, target_net):
        # target = (1-τ)*target + τ*online
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        #


        # ===== 1. 从 batch dict 中取出 N 条数据，并转成 tensor =====
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)  # 动作不再是float类型  离散动作索引
        latent_actions = torch.tensor(transition_dict['latent_action_probs'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        next_latent_actions = torch.tensor(transition_dict['next_latent_action_probs'], dtype=torch.float).to(
            self.device)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Updated the parameters of the two Q0 and Q1 networks
        # ===== 2. 更新 Critic（两套 Q 网络） =====
        # Critic 更新  用当前 Actor + target Q 算出 Q_target  让 critic_1(states, actions) 和 critic_2(states, actions) 去拟合这个 Q_target  用 MSE loss + 反向传播更新参数
        # 2.1 计算 TD 目标 Q_target
        target_q_values = self.calc_target_q(rewards, next_states, next_latent_actions)  # Calculate the target q values
        # 2.2 Critic_1 的 Q(s,a)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, target_q_values.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        # 2.3 Critic_2 同理
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, target_q_values.detach()))
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Updated the parameters of the actor network
        # ===== 3. 更新 Actor（FDN 扩散网络） =====
        # Actor 更新  用当前 Actor 重新输出 probs = π(a|s,x)   算策略的熵 H(π) 和 minQ 的期望 E_a[minQ]  反向传播更新 Actor（也就是 FDN 里的所有参数）
        probs = self.actor(states, latent_actions) # 当前策略 π(a|s,x)
        log_probs = torch.log(probs + 1e-8)  # 1e-8 is used to ensure the definition sense of log function
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  # Calculate the entropy, a positive value
        q1_value = self.critic_1(states) # Q1(s,·)
        q2_value = self.critic_2(states) # Q2(s,·)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)  # Calculate the expectation   E_a[minQ(s,a)]
        # Actor 的目标是最大化：α·entropy + E[minQ]，所以 loss 取负号
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the alpha value, i.e., log(alpha)
        # ===== 4. 更新 alpha（熵温度） =====
        # Alpha 更新  希望熵接近 target_entropy  通过优化 log_alpha 来自适应调节探索程度
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # print(alpha_loss.item())
        # print(entropy[-1])
        # print(self.log_alpha.exp())

        # By soft operation, update the parameters of the V and target V networks.
        # ===== 5. 软更新 target Q 网络 =====
        self.soft_update(self.critic_1, self.target_1)  # Update the V network parameters
        self.soft_update(self.critic_2, self.target_2)  # Update the target V network parameters
        # 记录一下 critic_2 的 loss，可以用于画损失曲线
        self.history_loss.append(critic_2_loss.item())  # store history cost
