import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 实现了反馈扩散处理
from helpers import (
    cosine_beta_schedule, # 余弦调度的 beta 序列
    linear_beta_schedule, # 线性调度的 beta 序列
    vp_beta_schedule,# VP 调度（variance preserving）的 beta 序列
    extract_into_tensor,# 从一维系数数组中，按时间步 t 抽取系数并 reshape
    Losses# 一个 dict，映射字符串到不同的损失类
)


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, beta_schedule='linear', denoising_steps=5,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True):
        """
                Diffusion：扩散模型封装，用于 FDN Actor。
                - state_dim: RL 状态维度
                - action_dim: 动作维度（动作空间大小）
                - model: 具体的去噪网络（这里是 MLP_PolicyNet）
                - beta_schedule: 选用的 beta 调度方式（线性/余弦/VP）
                - denoising_steps: 去噪步数 T
                - loss_type: 'l1' 或 'l2'，决定训练时的噪声预测损失
                - clip_denoised: 是否在 [-1, 1] 截断 x0（这里没启用）
                - predict_epsilon: True 表示 model 输出噪声 ε
        """
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = model# 去噪模型：输入 (x_t, t, state)，输出噪声或 x0
        # 根据 beta_schedule 选择不同的 beta_t 序列生成方式，长度为 denoising_steps
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(denoising_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(denoising_steps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(denoising_steps)
        # α_t = 1 - β_t
        alphas = 1. - betas
        # ᾱ_t = ∏_{i=1}^t α_i
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        # ᾱ_{t-1}，第一个元素对应 t=0 时设为 1
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(denoising_steps) # 扩散总步数 T
        self.clip_denoised = clip_denoised# 是否裁剪 x0
        self.predict_epsilon = predict_epsilon# 是否预测噪声 ε
        # register_buffer：这些参数是常量，不参与训练，但会跟着模型一起存
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # ======== 为正向/反向扩散预计算的各种系数 ========
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod)) # sqrt(ᾱ_t)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))# sqrt(1 - ᾱ_t)
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))# log(1 - ᾱ_t)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))# sqrt(1 / ᾱ_t)
        # 会在从 x_t 和 ε 反推 x0 的公式中用到
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1)) # sqrt(1 / ᾱ_t - 1)

        # ======== 后验 q(x_{t-1} | x_t, x_0) 所需的系数 ========
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain

        # log(posterior_variance)，但要做一个下界裁剪，避免 log(0)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        # posterior_mean_coef1, posterior_mean_coef2
        # 后验均值 μ_t = coef1 * x0 + coef2 * x_t
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # 根据 loss_type 选择 L1 或 L2 损失（helpers.Losses 是一个 dict）
        self.loss_fn = Losses[loss_type]()

    # 下面是原本更通用的版本（根据 predict_epsilon 决定模型输出的含义），目前被注释掉
    # def predict_start_from_noise(self, x_t, t, noise):
    #     """
    #         if self.predict_epsilon, model output is (scaled) noise;
    #         otherwise, model predicts x0 directly
    #     """
    #     if self.predict_epsilon:
    #         return (
    #                 extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
    #                 extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    #         )
    #     else:
    #         return noise

    def predict_start_from_noise(self, x_t, t, noise):
        """
                从 x_t 和 模型输出的 noise（通常是噪声 ε）反推 x0：
                x0 ≈ sqrt(1/ᾱ_t) * x_t - sqrt(1/ᾱ_t - 1) * noise
                这里没有根据 predict_epsilon 分支，等价于默认 always 预测噪声。
        """
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_t_v(self, x_t, t, noise):
        """
                另一个版本的 x0 预测函数，多了 predict_epsilon 的分支判断。
                实现上和上面的原始注释掉版本类似，只是单独写成一个函数。
        """
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            # 模型输出的是噪声 ε：用标准 DDPM 公式反推 x0
            return (
                    extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            # 如果模型直接预测的是 x0，就直接返回
            return noise


    def q_posterior(self, x_start, x_t, t):
        """
                计算后验分布 q(x_{t-1} | x_t, x_0) 的:
                  - 均值 posterior_mean
                  - 方差 posterior_variance
                  - log 方差 posterior_log_variance_clipped
                使用 DDPM 的 closed-form 公式 μ_t = c1 * x0 + c2 * x_t
        """
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        """
                给定当前 x_t、时间步 t、状态 s，计算模型的 p(x_{t-1} | x_t) 的均值和方差：
                1. 先用 model(x_t, t, s) 预测噪声 ε_hat
                2. 用 predict_start_from_noise 从 (x_t, ε_hat) 估计 x0_hat
                3. 代入 q_posterior 得到对 x_{t-1} 的均值和方差
        """
        # model(x, t, s): MLP_PolicyNet，输出预测噪声
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))
        if self.clip_denoised:
            # 如果需要，可以把估计的 x0 裁剪到 [-1, 1]
            x_recon.clamp_(-1, 1)
        # 用估计的 x0_hat 和 x_t 计算后验的均值和方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        """
                从 p(x_{t-1} | x_t) 采样一个 x_{t-1}：
                  x_{t-1} = μ_t + σ_t * z, z ~ N(0, I)
                其中 μ_t, σ_t^2 来自 p_mean_variance。
        """
        # b 是 batch_size，device 是当前张量所在设备
        b, *_, device = *x.shape, x.device
        # 模型给出的均值和 log 方差
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        # 标准高斯噪声
        probs = torch.randn_like(x)
        # probs = x   # 这行被注释掉了，说明作者尝试过不加噪声

        # no noise when t == 0
        # nonzero_mask 用来在 t == 0 时不再加噪声
        # t shape: (batch_size,), reshape 成 (batch_size, 1, 1, ...)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # 当 t>0: 返回 μ + exp(0.5*log_var)*z
        # 当 t==0: nonzero_mask=0，只返回 model_mean，即最干净的一步
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * probs

    # 下面是一个不用 latent_action_probs 的版本采样循环，被注释掉了
    # def p_sample_loop(self, state, shape):
    #     device = self.betas.device
    #     batch_size = shape[0]
    #     x = torch.randn(shape, device=device)
    #     for i in reversed(range(0, self.n_timesteps)):
    #         timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
    #         x = self.p_sample(x, timesteps, state)
    #     return x

    def p_sample_loop(self, state, latent_action_probs):
        """
                反向扩散主循环：
                给定 state 和 latent_action_probs 的形状，从纯噪声开始，
                连续做 self.n_timesteps 步 p_sample(x_t -> x_{t-1})，得到最终的 x_0。
        """
        device = self.betas.device
        batch_size = state.shape[0]
        # x = torch.randn(batch_size, self.action_dim, device=device)
        # x = latent_action_probs

        # 初始化 x_T：
        #   这里只是用 latent_action_probs 的形状来生成随机噪声，
        #   并没有直接把 latent_action_probs 当作起点使用。
        #   也就是说，当前实现是“从纯噪声生成动作向量”。
        # x = torch.randn(batch_size, self.action_dim, device=device)
        # x = latent_action_probs
        x = torch.randn_like(latent_action_probs)
        # 从 t = T-1, ..., 0 依次做反向采样
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
        return x

    # 下面是不用 latent_action_probs 的接口，被注释掉了
    # def sample(self, state, *args, **kwargs):
    #     batch_size = state.shape[0]
    #     shape = (batch_size, self.action_dim)
    #     action = self.p_sample_loop(state, shape, *args, **kwargs)
    #     return F.softmax(action, dim=-1)

    def sample(self, state, latent_action_probs, *args, **kwargs):
        """
               对外采样接口：
               给定 state 和 latent_action_probs（仅用来提供初始形状），
               返回 softmax 之后的动作概率分布。
        """
        action = self.p_sample_loop(state, latent_action_probs, *args, **kwargs)
        return F.softmax(action, dim=-1)

    def q_sample(self, x_start, t, noise=None):
        """
                正向加噪过程 q(x_t | x_0)：
                  x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
                训练时用它把 x_0 加噪成 x_t。
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        """
                扩散训练的单步 loss：
                - 随机采样噪声 noise
                - 得到 x_noisy = q_sample(x_start, t, noise)
                - 模型输入 (x_noisy, t, state)，预测重建 x_recon（通常是噪声 ε_hat）
                - 根据 predict_epsilon 决定是用 (ε_hat, ε) 还是 (x_hat, x_start) 计算损失
        """
        noise = torch.randn_like(x_start)
        # 正向扩散得到 x_t（带噪）
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 用 model 预测噪声或 x0
        x_recon = self.model(x_noisy, t, state)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            # 模型预测的是噪声：loss(x_recon, noise)
            return self.loss_fn(x_recon, noise, weights)
        else:
            # 模型直接预测的是 x0：loss(x_recon, x_start)
            return self.loss_fn(x_recon, x_start, weights)

    def loss(self, x, state, weights=1.0):
        """
                对一个 batch 的 x（真实“干净信号”）计算平均扩散损失：
                - 随机为每个样本抽一个时间步 t
                - 调用 p_losses(x, state, t)
                这里 x 实际上就是“目标动作向量”（比如 one-hot 或概率）。
        """
        batch_size = len(x)
        # 对每个样本随机采样一个 t ∈ [0, n_timesteps)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, latent_action_probs, *args, **kwargs):
        """
                让 Diffusion 模块像普通 nn.Module 一样可调用：
                  probs = self.actor(state, latent_action_probs)
                实际内部调用的是 self.sample(...)
        """
        return self.sample(state, latent_action_probs, *args, **kwargs)
