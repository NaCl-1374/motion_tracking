# Motion Tracking - 代码架构与调用流程文档

本文档全面分析了运动跟踪代码库，包括架构、调用流程、训练过程和组件关系。

## 目录
1. [概述](#概述)
2. [项目结构](#项目结构)
3. [训练流水线](#训练流水线)
4. [核心组件](#核心组件)
5. [调用流程](#调用流程)
6. [数据流](#数据流)
7. [关键模块](#关键模块)

---

## 概述

本项目实现了一个**教师-学生知识蒸馏框架**，用于人形机器人的鲁棒全身运动跟踪。系统在训练期间使用特权信息（教师），通过适应模块学习在没有特权信息的情况下运行（学生）。

### 主要特点
- 三阶段训练流水线（训练、适应、微调）
- 基于 MuJoCo 的物理仿真
- PPO（近端策略优化）算法
- 用于 sim-to-real 迁移的域随机化
- 支持对称性的观测和动作

---

## 项目结构

```
motion_tracking/
├── active_adaptation/           # 核心训练框架
│   ├── envs/                   # 环境实现
│   │   ├── base.py            # 基础环境类
│   │   ├── locomotion.py      # 主要 SimpleEnv 类
│   │   ├── scene.py           # 场景设置（机器人、地形）
│   │   └── mdp/               # MDP 组件
│   │       ├── action.py      # 动作空间
│   │       ├── observations.py # 观测组
│   │       ├── rewards/       # 奖励函数
│   │       ├── terminations.py # 终止条件
│   │       ├── randomizations.py # 域随机化
│   │       └── commands/      # 运动跟踪命令
│   ├── learning/              # 强化学习算法
│   │   ├── ppo/              # PPO 实现
│   │   │   ├── ppo.py        # PPOPolicy 类
│   │   │   └── networks.py   # Actor-Critic 网络
│   │   └── modules/          # 神经网络模块
│   └── utils/                # 实用函数
│       ├── motion.py         # 运动数据加载
│       ├── symmetry.py       # 对称变换
│       ├── torchrl.py        # 数据收集
│       └── wandb.py          # WandB 集成
├── scripts/
│   ├── train.py              # 主训练脚本
│   ├── eval.py               # 评估脚本
│   └── utils/
│       └── helpers.py        # 辅助函数
├── cfg/                      # 配置文件
│   ├── train.yaml           # 主训练配置
│   ├── exp/                 # 实验配置
│   │   ├── train.yaml       # 阶段 1：训练
│   │   ├── adapt.yaml       # 阶段 2：适应
│   │   └── finetune.yaml    # 阶段 3：微调
│   └── task/                # 任务特定配置
│       └── G1/              # G1 人形机器人
│           ├── G1.yaml      # 基础 G1 配置
│           └── G1_tracking.yaml  # 运动跟踪任务
└── train.sh                 # 流水线编排脚本
```

---

## 训练流水线

该项目使用**三阶段训练流水线**来实现教师-学生知识蒸馏：

### 阶段 1：训练（TRAIN）（80亿帧）
**目的**：使用特权信息训练教师策略

- **配置**：`+exp=train` + `algo=ppo_train`
- **时长**：约 80 亿帧
- **激活的网络**：
  - `encoder_priv`：编码特权观测
  - `actor_teacher`：策略网络（使用特权信息）
  - `critic`：价值函数
  - `adapt_module`：已训练但不用于推演
- **推演策略**：`[encoder_priv, actor_teacher]`
- **优化**：更新教师 actor 和 critic

### 阶段 2：适应（ADAPT）（10亿帧）
**目的**：训练适应模块以预测特权特征

- **配置**：`+exp=adapt` + `algo=ppo_adapt`
- **时长**：约 10 亿帧
- **加载**：阶段 1 的检查点
- **激活的网络**：
  - `adapt_module`：学习从策略观测预测特权特征
  - `actor_student`：蒸馏策略（从教师学习）
  - `actor_teacher`：冻结（用于蒸馏目标）
  - `critic`：继续训练
- **推演策略**：`[adapt_module, actor_student]`
- **优化**：更新学生 actor 和适应模块

### 阶段 3：微调（FINETUNE）（40亿帧）
**目的**：微调学生策略以获得最终性能

- **配置**：`+exp=finetune` + `algo=ppo_finetune`
- **时长**：约 40 亿帧
- **加载**：阶段 2 的检查点
- **激活的网络**：
  - `adapt_module`：冻结（提供估计的特权特征）
  - `actor_student`：继续微调
  - `critic`：继续训练
- **推演策略**：`[adapt_module, actor_student]`
- **优化**：更新学生 actor 和 critic

### 流水线编排

`train.sh` 脚本编排所有三个阶段：

```bash
# 阶段 1：训练教师
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=train \
  wandb.id=${ID_TRAIN}

# 阶段 2：适应（蒸馏）
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=adapt \
  checkpoint_path=run:${PROJECT}/${ID_TRAIN} \
  wandb.id=${ID_ADAPT}

# 阶段 3：微调学生
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=finetune \
  checkpoint_path=run:${PROJECT}/${ID_ADAPT} \
  wandb.id=${ID_FINETUNE}
```

---

## 核心组件

### 1. SimpleEnv（环境）

**文件**：`active_adaptation/envs/locomotion.py`

处理仿真、观测、奖励和回合管理的主要环境类。

```python
class SimpleEnv(_Env):
    def __init__(self, cfg, ...):
        self.setup_scene(cfg)     # 初始化 MuJoCo 场景
        self.init_mdp(cfg)        # 设置 MDP 组件
        
    def setup_scene(self, cfg):
        # 创建机器人实体（G1 人形机器人）
        # 创建地形
        # 设置接触传感器
        
    def init_mdp(self, cfg):
        # 初始化动作空间
        # 初始化观测组（policy、priv、critic_priv）
        # 初始化奖励组
        # 初始化终止条件
        # 初始化域随机化
```

**主要职责**：
- 通过 MuJoCo 后端进行物理仿真
- 状态管理和重置逻辑
- 观测计算（策略、特权、评论家）
- 奖励计算（跟踪、运动等）
- 域随机化
- 回合终止检测

**MDP 组件**：
- **动作**：关节位置目标（PD 控制）
- **观测**： 
  - `policy`：可观测状态（本体感受、传感器）
  - `priv`：特权信息（真实动力学、摩擦等）
  - `critic_priv`：评论家的额外特权信息
- **奖励**：跟踪奖励、运动奖励、正则化
- **终止**：跌倒检测、时间限制

### 2. PPOPolicy（学习算法）

**文件**：`active_adaptation/learning/ppo/ppo.py`

实现带有教师-学生蒸馏的 PPO 的强化学习算法。

```python
class PPOPolicy:
    def __init__(self, cfg, ...):
        # 网络
        self.encoder_priv = ...     # 编码特权观测
        self.actor_teacher = ...    # 教师策略
        self.actor_student = ...    # 学生策略
        self.critic = ...           # 价值函数
        self.adapt_module = ...     # 适应估计器
        
        # 优化器（取决于阶段）
        self.opt_teacher = ...      # 用于教师
        self.opt_student = ...      # 用于学生
        self.opt_critic = ...       # 用于 critic
        self.opt_estimator = ...    # 用于 adapt_module
        
    def get_rollout_policy(self, mode):
        # 返回推演的网络链
        if mode == "train":
            return [encoder_priv, actor_teacher]
        else:  # adapt 或 finetune
            return [adapt_module, actor_student]
            
    def train_op(self, data, vecnorm):
        # 阶段特定的训练逻辑
        if self.phase == "train":
            return self._update_teacher(data)
        elif self.phase == "adapt":
            return self._update_student(data)
        elif self.phase == "finetune":
            return self._update2(data)
```

**关键方法**：
- `get_rollout_policy(mode)`：返回推理网络链
- `train_op(data, vecnorm)`：执行一个训练步骤
- `_ppo_update()`：核心 PPO 算法（优势估计、策略损失、价值损失）
- `train_estimator()`：训练适应模块以预测特权特征
- `step_schedule()`：更新学习率和熵系数调度

### 3. TensorDict 数据结构

**全局使用**：所有数据都以 TensorDict 格式存储以提高效率

```python
TensorDict {
    "policy": obs,              # 策略观测 (N, T, obs_dim)
    "priv": priv_obs,          # 特权观测 (N, T, priv_dim)
    "critic_priv": critic_obs, # Critic 特权观测 (N, T, critic_dim)
    "action": actions,         # 采取的动作 (N, T, action_dim)
    "state_value": values,     # V(s) 估计 (N, T, 1)
    "log_prob": log_probs,    # log π(a|s) (N, T, 1)
    "next": {
        "policy": next_obs,
        "priv": next_priv_obs,
        "state_value": next_values,
        "done": done_flags,
        "stats": {
            "rewards/tracking": ...,
            "rewards/locomotion": ...,
            # ... 其他奖励组件
        },
        "_weight": importance_weights
    }
}
```

其中：
- `N` = 并行环境数量
- `T` = 每次推演的时间步数

---

## 调用流程

### 主训练循环

**文件**：`scripts/train.py`

```
main(cfg)
  │
  ├─► make_env_policy(cfg)
  │    ├─► SimpleEnv(cfg.task)
  │    │    ├─► setup_scene()      # MuJoCo 初始化
  │    │    └─► init_mdp()         # MDP 组件
  │    │
  │    ├─► VecNorm(obs_keys)       # 观测归一化
  │    │
  │    └─► PPOPolicy(cfg.algo)
  │         ├─► 创建网络（encoder、actor、critic、adapt_module）
  │         └─► 创建优化器
  │
  ├─► env.reset() → carry（初始状态）
  │
  └─► for i in range(total_iters):
       │
       ├─► ROLLOUT（T=train_every 步）
       │    └─► for t in range(T):
       │         ├─► carry = rollout_policy(carry)
       │         │    ├─► encoder_priv(carry) 或 adapt_module(carry)
       │         │    └─► actor(encoded_features) → actions
       │         │
       │         ├─► td, carry = env.step_and_maybe_reset(carry)
       │         │    ├─► mujoco_step(actions)
       │         │    ├─► compute_observations()
       │         │    ├─► compute_rewards()
       │         │    ├─► check_terminations()
       │         │    └─► maybe_reset_envs()
       │         │
       │         ├─► critic(td) → state_value
       │         ├─► critic(td["next"]) → next_state_value
       │         └─► data_buf.write_step(t, td)
       │
       ├─► 训练步骤
       │    ├─► policy.step_schedule(i)  # 更新学习率、熵
       │    ├─► env.step_schedule(i)     # 更新随机化
       │    │
       │    └─► train_carry = policy.train_op(data, vecnorm)
       │         │
       │         ├─► 阶段：TRAIN
       │         │    └─► _update_teacher(data)
       │         │         ├─► 计算优势（GAE）
       │         │         ├─► PPO 更新（教师 + critic）
       │         │         └─► 训练估计器（并行）
       │         │
       │         ├─► 阶段：ADAPT
       │         │    └─► _update_student(data)
       │         │         ├─► 计算优势（GAE）
       │         │         ├─► PPO 更新（学生 + critic）
       │         │         └─► 从教师蒸馏损失
       │         │
       │         └─► 阶段：FINETUNE
       │              └─► _update2(data)
       │                   ├─► 计算优势（GAE）
       │                   └─► PPO 更新（学生 + critic）
       │
       ├─► 日志记录
       │    ├─► episode_stats.add(data)
       │    ├─► info = {metrics, fps, frames, ...}
       │    └─► wandb.log(info, step=i)
       │
       └─► 检查点（定期）
            ├─► 保存 policy.state_dict()
            ├─► 保存 env.state_dict()
            ├─► 保存 vecnorm.state_dict()
            └─► 上传到 W&B
```

### 环境步进流程

**文件**：`active_adaptation/envs/locomotion.py`

```
step_and_maybe_reset(carry)
  │
  ├─► 从 carry 提取动作
  │
  ├─► mujoco.step(actions)           # 物理仿真
  │
  ├─► compute_observations()
  │    ├─► obs_group["policy"].compute()      # 策略观测
  │    ├─► obs_group["priv"].compute()        # 特权观测
  │    └─► obs_group["critic_priv"].compute() # Critic 观测
  │
  ├─► compute_rewards()
  │    └─► reward_group.compute()
  │         ├─► tracking_rewards()
  │         ├─► locomotion_rewards()
  │         └─► regularization_rewards()
  │
  ├─► check_terminations()
  │    └─► termination_manager.compute()
  │         ├─► fall_detection()
  │         └─► time_limit()
  │
  ├─► 使用（obs、reward、done）创建 TensorDict
  │
  └─► maybe_reset()
       ├─► 如果有环境完成：
       │    ├─► randomize_env_parameters()
       │    ├─► reset_poses()
       │    └─► sample_new_commands()
       │
       └─► 返回（td、new_carry）
```

### PPO 训练步骤流程

**文件**：`active_adaptation/learning/ppo/ppo.py`

```
train_op(data, vecnorm)
  │
  ├─► vecnorm.update(data)           # 更新归一化统计
  │
  ├─► 重塑数据：(N×T) → (N*T,)      # 展平时间维度
  │
  ├─► 计算 GAE 优势
  │    ├─► δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
  │    └─► A_t = Σ (γλ)^k δ_{t+k}
  │
  ├─► 归一化优势
  │
  └─► for epoch in range(ppo_epochs):
       │
       └─► for minibatch in minibatches:
            │
            ├─► 重新评估策略：
            │    ├─► π_new(a|s) 通过 actor 网络
            │    └─► V_new(s) 通过 critic 网络
            │
            ├─► 计算损失：
            │    ├─► ratio = π_new(a|s) / π_old(a|s)
            │    ├─► L_policy = -min(ratio·A, clip(ratio)·A)
            │    ├─► L_value = (V_new - V_target)²
            │    └─► L_entropy = -H(π)
            │
            ├─► 总损失 = L_policy + c1·L_value - c2·L_entropy
            │
            ├─► 反向传播
            │    ├─► loss.backward()
            │    └─► optimizer.step()
            │
            └─► 记录指标（policy_loss、value_loss、entropy、KL等）
```

---

## 数据流

### 单次训练迭代 (i)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 推演阶段（T 步，推理模式）                                      │
└─────────────────────────────────────────────────────────────────┘
   │
   ├─► 初始化：carry = env.reset()
   │   └─► 包含：机器人状态、目标运动等
   │
   └─► For t = 0 to T-1:
        │
        ├─► carry = rollout_policy(carry)
        │    输入：carry["policy"]、carry["priv"]
        │    输出：carry["action"]
        │
        ├─► td, carry = env.step_and_maybe_reset(carry)
        │    输入：carry["action"]
        │    输出：td（观测、奖励、完成）
        │            carry（更新状态）
        │
        ├─► td["state_value"] = critic(td)
        ├─► td["next", "state_value"] = critic(td["next"])
        │
        └─► data_buf[t] = td
             └─► 存储：(obs、action、reward、value、log_prob、done)

┌─────────────────────────────────────────────────────────────────┐
│ 2. 训练阶段（数据处理）                                            │
└─────────────────────────────────────────────────────────────────┘
   │
   ├─► data = data_buf.td                    # 形状：(N, T, ...)
   │
   ├─► 计算 GAE 优势
   │    输入：rewards、values、next_values、dones
   │    输出：advantages、returns
   │
   ├─► 展平：(N, T, ...) → (N*T, ...)
   │
   └─► 分割为小批次（用于 PPO epochs）

┌─────────────────────────────────────────────────────────────────┐
│ 3. 优化阶段（PPO 更新）                                            │
└─────────────────────────────────────────────────────────────────┘
   │
   └─► For each PPO epoch:
        └─► For each minibatch:
             ├─► 前向传播（actor + critic）
             ├─► 计算损失
             ├─► 反向传播
             └─► 更新参数

┌─────────────────────────────────────────────────────────────────┐
│ 4. 日志记录阶段                                                    │
└─────────────────────────────────────────────────────────────────┘
   │
   ├─► 收集回合统计
   ├─► 计算 FPS
   └─► wandb.log(metrics, step=i)
```

### 数据张量形状

整个流程中，数据保持特定形状：

| 组件 | 形状 | 描述 |
|-----|------|-----|
| 推演 carry | `(N, *)` | N 个并行环境的状态 |
| 单步 TD | `(N, *)` | 一步的观测/奖励 |
| 数据缓冲区 | `(N, T, *)` | 完整推演缓冲区 |
| 展平数据 | `(N*T, *)` | 为 PPO 训练重塑 |
| 小批次 | `(B, *)` | SGD 更新的子集 |

其中：
- `N` = num_envs（例如 4096）
- `T` = train_every（例如训练阶段 32，适应阶段 16）
- `B` = minibatch_size（例如 N*T / num_minibatches）

---

## 关键模块

### 1. MDP 组件（`active_adaptation/envs/mdp/`）

#### 观测（`observations.py`）

定义支持对称性的观测组：

```python
class ObsGroup:
    """具有对称变换的观测组"""
    
    def __init__(self, *obs_terms):
        self.obs_terms = obs_terms
        
    def compute(self, state) -> torch.Tensor:
        # 连接所有观测项
        obs = [term.compute(state) for term in self.obs_terms]
        return torch.cat(obs, dim=-1)
        
    def symmetry_transform(self, obs, symmetry_mask):
        # 应用左右对称变换
        ...
```

**常见观测项**：
- `BaseLinVel`：基座线速度
- `BaseAngVel`：基座角速度
- `ProjectedGravity`：基座坐标系中的重力向量
- `Commands`：目标运动跟踪命令
- `DofPos`：关节位置
- `DofVel`：关节速度
- `Actions`：先前动作（历史）

#### 奖励（`rewards/`）

训练的奖励函数：

```python
class RewardGroup:
    """带权重的奖励项组"""
    
    def __init__(self, *reward_terms):
        self.reward_terms = reward_terms
        
    def compute(self, state) -> Dict[str, torch.Tensor]:
        rewards = {}
        for term in self.reward_terms:
            rewards[term.name] = term.weight * term.compute(state)
        return rewards
```

**常见奖励项**：
- `MotionTracking`：偏离目标运动的惩罚
- `BaseHeightTracking`：基座高度误差惩罚
- `LinearVelocityTracking`：速度跟踪误差惩罚
- `ActionRate`：大动作变化的惩罚
- `Torques`：大关节力矩的惩罚
- `DofAcceleration`：高关节加速度的惩罚
- `FeetAirTime`：正确脚接触时机的奖励
- `StumbleReward`：脚绊倒的惩罚

#### 域随机化（`randomizations.py`）

随机化仿真参数以获得鲁棒策略：

```python
class RandomizationManager:
    def randomize_on_reset(self, env_ids):
        # 为重置环境随机化
        self.randomize_robot_mass(env_ids)
        self.randomize_robot_com(env_ids)
        self.randomize_joint_friction(env_ids)
        self.randomize_joint_damping(env_ids)
        self.randomize_motor_strength(env_ids)
        self.randomize_base_mass(env_ids)
        # ... 等等
        
    def randomize_on_step(self):
        # 每步随机化（噪声、延迟等）
        self.add_noise_to_observations()
        self.add_action_delay()
```

### 2. 网络（`active_adaptation/learning/modules/`）

#### Actor 网络

```python
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 256]):
        # MLP 网络
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ELU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ELU(),
            nn.Linear(hidden_dims[2], action_dim)
        )
        
    def forward(self, obs):
        return self.layers(obs)
```

#### Critic 网络

```python
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims=[256, 256, 256]):
        # 类似的 MLP 用于价值估计
        self.layers = nn.Sequential(...)
        
    def forward(self, obs):
        return self.layers(obs)  # 返回 V(s)
```

#### 适应模块

```python
class AdaptModule(nn.Module):
    """从策略观测估计特权特征"""
    
    def __init__(self, policy_obs_dim, priv_feature_dim):
        self.estimator = nn.Sequential(
            nn.Linear(policy_obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, priv_feature_dim)
        )
        
    def forward(self, policy_obs):
        return self.estimator(policy_obs)
```

### 3. 实用工具

#### 运动加载（`utils/motion.py`）

```python
class MotionDataset:
    """加载和管理运动捕捉数据"""
    
    def __init__(self, dataset_root):
        self.motions = self.load_all_motions(dataset_root)
        
    def sample_motion(self, env_ids, frame_ids):
        # 采样运动帧用于跟踪
        return target_poses, target_velocities
```

#### 对称性（`utils/symmetry.py`）

```python
class SymmetryTransform:
    """处理人形机器人的左右对称"""
    
    def transform_obs(self, obs, left_joint_ids, right_joint_ids):
        # 交换左/右关节
        # 取反 Y 轴分量
        ...
        
    def transform_action(self, action, ...):
        # 对动作类似处理
        ...
```

---

## 关键配置参数

### 训练超参数

**PPO 算法**（定义在 `active_adaptation/learning/ppo/ppo.py` 中）：
```python
# ppo_train 配置（通过 Hydra ConfigStore 注册）
PPOConfig(
    phase="train", 
    vecnorm="train",
    ppo_epochs=5,                    # 每批次的 epoch
    num_minibatches=8,               # 每 epoch 的小批次数
    train_every=32,                  # 推演长度（训练阶段）
    clip_param=0.2,                  # PPO 裁剪参数
    lr=5e-4,                         # 学习率（训练阶段）
    entropy_coef_start=0.01,         # 初始熵正则化
    entropy_coef_end=0.0025,         # 最终熵正则化
    gamma=0.99,                      # 折扣因子（隐式）
    gae_lambda=0.95,                 # GAE lambda（隐式）
    reg_lambda=0.2,                  # 蒸馏权重
)
```

**适应阶段**（注册为 `ppo_adapt`）：
```python
PPOConfig(
    phase="adapt",
    vecnorm="eval",
    train_every=16,                  # 更短的推演
    lr=5e-4,                         # 相同学习率
    reg_lambda=0.2,                  # 蒸馏权重
)
```

**微调阶段**（注册为 `ppo_finetune`）：
```python
PPOConfig(
    phase="finetune",
    vecnorm="eval",
    lr=1e-4,                         # 更低的学习率
    train_every=16,
    entropy_coef_start=0.0025,       # 更低的熵
    entropy_coef_end=0.0005,
)
```

### 环境参数

**G1 机器人任务**（`cfg/task/G1/G1_tracking.yaml`）：
```yaml
num_envs: 4096                   # 并行环境
max_episode_length: 1000         # 每回合最大步数
dt: 0.02                         # 仿真时间步长（50 Hz）
decimation: 2                    # 动作重复（控制频率 25 Hz）

# 运动跟踪
motion_dataset: "amass_all"      # 使用的数据集
track_base_height: true
track_linear_velocity: true

# 域随机化
randomize_friction: true
randomize_mass: true
randomize_com: true
# ... 等等
```

---

## 总结

运动跟踪代码库实现了一个复杂的教师-学生框架：

1. **教师训练**：使用特权信息（真实动力学、摩擦等）学习鲁棒策略

2. **适应**：训练估计器从传感器观测预测特权特征，使无需特权信息即可部署

3. **学生微调**：打磨学生策略以获得最终性能

架构模块化且可扩展：
- **环境**：带 MDP 组件的物理仿真
- **学习**：带蒸馏的 PPO 算法
- **数据**：基于 TensorDict 的高效数据流
- **配置**：基于 Hydra 的配置系统

这种设计使复杂的人形运动跟踪任务能够进行鲁棒的 sim-to-real 迁移。
