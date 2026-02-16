# Motion Tracking - Code Architecture and Call Flow Documentation

This document provides a comprehensive analysis of the motion tracking codebase, including the architecture, call flow, training process, and component relationships.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Training Pipeline](#training-pipeline)
4. [Core Components](#core-components)
5. [Call Flow](#call-flow)
6. [Data Flow](#data-flow)
7. [Key Modules](#key-modules)

---

## Overview

This project implements a **teacher-student knowledge distillation framework** for robust whole-body motion tracking on humanoid robots. The system uses privileged information during training (teacher) and learns to operate without it during deployment (student) through an adaptation module.

### Key Features
- Three-stage training pipeline (train, adapt, finetune)
- MuJoCo-based physics simulation
- PPO (Proximal Policy Optimization) algorithm
- Domain randomization for sim-to-real transfer
- Symmetry-aware observations and actions

---

## Project Structure

```
motion_tracking/
├── active_adaptation/           # Core training framework
│   ├── envs/                   # Environment implementations
│   │   ├── base.py            # Base environment class
│   │   ├── locomotion.py      # Main SimpleEnv class
│   │   ├── scene.py           # Scene setup (robot, terrain)
│   │   └── mdp/               # MDP components
│   │       ├── action.py      # Action space
│   │       ├── observations.py # Observation groups
│   │       ├── rewards/       # Reward functions
│   │       ├── terminations.py # Termination conditions
│   │       ├── randomizations.py # Domain randomization
│   │       └── commands/      # Motion tracking commands
│   ├── learning/              # RL algorithms
│   │   ├── ppo/              # PPO implementation
│   │   │   ├── ppo.py        # PPOPolicy class
│   │   │   └── networks.py   # Actor-Critic networks
│   │   └── modules/          # Neural network modules
│   └── utils/                # Utility functions
│       ├── motion.py         # Motion data loading
│       ├── symmetry.py       # Symmetry transforms
│       ├── torchrl.py        # Data collection
│       └── wandb.py          # WandB integration
├── scripts/
│   ├── train.py              # Main training script
│   ├── eval.py               # Evaluation script
│   └── utils/
│       └── helpers.py        # Helper functions
├── cfg/                      # Configuration files
│   ├── train.yaml           # Main training config
│   ├── exp/                 # Experiment configs
│   │   ├── train.yaml       # Stage 1: Train
│   │   ├── adapt.yaml       # Stage 2: Adapt
│   │   └── finetune.yaml    # Stage 3: Finetune
│   └── task/                # Task-specific configs
│       └── G1/              # G1 humanoid robot
│           ├── G1.yaml      # Base G1 config
│           └── G1_tracking.yaml  # Motion tracking task
└── train.sh                 # Pipeline orchestration script
```

---

## Training Pipeline

The project uses a **three-stage training pipeline** to enable teacher-student knowledge distillation:

### Stage 1: TRAIN (8B frames)
**Purpose**: Train the teacher policy with privileged information

- **Config**: `+exp=train` + `algo=ppo_train`
- **Duration**: ~8 billion frames
- **Networks Active**:
  - `encoder_priv`: Encodes privileged observations
  - `actor_teacher`: Policy network (uses privileged info)
  - `critic`: Value function
  - `adapt_module`: Trained but not used for rollout
- **Rollout Policy**: `[encoder_priv, actor_teacher]`
- **Optimization**: Updates teacher actor and critic

### Stage 2: ADAPT (1B frames)
**Purpose**: Train the adaptation module to predict privileged features

- **Config**: `+exp=adapt` + `algo=ppo_adapt`
- **Duration**: ~1 billion frames
- **Loads**: Checkpoint from Stage 1
- **Networks Active**:
  - `adapt_module`: Learns to predict privileged features from policy obs
  - `actor_student`: Used for rollout (frozen during this phase)
  - `actor_teacher`: Frozen (not used)
  - `critic`: Frozen (not trained in this phase)
- **Rollout Policy**: `[adapt_module, actor_student]`
- **Optimization**: **Only trains adaptation module** via MSE loss between predicted and true privileged features
  - Does NOT perform PPO updates in this phase
  - Student actor and critic remain frozen
  - Trains for 2 mini-epochs per rollout

### Stage 3: FINETUNE (4B frames)
**Purpose**: Finetune the student policy for final performance

- **Config**: `+exp=finetune` + `algo=ppo_finetune`
- **Duration**: ~4 billion frames
- **Loads**: Checkpoint from Stage 2
- **Networks Active**:
  - `adapt_module`: Frozen (provides estimated privileged features)
  - `actor_student`: Trained via PPO
  - `critic`: Trained via PPO
- **Rollout Policy**: `[adapt_module, actor_student]`
- **Optimization**: PPO updates for student actor and critic
  - First 2.5% of training: Critic-only warmup
  - Remaining 97.5%: Joint actor-critic PPO updates

### Pipeline Orchestration

The `train.sh` script orchestrates all three stages:

```bash
# Stage 1: Train teacher
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=train \
  wandb.id=${ID_TRAIN}

# Stage 2: Adapt (distillation)
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=adapt \
  checkpoint_path=run:${PROJECT}/${ID_TRAIN} \
  wandb.id=${ID_ADAPT}

# Stage 3: Finetune student
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=finetune \
  checkpoint_path=run:${PROJECT}/${ID_ADAPT} \
  wandb.id=${ID_FINETUNE}
```

---

## Core Components

### 1. SimpleEnv (Environment)

**File**: `active_adaptation/envs/locomotion.py`

The main environment class that handles simulation, observations, rewards, and episode management.

```python
class SimpleEnv(_Env):
    def __init__(self, cfg, ...):
        self.setup_scene(cfg)     # Initialize MuJoCo scene
        self.init_mdp(cfg)        # Setup MDP components
        
    def setup_scene(self, cfg):
        # Create robot entity (G1 humanoid)
        # Create terrain
        # Setup contact sensors
        
    def init_mdp(self, cfg):
        # Initialize action space
        # Initialize observation groups (policy, priv, critic_priv)
        # Initialize reward groups
        # Initialize termination conditions
        # Initialize domain randomization
```

**Key Responsibilities**:
- Physics simulation via MuJoCo backend
- State management and reset logic
- Observation computation (policy, privileged, critic)
- Reward computation (tracking, locomotion, etc.)
- Domain randomization
- Episode termination detection

**MDP Components**:
- **Actions**: Joint position targets (PD control)
- **Observations**: 
  - `policy`: Observable states (proprioception, sensors)
  - `priv`: Privileged info (true dynamics, friction, etc.)
  - `critic_priv`: Additional privileged info for critic
- **Rewards**: Tracking rewards, locomotion rewards, regularization
- **Terminations**: Fall detection, time limits

### 2. PPOPolicy (Learning Algorithm)

**File**: `active_adaptation/learning/ppo/ppo.py`

The RL algorithm that implements PPO with teacher-student distillation.

```python
class PPOPolicy:
    def __init__(self, cfg, ...):
        # Networks
        self.encoder_priv = ...     # Encodes privileged obs
        self.actor_teacher = ...    # Teacher policy
        self.actor_student = ...    # Student policy
        self.critic = ...           # Value function
        self.adapt_module = ...     # Adaptation estimator
        
        # Optimizers (phase-dependent)
        self.opt_teacher = ...      # For teacher
        self.opt_student = ...      # For student
        self.opt_critic = ...       # For critic
        self.opt_estimator = ...    # For adapt_module
        
    def get_rollout_policy(self, mode):
        # Returns the network chain for rollout
        if mode == "train":
            return [encoder_priv, actor_teacher]
        else:  # adapt or finetune
            return [adapt_module, actor_student]
            
    def train_op(self, data, vecnorm):
        # Phase-specific training logic
        if self.phase == "train":
            return self._update_teacher(data)
        elif self.phase == "adapt":
            return self._update_student(data)
        elif self.phase == "finetune":
            return self._update2(data)
```

**Key Methods**:
- `get_rollout_policy(mode)`: Returns inference network chain
- `train_op(data, vecnorm)`: Executes one training step
- `_ppo_update()`: Core PPO algorithm (advantage estimation, policy loss, value loss)
- `train_estimator()`: Trains adaptation module to predict privileged features
- `step_schedule()`: Updates learning rate and entropy coefficient schedules

### 3. TensorDict Data Structure

**Used throughout**: All data is stored in TensorDict format for efficiency

```python
TensorDict {
    "policy": obs,              # Policy observations (N, T, obs_dim)
    "priv": priv_obs,          # Privileged observations (N, T, priv_dim)
    "critic_priv": critic_obs, # Critic privileged obs (N, T, critic_dim)
    "action": actions,         # Actions taken (N, T, action_dim)
    "state_value": values,     # V(s) estimates (N, T, 1)
    "log_prob": log_probs,    # log π(a|s) (N, T, 1)
    "next": {
        "policy": next_obs,
        "priv": next_priv_obs,
        "state_value": next_values,
        "done": done_flags,
        "stats": {
            "rewards/tracking": ...,
            "rewards/locomotion": ...,
            # ... other reward components
        },
        "_weight": importance_weights
    }
}
```

Where:
- `N` = number of parallel environments
- `T` = number of timesteps per rollout

---

## Call Flow

### Main Training Loop

**File**: `scripts/train.py`

```
main(cfg)
  │
  ├─► make_env_policy(cfg)
  │    ├─► SimpleEnv(cfg.task)
  │    │    ├─► setup_scene()      # MuJoCo initialization
  │    │    └─► init_mdp()         # MDP components
  │    │
  │    ├─► VecNorm(obs_keys)       # Observation normalization
  │    │
  │    └─► PPOPolicy(cfg.algo)
  │         ├─► create networks (encoder, actor, critic, adapt_module)
  │         └─► create optimizers
  │
  ├─► env.reset() → carry (initial state)
  │
  └─► for i in range(total_iters):
       │
       ├─► ROLLOUT (T=train_every steps)
       │    └─► for t in range(T):
       │         ├─► carry = rollout_policy(carry)
       │         │    ├─► encoder_priv(carry) OR adapt_module(carry)
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
       ├─► TRAINING STEP
       │    ├─► policy.step_schedule(i)  # Update LR, entropy
       │    ├─► env.step_schedule(i)     # Update randomization
       │    │
       │    └─► train_carry = policy.train_op(data, vecnorm)
       │         │
       │         ├─► Phase: TRAIN
       │         │    └─► _update_teacher(data)
       │         │         ├─► Compute advantages (GAE)
       │         │         ├─► PPO updates (teacher + critic)
       │         │         └─► Train estimator (adaptation module)
       │         │
       │         ├─► Phase: ADAPT
       │         │    └─► train_estimator(data)
       │         │         ├─► Forward: encoder_priv → priv_features
       │         │         ├─► Forward: adapt_module → priv_pred
       │         │         └─► MSE loss between priv_pred and priv_features
       │         │              (NO PPO updates, only adaptation module trained)
       │         │
       │         └─► Phase: FINETUNE
       │              └─► _ppo_update(data, update_student)
       │                   ├─► First 2.5%: Critic-only warmup
       │                   ├─► After 2.5%: Full actor-critic PPO
       │                   └─► Compute advantages (GAE)
       │
       ├─► LOGGING
       │    ├─► episode_stats.add(data)
       │    ├─► info = {metrics, fps, frames, ...}
       │    └─► wandb.log(info, step=i)
       │
       └─► CHECKPOINTING (periodic)
            ├─► save policy.state_dict()
            ├─► save env.state_dict()
            ├─► save vecnorm.state_dict()
            └─► upload to W&B
```

### Environment Step Flow

**File**: `active_adaptation/envs/locomotion.py`

```
step_and_maybe_reset(carry)
  │
  ├─► Extract actions from carry
  │
  ├─► mujoco.step(actions)           # Physics simulation
  │
  ├─► compute_observations()
  │    ├─► obs_group["policy"].compute()      # Policy obs
  │    ├─► obs_group["priv"].compute()        # Privileged obs
  │    └─► obs_group["critic_priv"].compute() # Critic obs
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
  ├─► Create TensorDict with (obs, reward, done)
  │
  └─► maybe_reset()
       ├─► If any envs are done:
       │    ├─► randomize_env_parameters()
       │    ├─► reset_poses()
       │    └─► sample_new_commands()
       │
       └─► Return (td, new_carry)
```

### PPO Training Step Flow

**File**: `active_adaptation/learning/ppo/ppo.py`

```
train_op(data, vecnorm)
  │
  ├─► vecnorm.update(data)           # Update normalization stats
  │
  ├─► Reshape data: (N×T) → (N*T,)  # Flatten time dimension
  │
  ├─► Compute GAE advantages
  │    ├─► δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
  │    └─► A_t = Σ (γλ)^k δ_{t+k}
  │
  ├─► Normalize advantages
  │
  └─► for epoch in range(ppo_epochs):
       │
       └─► for minibatch in minibatches:
            │
            ├─► Re-evaluate policy:
            │    ├─► π_new(a|s) via actor network
            │    └─► V_new(s) via critic network
            │
            ├─► Compute losses:
            │    ├─► ratio = π_new(a|s) / π_old(a|s)
            │    ├─► L_policy = -min(ratio·A, clip(ratio)·A)
            │    ├─► L_value = (V_new - V_target)²
            │    └─► L_entropy = -H(π)
            │
            ├─► Total loss = L_policy + c1·L_value - c2·L_entropy
            │
            ├─► Backpropagation
            │    ├─► loss.backward()
            │    └─► optimizer.step()
            │
            └─► Log metrics (policy_loss, value_loss, entropy, KL, etc.)
```

---

## Data Flow

### Single Training Iteration (i)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. ROLLOUT PHASE (T steps, inference mode)                      │
└─────────────────────────────────────────────────────────────────┘
   │
   ├─► Initialize: carry = env.reset()
   │   └─► Contains: robot states, target motions, etc.
   │
   └─► For t = 0 to T-1:
        │
        ├─► carry = rollout_policy(carry)
        │    Input:  carry["policy"], carry["priv"]
        │    Output: carry["action"]
        │
        ├─► td, carry = env.step_and_maybe_reset(carry)
        │    Input:  carry["action"]
        │    Output: td (observations, rewards, done)
        │             carry (updated states)
        │
        ├─► td["state_value"] = critic(td)
        ├─► td["next", "state_value"] = critic(td["next"])
        │
        └─► data_buf[t] = td
             └─► Stores: (obs, action, reward, value, log_prob, done)

┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAINING PHASE (data processing)                             │
└─────────────────────────────────────────────────────────────────┘
   │
   ├─► data = data_buf.td                    # Shape: (N, T, ...)
   │
   ├─► Compute GAE advantages
   │    Input:  rewards, values, next_values, dones
   │    Output: advantages, returns
   │
   ├─► Flatten: (N, T, ...) → (N*T, ...)
   │
   └─► Split into minibatches (for PPO epochs)

┌─────────────────────────────────────────────────────────────────┐
│ 3. OPTIMIZATION PHASE (PPO updates)                             │
└─────────────────────────────────────────────────────────────────┘
   │
   └─► For each PPO epoch:
        └─► For each minibatch:
             ├─► Forward pass (actor + critic)
             ├─► Compute losses
             ├─► Backward pass
             └─► Update parameters

┌─────────────────────────────────────────────────────────────────┐
│ 4. LOGGING PHASE                                                 │
└─────────────────────────────────────────────────────────────────┘
   │
   ├─► Collect episode statistics
   ├─► Compute FPS
   └─► wandb.log(metrics, step=i)
```

### Data Tensor Shapes

Throughout the pipeline, data maintains specific shapes:

| Component | Shape | Description |
|-----------|-------|-------------|
| Rollout carry | `(N, *)` | State for N parallel environments |
| Single step TD | `(N, *)` | Observations/rewards for one step |
| Data buffer | `(N, T, *)` | Full rollout buffer |
| Flattened data | `(N*T, *)` | Reshaped for PPO training |
| Minibatch | `(B, *)` | Subset for SGD update |

Where:
- `N` = num_envs (e.g., 4096)
- `T` = train_every (e.g., 32 for train, 16 for adapt)
- `B` = minibatch_size (e.g., N*T / num_minibatches)

---

## Key Modules

### 1. MDP Components (`active_adaptation/envs/mdp/`)

#### Observations (`observations.py`)

Defines observation groups with symmetry support:

```python
class ObsGroup:
    """Group of observations with symmetry transforms"""
    
    def __init__(self, *obs_terms):
        self.obs_terms = obs_terms
        
    def compute(self, state) -> torch.Tensor:
        # Concatenate all observation terms
        obs = [term.compute(state) for term in self.obs_terms]
        return torch.cat(obs, dim=-1)
        
    def symmetry_transform(self, obs, symmetry_mask):
        # Apply left-right symmetry transforms
        ...
```

**Common observation terms**:
- `BaseLinVel`: Base linear velocity
- `BaseAngVel`: Base angular velocity
- `ProjectedGravity`: Gravity vector in base frame
- `Commands`: Target motion tracking commands
- `DofPos`: Joint positions
- `DofVel`: Joint velocities
- `Actions`: Previous actions (history)

#### Rewards (`rewards/`)

Reward functions for training:

```python
class RewardGroup:
    """Group of reward terms with weights"""
    
    def __init__(self, *reward_terms):
        self.reward_terms = reward_terms
        
    def compute(self, state) -> Dict[str, torch.Tensor]:
        rewards = {}
        for term in self.reward_terms:
            rewards[term.name] = term.weight * term.compute(state)
        return rewards
```

**Common reward terms**:
- `MotionTracking`: Penalty for deviation from target motion
- `BaseHeightTracking`: Penalty for base height error
- `LinearVelocityTracking`: Penalty for velocity tracking error
- `ActionRate`: Penalty for large action changes
- `Torques`: Penalty for large joint torques
- `DofAcceleration`: Penalty for high joint accelerations
- `FeetAirTime`: Reward for proper foot contact timing
- `StumbleReward`: Penalty for foot stumbling

#### Domain Randomization (`randomizations.py`)

Randomizes simulation parameters for robust policies:

```python
class RandomizationManager:
    def randomize_on_reset(self, env_ids):
        # Randomize for reset environments
        self.randomize_robot_mass(env_ids)
        self.randomize_robot_com(env_ids)
        self.randomize_joint_friction(env_ids)
        self.randomize_joint_damping(env_ids)
        self.randomize_motor_strength(env_ids)
        self.randomize_base_mass(env_ids)
        # ... etc
        
    def randomize_on_step(self):
        # Randomize every step (noise, delays, etc.)
        self.add_noise_to_observations()
        self.add_action_delay()
```

### 2. Networks (`active_adaptation/learning/modules/`)

#### Actor Network

```python
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 256]):
        # MLP network
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

#### Critic Network

```python
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims=[256, 256, 256]):
        # Similar MLP for value estimation
        self.layers = nn.Sequential(...)
        
    def forward(self, obs):
        return self.layers(obs)  # Returns V(s)
```

#### Adaptation Module

```python
class AdaptModule(nn.Module):
    """Estimates privileged features from policy observations"""
    
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

### 3. Utilities

#### Motion Loading (`utils/motion.py`)

```python
class MotionDataset:
    """Loads and manages motion capture data"""
    
    def __init__(self, dataset_root):
        self.motions = self.load_all_motions(dataset_root)
        
    def sample_motion(self, env_ids, frame_ids):
        # Sample motion frames for tracking
        return target_poses, target_velocities
```

#### Symmetry (`utils/symmetry.py`)

```python
class SymmetryTransform:
    """Handles left-right symmetry for humanoid"""
    
    def transform_obs(self, obs, left_joint_ids, right_joint_ids):
        # Swap left/right joints
        # Negate Y-axis components
        ...
        
    def transform_action(self, action, ...):
        # Similar for actions
        ...
```

---

## Key Configuration Parameters

### Training Hyperparameters

**PPO Algorithm** (defined in `active_adaptation/learning/ppo/ppo.py`):
```python
# ppo_train config (registered via Hydra ConfigStore)
PPOConfig(
    phase="train", 
    vecnorm="train",
    ppo_epochs=5,                    # Epochs per batch
    num_minibatches=8,               # Minibatches per epoch
    train_every=32,                  # Rollout length (train phase)
    clip_param=0.2,                  # PPO clipping parameter
    lr=5e-4,                         # Learning rate (train phase)
    entropy_coef_start=0.01,         # Initial entropy regularization
    entropy_coef_end=0.0025,         # Final entropy regularization
    gamma=0.99,                      # Discount factor (implicit)
    gae_lambda=0.95,                 # GAE lambda (implicit)
    reg_lambda=0.2,                  # Distillation weight
)
```

**Adaptation Phase** (registered as `ppo_adapt`):
```python
PPOConfig(
    phase="adapt",
    vecnorm="eval",
    train_every=16,                  # Shorter rollouts
    lr=5e-4,                         # Same learning rate
    reg_lambda=0.2,                  # Distillation weight
)
```

**Finetune Phase** (registered as `ppo_finetune`):
```python
PPOConfig(
    phase="finetune",
    vecnorm="eval",
    lr=1e-4,                         # Lower learning rate
    train_every=16,
    entropy_coef_start=0.0025,       # Lower entropy
    entropy_coef_end=0.0005,
)
```

### Environment Parameters

**G1 Robot Task** (`cfg/task/G1/G1_tracking.yaml`):
```yaml
num_envs: 4096                   # Parallel environments
max_episode_length: 1000         # Max steps per episode
dt: 0.02                         # Simulation timestep (50 Hz)
decimation: 2                    # Action repeat (control at 25 Hz)

# Motion tracking
motion_dataset: "amass_all"      # Dataset to use
track_base_height: true
track_linear_velocity: true

# Domain randomization
randomize_friction: true
randomize_mass: true
randomize_com: true
# ... etc
```

---

## Summary

The motion tracking codebase implements a sophisticated teacher-student framework:

1. **Teacher Training (Stage 1)**: Learns a robust policy using privileged information (ground truth dynamics, friction, etc.). Simultaneously trains the adaptation module to predict privileged features from policy observations.

2. **Adaptation (Stage 2)**: Focuses exclusively on training the adaptation module via supervised learning (MSE loss). The module learns to estimate privileged features from observable sensor data. The student actor and critic remain frozen during this phase, using weights from Stage 1.

3. **Student Finetuning (Stage 3)**: Uses the frozen adaptation module to provide estimated privileged features, then trains the student actor and critic via PPO. Includes a 2.5% critic-only warmup period before full actor-critic training.

The architecture is modular and extensible:
- **Environment**: Physics simulation with MDP components
- **Learning**: PPO algorithm with privileged information estimation
- **Data**: Efficient TensorDict-based data flow
- **Config**: Hydra-based configuration system

This design enables robust sim-to-real transfer for complex humanoid motion tracking tasks through a carefully staged training process that progressively transitions from privileged to sensory-only observations.
