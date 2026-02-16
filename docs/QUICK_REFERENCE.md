# Quick Reference: Motion Tracking Training Pipeline

## Three-Stage Training Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: TRAIN (8B frames, ~15 hours on 4×A100)                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Purpose: Train teacher policy with privileged information              │
│                                                                         │
│ Rollout:  [encoder_priv → actor_teacher] + privileged observations     │
│                                                                         │
│ Training:                                                               │
│   ✓ PPO updates: actor_teacher + critic                                │
│   ✓ MSE loss: adapt_module (learns to predict priv features)           │
│                                                                         │
│ Config: +exp=train, vecnorm=train, lr=5e-4                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        Checkpoint: teacher weights +
                                adapt_module (partially trained)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: ADAPT (1B frames, ~2 hours on 4×A100)                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Purpose: Train adaptation module to predict privileged features        │
│                                                                         │
│ Rollout:  [adapt_module → actor_student] + policy observations only    │
│           (student uses Stage 1 weights, frozen)                       │
│                                                                         │
│ Training:                                                               │
│   ✓ MSE loss ONLY: adapt_module predicts priv_features                 │
│   ✗ NO PPO updates                                                      │
│   ✗ actor_student FROZEN (no weight updates)                           │
│   ✗ critic FROZEN (no weight updates)                                  │
│                                                                         │
│ Key Point: This is SUPERVISED LEARNING, not RL!                        │
│            Trains estimator with 2 mini-epochs per rollout             │
│                                                                         │
│ Config: +exp=adapt, vecnorm=eval, lr=5e-4, train_every=16              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        Checkpoint: adapt_module (trained) +
                                student (still from Stage 1)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: FINETUNE (4B frames, ~8 hours on 4×A100)                      │
├─────────────────────────────────────────────────────────────────────────┤
│ Purpose: Finetune student policy with frozen adaptation module         │
│                                                                         │
│ Rollout:  [adapt_module (FROZEN) → actor_student]                      │
│                                                                         │
│ Training:                                                               │
│   Phase A (first 2.5% of training):                                    │
│     ✓ PPO updates: critic ONLY                                         │
│     ✗ actor_student FROZEN                                             │
│                                                                         │
│   Phase B (remaining 97.5%):                                           │
│     ✓ PPO updates: actor_student + critic                              │
│                                                                         │
│ Config: +exp=finetune, vecnorm=eval, lr=1e-4, train_every=16           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        Final Model: deploy-ready student policy
```

## Key Network States by Stage

| Network         | Stage 1 (TRAIN) | Stage 2 (ADAPT) | Stage 3 (FINETUNE) |
|-----------------|-----------------|-----------------|---------------------|
| encoder_priv    | ✅ Trained      | 🔒 Not used     | 🔒 Not used        |
| actor_teacher   | ✅ Trained      | 🔒 Not used     | 🔒 Not used        |
| actor_student   | 🔒 Not trained  | 🔒 Frozen       | ✅ Trained         |
| critic          | ✅ Trained      | 🔒 Frozen       | ✅ Trained         |
| adapt_module    | ✅ Trained      | ✅ Trained      | 🔒 Frozen          |

## Data Flow Per Stage

### Stage 1: TRAIN
```
Policy Obs → encoder_priv → priv_features → actor_teacher → actions
Priv Obs  ↗                                                      ↓
                                                            Environment
Critic Obs → critic → value estimate                            ↓
                                                              Next State
                                                                 ↓
Policy Obs → adapt_module → priv_pred ─┐                   Compute Rewards
True Priv  → encoder_priv → priv_true ─┴─→ MSE Loss           ↓
                                            ↓               PPO Update
                                         Update              (Teacher)
```

### Stage 2: ADAPT
```
Policy Obs → adapt_module → priv_pred → actor_student → actions
                                 ↓                           ↓
True Priv  → encoder_priv → priv_true              Environment (collect data)
                                 ↓                           ↓
                            MSE Loss                     No RL Training!
                                 ↓                    (Just data collection)
                         Update adapt_module
```

### Stage 3: FINETUNE
```
Policy Obs → adapt_module → priv_pred → actor_student → actions
           (FROZEN)                    (UNFROZEN)          ↓
                                                      Environment
Critic Obs → critic → value estimate                      ↓
          (UNFROZEN)                                  Compute Rewards
                                                           ↓
                                                      PPO Update
                                                   (Student + Critic)
```

## Command Examples

### Run Full Pipeline
```bash
bash train.sh
```

### Run Individual Stages
```bash
# Stage 1: Train
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=train \
  wandb.id=my_train_run

# Stage 2: Adapt
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=adapt \
  checkpoint_path=run:PROJECT/my_train_run \
  wandb.id=my_adapt_run

# Stage 3: Finetune
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tracking +exp=finetune \
  checkpoint_path=run:PROJECT/my_adapt_run \
  wandb.id=my_finetune_run
```

### Evaluate Policy
```bash
# Play in simulation
uv run scripts/eval.py --run_path PROJECT/my_finetune_run -p

# Export to ONNX for deployment
uv run scripts/eval.py --run_path PROJECT/my_finetune_run -p --export
```

## Key Configuration Parameters

| Parameter       | Train   | Adapt   | Finetune |
|-----------------|---------|---------|----------|
| total_frames    | 8B      | 1B      | 4B       |
| train_every     | 32      | 16      | 16       |
| lr              | 5e-4    | 5e-4    | 1e-4     |
| entropy_coef    | 0.01→0.0025 | 0.005→0.002 | 0.0025→0.0005 |
| vecnorm         | train   | eval    | eval     |
| ppo_epochs      | 5       | -       | 5        |
| num_minibatches | 8       | 8       | 8        |

## Important Notes

1. **Stage 2 (ADAPT) is NOT reinforcement learning** - it's supervised learning to train the estimator. The policy networks remain frozen.

2. **The student actor is never directly trained during ADAPT** - it only gets updated in FINETUNE stage.

3. **Privileged information is only available during training** - at deployment, only the student policy with adaptation module is used.

4. **The 2.5% critic warmup in FINETUNE** helps stabilize value estimates before updating the actor.

5. **Training order matters** - each stage builds on the previous checkpoint, so they must be run sequentially.

## File Structure Quick Reference

```
active_adaptation/
├── envs/
│   ├── locomotion.py          # SimpleEnv: main environment
│   ├── mdp/
│   │   ├── observations.py    # Observation groups (policy, priv, critic_priv)
│   │   ├── rewards/           # Reward functions
│   │   ├── terminations.py    # Episode termination
│   │   └── randomizations.py  # Domain randomization
│   └── scene.py               # MuJoCo scene setup
├── learning/
│   ├── ppo/
│   │   └── ppo.py            # PPOPolicy: main RL algorithm
│   └── modules/              # Neural network modules
└── utils/
    ├── motion.py             # Motion dataset loading
    └── symmetry.py           # Left-right symmetry transforms

scripts/
├── train.py                  # Main training script
└── eval.py                   # Evaluation script

cfg/
├── train.yaml               # Base training config
├── exp/
│   ├── train.yaml          # Stage 1 config
│   ├── adapt.yaml          # Stage 2 config
│   └── finetune.yaml       # Stage 3 config
└── task/
    └── G1/                 # G1 robot configs
```

## Troubleshooting

**Q: Training is slow / OOM errors**
- Reduce `num_envs` in `cfg/task/G1/G1.yaml`
- Reduce `NPROC` in `train.sh`
- May increase training time

**Q: Adapt stage seems to do nothing**
- This is expected! ADAPT only trains the estimator, not the policy
- Check WandB for `adapt/estimator_loss` - it should decrease
- Policy performance will improve in FINETUNE stage

**Q: Student performs worse than teacher**
- Increase ADAPT training frames
- Increase FINETUNE training frames
- Check that adaptation module loss converged in Stage 2

**Q: How to resume training?**
- Set `checkpoint_path` to load from W&B run
- Can resume any stage from its checkpoint

## Performance Expectations

**Training Time** (4×A100 GPUs):
- Stage 1 (TRAIN): ~15 hours
- Stage 2 (ADAPT): ~2 hours
- Stage 3 (FINETUNE): ~8 hours
- **Total**: ~25 hours

**GPU Memory**:
- ~30-40GB per GPU with 4096 envs
- Can be reduced by lowering `num_envs`

**Final Performance**:
- Student should achieve 85-95% of teacher performance
- Real robot deployment uses student policy only
