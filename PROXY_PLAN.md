# UCVLA Proxy Experiment — Setup Plan

## Goal
Validate UCVLA bias learning (triplet/ortho losses) on a lightweight model before
committing to a full RDT2 retrain. Single GPU, ~20 min per run.

**Stack:** Python 3.12, uv, PyTorch, Accelerate, wandb

---

## 1. Create Repo

```bash
gh repo create ucvla-proxy --private --clone
cd ucvla-proxy
uv init --python 3.12
```

---

## 2. Dependencies (`pyproject.toml`)

```toml
[project]
name = "ucvla-proxy"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    "torch>=2.5.0",
    "torchvision",
    "accelerate>=1.0.0",
    "wandb",
    "open-clip-torch",
    "webdataset",
    "einops",
    "timm",
    "pyyaml",
    "numpy",
]
```

---

## 3. Repo Structure

```
ucvla-proxy/
├── pyproject.toml
├── configs/
│   └── dp_ucvla.yaml
├── data/                        # symlink: ln -s /path/to/mug_handover_webdataset data
├── models/
│   ├── bias.py                  # UserBias + BiasProj (ported from RDT2)
│   ├── clip_encoder.py          # frozen CLIP-ViT-B/16
│   └── dp/
│       ├── model.py             # minimal DiT diffusion policy
│       └── runner.py            # UCVLADPRunner
├── scripts/
│   ├── train.py                 # main Accelerate training script
│   ├── train.sh                 # launch script
│   └── eval_confusion.py        # cross-user confusion matrix
└── wandb/
```

---

## 4. Architecture

```
image  → frozen CLIP-ViT-B/16 → [B, 196, 512] → linear → [B, 196, D]  ┐
state  → linear               → [B, 1, D]                               ├─ cross-attn context
                                                                         ┘
noisy_action → linear → [B, T, D]
t_embed + bias_proj(user_bias[user_id]) → adaLN → DiT blocks → action
```

**Sizes:**
- CLIP ViT-B/16: 86M params, **frozen**
- DiT (8 layers, hidden=512): ~10M params, **frozen**
- `user_bias`: `nn.Embedding(n_users, 64)`
- `bias_proj`: `nn.Linear(64, 512)`
- **Trainable in Stage 1**: only `user_bias` + `bias_proj` (~100K params)

Bias injection is identical to RDT2 — validated loss design ports directly.

---

## 5. Config (`configs/dp_ucvla.yaml`)

```yaml
n_users: 3
bias_dim: 64
hidden_size: 512
depth: 8
num_heads: 8
pred_horizon: 64
action_dim: 14
state_dim: 14

lr: 1e-3
warmup_steps: 500
max_steps: 10000
batch_size: 64
weight_decay: 0.01

lambda_triplet: 0.1
lambda_ortho: 0.01
triplet_margin: 0.5

log_every: 10
val_every: 500
save_every: 2000

wandb_project: ucvla-proxy
```

---

## 6. Loss Design

```python
# Base: flow matching
loss = F.mse_loss(pred_velocity, target_velocity)

# Triplet: same-user bias closer than cross-user
loss += lambda_triplet * F.triplet_margin_loss(
    anchor=bias[user_id],
    positive=bias[user_id],        # same user, different sample
    negative=bias[other_user_id],  # different user
    margin=triplet_margin,
)

# Orthogonality: push all bias vectors apart
B = user_bias.weight               # [n_users, bias_dim]
gram = B @ B.T                     # [n_users, n_users]
loss += lambda_ortho * (gram - torch.eye(n_users, device=gram.device)).norm()
```

**Ablations to run (4 runs total):**
1. MSE only (baseline — what Stage 1 currently uses)
2. MSE + triplet
3. MSE + ortho
4. MSE + triplet + ortho

---

## 7. Training Script (`scripts/train.sh`)

```bash
#!/bin/bash
PYTHONPATH="$(pwd)" uv run accelerate launch \
    --num_processes=1 \
    scripts/train.py \
    --config configs/dp_ucvla.yaml \
    --data_dir data/ \
    --output_dir outputs/dp_ucvla/stage1
```

---

## 8. Implementation Steps

1. **Init repo** — `gh repo create`, `uv init`
2. **Port from RDT2**
   - `models/bias.py` ← extract `UserBias` + `BiasProj` from `models/ucvla/ucvla_rdt.py`
   - `data/` loading ← adapt `rdt/dataset.py` webdataset pipeline
   - `scripts/eval_confusion.py` ← adapt `scripts/eval_ucvla.py`
3. **Implement `models/clip_encoder.py`** — load `open_clip` ViT-B/16, freeze, return patch tokens
4. **Implement `models/dp/model.py`** — minimal DiT with adaLN + cross-attn to CLIP+state tokens
5. **Implement `models/dp/runner.py`** — `compute_loss()`, `predict_action()`, `forward()`
6. **Write `scripts/train.py`** — copy Accelerate structure from `rdt/train_ucvla_stage1.py`, add triplet/ortho losses
7. **Run 4 ablations**, check confusion matrix each time

---

## 9. Success Criteria

Confusion matrix with all 3 users winning on correct bias:

```
        bias_0   bias_1   bias_2
user_0: [0.00002] 0.00005  0.00005
user_1:  0.00005 [0.00003] 0.00005
user_2:  0.00005  0.00005 [0.00002]
```

→ port winning loss combo to `rdt/train_ucvla_stage1.py` → retrain RDT2

---

## 10. Timeline

| Task                          | Time        |
|-------------------------------|-------------|
| Repo setup + boilerplate port | 1-2 hours   |
| DiT + CLIP encoder impl       | 2-3 hours   |
| First training run (10k steps)| ~20 min     |
| 4 ablation runs               | ~2 hours    |
| Port winning loss to RDT2     | 30 min      |
| RDT2 retrain                  | ~15 hours   |
