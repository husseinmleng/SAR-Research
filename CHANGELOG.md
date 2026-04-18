# Changelog — SAR ATR / GNN-OSR Pipeline

Reference paper: Zhou et al., "SAR Target Recognition with Limited Training Samples in Open Set Conditions", *Sensors* 2023.

---

## [2026-04-12] Full Implementation — spec 001-gnn-sar-osr-alignment

### New Files

| File | Purpose |
|------|---------|
| `augment.py` | `SpeckleNoise(sigma)` — multiplicative Rayleigh noise; `RandomRotation360()` — uniform 0–360° rotation |
| `evaluate.py` | `compute_metrics()` / `save_report()` / `print_comparison_table()` — per-class F1, confusion matrix, JSON reports |
| `gan.py` | Conditional DCGAN: `ConditionalGenerator` + `ConditionalDiscriminator` + training loop + quality check |
| `requirements.txt` | `torch torchvision numpy Pillow scikit-learn matplotlib` |
| `README.md` | Full argument reference, quick-start recipes, output structure, architecture summary |
| `EXPLANATION.md` | Plain-language explanation of SAR/ATR problem, FSL, CNN+SE, GNN, GAN, physics loss, augmentation |
| `pipeline.ipynb` | Vast.ai Jupyter notebook — dataset download, full pipeline training, result export |

### Modified Files

#### `argument.py`
Added 11 new CLI arguments after `--freeze_cnn`:
```
--unseen_class     str,   default='T72'
--unseen_ratio     float, default=1.0       # 1.0 = 1:1 seen/unseen query ratio
--gan_augment      flag                     # augment support set with GAN images
--gan_output_dir   str,   default='gan_output'
--physics_lambda   float, default=0.0       # 0.0 = disabled
--augment_rotation flag
--augment_speckle  flag
--speckle_sigma    float, default=0.1
--eval_only        flag                     # skip training, eval checkpoint once
--eval_output      str,   default='results'
--baseline_kshot   flag                     # subsample K images/class for CNN baseline
```
Validation: `unseen_ratio > 0`, `physics_lambda >= 0`.

#### `cnn.py` (full rewrite)
- Replaced 3-layer doubling-channel CNN with 4 × `[Conv2d(in,32,3,pad=1) + BN + ReLU + SEBlock + MaxPool2d(2)]`
- `SEBlock`: `AdaptiveAvgPool2d(1) → Linear(32,32) → ReLU → Sigmoid → channel-wise multiply`
- Final projection: `Conv2d(32,64,spatial) + AdaptiveAvgPool2d(1)` → 64-D embedding
- Removed dead classes: `ChannelAttention`, `SpatialAttention`, `BasicBlock`, `conv3x3`

#### `gnn.py` (full rewrite)
- `AdjacencyModule`: pairwise `|vi−vj|` → `Conv2d(D,64,1)/LReLU → Conv2d(64,32,1)/LReLU → Conv2d(32,1,1)` → softmax over neighbors
- `UpdateModule`: `bmm(A, V) → Linear(D, 16) + ReLU`
- `GNN_module.forward`: 3-adj + 2-update dense structure with feature concatenation after each update:
  ```
  R0=Adj(V); V'=Upd(R0,V); V=cat(V,V')
  R1=Adj(V); V'=Upd(R1,V); V=cat(V,V')
  R2=Adj(V); logits=FC(R2[:,0,:])  →  (B, C+1)
  ```
- Removed: `Graph_conv_block`, `Adjacency_layer`

#### `data.py` (full rewrite)
- **Bug fix**: `ImageFolder(root=root)` — was `root=os.path.join(root,"MSTAR")` (non-existent subdir)
- **Bug fix**: unseen class lookup `dataset.class_to_idx[args.unseen_class]` — was hardcoded `[4]` (BTR70, wrong)
- Configurable seen/unseen query ratio: `unseen_prob = ratio / (1 + ratio)`
- GAN augment hook: loads from `gan_output_dir/<class>/` and appends to support set
- Per-sample augment: applies `RandomRotation360` / `SpeckleNoise` at batch generation time
- 80/20 train/test split preserved

#### `trainer.py` (major additions)
- `eval_sample` increased 400 → **5000** tasks
- Checkpoint criterion: `best_loss` → `best_acc` (overall accuracy, not loss)
- `physics_loss(support_features, support_labels, nway)`: intra-class embedding variance penalty
- `train_batch()`: when `physics_lambda > 0`, `L_total = L_NLL + λ * physics_loss()`; returns `(cls_loss, phys_loss)`
- `eval()`: returns `(avg_loss, overall_acc, seen_acc, unseen_acc, y_true_all, y_pred_all)`; integrates `evaluate.py` for per-class F1 + JSON save
- `eval_augmented()`: second eval pass with `RandomRotation360 + SpeckleNoise` on query images
- `TrainerBaseline`: standard cross-entropy CNN classifier with optional K-shot subsampling
- Log format: `seen=XX.XX%  unseen=XX.XX%  overall=XX.XX%`

#### `main.py`
- `--eval_only` mode: load checkpoint → `trainer.eval()` once → save JSON → exit
- `--model_type cnn` routes to `TrainerBaseline`
- All new args passed through to `self_DataLoader`

---

### Architecture Summary

```
CNN: 4 × [Conv2d(32) + BN + ReLU + SE-attention + MaxPool] → Conv2d(64) → 64-D embedding
GNN: 3 AdjacencyModules + 2 UpdateModules, dense concat, output (C+1) logits
Loss: NLL(C+1 classes) + λ × intra-class variance penalty
Eval: 5000 episodic tasks → seen / unseen / overall accuracy + per-class F1 + confusion matrix
```

---

### Key Bug Fixes

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Data not loading | `ImageFolder` pointed to non-existent `data/MSTAR/` subdir | Use `root` directly |
| Wrong unseen class | Hardcoded index `4` (BTR70) instead of T72 | `dataset.class_to_idx['T72']` |
| SE-attention ignored | Module defined but commented out | Re-activated and wired into each CNN block |
| GNN output wrong | Old spectral conv formula `A·X·W` | Paper's pairwise-diff adjacency + weighted aggregation |
| Checkpoint on wrong metric | Saved on lowest loss | Save on highest overall accuracy |

---

### Jupyter Notebook — Vast.ai (pipeline.ipynb)

**Critical fix — kernel/pip mismatch:**
- `!pip install` targets system pip which may differ from the Jupyter kernel's Python
- Fix: use `!{sys.executable} -m pip install` everywhere
- Same for training: `!{sys.executable} main.py` instead of `!python main.py`

**Diagnostic cell (run first):**
```python
import sys
print('Kernel Python:', sys.executable)
print('Python version:', sys.version)
```

**Verify cell (after install):**
```python
import importlib
for mod, pkg in {'torch':'torch','sklearn':'scikit-learn','gdown':'gdown'}.items():
    try: importlib.import_module(mod); print(f'OK  {pkg}')
    except ImportError: print(f'MISSING  {pkg}  <-- re-run install cell')
```

---

### Google Drive Dataset Download

**Structure on Drive:** `SAR/data/MSTAR/<class_folders>`

```bash
# Download MSTAR folder (right-click MSTAR in Drive → Get link → copy folder ID)
gdown --folder "https://drive.google.com/drive/folders/YOUR_MSTAR_FOLDER_ID" -O .
```

```python
import os
os.rename("MSTAR", "mstar_sampled_data")   # gdown keeps Drive folder name
```

**If data is a zip:**
```bash
gdown YOUR_FILE_ID -O mstar.zip
```
```python
import zipfile, os
with zipfile.ZipFile("mstar.zip") as z:
    z.extractall(".")
os.rename("MSTAR", "mstar_sampled_data")   # adjust if zip extracts to different name
```

---

### Pending (requires GPU on Vast.ai)

| Task | Command |
|------|---------|
| T010 — 3-way 20-shot baseline | `python main.py --nway 3 --shots 20 --data_root mstar_sampled_data --save` |
| T012 — CNN baseline | `python main.py --model_type cnn --nway 9 --data_root mstar_sampled_data` |
| T022 — Train GAN | `python gan.py --data_root mstar_sampled_data --output_dir gan_output --epochs 200` |
| T023 — GAN augment run | `python main.py --nway 3 --shots 10 --gan_augment --data_root mstar_sampled_data --save` |
| T026 — Physics ablation | `python main.py --nway 3 --shots 20 --physics_lambda 0.05 --data_root mstar_sampled_data --save` |
| T029 — Augmented training | `python main.py --nway 3 --shots 20 --augment_rotation --augment_speckle --save` |
| T031 — 5-way / 7-way sweep | `for NWAY in 5 7; do python main.py --nway $NWAY --shots 20 --save; done` |
| T032 — Comparison table | `from evaluate import print_comparison_table; print_comparison_table(glob('results/*.json'))` |
| T034 — eval_only round-trip | `python main.py --eval_only --load model/3way_20shot_gnn_/model.pth --nway 3 --shots 20` |

Expected targets: seen ≥ 95%, unseen ≥ 62%, overall ≥ 85% (3-way 20-shot).
