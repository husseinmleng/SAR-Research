# SAR_Dr.Ahmed Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-12

## Active Technologies

- Python 3.7+, PyTorch, torchvision, NumPy, Pillow
- CUDA GPU training (NVIDIA RTX 3060 / 12GB recommended)
- MSTAR dataset: 10-class SAR vehicle images, 100×100 grayscale

## Project Structure

```text
main.py          # Entry point — parse args, init data/trainer, run training
argument.py      # argparse config — all hyperparameters
cnn.py           # EmbeddingCNN — SAR image feature extractor (64-D output)
gnn.py           # GNN_module — relation space for few-shot comparison
data.py          # self_DataLoader — few-shot episode batch generator
trainer.py       # Trainer — training loop, evaluation, checkpointing
utils.py         # Logger, directory utilities

mstar_sampled_data/   # Dataset (read-only): 10 classes × 100 images
model/                # Checkpoint output
log/                  # Training logs
specs/001-gnn-sar-osr-alignment/   # Active feature plan
```

## Commands

```bash
# Train (paper default: 3-way 20-shot, T72 unseen)
python main.py --nway 3 --shots 20 --data_root mstar_sampled_data

# 5-way or 7-way
python main.py --nway 5 --shots 20 --data_root mstar_sampled_data

# Resume from checkpoint
python main.py --load model/3way_20shot_gnn_/best_model.pt --nway 3 --shots 20 --data_root mstar_sampled_data
```

## Code Style

Python: Follow existing conventions (no type hints, no docstrings unless already present). Keep all changes in-place — no new top-level files.

## Active Feature

**001-gnn-sar-osr-alignment**: Align CNN+GNN architecture with Zhou et al. (Sensors 2023).  
Key changes: 4-layer CNN + SE attention, 3-adj+2-update GNN, T72 as unseen class, 5000-task eval.  
Plan: `specs/001-gnn-sar-osr-alignment/plan.md`

## Recent Changes

- 001-gnn-sar-osr-alignment: Added (2026-04-12)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
