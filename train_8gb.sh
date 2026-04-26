#!/bin/bash

# SAR-GNN Training Script for RTX 4060 (8GB VRAM)
# Usage: bash train_8gb.sh [3|5|7] [shots] [tag]

NWAY=${1:-3}
SHOTS=${2:-20}
TAG=${3:-rtx4060}

if [ "$NWAY" -eq 3 ]; then
    BATCH_SIZE=4
    EVAL_BATCH_SIZE=2
    EVAL_SAMPLE=2000
elif [ "$NWAY" -eq 5 ]; then
    BATCH_SIZE=2
    EVAL_BATCH_SIZE=1
    EVAL_SAMPLE=1500
elif [ "$NWAY" -eq 7 ]; then
    BATCH_SIZE=2
    EVAL_BATCH_SIZE=1
    EVAL_SAMPLE=1000
else
    echo "Unsupported nway=$NWAY (use 3, 5, or 7)"
    exit 1
fi

echo "=========================================="
echo "SAR-GNN Training (RTX 4060 8GB)"
echo "=========================================="
echo "nway=$NWAY, shots=$SHOTS"
echo "batch_size=$BATCH_SIZE, eval_batch_size=$EVAL_BATCH_SIZE"
echo "eval_sample=$EVAL_SAMPLE"
echo "tag=$TAG"
echo "=========================================="
echo ""

python main.py \
    --nway $NWAY \
    --shots $SHOTS \
    --data_root mstar_sampled_data \
    --model_type gnn \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --eval_sample_8gb $EVAL_SAMPLE \
    --gradient_checkpointing \
    --amp \
    --max_iteration 100000 \
    --eval_interval 500 \
    --early_stop 10 \
    --unseen_class T72 \
    --unseen_ratio 1.0 \
    --warmup_iters 2000 \
    --save \
    --affix $TAG

echo ""
echo "Training complete! Check log/ and model/ directories."
