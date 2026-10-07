#!/bin/bash
# HW1 Mini-LLM experiments (small CPU config of config/train_shakespeare_char.py).
# Every run uses the same model/data settings; only the flags after COMMON change.
# Each run writes out-<name>/loss.csv (iter,train_loss,val_loss).
#
# Usage:  bash run_experiments.sh <section>
#   baseline | rmsnorm | swiglu | nope | rope | gqa
# Compare runs:  python3 plot.py out-baseline-lr3e-3 out-rmsnorm-lr5e-3

set -e

python3 data/shakespeare_char/prepare.py

COMMON="config/train_shakespeare_char.py --device=cpu --compile=False
  --eval_iters=20 --log_interval=100 --block_size=64 --batch_size=12
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000
  --dropout=0.0"

# run <name> <lr> [extra train.py flags...]; min_lr = lr/10 as in the original config
run() {
  name=$1; lr=$2; shift 2
  python3 train.py $COMMON --learning_rate=$lr --min_lr=$(python3 -c "print($lr/10)") \
    "$@" --out_dir=out-$name-lr$lr
}

case "$1" in
  baseline)  # Step 1: LayerNorm + GELU + learned pos emb + MHA
    # (the very first baseline run used the default lr=1e-3 and out_dir=out-baseline)
    for lr in 5e-4 1e-3 2e-3 3e-3 5e-3; do run baseline $lr --norm_type=layernorm; done ;;
  rmsnorm)   # Step 2
    for lr in 5e-4 1e-3 2e-3 3e-3 5e-3 7e-3; do run rmsnorm $lr --norm_type=rmsnorm; done ;;
  swiglu)    for lr in 2e-3 3e-3 5e-3; do run swiglu $lr --mlp_type=swiglu; done ;;
  nope)      for lr in 1e-3 2e-3 3e-3 5e-3; do run nope $lr --pos_type=none; done ;;
  rope)      for lr in 2e-3 3e-3 5e-3; do run rope $lr --pos_type=rope; done ;;
  gqa)       for lr in 2e-3 3e-3 5e-3 7e-3; do run gqa $lr --n_kv_head=2; done ;;
  *) echo "usage: bash run_experiments.sh baseline|rmsnorm|swiglu|nope|rope|gqa"; exit 1 ;;
esac
