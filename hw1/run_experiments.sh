#!/bin/bash
# HW1 Mini-LLM experiments on config/train_shakespeare_char.py.
# Every run in a profile uses the same model/data settings; only the variant flag and lr change.
# Each run writes <prefix>-<name>-lr<lr>/loss.csv (iter,train_loss,val_loss).
#
# Usage:  [PROFILE=cpu|gpu] [LRS="..."] [EXTRA="..."] bash run_experiments.sh <section>
#   section: baseline | rmsnorm | swiglu | nope | rope | gqa | all
#
# Profiles:
#   cpu (default): README "macbook" settings - 4 layers, 4 heads, 128-dim, context 64, 2000 iters,
#                  dropout 0. Output dirs out-<name>-lr<lr>. These are the runs in WRITEUP.md.
#   gpu:           the unmodified default config - 6 layers, 6 heads, 384-dim, context 256,
#                  5000 iters, dropout 0.2, cuda + torch.compile. Output dirs out-gpu-<name>-lr<lr>.
#
# Env overrides:
#   LRS="3e-4 5e-4"            learning rates to sweep instead of the profile's default list
#   EXTRA="--compile=False"    extra train.py flags appended to every run
#   OUT_PREFIX=out-gpu2        output dir prefix
#
# Compare runs:  python3 plot.py out-baseline-lr3e-3 out-rmsnorm-lr5e-3

set -e

PROFILE=${PROFILE:-cpu}
case "$PROFILE" in
  cpu)
    COMMON="config/train_shakespeare_char.py --device=cpu --compile=False
      --eval_iters=20 --log_interval=100 --block_size=64 --batch_size=12
      --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000
      --dropout=0.0"
    N_HEAD=4
    OUT_PREFIX=${OUT_PREFIX:-out} ;;
  gpu)
    COMMON="config/train_shakespeare_char.py --log_interval=100"
    N_HEAD=6
    OUT_PREFIX=${OUT_PREFIX:-out-gpu} ;;
  *) echo "unknown PROFILE=$PROFILE (use cpu or gpu)"; exit 1 ;;
esac
# GQA with group size 2: every 2 query heads share one kv head
KV_HEAD=$((N_HEAD / 2))

[ -f data/shakespeare_char/train.bin ] || python3 data/shakespeare_char/prepare.py

# run <name> <lr> [extra train.py flags...]; min_lr = lr/10 as in the original config
run() {
  name=$1; lr=$2; shift 2
  python3 train.py $COMMON --learning_rate=$lr --min_lr=$(python3 -c "print('%g' % ($lr/10))") \
    "$@" $EXTRA --out_dir=$OUT_PREFIX-$name-lr$lr
}

# sweep <name> <cpu lrs> <gpu lrs> [variant flags...]
sweep() {
  name=$1; cpu_lrs=$2; gpu_lrs=$3; shift 3
  if [ "$PROFILE" = gpu ]; then default_lrs=$gpu_lrs; else default_lrs=$cpu_lrs; fi
  for lr in ${LRS:-$default_lrs}; do run $name $lr "$@"; done
}

# cpu lrs are the ones already run (see WRITEUP.md). gpu lrs are a starting bracket around the
# config default 1e-3: extend with LRS=... if the best one is at either end.
GPU_LRS="5e-4 1e-3 2e-3"
section() {
  case "$1" in
    baseline)  # Step 1: LayerNorm + GELU + learned pos emb + MHA
      # (the very first cpu baseline run used the default lr=1e-3 and out_dir=out-baseline)
      sweep baseline "5e-4 1e-3 2e-3 3e-3 5e-3" "$GPU_LRS" --norm_type=layernorm ;;
    rmsnorm) sweep rmsnorm "5e-4 1e-3 2e-3 3e-3 5e-3 7e-3" "$GPU_LRS" --norm_type=rmsnorm ;;
    swiglu)  sweep swiglu  "2e-3 3e-3 5e-3"                "$GPU_LRS" --mlp_type=swiglu ;;
    nope)    sweep nope    "1e-3 2e-3 3e-3 5e-3"           "$GPU_LRS" --pos_type=none ;;
    rope)    sweep rope    "2e-3 3e-3 5e-3"                "$GPU_LRS" --pos_type=rope ;;
    gqa)     sweep gqa     "2e-3 3e-3 5e-3 7e-3"           "$GPU_LRS" --n_kv_head=$KV_HEAD ;;
    all)     for s in baseline rmsnorm swiglu nope rope gqa; do section $s; done ;;
    *) echo "usage: [PROFILE=cpu|gpu] bash run_experiments.sh baseline|rmsnorm|swiglu|nope|rope|gqa|all"; exit 1 ;;
  esac
}
section "$1"
