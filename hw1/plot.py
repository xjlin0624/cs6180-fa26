"""Plot train/val loss from out_dir/loss.csv files.

Usage:
  python plot.py results/gpu/baseline-lr2e-3       # one run: train vs val -> plot_baseline-lr2e-3.png
  python plot.py results/gpu/a results/gpu/b ...   # overlay: val (solid) + train (dashed) -> plot_compare.png
  python plot.py -o plots/gpu/rmsnorm.png a b      # -o / --output sets the output filename
  python plot.py --val-only --ylim 1.6 2.5 a b           # val curves only, zoomed y-axis
  python plot.py --skip-first a b                  # drop iter 0 (the ~4.2 starting loss)
  python plot.py --mark-best a b                   # star each val curve's minimum
"""
import sys
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('dirs', nargs='+', help='run directories containing loss.csv')
parser.add_argument('-o', '--output', default=None, help='output image filename')
parser.add_argument('--val-only', action='store_true', help='overlay: plot only val curves')
parser.add_argument('--ylim', type=float, nargs=2, metavar=('LO', 'HI'), default=None, help='y-axis range')
parser.add_argument('--skip-first', action='store_true', help='drop the iter-0 row from the plot')
parser.add_argument('--mark-best', action='store_true', help='mark the minimum of each val curve')
args = parser.parse_args()
dirs = args.dirs
dfs = {d: pd.read_csv(os.path.join(d, 'loss.csv')) for d in dirs}
plot_dfs = {d: (df.iloc[1:] if args.skip_first else df) for d, df in dfs.items()}

fig, ax = plt.subplots(figsize=(7, 4.5))
if len(dirs) == 1:
    df = plot_dfs[dirs[0]]
    ax.plot(df.iter, df.train_loss, marker='o', label='train')
    ax.plot(df.iter, df.val_loss, marker='o', label='val')
    if args.mark_best:
        b = df.loc[df.val_loss.idxmin()]
        ax.plot(b.iter, b.val_loss, marker='*', markersize=16, color='C1', markeredgecolor='black', zorder=5,
                linestyle='none', label=f'best val {b.val_loss:.4f} @ {int(b.iter)}')
    out = f'plot_{os.path.basename(dirs[0].rstrip("/"))}.png'
else:
    for i, (d, df) in enumerate(plot_dfs.items()):
        name = os.path.basename(d.rstrip('/')).removeprefix('out-')
        c = f'C{i}'
        ax.plot(df.iter, df.val_loss, color=c, marker='o', label=name if args.val_only else f'{name} val')
        if args.mark_best:
            b = df.loc[df.val_loss.idxmin()]
            ax.plot(b.iter, b.val_loss, marker='*', markersize=16, color=c, markeredgecolor='black', zorder=5,
                    linestyle='none', label=f'{name} best {b.val_loss:.4f} @ {int(b.iter)}')
        if not args.val_only:
            ax.plot(df.iter, df.train_loss, color=c, linestyle='--', alpha=0.6, label=f'{name} train')
    out = 'plot_compare.png'

if args.output:
    out = args.output

if args.ylim:
    ax.set_ylim(*args.ylim)
ax.set_xlabel('iteration')
ax.set_ylabel('val loss' if args.val_only and len(dirs) > 1 else 'loss')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=150)
print('saved', out)
for d, df in dfs.items():
    r = df.loc[df.val_loss.idxmin()]
    print(f'{d}: best val {r.val_loss:.4f} @ iter {int(r.iter)}, final train {df.train_loss.iloc[-1]:.4f}')
