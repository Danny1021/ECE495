#!/usr/bin/env python3
import argparse, csv, os
import matplotlib.pyplot as plt

def load_metrics(csv_path):
    epochs, train_acc, val_acc, val_loss = [], [], [], []
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row['epoch']))
            train_acc.append(float(row['train_acc']))
            val_acc.append(float(row['val_acc']))
            val_loss.append(float(row['val_loss']))
    return epochs, train_acc, val_acc, val_loss

def best_by_min_valloss(epochs, val_acc, val_loss):
    best_idx = min(range(len(epochs)), key=lambda i: val_loss[i])
    return {
        'epoch': epochs[best_idx],
        'val_acc': val_acc[best_idx],
        'val_loss': val_loss[best_idx],
        'idx': best_idx
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plain', required=True, help='Directory with metrics.csv for the plain MLP run')
    ap.add_argument('--res', required=True, help='Directory with metrics.csv for the residual MLP run')
    ap.add_argument('--out', default='compare_acc.png', help='Output figure filename')
    args = ap.parse_args()

    plain_csv = os.path.join(args.plain, 'metrics.csv')
    res_csv   = os.path.join(args.res,   'metrics.csv')
    assert os.path.exists(plain_csv), f"Not found: {plain_csv}"
    assert os.path.exists(res_csv),   f"Not found: {res_csv}"

    # Load
    ep_p, tr_p, va_p, vl_p = load_metrics(plain_csv)
    ep_r, tr_r, va_r, vl_r = load_metrics(res_csv)

    # Best (by minimum val loss)
    best_p = best_by_min_valloss(ep_p, va_p, vl_p)
    best_r = best_by_min_valloss(ep_r, va_r, vl_r)

    print(f"[PLAIN ] best@epoch {best_p['epoch']:>3} | val_acc={best_p['val_acc']:.4f} | val_loss={best_p['val_loss']:.4f}")
    print(f"[RES   ] best@epoch {best_r['epoch']:>3} | val_acc={best_r['val_acc']:.4f} | val_loss={best_r['val_loss']:.4f}")

    # Plot
    fig = plt.figure(figsize=(10,5))

    # Left: val-accuracy curves with best markers
    ax1 = plt.subplot(1,2,1)
    ax1.plot(ep_p, va_p, label='Plain MLP (val acc)')
    ax1.plot(ep_r, va_r, label='Residual MLP (val acc)')
    ax1.scatter([best_p['epoch']], [best_p['val_acc']], marker='o', zorder=5)
    ax1.scatter([best_r['epoch']], [best_r['val_acc']], marker='o', zorder=5)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Validation Accuracy vs Epoch')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Right: bar chart of best val acc
    ax2 = plt.subplot(1,2,2)
    models = ['Plain', 'Residual']
    accs = [best_p['val_acc'], best_r['val_acc']]
    bars = ax2.bar(models, accs)
    ax2.set_ylim(0, max(accs)*1.15 if accs else 1.0)
    ax2.set_ylabel('Best Validation Accuracy')
    ax2.set_title('Best Val Accuracy (min val loss)')
    for b, a in zip(bars, accs):
        ax2.text(b.get_x()+b.get_width()/2, a, f"{a:.3f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {args.out}")

if __name__ == '__main__':
    main()
