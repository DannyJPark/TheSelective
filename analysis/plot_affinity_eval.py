"""Plot scatter (head output vs experimental pK) for each test set / head,
and print a markdown summary table. Reads outputs of eval_affinity_pdbbind.py."""
import os, csv, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def read_csv(path):
    pid, lab, h1, h2 = [], [], [], []
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            pid.append(row[0]); lab.append(float(row[1]))
            h1.append(float(row[2])); h2.append(float(row[3]))
    return pid, np.array(lab), np.array(h1), np.array(h2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='./analysis/affinity_eval')
    args = ap.parse_args()
    sets = ['CASF-2013', 'CASF-2016', 'test2019']
    sets = [s for s in sets if os.path.exists(os.path.join(args.dir, f'{s}_predictions.csv'))]

    fig, axes = plt.subplots(2, len(sets), figsize=(5 * len(sets), 9))
    if len(sets) == 1:
        axes = axes.reshape(2, 1)
    for j, s in enumerate(sets):
        _, lab, h1, h2 = read_csv(os.path.join(args.dir, f'{s}_predictions.csv'))
        for i, (h, name) in enumerate([(h1, 'Head1'), (h2, 'Head2')]):
            ax = axes[i, j]
            ax.scatter(h, lab, s=10, alpha=0.4, edgecolors='none')
            pr = pearsonr(h, lab)[0]; sr = spearmanr(h, lab)[0]
            a, b = np.polyfit(h, lab, 1)
            xs = np.linspace(h.min(), h.max(), 50)
            ax.plot(xs, a * xs + b, 'r-', lw=1.2)
            ax.set_title(f'{s} | {name}\nPearson={pr:.3f}  Spearman={sr:.3f}  n={len(lab)}')
            ax.set_xlabel('Head output (Vina-normalized, 1=strong)')
            ax.set_ylabel('Experimental pK (-logKd/Ki)')
    plt.tight_layout()
    out = os.path.join(args.dir, 'affinity_scatter.png')
    plt.savefig(out, dpi=130)
    print('[INFO] saved', out)

    # markdown table
    summ = json.load(open(os.path.join(args.dir, 'summary.json')))
    print('\n| Test set | N | Head | Pearson | Spearman | Kendall | RMSE* |')
    print('|---|---|---|---|---|---|---|')
    for s in sets:
        if s not in summ:
            continue
        for hk in ('head1', 'head2'):
            m = summ[s][hk]
            print(f'| {s} | {summ[s]["n"]} | {hk[-1]} | {m["pearson"]:.4f} | '
                  f'{m["spearman"]:.4f} | {m["kendall"]:.4f} | {m["rmse"]:.3f} |')


if __name__ == '__main__':
    main()
