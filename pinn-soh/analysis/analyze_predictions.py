"""
Chỉ số mở rộng và hình dự đoán SOH, tính lại từ dự đoán test đã lưu của E13.

    python analysis/analyze_predictions.py     # -> results/pred_metrics.csv|md, results/fig_pred_*.png|pdf

Mô hình: MLP và PINN-semi ở 30 % và 70 % nhãn. Mỗi cell được dự đoán 5 lần (5 lần lặp kiểm định
chéo); bảng chỉ số gộp toàn bộ, hình dùng lần lặp 0 (mỗi cell xuất hiện đúng một lần).
Hình không mang chữ chú thích.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import glob, json, os
import numpy as np
import pandas as pd

RUNS, OUT = 'runs/E13', 'results'
DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
ARMS = {'A': ('mlp', 0.3, 'MLP 30 %'), 'C': ('pinn_semi', 0.3, 'PINN-semi 30 %'),
        'F': ('mlp', 0.7, 'MLP 70 %'), 'G': ('pinn_semi', 0.7, 'PINN-semi 70 %')}


def load():
    rows = []
    for arm, (model, frac, _) in ARMS.items():
        for f in sorted(glob.glob(f'{RUNS}/{arm}_*_{model}_f{frac}_*.json')):
            r = json.load(open(f)); c = r['config']; tp = r['test_predictions']
            ids = tp['cell_ids']
            rows.append(pd.DataFrame(dict(arm=arm, dataset=c['dataset'], repeat=c['repeat'], fold=c['fold'],
                                          cell=[ids[k] for k in tp['cell']], cycle=tp['cycle'],
                                          y=tp['y'], p=tp['p'])))
    return pd.concat(rows, ignore_index=True)


def pick_cell(r0, ds):
    """Cell có MAE của MLP 30 % ở trung vị của bộ (lần lặp 0), bỏ qua cell có nhãn nhảy quá 0,02 giữa hai chu kỳ
    liền nhau (lỗi đo đơn lẻ làm hình khó đọc). Không chọn theo sai số của PINN."""
    g = r0[(r0.dataset == ds) & (r0.arm == 'A')]
    clean = [c for c, h in g.groupby('cell') if np.abs(np.diff(h.sort_values('cycle').y.values)).max() <= 0.02]
    g = g[g.cell.isin(clean)]
    err = g.groupby('cell').apply(lambda h: np.abs(h.p - h.y).mean(), include_groups=False).sort_values()
    return err.index[(len(err) - 1) // 2]


def load_levels():
    """Dự đoán test lần lặp 0 của MLP và PINN-semi ở 10, 30, 70 % nhãn (cho hình quỹ đạo)."""
    pref = {('mlp', 0.1): 'L', ('pinn_semi', 0.1): 'L', ('mlp', 0.3): 'A', ('pinn_semi', 0.3): 'C',
            ('mlp', 0.7): 'F', ('pinn_semi', 0.7): 'G'}
    rows = []
    for (model, frac), arm in pref.items():
        for f in sorted(glob.glob(f'{RUNS}/{arm}_*_{model}_f{frac}_*cv0-*.json')):
            r = json.load(open(f)); c = r['config']; tp = r['test_predictions']; ids = tp['cell_ids']
            rows.append(pd.DataFrame(dict(model=model, frac=frac, dataset=c['dataset'], cell=[ids[k] for k in tp['cell']],
                                          cycle=tp['cycle'], y=tp['y'], p=tp['p'])))
    return pd.concat(rows, ignore_index=True)


def metrics(y, p):
    e = p - y
    ss_res = (e ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                MAPE=100 * np.abs(e / y).mean(), R2=1 - ss_res / ss_tot, MaxAE=np.abs(e).max(),
                Bias=e.mean())


def table(df):
    out = []
    for (ds, arm), g in df.groupby(['dataset', 'arm']):
        m = metrics(g.y.values, g.p.values)
        g10 = g[g.cycle >= 10]
        m10 = metrics(g10.y.values, g10.p.values)
        cell = g.groupby(['cell', 'repeat']).apply(lambda h: metrics(h.y.values, h.p.values), include_groups=False)
        cell = pd.DataFrame(list(cell))
        out.append(dict(bo=ds, mo_hinh=ARMS[arm][2], arm=arm, n_chu_ky=len(g) // g.repeat.nunique(),
                        MAE=m['MAE'], RMSE=m['RMSE'], MAPE=m['MAPE'], R2=m['R2'], MaxAE=m['MaxAE'],
                        Bias=m['Bias'], R2_tu10=m10['R2'], RMSE_tu10=m10['RMSE'], MAE_cell=cell.MAE.mean(), RMSE_cell=cell.RMSE.mean(),
                        R2_cell_median=cell.R2.median()))
    t = pd.DataFrame(out)
    t['bo'] = pd.Categorical(t.bo, DATASETS); t['arm'] = pd.Categorical(t.arm, list(ARMS))
    return t.sort_values(['bo', 'arm']).reset_index(drop=True)


def to_md(t, path):
    head = ['Bộ', 'Mô hình', 'MAE (%)', 'RMSE (%)', 'MAPE (%)', 'R²', 'R² từ chu kỳ 10', 'MaxAE (%)',
            'MAE theo cell (%)', 'R² từng cell (trung vị)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for _, r in t.iterrows():
        v = [str(r.bo), r.mo_hinh, f'{100 * r.MAE:.2f}', f'{100 * r.RMSE:.2f}', f'{r.MAPE:.2f}', f'{r.R2:.3f}',
             f'{r.R2_tu10:.3f}', f'{100 * r.MaxAE:.1f}', f'{100 * r.MAE_cell:.2f}', f'{r.R2_cell_median:.3f}']
        lines.append('| ' + ' | '.join(v).replace('.', ',') + ' |')
    open(path, 'w').write('\n'.join(lines) + '\n')


# ───────────────────────────────────────────────────────────── hình
def figures(df):
    import matplotlib, matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, MODEL_COLOR, GRID, CHEM, panel_title, save
    apply_theme()
    r0 = df[df.repeat == 0]
    COL = {'A': MODEL_COLOR['mlp'], 'C': MODEL_COLOR['pinn_semi']}

    # 1. dự đoán so với thực đo (parity), hàng trên MLP, hàng dưới PINN-semi
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for j, ds in enumerate(DATASETS):
        g = r0[r0.dataset == ds]
        lo, hi = g.y.min() - 0.02, g.y.max() + 0.02        # khung theo dải SOH thực đo
        for i, arm in enumerate(['A', 'C']):
            ax = axes[i, j]; h = g[g.arm == arm]
            ax.plot([lo, hi], [lo, hi], color='#9aa5ad', lw=0.8, zorder=1)
            ax.scatter(h.y, h.p, s=1.2, color=COL[arm], alpha=0.18, lw=0, rasterized=True, zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
            ax.grid(True, color=GRID, lw=0.5); ax.locator_params(nbins=4); ax.tick_params(labelsize=6.5)
            if i == 0:
                panel_title(ax, ds, CHEM[ds], pad=4)
            if j == 0:
                ax.set_ylabel('SOH dự đoán\n' + ('MLP' if arm == 'A' else 'PINN-semi'), fontsize=7.5)
    fig.supxlabel('SOH thực đo', fontsize=8.5, y=0.02)
    fig.tight_layout()
    save(fig, f'{OUT}/fig_pred_parity')

    # 2. quỹ đạo SOH của cùng một cell test ở 10, 30 và 70 % nhãn. Cell được chọn theo sai số của MLP ở 30 %
    #    (trung vị trong bộ, lần lặp 0) — không chọn theo PINN.
    lv = load_levels()
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.6), sharey='row')
    for i, ds in enumerate(DATASETS):
        cid = pick_cell(r0, ds)
        for j, frac in enumerate((0.1, 0.3, 0.7)):
            ax = axes[i, j]
            h = lv[(lv.dataset == ds) & (lv.frac == frac) & (lv.cell == cid)]
            t = h[h.model == 'mlp'].sort_values('cycle')
            ax.plot(t.cycle, t.y, color=MODEL_COLOR['truth'], lw=1.7, zorder=2)
            for m in ('mlp', 'pinn_semi'):
                q = h[h.model == m].sort_values('cycle')
                ax.plot(q.cycle, q.p, color=MODEL_COLOR[m], lw=0.85, alpha=0.75 if m == 'mlp' else 0.95, zorder=3)
            panel_title(ax, ds, f'{int(round(100 * frac))} % nhãn', pad=3)
            lo, hi = t.y.min(), t.y.max(); pad = max(0.03, 0.25 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
            ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=4)
            ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
            if j == 0:
                ax.set_ylabel('SOH')
    fig.legend(handles=[plt.Line2D([], [], color=MODEL_COLOR['truth'], lw=1.6, label='thực đo'),
                        plt.Line2D([], [], color=MODEL_COLOR['mlp'], lw=1.4, label='MLP'),
                        plt.Line2D([], [], color=MODEL_COLOR['pinn_semi'], lw=1.4, label='PINN-semi')],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.text(0.5, 0.035, 'chu kỳ', ha='center', fontsize=8.5)
    save(fig, f'{OUT}/fig_pred_traj')

    # 3. phân bố sai số (dự đoán − thực đo)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=False)
    for j, ds in enumerate(DATASETS):
        ax = axes[j]; g = r0[r0.dataset == ds]
        e_all = 100 * (g.p - g.y)
        lim = np.percentile(np.abs(e_all), 99)
        bins = np.linspace(-lim, lim, 50)
        for arm in ['A', 'C']:
            e = 100 * (g[g.arm == arm].p - g[g.arm == arm].y)
            ax.hist(e, bins=bins, histtype='step', lw=1.3, color=COL[arm], density=True)
        ax.axvline(0, color='#9aa5ad', lw=0.8)
        panel_title(ax, ds, CHEM[ds], pad=4); ax.set_yticks([])
        ax.spines['left'].set_visible(False); ax.grid(False); ax.tick_params(labelsize=6.5)
    axes[1].set_xlabel('sai số dự đoán (điểm % SOH)', x=1.05)
    fig.legend(handles=[plt.Line2D([], [], color=COL['A'], lw=1.4, label='MLP 30 %'),
                        plt.Line2D([], [], color=COL['C'], lw=1.4, label='PINN-semi 30 %')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, f'{OUT}/fig_pred_error_hist')

    # 4. sai số theo mức SOH
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    edges = np.array([0.0, 0.80, 0.85, 0.90, 0.95, 1.00, 2.0])
    labels = ['<0,80', '0,80–0,85', '0,85–0,90', '0,90–0,95', '0,95–1,00', '>1,00']
    for j, ds in enumerate(DATASETS):
        ax = axes[j]; g = df[df.dataset == ds]
        for k, arm in enumerate(['A', 'C']):
            h = g[g.arm == arm]
            b = np.digitize(h.y, edges) - 1
            mae = [100 * np.abs(h.p - h.y)[b == q].mean() if (b == q).sum() > 50 else np.nan for q in range(len(labels))]
            ax.plot(range(len(labels)), mae, marker='o', ms=3, lw=1.5, color=COL[arm])
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, fontsize=6)
        panel_title(ax, ds, CHEM[ds], pad=4); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.set_ylim(bottom=0); ax.tick_params(axis='y', labelsize=6.5)
        if j == 0:
            ax.set_ylabel('MAE (điểm % SOH)')
    axes[1].set_xlabel('SOH thực đo', x=1.05)
    fig.legend(handles=[plt.Line2D([], [], color=COL['A'], lw=1.4, marker='o', ms=3, label='MLP 30 %'),
                        plt.Line2D([], [], color=COL['C'], lw=1.4, marker='o', ms=3, label='PINN-semi 30 %')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    save(fig, f'{OUT}/fig_pred_error_soh')

    # 5. R² theo bộ dữ liệu và mô hình: mọi chu kỳ | từ chu kỳ 10 (bộ lọc đã đủ lịch sử)
    t = table(df)
    cols = {'A': COL['A'], 'C': COL['C'], 'F': '#8fb6e8', 'G': '#8fd6b8'}
    mk = {'A': 'o', 'C': 'o', 'F': 's', 'G': 's'}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, col, title in [(axes[0], 'R2', 'mọi chu kỳ'), (axes[1], 'R2_tu10', 'từ chu kỳ 10')]:
        for k, arm in enumerate(['A', 'C', 'F', 'G']):
            v = [t[(t.bo == ds) & (t.arm == arm)][col].iloc[0] for ds in DATASETS]
            ax.plot(np.arange(4) + (k - 1.5) * 0.12, v, ls='none', marker=mk[arm], ms=5, color=cols[arm],
                    mec='white', mew=0.6)
        ax.set_xticks(range(4)); ax.set_xticklabels(DATASETS); ax.set_xlim(-0.5, 3.5)
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, title, pad=4)
    axes[0].set_ylabel('R²'); axes[0].set_ylim(0.35, 1.0)
    fig.legend(handles=[plt.Line2D([], [], ls='none', marker=mk[a], ms=5, color=cols[a], label=ARMS[a][2])
                        for a in ['A', 'C', 'F', 'G']], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, f'{OUT}/fig_pred_r2')


def main():
    df = load()
    print(f'{len(df):,} dự đoán từ {df.groupby(["arm", "dataset", "repeat", "fold"]).ngroups} lượt')
    t = table(df)
    t.to_csv(f'{OUT}/pred_metrics.csv', index=False)
    to_md(t, f'{OUT}/pred_metrics.md')
    print(open(f'{OUT}/pred_metrics.md').read())
    figures(df)
    print('đã ghi results/pred_metrics.*, fig_pred_{parity,traj,error_hist,error_soh,r2}')


if __name__ == '__main__':
    main()
