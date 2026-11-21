"""
Hình tổng quan: sai số của từng mô hình khi có 30 % và 70 % số cell có nhãn (E20), cạnh chế độ đo thưa (E28).

    python analysis/analyze_overview.py     # -> results/fig_overview.pdf|png, results/overview.csv

Mỗi bộ dữ liệu một panel, mỗi mô hình một hàng; trục ngang là MAE theo cell (thang log). Vòng tròn rỗng là 30 % nhãn,
vòng tròn đặc là 70 % nhãn, nối bằng một đoạn thẳng; hình thoi là cùng mô hình học từ các lần đo thưa trên mọi cell train
(mức q nhỏ nhất của E28, khoảng vài phần trăm số chu kỳ; GBDT học nhãn nội suy giữa các lần đo). Đọc theo chiều ngang:
điểm nào nằm bên trái điểm khác là sai số thấp hơn.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import os
import numpy as np
import pandas as pd

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
MODELS = [('mlp', 'MLP'), ('pinn', 'PINN-semi'), ('gbdt', 'GBDT'), ('pinn_lai', 'PINN-lai'),
          ('pinn_lai_v2', 'PINN-lai tinh chỉnh')]
SPARSE = {'mlp': 'mlp', 'pinn': 'pinn', 'gbdt': 'gbdt_interp', 'pinn_lai': 'pinn_lai', 'pinn_lai_v2': 'pinn_lai_v2'}


def load():
    s = pd.read_csv(f'{OUT}/e20_summary.csv')
    rows = [dict(dataset=r.dataset, model=r.model, regime=f'f{r.frac}', mae=r.MAE_cell)
            for r in s.itertuples() if r.frac in (0.3, 0.7) and r.model in dict(MODELS)]
    if os.path.exists(f'{OUT}/e20_tuned.csv'):
        t = pd.read_csv(f'{OUT}/e20_tuned.csv')
        rows += [dict(dataset=r.ds, model='pinn_lai_v2', regime=f'f{r.f}', mae=r.v2) for r in t.itertuples() if r.f in (0.3, 0.7)]
    sparse = None
    if os.path.exists(f'{OUT}/e28_summary.csv'):
        e = pd.read_csv(f'{OUT}/e28_summary.csv')
        e = e[e.kind == 'do_thua']
        if len(e):
            qmin = e.regime.map(lambda x: float(x[1:])).min()
            sparse = e[e.regime == f'q{qmin}']
            for m, src in SPARSE.items():
                for r in sparse[sparse.model == src].itertuples():
                    rows.append(dict(dataset=r.dataset, model=m, regime='sparse', mae=r.mae, frac_rows=r.frac_rows))
    return pd.DataFrame(rows)


def figure(D):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save
    apply_theme()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.45), sharey=True)
    col = {'mlp': '#9aa5ad', 'pinn': '#1baf7a', 'gbdt': '#eb6834', 'pinn_lai': '#2a78d6', 'pinn_lai_v2': '#0b3d73'}
    for ax, ds in zip(axes, DS):
        for i, (m, _) in enumerate(MODELS):
            g = D[(D.dataset == ds) & (D.model == m)].set_index('regime').mae * 1e3
            if 'f0.3' in g and 'f0.7' in g:
                ax.plot([g['f0.3'], g['f0.7']], [i, i], color=col[m], lw=1.2, zorder=1)
                ax.scatter([g['f0.3']], [i], s=22, facecolor='white', edgecolor=col[m], lw=1.2, zorder=2)
                ax.scatter([g['f0.7']], [i], s=22, color=col[m], zorder=2)
            if 'sparse' in g:
                ax.scatter([g['sparse']], [i + 0.3], s=20, marker='D', color=col[m], edgecolor='white', lw=0.5, zorder=3)
        ref = D[(D.dataset == ds) & (D.model == 'pinn_lai_v2') & (D.regime == 'f0.3')].mae
        if not len(ref):
            ref = D[(D.dataset == ds) & (D.model == 'pinn_lai') & (D.regime == 'f0.3')].mae
        if len(ref):
            ax.axvline(1e3 * float(ref.iloc[0]), color='#c9d3dc', lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_xscale('log')
        lo, hi = ax.get_xlim()
        ax.set_xticks([t for t in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30) if lo <= t <= hi])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(False); ax.xaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        ax.set_ylim(-0.6, len(MODELS) - 0.2)
        panel_title(ax, ds, pad=4)
    axes[0].set_yticks(range(len(MODELS))); axes[0].set_yticklabels([l for _, l in MODELS], fontsize=7)
    hd = [plt.Line2D([], [], ls='', marker='o', mfc='white', mec='#4a5563', ms=4.5, label='30 % cell có nhãn'),
          plt.Line2D([], [], ls='', marker='o', color='#4a5563', ms=4.5, label='70 % cell có nhãn'),
          plt.Line2D([], [], ls='', marker='D', color='#4a5563', ms=4.5, label='đo thưa mọi cell')]
    fig.legend(handles=hd, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=7)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.text(0.5, 0.1, 'MAE theo cell trên test (×10$^{-3}$, thang log)', ha='center', fontsize=7.5)
    save(fig, f'{OUT}/fig_overview')


def main():
    D = load()
    D.to_csv(f'{OUT}/overview.csv', index=False)
    print((D.assign(mae=1e3 * D.mae).pivot_table(index=['dataset', 'model'], columns='regime', values='mae')).round(2))
    figure(D)


if __name__ == '__main__':
    main()
