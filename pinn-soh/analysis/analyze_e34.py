"""
E34: HiGBDT cho bài toán cập nhật — bảng, kiểm định, đối chiếu với lần chạy trước ở kho G-doan, hình đánh đổi giữa số
lần đo và sai số.

    python analysis/analyze_e34.py      # -> results/e34_anchor.csv, e34_cost.csv, e34_repro.csv, fig_e34_cost.pdf|png

Sai số của một cell (như analyze.py của G-doan): MAE trên các chu kỳ được chấm của lịch đo; trung bình qua cell mỗi lần
lặp, rồi trung bình ± SD qua 3 lần lặp. Kiểm định: Wilcoxon ghép cặp theo cell (sai số trung bình qua lần lặp) so với
GBDT-cập nhật, không hiệu chỉnh. Khi runs/E34 chưa đủ 60 lượt, dùng kết quả đã lưu của kho G-doan (biến GDOAN).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.update_hist import ROOT, TIERS
from analyze_e33 import a_wins, flip_k

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
GDOAN = os.environ.get('GDOAN', '/home/user/huynhquocthang1805/g-doan')
TL = list(TIERS)
MODELS = ['hold', 'linear', 'e32_style', 'history_delta', 'history_residual', 'selected']
LABEL = {'hold': 'Giữ lần đo', 'linear': 'Tuyến tính', 'e32_style': 'GBDT-cập nhật', 'history_delta': 'HiGBDT-Δ',
         'history_residual': 'HiGBDT-R', 'selected': 'HiGBDT-V'}
N_RUNS = 60


def files(source):
    if source == 'E34':
        return sorted(glob.glob(f'{ROOT}/up_*_cv[0-2]-[0-4]of5.npz'))
    return sorted(glob.glob(f'{GDOAN}/improvements/anchors/*_r[0-2]_f[0-4].npz'))


def cells(fs):
    """Sai số theo (bộ, lần lặp, lịch đo, cell) và số lần đo của cell ở lịch đó."""
    rows = []
    for f in fs:
        q = np.load(f); i = json.loads(str(q['info']))
        d = pd.DataFrame(q['data'], columns=i['columns'])
        for (ci, ti), g in d.groupby(['cell', 'tier']):
            rows.append(dict(ds=i['dataset'], r=i['repeat'], tier=TL[int(ti)], cid=i['split']['test'][int(ci)],
                             n_measured=float(g.n_measured.iloc[0]), n_scored=len(g),
                             **{m: float(np.abs(g[m] - g.y).mean()) for m in MODELS}))
    return pd.DataFrame(rows)


def summary(A):
    rep = A.groupby(['ds', 'tier', 'r'])[MODELS].mean()
    S = rep.groupby(['ds', 'tier']).agg(['mean', 'std'])
    S.columns = [f'{a}' if b == 'mean' else f'{a}_sd' for a, b in S.columns]
    S = S.reset_index()
    pc = A.groupby(['ds', 'tier', 'cid'])[MODELS].mean().reset_index()
    for c in ('history_delta', 'history_residual', 'selected'):
        S[f'p_{c}'] = [stats.wilcoxon(g[c] - g['e32_style']).pvalue for _, g in pc.groupby(['ds', 'tier'])]
        S[f'better_{c}'] = [int((g[c] < g['e32_style']).sum()) for _, g in pc.groupby(['ds', 'tier'])]
        S[f'wins_{c}'] = [a_wins(g[c] - g['e32_style']) for _, g in pc.groupby(['ds', 'tier'])]
        S[f'flip_{c}'] = [flip_k(g[c] - g['e32_style'])[0] for _, g in pc.groupby(['ds', 'tier'])]
    S['n_cell'] = [len(g) for _, g in pc.groupby(['ds', 'tier'])]
    return S


def cost(A):
    """Số lần đo trung bình mỗi cell và MAE của từng mô hình, theo (bộ, lịch đo)."""
    C = A.groupby(['ds', 'tier', 'r']).agg(n_measured=('n_measured', 'mean'), n_scored=('n_scored', 'mean'),
                                           **{m: (m, 'mean') for m in MODELS})
    return C.groupby(['ds', 'tier']).mean().reset_index()


def fig_cost(C):
    """MAE theo số lần đo dung lượng trung bình mỗi cell, bốn lịch đo, mỗi bộ một panel (thang log hai trục)."""
    import figures_report as FR
    plt = FR.setup()
    from pinnsoh.plotstyle import panel_title, save, CHEM, GRID
    style = [('hold', '#9aa5ad', (0, (4, 2)), 'v'), ('e32_style', '#2a78d6', (0, (3, 1.5)), 's'),
             ('selected', '#d9480f', '-', 'o')]
    fig, axes = FR._grid(plt, 4.6)
    for ax, ds in zip(axes, DS):
        g = C[C.ds == ds].set_index('tier').reindex(TL)
        for m, col, ls, mk in style:
            ax.plot(g.n_measured, 1e3 * g[m], color=col, ls=ls, marker=mk, ms=3.8, lw=1.3 if m == 'selected' else 1.0,
                    label=LABEL[m], zorder=3 if m == 'selected' else 2)
        ax.set_xscale('log'); ax.set_yscale('log')
        for axis, lim, cand in ((ax.xaxis, ax.get_xlim(), (3, 10, 30, 100, 300, 1000, 3000)),
                                (ax.yaxis, ax.get_ylim(), (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50))):
            axis.set_major_locator(plt.FixedLocator([t for t in cand if lim[0] <= t <= lim[1]]))
            axis.set_minor_locator(plt.NullLocator())
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        panel_title(ax, f'{ds} ({CHEM[ds]})', pad=4)
    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.set_ylabel('MAE (×10$^{-3}$)')
        if i >= 2:
            ax.set_xlabel('số lần đo dung lượng trung bình mỗi cell')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=1.2, w_pad=1.2)
    save(fig, f'{OUT}/fig_e34_cost')


def main():
    os.makedirs(OUT, exist_ok=True)
    own = files('E34')
    src = 'E34' if len(own) >= N_RUNS else 'G-doan'
    fs = own if src == 'E34' else files('G-doan')
    if not fs:
        print('chưa có kết quả E34 (python run.py E34) và không tìm thấy kết quả đã lưu của kho G-doan')
        return None
    A = cells(fs)
    S = summary(A); S['source'] = src
    S.to_csv(f'{OUT}/e34_anchor.csv', index=False)
    C = cost(A); C['source'] = src
    C.to_csv(f'{OUT}/e34_cost.csv', index=False)
    fig_cost(C)
    old = files('G-doan')
    if src == 'E34' and old:                          # đối chiếu với lần chạy trước ở kho G-doan, cùng cell và lịch đo
        B = cells(old)
        j = A.merge(B, on=['ds', 'r', 'tier', 'cid'], suffixes=('', '_g'))
        R = j.groupby('ds').apply(lambda x: pd.Series(dict(
            n=len(x), **{f'corr_{m}': np.corrcoef(x[m], x[f'{m}_g'])[0, 1] for m in ('e32_style', 'selected')},
            **{f'mae_{m}': x[m].mean() for m in ('e32_style', 'selected')},
            **{f'mae_{m}_g': x[f'{m}_g'].mean() for m in ('e32_style', 'selected')},
            same_hold=float(np.allclose(x.hold, x.hold_g)))), include_groups=False).reset_index()
        R.to_csv(f'{OUT}/e34_repro.csv', index=False)
    pd.set_option('display.width', 200)
    show = S[['ds', 'tier'] + MODELS + ['p_selected', 'better_selected', 'n_cell']].copy()
    for m in MODELS:
        show[m] = (1e3 * show[m]).round(2)
    print(f'nguồn: {src} ({len(fs)} lượt)')
    print(show.to_string(index=False))
    return S


if __name__ == '__main__':
    main()
