"""
E29: mô hình dùng lịch sử của cell — đánh giá cuối (tai-lieu/de-cuong-E29.md).

    python analysis/analyze_e29.py     # -> results/e29_summary.csv, e29_tests.csv, fig_e29_hist

Cùng fold, hạt giống và mức nhãn với E20 (3 lần lặp × 5 fold × 4 mức nhãn), trên các bộ có dạng lịch sử được chọn
(XJTU, TJU, HUST). Mô hình (hậu tố _h: có lịch sử):
    pinn, pinn_h          PINN-semi, thành viên 0
    mlp, mlp_h            MLP, thành viên 0 (mlp_h cùng dạng lịch sử với pinn_h)
    gbdt, gbdt_h          GBDT; gbdt_h chỉ ở XJTU và HUST — ở TJU, GBDT của bộ đã chọn là gbdt
    pinn_lai              E20: ½ trung bình 3 PINN-semi + ½ GBDT, bộ lọc xu thế w = 10
    pinn_lai_h            cùng công thức với 3 PINN-semi có lịch sử và GBDT đã chọn cho bộ
    pinn_lai_v2, _h_v2    bản tinh chỉnh: (ω, w) chọn trên validation như analyze_tuning.py, riêng cho từng (bộ, mức nhãn)
Sai số: MAE theo cell trên test, trung bình 3 lần lặp; Wilcoxon ghép cặp theo cell.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import collections, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.experiments import E29, E29_SPEC_PINN, E29_SPEC_GBDT
from pinnsoh.postproc import by_cell, causal_trend
from pinnsoh.runcache import cached_map
from analyze_tuning import OMEGAS, WINDOWS

OUT = 'results'
DS = list(E29_SPEC_PINN)
W = 10


def collect():
    """{(bộ, mức nhãn, lần lặp, fold): {'H': [3 đường dẫn], 'HM': đường dẫn, 'G': đường dẫn | None}}"""
    G = collections.defaultdict(dict)
    for j in E29():
        a = j.args
        if j.kind == 'train':
            key = (a['dataset'], a['label_frac'], a['repeat'], a['fold'])
            if a['model'] == 'pinn_semi':
                G[key].setdefault('H', [None] * 3)[a['member']] = j.out
            else:
                G[key]['HM'] = j.out
        else:
            G[(a['ds'], a['frac'], a['r'], a['k'])]['G'] = j.out
    return G


def cell_errors(y, p, cell):
    return {c: float(np.abs(p[cell == c] - y[cell == c]).mean()) for c in np.unique(cell)}


def build():
    import analyze_e20 as A
    P, _ = A.build()
    G = collect()
    paths = sorted({p for g in G.values() for v in g.values() for p in (v if isinstance(v, list) else [v])
                    if p and os.path.exists(p)})
    data = dict(zip(paths, cached_map(paths, A._both, 'e29_preds_v1')))
    rows, grid = [], []
    for key, g in sorted(G.items()):
        ds, f, r, k = key
        need = g['H'] + [g['HM']] + ([g['G']] if 'G' in g else [])
        if key not in P or not all(p in data for p in need):
            continue
        e20 = P[key]
        for part in ('test', 'val'):
            base = e20[part]
            y, cell, cyc = base['y'], base['cell'], base['cyc']
            H = [data[p][part] for p in g['H']]
            for o in H:
                assert np.array_equal(o['cell'], cell) and np.array_equal(o['cyc'], cyc), key
            pinn3_h = np.mean([o['p'] for o in H], axis=0)
            gsel = data[g['G']][part]['p'] if 'G' in g else base['pred']['gbdt']
            pred = dict(pinn=base['pred']['pinn'], pinn_h=H[0]['p'], pinn3=base['pred']['pinn3'], pinn3_h=pinn3_h,
                        gbdt=base['pred']['gbdt'], gbdt_sel=gsel, pinn_lai=base['pred']['pinn_lai'],
                        pinn_lai_h=by_cell(causal_trend, 0.5 * pinn3_h + 0.5 * gsel, cell, cyc, w=W))
            if part == 'test':
                hm = data[g['HM']]['test']
                assert np.array_equal(hm['cell'], cell) and np.array_equal(hm['cyc'], cyc), key
                pred.update(mlp=base['pred']['mlp'], mlp_h=hm['p'])
                for m, p in pred.items():
                    for c, e in cell_errors(y, p, cell).items():
                        rows.append(dict(dataset=ds, f=f, r=r, k=k, model=m, cell=c, mae=e))
            for src, (p3, gb) in dict(e20=(base['pred']['pinn3'], base['pred']['gbdt']), hist=(pinn3_h, gsel)).items():
                for om in OMEGAS:
                    for w in WINDOWS:
                        p = by_cell(causal_trend, om * p3 + (1 - om) * gb, cell, cyc, w=w)
                        for c, e in cell_errors(y, p, cell).items():
                            grid.append(dict(dataset=ds, f=f, r=r, k=k, part=part, src=src, om=om, w=w, cell=c, mae=e))
    C, Gd = pd.DataFrame(rows), pd.DataFrame(grid)
    V = Gd[Gd.part == 'val'].groupby(['dataset', 'f', 'src', 'om', 'w']).mae.mean().reset_index()
    best = V.loc[V.groupby(['dataset', 'f', 'src']).mae.idxmin()]
    Tb = Gd[Gd.part == 'test'].merge(best[['dataset', 'f', 'src', 'om', 'w']], on=['dataset', 'f', 'src', 'om', 'w'])
    Tb['model'] = Tb.src.map(dict(e20='pinn_lai_v2', hist='pinn_lai_h_v2'))
    C = pd.concat([C, Tb[['dataset', 'f', 'r', 'k', 'model', 'cell', 'mae']]], ignore_index=True)
    return C, best


def per_cell_mean(C):
    return C.groupby(['dataset', 'f', 'model', 'cell']).mae.mean().reset_index()


PAIRS = [('pinn_h', 'pinn'), ('pinn_lai_h', 'pinn_lai'), ('pinn_lai_h_v2', 'pinn_lai_v2'), ('pinn_lai_h', 'gbdt_sel'),
         ('pinn_lai_h_v2', 'gbdt_sel'), ('pinn_h', 'mlp_h'), ('pinn', 'mlp'), ('mlp_h', 'mlp'), ('gbdt_sel', 'gbdt')]


def tests(C):
    a = per_cell_mean(C)
    out = []
    for ds in DS:
        for f in sorted(a.f.unique()):
            for m, ref in PAIRS:
                if m == 'gbdt_sel' and ref == 'gbdt' and ds not in E29_SPEC_GBDT:
                    continue
                x = a[(a.dataset == ds) & (a.f == f) & (a.model == m)].set_index('cell').mae
                z = a[(a.dataset == ds) & (a.f == f) & (a.model == ref)].set_index('cell').mae
                cells = sorted(set(x.index) & set(z.index))
                if not cells:
                    continue
                d = x.reindex(cells).values - z.reindex(cells).values
                out.append(dict(dataset=ds, f=f, model=m, ref=ref, n_cells=len(cells), mae=x.reindex(cells).mean(),
                                mae_ref=z.reindex(cells).mean(), ratio=x.reindex(cells).mean() / z.reindex(cells).mean(),
                                better=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    return pd.DataFrame(out)


def summary(C):
    S = per_cell_mean(C).groupby(['dataset', 'f', 'model']).mae.mean().reset_index()
    S['spec_pinn'] = S.dataset.map(E29_SPEC_PINN)
    S['spec_gbdt'] = S.dataset.map(lambda d: E29_SPEC_GBDT.get(d, 'global'))
    return S


def figure(S):
    """Ba panel (XJTU, TJU, HUST): MAE theo mức nhãn của PINN-semi, GBDT đã chọn, PINN-lai — không / có lịch sử."""
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save
    apply_theme(base=9)
    spec = [('pinn', '#9fd8c1', (0, (4, 2)), 'PINN-semi'),
            ('pinn_h', '#1baf7a', '-', 'PINN-semi + lịch sử'),
            ('gbdt_sel', '#eb6834', '-', 'GBDT'),
            ('pinn_lai', '#9cc3e6', (0, (4, 2)), 'PINN-lai'),
            ('pinn_lai_h', '#0b3d73', '-', 'PINN-lai + lịch sử')]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.75))
    for ax, ds in zip(axes, DS):
        for m, col, ls, lab in spec:
            g = S[(S.dataset == ds) & (S.model == m)].sort_values('f')
            ax.plot(100 * g.f, 1e3 * g.mae, color=col, ls=ls, lw=1.6, marker='o', ms=3.2, label=lab)
        ax.set_xticks([10, 30, 50, 70]); ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.set_xlabel('% cell có nhãn')
        panel_title(ax, ds, pad=4)
    axes[0].set_ylabel('MAE theo cell (×10$^{-3}$)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    save(fig, f'{OUT}/fig_e29_hist')


def main():
    C, best = build()
    if C.empty:
        print('chưa có kết quả E29')
        return
    os.makedirs(OUT, exist_ok=True)
    S = summary(C); S.to_csv(f'{OUT}/e29_summary.csv', index=False)
    T = tests(C); T.to_csv(f'{OUT}/e29_tests.csv', index=False)
    best.to_csv(f'{OUT}/e29_tuned.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 400):
        print((S.assign(mae=1e3 * S.mae).pivot_table(index=['dataset', 'f'], columns='model', values='mae')).round(2))
        print(T.round(4).to_string(index=False))
    figure(S)


if __name__ == '__main__':
    main()
