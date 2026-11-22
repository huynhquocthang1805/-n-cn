"""
Bộ lọc quỹ đạo của PINN-lai có thật sự không cho SOH tăng không, và ép đơn điệu thật thì được hay mất gì.

    python analysis/analyze_mono.py     # -> results/mono_stats.csv, mono_filter.csv, fig_mono_example

1. Bộ lọc xu thế (postproc.causal_trend, w = 10) khớp đường thẳng có độ dốc <= 0 trên w chu kỳ gần nhất; mỗi chu kỳ dùng
   một cửa sổ khác nên chuỗi đầu ra vẫn có thể tăng. Đếm số bước tăng trên test của E20 (PINN-lai, ½ PINN + ½ GBDT).
2. Nhãn đo được có tăng không: tỉ lệ bước tăng quá 0,5 % SOH, mức vượt lên trên cực tiểu luỹ tiến trước đó, mức tăng
   trong 30 chu kỳ đầu.
3. Bộ lọc có bảo đảm (postproc.causal_mono): (w, a) chọn trên VALIDATION riêng cho từng bộ, lưới w ∈ {5, 10, 20, 40},
   a ∈ {1; 0,5; 0,3; 0,1}; so với bộ lọc xu thế w = 10 trên test, Wilcoxon ghép cặp theo cell (trung bình 3 lần lặp).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.postproc import by_cell, causal_trend, causal_mono

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
WINDOWS, ALPHAS = (5, 10, 20, 40), (1.0, 0.5, 0.3, 0.1)


def label_stats():
    from pinnsoh.baselines import cells_of
    rows = []
    for ds in DS:
        for c in cells_of(ds):
            y = c.soh
            rows.append(dict(dataset=ds, up=float((np.diff(y) > 0.005).mean()),
                             above_min=float((y - np.minimum.accumulate(y)).max()),
                             early_rise=float(y[:min(30, len(y))].max() - y[0])))
    R = pd.DataFrame(rows)
    return R.groupby('dataset').agg(label_up=('up', 'mean'), above_min_med=('above_min', 'median'),
                                    above_min_max=('above_min', 'max'), early_rise_med=('early_rise', 'median'))


def per_cell(y, p, c):
    return {k: float(np.abs(p[c == k] - y[c == k]).mean()) for k in np.unique(c)}


def main():
    import analyze_e20 as A
    P, _ = A.build()
    grid, steps, examples = [], [], {}
    for (ds, f, r, k), rec in P.items():
        for part in ('val', 'test'):
            t = rec[part]; y, c, cyc = t['y'], t['cell'], t['cyc']
            raw = 0.5 * t['pred']['pinn3'] + 0.5 * t['pred']['gbdt']
            tr = by_cell(causal_trend, raw, c, cyc, w=10)
            for cell, e in per_cell(y, tr, c).items():
                grid.append(dict(dataset=ds, f=f, r=r, k=k, part=part, kind='trend', w=10, a=0.0, cell=cell, mae=e))
            for w in WINDOWS:
                for a in ALPHAS:
                    p = by_cell(causal_mono, raw, c, cyc, w=w, a=a)
                    for cell, e in per_cell(y, p, c).items():
                        grid.append(dict(dataset=ds, f=f, r=r, k=k, part=part, kind='mono', w=w, a=a, cell=cell, mae=e))
            if part == 'test':
                for cell in np.unique(c):
                    idx = np.flatnonzero(c == cell); idx = idx[np.argsort(cyc[idx], kind='stable')]
                    d = np.diff(tr[idx])
                    steps.append(dict(dataset=ds, n=len(d), up=int((d > 1e-6).sum()), up_max=float(d.max(initial=0.0))))
                    if f == 0.3 and r == 0 and (ds, cell) not in examples:
                        examples[(ds, cell)] = (cyc[idx], y[idx], raw[idx], tr[idx])
    G = pd.DataFrame(grid)
    # chọn (w, a) của bộ lọc đơn điệu trên validation, riêng cho từng bộ (trung bình tỉ số qua các mức nhãn)
    V = G[G.part == 'val'].groupby(['dataset', 'f', 'kind', 'w', 'a']).mae.mean().reset_index()
    ref = V[V.kind == 'trend'].set_index(['dataset', 'f']).mae
    M = V[V.kind == 'mono'].copy()
    M['r'] = M.mae.values / ref.reindex(list(zip(M.dataset, M.f))).values
    best = M.groupby(['dataset', 'w', 'a']).r.mean().reset_index().sort_values('r').groupby('dataset').head(1)
    best = best.set_index('dataset')
    T = G[G.part == 'test'].groupby(['dataset', 'f', 'kind', 'w', 'a', 'cell']).mae.mean().reset_index()
    rows = []
    for ds in DS:
        b = best.loc[ds]
        for f in sorted(T.f.unique()):
            x = T[(T.dataset == ds) & (T.f == f) & (T.kind == 'mono') & (T.w == b.w) & (T.a == b.a)].set_index('cell').mae
            z = T[(T.dataset == ds) & (T.f == f) & (T.kind == 'trend')].set_index('cell').mae
            cells = sorted(set(x.index) & set(z.index))
            d = x.reindex(cells).values - z.reindex(cells).values
            rows.append(dict(dataset=ds, f=f, w=int(b.w), a=float(b.a), mono=x.reindex(cells).mean(),
                             trend=z.reindex(cells).mean(), ratio=x.reindex(cells).mean() / z.reindex(cells).mean(),
                             p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    F = pd.DataFrame(rows)
    S = pd.DataFrame(steps).groupby('dataset').agg(n=('n', 'sum'), up=('up', 'sum'), up_max=('up_max', 'max'))
    S['frac_up'] = S.up / S.n
    S = S.join(label_stats(), how='outer').reset_index()
    os.makedirs(OUT, exist_ok=True)
    S.to_csv(f'{OUT}/mono_stats.csv', index=False); F.to_csv(f'{OUT}/mono_filter.csv', index=False)
    with pd.option_context('display.width', 200):
        print(S.round(4).to_string(index=False)); print(F.round(4).to_string(index=False))
    figure(examples, best)


def figure(examples, best):
    """Một cell XJTU có dung lượng tăng đầu đời và một cell TJU suy giảm đều (30 % nhãn, lần lặp 0): nhãn, dự đoán
    chưa lọc, hai bộ lọc."""
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save
    apply_theme(base=9)

    def pick(ds):
        """XJTU: cell có dung lượng tăng đầu đời rõ nhất trong số cell mà dự đoán 5 chu kỳ đầu lệch không quá 0,03
        (để không lẫn lỗi dự đoán đầu đời với tác động của bộ lọc); TJU: cell dài nhất."""
        best_cell, score = None, -np.inf
        for (d, cell), (cyc, y, raw, tr) in examples.items():
            if d != ds or len(y) < 80:
                continue
            if ds == 'XJTU':
                if abs(raw[:5].mean() - y[:5].mean()) > 0.03:
                    continue
                s = y[:40].max() - y[0]
            else:
                s = len(y)
            if s > score:
                best_cell, score = (d, cell), s
        return best_cell
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for ax, ds in zip(axes, ('XJTU', 'TJU')):
        key = pick(ds)
        cyc, y, raw, tr = examples[key]
        b = best.loc[ds]
        mono = causal_mono(raw, cyc, int(b.w), float(b.a))
        ax.plot(cyc, y, color='#3f4a52', lw=1.6, label='SOH đo được')
        ax.plot(cyc, raw, color='#c3cbd1', lw=0.8, label='dự đoán chưa lọc')
        ax.plot(cyc, tr, color='#2a78d6', lw=1.3, label='lọc xu thế (hiện tại)')
        ax.plot(cyc, mono, color='#eb6834', lw=1.3, ls=(0, (4, 2)), label='lọc đơn điệu (có bảo đảm)')
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5)
        ax.set_xlabel('chu kỳ'); panel_title(ax, f'{ds}, một cell test', pad=4)
    axes[0].set_ylabel('SOH')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    save(fig, f'{OUT}/fig_mono_example')


if __name__ == '__main__':
    main()
