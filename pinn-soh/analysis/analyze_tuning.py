"""
Tinh chỉnh PINN-lai không cần huấn luyện lại: chọn trọng số ghép ω và cửa sổ bộ lọc quỹ đạo w trên tập VALIDATION.

    python analysis/analyze_tuning.py       # -> results/e20_tuned.csv

PINN-lai của E20 cố định ω = ½ (trung bình 3 PINN-semi) và w = 10 theo đề cương. Ở đây, với mỗi bộ dữ liệu và mức nhãn,
(ω, w) được chọn trên lưới ω ∈ {0,3; 0,4; 0,5; 0,6; 0,7}, w ∈ {5, 10, 20, 40} bằng MAE theo cell trung bình trên cell
VALIDATION của 3 lần lặp × 5 fold (dự đoán validation của E20); tập test chỉ dùng một lần để đánh giá. So sánh ghép cặp
theo cell (trung bình 3 lần lặp, Wilcoxon) với PINN-lai gốc và với GBDT cùng mức nhãn và ở 70 % nhãn.
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

from pinnsoh.postproc import by_cell, causal_trend

OUT = 'results'
OMEGAS = (0.3, 0.4, 0.5, 0.6, 0.7)
WINDOWS = (5, 10, 20, 40)


def per_cell(y, p, c):
    return {k: float(np.abs(p[c == k] - y[c == k]).mean()) for k in np.unique(c)}


def main():
    import analyze_e20 as A
    P, _ = A.build()
    val, test = [], []
    for (ds, f, r, k), rec in P.items():
        for part, sink in (('val', val), ('test', test)):
            t = rec[part]; y, c, cyc = t['y'], t['cell'], t['cyc']
            p3, g = t['pred']['pinn3'], t['pred']['gbdt']
            for om in OMEGAS:
                raw = om * p3 + (1 - om) * g
                for w in WINDOWS:
                    for cell, e in per_cell(y, by_cell(causal_trend, raw, c, cyc, w=w), c).items():
                        sink.append(dict(ds=ds, f=f, r=r, om=om, w=w, cell=cell, mae=e))
            if part == 'test':
                for cell, e in per_cell(y, g, c).items():
                    sink.append(dict(ds=ds, f=f, r=r, om=-1, w=-1, cell=cell, mae=e))       # GBDT
    V = pd.DataFrame(val).groupby(['ds', 'f', 'om', 'w']).mae.mean().reset_index()
    best = V.loc[V.groupby(['ds', 'f']).mae.idxmin()].set_index(['ds', 'f'])
    Tt = pd.DataFrame(test).groupby(['ds', 'f', 'om', 'w', 'cell']).mae.mean().reset_index()
    sel = lambda ds, f, om, w: Tt[(Tt.ds == ds) & (Tt.f == f) & (Tt.om == om) & (Tt.w == w)].set_index('cell').mae
    rows = []
    for (ds, f), b in best.iterrows():
        v2, v1, gb = sel(ds, f, b.om, b.w), sel(ds, f, 0.5, 10), sel(ds, f, -1, -1)
        gb70 = sel(ds, 0.7, -1, -1)
        cells = sorted(set(v2.index) & set(v1.index) & set(gb.index) & set(gb70.index))
        a, o, g, g7 = (s.reindex(cells).values for s in (v2, v1, gb, gb70))
        test_ = lambda d: stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append(dict(ds=ds, f=f, omega=b.om, w=int(b.w), val=b.mae, n_cells=len(cells), v1=o.mean(), v2=a.mean(),
                         gbdt=g.mean(), gbdt70=g7.mean(), r_v1=a.mean() / o.mean(), p_v1=test_(a - o),
                         r_gbdt=a.mean() / g.mean(), p_gbdt=test_(a - g), r_gbdt70=a.mean() / g7.mean(),
                         p_gbdt70=test_(a - g7)))
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/e20_tuned.csv', index=False)
    with pd.option_context('display.width', 200):
        print(R.assign(v1=1e3 * R.v1, v2=1e3 * R.v2, gbdt=1e3 * R.gbdt, gbdt70=1e3 * R.gbdt70)
              .drop(columns=['val']).round(4).to_string(index=False))
    print('trung bình nhân v2/v1:', round(float(np.exp(np.log(R.r_v1).mean())), 4))


if __name__ == '__main__':
    main()
