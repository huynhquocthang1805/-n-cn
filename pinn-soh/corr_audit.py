"""
Tính lại bảng tương quan đặc trưng–SOH, và so ba hệ số tương quan khác nhau.

    python corr_audit.py

Sinh:
  results/corr_table.csv   — trung vị Spearman trong từng cell, mọi đặc trưng × mọi bộ
  results/corr_compare.csv — Pearson / Spearman / Kendall cạnh nhau, để trả lời
                             câu "vì sao dùng Spearman"
  results/corr_percell.csv — phân bố theo từng cell (min/q1/median/q3/max) cho các
                             đặc trưng được nhắc trong báo cáo

Quy trình đúng như mô tả trong báo cáo: tương quan tính TRONG TỪNG CELL (không gộp
cell), rồi lấy TRUNG VỊ trên các cell của bộ đó. Gộp cell sẽ trộn lẫn biến thiên
giữa-cell với biến thiên trong-cell và cho ra con số vô nghĩa.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
from pinnsoh.data import load_dataset, FEATURES

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
SHOWN = ['CC Q', 'CC charge time', 'CV Q', 'CV charge time', 'voltage mean', 'current kurtosis']


def per_cell_corr(cells, method):
    """Trả về ma trận (n_cell, 16) hệ số tương quan giữa từng đặc trưng và SOH."""
    fn = {'spearman': stats.spearmanr, 'pearson': stats.pearsonr, 'kendall': stats.kendalltau}[method]
    out = []
    for c in cells:
        if len(c.soh) < 10:
            continue
        row = []
        for j in range(len(FEATURES)):
            x = c.X[:, j]
            if np.std(x) < 1e-12 or np.std(c.soh) < 1e-12:
                row.append(np.nan); continue
            row.append(float(fn(x, c.soh)[0]))
        out.append(row)
    return np.array(out, dtype=float)


def main():
    med = {}          # (bộ, phương pháp) -> vector 16 trung vị
    percell = []
    ncells = {}
    for ds in DATASETS:
        cells = load_dataset('PINN4SOH/data', ds)
        ncells[ds] = len(cells)
        for method in ['spearman', 'pearson', 'kendall']:
            C = per_cell_corr(cells, method)
            med[(ds, method)] = np.nanmedian(C, axis=0)
            if method == 'spearman':
                for j, f in enumerate(FEATURES):
                    v = C[:, j][np.isfinite(C[:, j])]
                    percell.append(dict(dataset=ds, feature=f, n_cell=len(v),
                                        min=v.min(), q1=np.percentile(v, 25),
                                        median=np.median(v), q3=np.percentile(v, 75),
                                        max=v.max(),
                                        frac_positive=float((v > 0).mean())))
        print(f'{ds}: {len(cells)} cell')

    # bảng chính — trung vị Spearman
    t = pd.DataFrame({ds: med[(ds, 'spearman')] for ds in DATASETS}, index=FEATURES).round(2)
    t.index.name = 'feature'
    t.to_csv('results/corr_table.csv')
    print('\n=== TRUNG VỊ SPEARMAN TRONG TỪNG CELL ===')
    print(t.to_string())

    # so ba hệ số cho các đặc trưng được trích trong báo cáo
    rows = []
    for f in SHOWN:
        j = FEATURES.index(f)
        for ds in DATASETS:
            rows.append(dict(feature=f, dataset=ds,
                             pearson=round(med[(ds, 'pearson')][j], 2),
                             spearman=round(med[(ds, 'spearman')][j], 2),
                             kendall=round(med[(ds, 'kendall')][j], 2)))
    cmp = pd.DataFrame(rows)
    cmp.to_csv('results/corr_compare.csv', index=False)
    print('\n=== PEARSON so SPEARMAN so KENDALL (trung vị trong từng cell) ===')
    print(cmp.pivot(index='feature', columns='dataset',
                    values=['pearson', 'spearman', 'kendall']).to_string())

    pc = pd.DataFrame(percell)
    pc.to_csv('results/corr_percell.csv', index=False)
    print('\n=== PHÂN BỐ THEO CELL, các đặc trưng được trích ===')
    print(pc[pc.feature.isin(SHOWN)].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
