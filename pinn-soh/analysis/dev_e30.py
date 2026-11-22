"""
E30dev: chọn độ dài cửa sổ L của CNN, LSTM, Transformer trên VALIDATION (lần lặp 0, 30 và 70 % nhãn, 5 fold, bốn bộ).

    python analysis/dev_e30.py     # -> results/e30_dev.csv

Quy tắc (docs/de-cuong-E30.md): với mỗi mô hình, tỉ số MAE theo cell trên validation của L = 48 so với L = 16, trung
bình theo fold rồi qua bộ và mức nhãn; chọn L = 48 nếu tỉ số < 1, ngược lại L = 16. Một giá trị L cho mỗi mô hình.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import json, os
import numpy as np
import pandas as pd

from pinnsoh.experiments import E30dev, E30_DEV_L

OUT = 'results'


def val_cell(path):
    vp = json.load(open(path))['val_predictions']
    y, p, c = np.asarray(vp['y']), np.asarray(vp['p']), np.asarray(vp['cell'])
    return float(np.mean([np.abs(p[c == k] - y[c == k]).mean() for k in np.unique(c)]))


def main():
    rows = []
    for j in E30dev():
        if os.path.exists(j.out):
            a = j.args
            rows.append(dict(model=a['model'], L=a['L'], dataset=a['ds'], f=a['frac'], k=a['k'], val=val_cell(j.out),
                             test=json.load(open(j.out))['test']['MAE_cell']))
    D = pd.DataFrame(rows)
    lo, hi = E30_DEV_L
    P = D.pivot_table(index=['model', 'dataset', 'f', 'k'], columns='L', values='val').dropna()
    P['ratio'] = P[hi] / P[lo]
    R = P.groupby(['model', 'dataset', 'f']).ratio.mean().reset_index()
    C = R.groupby('model').ratio.mean().reset_index()
    C['chosen'] = np.where(C.ratio < 1, hi, lo)
    C['n_runs'] = D.groupby('model').size().reindex(C.model).values
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/e30_dev_detail.csv', index=False)
    C.to_csv(f'{OUT}/e30_dev.csv', index=False)
    with pd.option_context('display.width', 200):
        print(R.pivot_table(index='model', columns=['dataset', 'f'], values='ratio').round(3).to_string())
        print(C.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
