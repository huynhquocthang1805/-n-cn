"""
E29dev: chọn dạng lịch sử của đầu vào trên VALIDATION (lần lặp 0, 5 fold, 30 và 70 % nhãn; NASA thêm cách chia NASA-L).

    python analysis/dev_e29.py     # -> results/e29_dev.csv, results/e29_choice.csv

Tỉ số = MAE theo cell trên validation của mô hình có lịch sử / cùng mô hình với đặc trưng gốc, cùng fold và hạt giống
(PINN-semi thành viên 0 của E20, E24; GBDT của E20, E24). Quy tắc chọn (tai-lieu/de-cuong-E29.md): với mỗi bộ và mỗi
loại mô hình, dạng có tỉ số trung bình nhỏ nhất, chỉ giữ khi tỉ số đó <= 0,98.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import json, os
import numpy as np
import pandas as pd

from pinnsoh.experiments import E29dev, NASA_SETS
from pinnsoh.train import Config, run_name
from pinnsoh.baselines import cv_path, samba_path

OUT = 'results'
KEEP = 0.98


def val_cell(path):
    j = json.load(open(path))
    if 'val_predictions' not in j:
        return j['val']['MAE_cell']
    vp = j['val_predictions']
    y, p, c = np.asarray(vp['y']), np.asarray(vp['p']), np.asarray(vp['cell'])
    return float(np.mean([np.abs(p[c == k] - y[c == k]).mean() for k in np.unique(c)]))


def rows():
    out = []
    for j in E29dev():
        if not os.path.exists(j.out):
            continue
        a = j.args
        if j.kind == 'train':
            ds = a['dataset']; samba = a.get('split') == 'samba'
            src = 'runs/E24' if ds in NASA_SETS else 'runs/E20'
            base = os.path.join(src, run_name(Config(**{**a, 'norm': 'global', 'tag': 'S' if samba else 'P'})) + '.json')
            split = 'samba' if samba else f"cv{a['label_frac']}"
            model, spec = 'pinn_semi', a['norm']
        elif j.kind == 'baseline':
            ds = a['ds']; src = 'runs/E24' if ds in NASA_SETS else 'runs/E20'
            base = cv_path('gbdt', ds, a['frac'], a['r'], a['k'], src)
            split, model, spec = f"cv{a['frac']}", 'gbdt', a['norm']
        else:
            ds = a['ds']
            base = samba_path('gbdt', ds, a['seed'], 'runs/E24')
            split, model, spec = 'samba', 'gbdt', a['norm']
        if not os.path.exists(base):
            continue
        out.append(dict(model=model, dataset=ds, split=split, spec=spec, ratio=val_cell(j.out) / val_cell(base)))
    return pd.DataFrame(out)


def main():
    D = rows()
    if D.empty:
        print('chưa có kết quả E29dev')
        return
    R = D.groupby(['model', 'dataset', 'spec', 'split']).ratio.mean().reset_index()
    M = R.groupby(['model', 'dataset', 'spec']).ratio.mean().reset_index()
    best = M.loc[M.groupby(['model', 'dataset']).ratio.idxmin()].copy()
    best['chosen'] = np.where(best.ratio <= KEEP, best.spec, 'global')
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/e29_dev.csv', index=False); best.to_csv(f'{OUT}/e29_choice.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(R.pivot_table(index=['model', 'spec'], columns=['dataset', 'split'], values='ratio').round(3).to_string())
        print(best.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
