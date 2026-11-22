"""
Phân rã sai số theo cell của các mô hình đã báo cáo (E13, E16): lệch mức, trôi chậm, dao động nhanh.

    python analysis/error_decomp.py     # -> results/e20_error_decomp.csv

Với mỗi cell test, sai số e_n = dự đoán − thực đo theo thứ tự chu kỳ được tách thành
    b       = trung bình e_n                         (lệch mức của cả cell)
    s_n     = trung bình trượt CĂN GIỮA 21 điểm của e_n − b   (trôi chậm theo vòng đời)
    f_n     = e_n − b − s_n                          (dao động giữa các chu kỳ liền nhau)
và MSE của cell = b² + mean(s²) + mean(f²) + 2·mean(s·f) (số hạng chéo nhỏ, ghi riêng).
Đây là phân tích chẩn đoán trên kết quả đã công bố, không dùng để chọn mô hình.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import glob, json
import numpy as np
import pandas as pd

from pinnsoh.runcache import cached_map

DS = ['XJTU', 'TJU', 'MIT', 'HUST']
W = 21


def _cells(tp):
    y, p, cell, cyc = (np.asarray(tp[k]) for k in ('y', 'p', 'cell', 'cycle'))
    out = []
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        idx = idx[np.argsort(cyc[idx], kind='stable')]
        e = p[idx] - y[idx]
        b = e.mean()
        r = e - b
        k = min(W, len(r) - (1 - len(r) % 2)) if len(r) > 2 else 1
        s = np.convolve(np.pad(r, (k // 2, k // 2), mode='edge'), np.ones(k) / k, mode='valid') if k > 1 else np.zeros_like(r)
        f = r - s
        out.append((b * b, float((s * s).mean()), float((f * f).mean()), float((e * e).mean()), float(np.abs(e).mean())))
    return out


def _parse(path):
    d = json.load(open(path))
    return _cells(d['test_predictions'])


def main():
    sel = {('mlp', 'A', 0.3): 'runs/E13/A_{ds}_mlp_f0.3_*.json', ('pinn_semi', 'C', 0.3): 'runs/E13/C_{ds}_pinn_semi_f0.3_*.json',
           ('mlp', 'L', 0.1): 'runs/E13/L_{ds}_mlp_f0.1_*.json', ('pinn_semi', 'L', 0.1): 'runs/E13/L_{ds}_pinn_semi_f0.1_*.json',
           ('mlp', 'F', 0.7): 'runs/E13/F_{ds}_mlp_f0.7_*.json', ('pinn_semi', 'G', 0.7): 'runs/E13/G_{ds}_pinn_semi_f0.7_*.json',
           ('gbdt', 'K', 0.1): 'runs/E16/gbdt_{ds}_f0.1_*.json', ('gbdt', 'K', 0.3): 'runs/E16/gbdt_{ds}_f0.3_*.json',
           ('gbdt', 'K', 0.7): 'runs/E16/gbdt_{ds}_f0.7_*.json'}
    groups = {(m, frac, ds): sorted(glob.glob(pat.format(ds=ds))) for (m, _, frac), pat in sel.items() for ds in DS}
    allf = [f for fs in groups.values() for f in fs]
    parsed = dict(zip(allf, cached_map(allf, _parse, 'error_decomp_v1')))
    rows = []
    for (model, frac, ds), files in groups.items():
            arr = np.array([c for f in files for c in parsed[f]])
            b2, s2, f2, mse, mae = arr.mean(0)
            rows.append(dict(model=model, frac=frac, dataset=ds, n_cell=len(arr), MAE_cell=mae, MSE=mse,
                             lech_muc=b2 / mse, troi_cham=s2 / mse, dao_dong=f2 / mse))
    df = pd.DataFrame(rows)
    df.to_csv('results/e20_error_decomp.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
