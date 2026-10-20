"""
E20, giai đoạn phát triển: so các cải tiến CHỈ trên tập validation (lần lặp 0, mức nhãn 10 % và 50 %).

    python analysis/dev_e20.py      # -> results/e20_dev.csv, in bảng tỉ số so với PINN-semi gốc

Mỗi fold có ~10 % số cell làm validation, tách khỏi cả train lẫn test. Bảng in ra là MAE theo cell trên
validation, gộp 5 fold; tỉ số < 1 nghĩa là tốt hơn PINN-semi gốc. Không đọc dự đoán test ở đây.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import glob, json, re
import numpy as np
import pandas as pd

from pinnsoh.postproc import by_cell, causal_trend, causal_iso

DS = ['XJTU', 'TJU', 'MIT', 'HUST']
OUT = 'runs/E20dev'


def mae_cell(y, p, cell):
    y, p, cell = map(np.asarray, (y, p, cell))
    return float(np.mean([np.abs(p[cell == c] - y[cell == c]).mean() for c in np.unique(cell)]))


def load():
    V = {}
    for f in glob.glob(f'{OUT}/*.json'):
        d = json.load(open(f))
        vp = d.get('val_predictions')
        if vp is None:
            continue
        name = _os.path.basename(f)
        if name.startswith('gbdt'):
            m = re.match(r'gbdt(-both)?_(\w+)_f([\d.]+)_cv(\d+)-(\d+)of5', name)
            ds, frac, r, k = m.group(2), float(m.group(3)), int(m.group(4)), int(m.group(5))
            arm = 'GBDTrel' if m.group(1) else 'GBDT'
        else:
            c = d['config']
            ds, frac, r, k = c['dataset'], c['label_frac'], c['repeat'], c['fold']
            arm = c['tag'] + (f"_m{c['member']}" if c['tag'] == 'P' else '')
        V[(arm, ds, frac, r, k)] = vp
    return V


def main():
    V = load()
    keys = sorted({k[1:] for k in V})
    rows = []
    for ds, frac, r, k in keys:
        get = lambda a: V.get((a, ds, frac, r, k))
        base = get('P_m0')
        if base is None:
            continue
        y, cell, cyc = np.asarray(base['y']), np.asarray(base['cell']), np.asarray(base['cycle'])
        P = {a: np.asarray(get(a)['p']) for a in ['P_m0', 'P_m1', 'P_m2', 'KD03', 'KD1', 'EU', 'M2', 'REL', 'GBDT', 'GBDTrel']
             if get(a)}
        for a, v in P.items():
            assert len(v) == len(y), (a, ds, frac, k)
        if all(f'P_m{i}' in P for i in range(3)):
            P['ENS3'] = (P['P_m0'] + P['P_m1'] + P['P_m2']) / 3
            P['ENS2'] = (P['P_m0'] + P['P_m1']) / 2
        if 'GBDT' in P:
            P['HYB'] = 0.5 * P['P_m0'] + 0.5 * P['GBDT']
            if 'ENS3' in P:
                P['ENS3+GBDT'] = 0.5 * P['ENS3'] + 0.5 * P['GBDT']
                P['ENS3+GBDT_w0.6'] = 0.4 * P['ENS3'] + 0.6 * P['GBDT']
            if 'KD03' in P:
                P['KD03+GBDT'] = 0.5 * P['KD03'] + 0.5 * P['GBDT']
        if 'GBDTrel' in P and 'REL' in P:
            P['REL+GBDTrel'] = 0.5 * P['REL'] + 0.5 * P['GBDTrel']
        for src in [a for a in ['P_m0', 'ENS3', 'GBDT', 'HYB', 'ENS3+GBDT', 'KD1', 'REL', 'REL+GBDTrel'] if a in P]:
            for w in (5, 10, 20, 40):
                P[f'{src}|trend{w}'] = by_cell(causal_trend, P[src], cell, cyc, w=w)
            P[f'{src}|iso'] = by_cell(causal_iso, P[src], cell, cyc)
        for a, p in P.items():
            rows.append(dict(arm=a, dataset=ds, frac=frac, fold=k, val_MAE_cell=mae_cell(y, p, cell)))
    df = pd.DataFrame(rows)
    _os.makedirs('results', exist_ok=True)
    df.to_csv('results/e20_dev.csv', index=False)
    g = df.groupby(['arm', 'dataset', 'frac']).val_MAE_cell.mean().unstack(['frac', 'dataset'])
    base = g.loc['P_m0']
    ratio = g / base
    ratio['geo'] = np.exp(np.log(ratio).mean(axis=1))
    ratio['n'] = df.groupby('arm').fold.count()
    pd.set_option('display.width', 250, 'display.max_rows', 200)
    print((g * 1e3).round(2).to_string())
    print()
    print(ratio.sort_values('geo').round(3).to_string())


if __name__ == '__main__':
    main()
