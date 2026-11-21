"""
Độ nhạy của bộ lọc ngoại lai nhân quả (Mục Tiền xử lý): ngưỡng k và số chu kỳ lịch sử tối thiểu.

    python analysis/filter_sensitivity.py [--workers 4]     # -> runs/FS/*.npz, results/filter_sensitivity.csv
    python analysis/filter_sensitivity.py --analyze-only

Bộ lọc của đồ án (pinnsoh/data.py) được ấn định trước mọi lượt huấn luyện: cửa sổ 50 chu kỳ quá khứ, cần ít nhất 10 chu
kỳ lịch sử, loại chu kỳ khi |z bền vững| > 5 ở một đặc trưng bất kỳ. Đây là kiểm tra sau: với mỗi cách đặt (k, số chu kỳ
lịch sử tối thiểu), đọc lại bộ dữ liệu, tính tỉ lệ chu kỳ bị loại, rồi học GBDT (cấu hình E16, 30 % nhãn, 5 fold) trên
dữ liệu đã lọc theo cách đặt đó. MAE theo cell được chấm trên CÙNG một tập hàng test cho mọi cách đặt (các chu kỳ mà mọi
cách đặt đều giữ), để cách lọc mạnh hơn không được lợi vì bỏ bớt chính các hàng test khó. Kiểm định: Wilcoxon ghép cặp
theo cell so với cách đặt của đồ án (k = 5, 10 chu kỳ), MAE của cell lấy trung bình qua lần lặp, Holm qua bốn bộ.
"""
from __future__ import annotations
import os as _os, sys as _sys
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import argparse, glob, os
from multiprocessing import get_context
import numpy as np
import pandas as pd

RUN = 'runs/FS'
OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRAC = 0.3
# tên, k, số chu kỳ lịch sử tối thiểu; 'none': chỉ bỏ hàng có đặc trưng không hữu hạn
SETTINGS = [('k3', 3.0, 10), ('k4', 4.0, 10), ('k5', 5.0, 10), ('k6', 6.0, 10), ('k8', 8.0, 10),
            ('none', np.inf, 10), ('h5', 5.0, 5), ('h20', 5.0, 20)]
REF = 'k5'


def load(ds, k, hist):
    import pinnsoh.data as D
    D.CAUSAL_K, D.CAUSAL_MIN_HIST = k, hist
    return D._load_dataset_csv(D.resolve_root('PINN4SOH/data'), ds, 'causal')


def path(ds, name, r):
    return f'{RUN}/{ds}_{name}_r{r}.npz'


def run_job(args):
    """Một bộ, một cách đặt, một lần lặp: 5 fold; lưu dự đoán trên các hàng test có nhãn kèm (cell, chu kỳ)."""
    from pinnsoh.data import cv_split, choose_labelled, Normalizer, make_pointset
    from features import gbdt
    ds, name, k, hist, r = args
    out = path(ds, name, r)
    if os.path.exists(out):
        return out
    cells = load(ds, k, hist); by = {c.cid: c for c in cells}
    rows = []
    for f in range(5):
        sp = cv_split(cells, f, r, 5, 0.10)
        seed = 100 * r + f
        lab, unl = choose_labelled(sp['train'], FRAC, len(cells), seed, by)
        norm = Normalizer('global').fit([by[i] for i in lab + unl])
        arr = []
        for ids in (lab, sp['val'], sp['test']):
            S = make_pointset([by[i] for i in ids], norm, 'global', labelled_only=True)
            arr.append((np.hstack([S.X, S.T]), S.Y.ravel(), np.array(ids)[S.cell], S.cyc))
        (Xtr, ytr, _, _), (Xva, yva, _, _), (Xte, yte, cte, nte) = arr
        pred, it = gbdt(Xtr, ytr, Xva, yva, seed)
        p = pred(Xte)
        rows.append(pd.DataFrame(dict(fold=f, cell=cte, cycle=nte, y=yte, p=p, n_trees=it)))
    D = pd.concat(rows, ignore_index=True)
    n_rows = sum(len(c.cycle) for c in cells)
    os.makedirs(RUN, exist_ok=True)
    tmp = out + f'.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp, fold=D.fold.values, cell=np.array(D.cell.tolist(), dtype=str), cycle=D.cycle.values,
                        y=D.y.values, p=D.p.values, n_trees=D.n_trees.values, n_rows=n_rows)
    os.replace(tmp, out)
    return out


def analyze(repeats):
    from scipy import stats
    res = []
    for ds in DS:
        P = {}
        for name, k, hist in SETTINGS:
            parts = []
            for r in repeats:
                q = np.load(path(ds, name, r), allow_pickle=True)      # tệp do chính script này ghi
                parts.append(pd.DataFrame(dict(r=r, fold=q['fold'], cell=q['cell'], cycle=q['cycle'], y=q['y'],
                                               p=q['p'])))
                n_rows = int(q['n_rows'])
            P[name] = (pd.concat(parts, ignore_index=True), n_rows)
        base_rows = P['none'][1]
        keys = None                                  # hàng test mà mọi cách đặt đều giữ
        for name, (d, _) in P.items():
            kk = set(zip(d.r, d.cell, d.cycle))
            keys = kk if keys is None else keys & kk
        cellmae = {}
        for name, (d, n_rows) in P.items():
            m = [(a, b, c) in keys for a, b, c in zip(d.r, d.cell, d.cycle)]
            e = d[m].assign(ae=lambda x: np.abs(x.p - x.y))
            cellmae[name] = e.groupby(['r', 'cell']).ae.mean().groupby('cell').mean()   # trung bình qua lần lặp
        for name, k, hist in SETTINGS:
            a, b = cellmae[name], cellmae[REF].reindex(cellmae[name].index)
            p = stats.wilcoxon(a - b).pvalue if name != REF and np.any(a != b) else 1.0
            res.append(dict(ds=ds, setting=name, k=k, min_hist=hist, drop=1 - P[name][1] / base_rows,
                            mae=float(a.mean()), ratio=float(a.mean() / b.mean()), p=float(p),
                            n_cell=len(a), n_rows_scored=len(keys), n_fits=5 * len(repeats)))
    R = pd.DataFrame(res)
    from analyze_e33 import holm                     # Holm qua bốn bộ cho mỗi cách đặt, như các kiểm định khác
    R['p_holm'] = 1.0
    for name, g in R.groupby('setting'):
        if name != REF:
            R.loc[g.index, 'p_holm'] = holm(list(g.p))
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/filter_sensitivity.csv', index=False)
    show = R.copy(); show['drop'] = (100 * show['drop']).round(2); show['mae'] = (1e3 * show.mae).round(3)
    print(show.to_string(index=False))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--repeats', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    if not a.analyze_only:
        jobs = [(ds, n, k, h, r) for r in a.repeats for ds in DS for n, k, h in SETTINGS
                if not os.path.exists(path(ds, n, r))]
        jobs.sort(key=lambda j: -dict(HUST=4, TJU=3, MIT=2, XJTU=1)[j[0]])       # bộ lớn chạy trước
        with get_context('spawn').Pool(a.workers) as pool:
            for i, out in enumerate(pool.imap_unordered(run_job, jobs), 1):
                print(f'[{i}/{len(jobs)}] {out}', flush=True)
    analyze(a.repeats)


if __name__ == '__main__':
    main()
