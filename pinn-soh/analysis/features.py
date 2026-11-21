"""
Thí nghiệm F — đặc trưng đầu vào: dùng đặc trưng nào, bỏ đặc trưng nào, và số liệu chứng minh.

    python analysis/features.py [--workers 2]      # 120 lượt (4 bộ × 2 mức nhãn × 3 lần lặp × 5 fold), ~2 giờ CPU
    python analysis/features.py --analyze-only      # chỉ tổng hợp lại từ runs/F

Mọi lượt dùng đúng giao thức không rò rỉ của các thí nghiệm chính (E16): chia 5 fold theo cell, phân tầng theo chế độ
thí nghiệm, lần lặp 0–2, cùng tập cell có nhãn (hạt giống 100·r + k), normaliser fit trên đặc trưng của cell train, GBDT
cùng cấu hình E16 (số cây chọn trên validation). Biến thể 'all' vì vậy trùng từng bit với GBDT của E16. GBDT được dùng
vì là mô hình đơn lẻ mạnh nhất của đồ án và đủ nhanh để huấn luyện lại cho từng biến thể.

Phần 1, không cần mô hình (mọi cell của bộ): tương quan hạng Spearman giữa từng đặc trưng và SOH, trong từng cell
(trung vị qua các cell, tỉ lệ cell có |rho| >= 0,5) và gộp mọi chu kỳ của bộ; các cặp đặc trưng gần như trùng thông tin
(|rho| >= 0,95 trên mọi chu kỳ của bộ).
Phần 2, bỏ hoặc thêm đặc trưng, mỗi biến thể huấn luyện lại GBDT trên đúng fold đó:
    all      16 đặc trưng sạc + chỉ số chu kỳ t = n/1000 (cấu hình của đồ án)
    no_t     bỏ t                                  t_only   chỉ t
    no_V     bỏ 6 thống kê điện áp pha CC          no_I     bỏ 6 thống kê dòng điện pha CV
    no_QT    bỏ điện lượng và thời gian hai pha    QT_t     chỉ điện lượng, thời gian hai pha và t
    rel      thêm 16 đặc trưng tương đối so với 3 chu kỳ đầu của cell (ứng viên của E20)
    prune    bỏ các đặc trưng mà hoán vị trên VALIDATION làm MAE tăng dưới 1 % (chọn đặc trưng kiểu wrapper)
    with_T   thêm 1/T (chỉ TJU, bộ duy nhất có nhiều mức nhiệt độ)
Phần 3: độ quan trọng hoán vị trên test của mô hình 'all': hoán vị một đặc trưng, hoặc cả một nhóm, giữa các hàng test;
tỉ số MAE theo cell sau và trước hoán vị.
Kiểm định: Wilcoxon ghép cặp theo cell (MAE của cell lấy trung bình qua 3 lần lặp), Holm qua 4 bộ cho mỗi biến thể và
mức nhãn.
"""
from __future__ import annotations
import os as _os, sys as _sys
_os.environ.setdefault('OMP_NUM_THREADS', '1')          # GBDT một luồng: nhiều tiến trình không tranh lõi
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])
import argparse, glob, json, os
from itertools import islice
from multiprocessing import get_context
import numpy as np
import pandas as pd

from pinnsoh.data import FEATURES, cv_split, choose_labelled, Normalizer, make_pointset

RUN = 'runs/F'
OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.7]
REPEATS = [0, 1, 2]
NAMES = FEATURES + ['t']
GROUPS = dict(V=[0, 1, 2, 3, 6, 7], I=[8, 9, 10, 11, 14, 15], QT=[4, 5, 12, 13], t=[16])
ALL = list(range(17))
SUBSETS = dict(all=ALL, no_t=list(range(16)), t_only=[16],
               no_V=[j for j in ALL if j not in GROUPS['V']], no_I=[j for j in ALL if j not in GROUPS['I']],
               no_QT=[j for j in ALL if j not in GROUPS['QT']], QT_t=GROUPS['QT'] + [16])
PRUNE_MIN = 1.01            # giữ đặc trưng khi hoán vị nó trên validation làm MAE tăng ít nhất 1 %


def gbdt(Xtr, ytr, Xva, yva, seed, fast=False):
    """GBDT đúng như E16 (pinnsoh/baselines.fit_predict); trả hàm dự đoán tại số cây chọn trên validation.
    fast=True (mô hình dùng cho hoán vị, cần dự đoán nhiều lần): bản sao chỉ giữ `it` cây đầu, cộng cây theo đúng thứ tự
    như staged_predict nên cho cùng kết quả (kiểm tra trên validation), nhưng không phải đi qua mọi tầng mỗi lần."""
    import copy
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                      l2_regularization=1.0, early_stopping=False, max_iter=50,
                                      warm_start=True, random_state=seed)
    best, bad = None, 0
    for it in range(50, 1001, 50):
        m.set_params(max_iter=it); m.fit(Xtr, ytr); v = float(np.abs(m.predict(Xva) - yva).mean())
        if best is None or v < best[0] - 1e-6:
            best, bad = (v, it), 0
        else:
            bad += 1
            if bad >= 4:
                break
    it = best[1]
    staged = lambda X: next(islice(m.staged_predict(X), it - 1, None))
    if fast:
        m2 = copy.copy(m); m2._predictors = m._predictors[:it]
        if np.array_equal(m2.predict(Xva), staged(Xva)):
            return m2.predict, it
    return staged, it


def per_cell(p, y, c):
    return {int(k): float(np.abs(p[c == k] - y[c == k]).mean()) for k in np.unique(c)}


def mae_cell(p, y, c):
    return float(np.mean(list(per_cell(p, y, c).values())))


def job_path(ds, frac, r, k):
    return f'{RUN}/{ds}_f{frac}_cv{r}-{k}.json'


def run_job(args):
    from pinnsoh.baselines import cells_of
    from pinnsoh.train import write_json
    ds, frac, r, k = args
    path = job_path(ds, frac, r, k)
    if os.path.exists(path):
        return path
    cells = cells_of(ds); by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, 0.10)
    seed = 100 * r + k
    lab, unl = choose_labelled(sp['train'], frac, len(cells), seed, by)
    te = [by[i] for i in sp['test']]

    def arrays(mode):
        """(X, y, chỉ số cell) của train có nhãn, validation, test; cột: đặc trưng, t, 1/T."""
        norm = Normalizer(mode).fit([by[i] for i in lab + unl])            # cùng pool như E16
        out = []
        for group in ([by[i] for i in lab], [by[i] for i in sp['val']], te):
            S = make_pointset(group, norm, 'global', labelled_only=True)
            out.append((np.hstack([S.X, S.T, S.invT]), S.Y.ravel(), S.cell))
        return out

    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, cte) = arrays('global')
    res = dict(dataset=ds, frac=frac, repeat=r, fold=k, seed=seed, variants={}, perm_test={}, perm_val={})

    def fit(name, cols, A=None):
        (a, ya, _), (b, yb, _), (c, yc, cc) = A or ((Xtr, ytr, None), (Xva, yva, None), (Xte, yte, cte))
        f, it = gbdt(a[:, cols], ya, b[:, cols], yb, seed, fast=name == 'all')
        res['variants'][name] = dict(mae={te[i].cid: v for i, v in per_cell(f(c[:, cols]), yc, cc).items()},
                                     n_trees=it, cols=[int(j) for j in cols])
        return f

    for name, cols in SUBSETS.items():
        f = fit(name, cols)
        if name == 'all':
            f_all = f
    # độ quan trọng hoán vị của mô hình đầy đủ: một hoán vị hàng cho mỗi đặc trưng / nhóm
    rng = np.random.RandomState(seed)
    base_te = mae_cell(f_all(Xte[:, ALL]), yte, cte)
    base_va = float(np.abs(f_all(Xva[:, ALL]) - yva).mean())
    for name, cols in [(n, [j]) for j, n in enumerate(NAMES)] + [(f'[{g}]', c) for g, c in GROUPS.items()]:
        P = Xte[:, ALL].copy(); P[:, cols] = P[rng.permutation(len(P))][:, cols]
        res['perm_test'][name] = mae_cell(f_all(P), yte, cte) / base_te
    for j, n in enumerate(NAMES):
        P = Xva[:, ALL].copy(); P[:, j] = P[rng.permutation(len(P)), j]
        res['perm_val'][n] = float(np.abs(f_all(P) - yva).mean()) / base_va
    order = sorted(range(17), key=lambda j: -res['perm_val'][NAMES[j]])
    keep = sorted([j for j in order if res['perm_val'][NAMES[j]] >= PRUNE_MIN] or order[:2])
    fit('prune', keep)
    fit('rel', list(range(33)), arrays('both'))                  # 16 tuyệt đối + 16 tương đối + t
    if ds == 'TJU':
        fit('with_T', ALL + [17])
    write_json(res, path)
    return path


# --------------------------------------------------------------------------- phần 1: liên quan và trùng lặp
def relevance():
    from scipy.stats import spearmanr
    from pinnsoh.baselines import cells_of
    rows, red = [], []
    for ds in DS:
        per = {n: [] for n in NAMES}
        Xs, Ys = [], []
        for c in cells_of(ds):
            m = np.isfinite(c.soh)
            if m.sum() < 20:
                continue
            X = np.hstack([c.X[m], (c.cycle[m] / 1000.0)[:, None]]); y = c.soh[m]
            Xs.append(X); Ys.append(y)
            for j, n in enumerate(NAMES):
                if np.ptp(X[:, j]) > 0 and np.ptp(y) > 0:
                    per[n].append(spearmanr(X[:, j], y)[0])
        X = np.vstack(Xs); y = np.concatenate(Ys)
        for j, n in enumerate(NAMES):
            r = np.array(per[n], dtype=float); r = r[np.isfinite(r)]
            rows.append(dict(dataset=ds, feature=n, rho_within=float(np.median(r)),
                             share_abs_ge_05=float(np.mean(np.abs(r) >= 0.5)),
                             rho_pooled=float(spearmanr(X[:, j], y)[0]), n_cells=len(r)))
        R = spearmanr(X[:, :16]).correlation
        for a in range(16):
            for b in range(a + 1, 16):
                if abs(R[a, b]) >= 0.95:
                    red.append(dict(dataset=ds, f1=NAMES[a], f2=NAMES[b], rho=float(R[a, b])))
    pd.DataFrame(rows).to_csv(f'{OUT}/features_relevance.csv', index=False, float_format='%.4f')
    pd.DataFrame(red, columns=['dataset', 'f1', 'f2', 'rho']).to_csv(f'{OUT}/features_redundancy.csv', index=False,
                                                                     float_format='%.4f')


# --------------------------------------------------------------------------- tổng hợp phần 2, 3
def analyze():
    from scipy.stats import wilcoxon
    _sys.path.insert(0, 'analysis')
    from analyze_e20 import holm
    recs, perm, kept = [], [], []
    for f in sorted(glob.glob(f'{RUN}/*.json')):
        d = json.load(open(f))
        key = dict(dataset=d['dataset'], frac=d['frac'], repeat=d['repeat'], fold=d['fold'])
        for v, e in d['variants'].items():
            recs += [dict(key, variant=v, cell=cid, mae=m, n_trees=e['n_trees']) for cid, m in e['mae'].items()]
        perm += [dict(key, feature=n, ratio=x, part='test') for n, x in d['perm_test'].items()]
        perm += [dict(key, feature=n, ratio=x, part='val') for n, x in d['perm_val'].items()]
        kept.append(dict(key, n_kept=len(d['variants']['prune']['cols']),
                         dropped=';'.join(NAMES[j] for j in ALL if j not in d['variants']['prune']['cols'])))
    C = pd.DataFrame(recs)
    C.to_csv(f'{OUT}/features_cells.csv', index=False, float_format='%.6g')
    pc = C.groupby(['dataset', 'frac', 'variant', 'cell']).mae.mean().reset_index()
    rows = []
    for (ds, fr), g in pc.groupby(['dataset', 'frac']):
        base = g[g.variant == 'all'].set_index('cell').mae
        for v, h in g.groupby('variant'):
            a = h.set_index('cell').mae
            j = a.index.intersection(base.index)
            dd = a[j].values - base[j].values
            p = wilcoxon(dd).pvalue if v != 'all' and np.any(dd != 0) else np.nan
            rows.append(dict(dataset=ds, frac=fr, variant=v, mae=a[j].mean(), ratio=a[j].mean() / base[j].mean(),
                             n_cells=len(j), cells_better=int((dd < 0).sum()), cells_worse=int((dd > 0).sum()), p=p))
    T = pd.DataFrame(rows)
    T['p_holm'] = np.nan
    for _, idx in T[T.p.notna()].groupby(['frac', 'variant']).groups.items():
        T.loc[idx, 'p_holm'] = holm(T.loc[idx, 'p'].values)
    T.to_csv(f'{OUT}/features_ablation.csv', index=False, float_format='%.6g')
    P = pd.DataFrame(perm)
    P.groupby(['part', 'dataset', 'frac', 'feature']).ratio.agg(['mean', 'min', 'max']).reset_index() \
        .to_csv(f'{OUT}/features_permutation.csv', index=False, float_format='%.4f')
    K = pd.DataFrame(kept)
    K.to_csv(f'{OUT}/features_prune.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        show = T.assign(mae=1e3 * T.mae).round(4)
        print(show.pivot_table(index=['frac', 'variant'], columns='dataset', values='ratio').round(3))
        print(show.pivot_table(index=['frac', 'variant'], columns='dataset', values='p_holm').round(4))
        print(K.groupby(['dataset', 'frac']).n_kept.describe()[['min', '50%', 'max']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(RUN, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    if not a.analyze_only:
        jobs = [(ds, f, r, k) for f in FRACS for ds in ('HUST', 'MIT', 'TJU', 'XJTU') for r in REPEATS for k in range(5)]
        todo = [j for j in jobs if not os.path.exists(job_path(*j))]
        print(f'{len(jobs)} lượt, cần chạy {len(todo)}', flush=True)
        with get_context('spawn').Pool(a.workers) as pool:
            for i, p in enumerate(pool.imap_unordered(run_job, todo), 1):
                print(f'[{i}/{len(todo)}] {p}', flush=True)
    relevance()
    analyze()


if __name__ == '__main__':
    main()
