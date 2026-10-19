"""
E16 — baseline học có nhãn cổ điển, CÙNG giao thức với E13.

    python run_baselines.py --workers 2      # -> runs/E16/<model>_<bộ>_f<nhãn>_cv<r>-<k>of5.json

Mỗi lượt dùng đúng cách chia (kiểm định chéo 5 fold × 5 lần lặp theo cell), đúng tập cell có nhãn
(cùng seed 100·r + k), đúng normaliser (fit trên đặc trưng mọi cell train) và cùng đầu vào [x̃, t] như
MLP/PINN. Chỉ học từ cell có nhãn. Siêu tham số chọn trên cell validation, không bao giờ trên test.

    ridge : hồi quy tuyến tính có chuẩn hoá L2, alpha ∈ {1e-3 … 1e3} chọn theo MAE validation
    gbdt  : gradient boosting trên cây (HistGradientBoosting), số cây chọn theo MAE validation
    gpr   : hồi quy Gauss, nhân RBF + nhiễu trắng, 1 500 điểm có nhãn lấy ngẫu nhiên (O(n³))
"""
from __future__ import annotations
import argparse, copy, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.3, 0.5, 0.7]
MODELS = ['ridge', 'gbdt', 'gpr']
OUT = 'runs/E16'
_CACHE = {}


def _cells(ds):
    from pinnsoh.data import load_dataset
    if ds not in _CACHE:
        _CACHE[ds] = load_dataset('PINN4SOH/data', ds, clean='causal')
    return _CACHE[ds]


def _xy(cells, norm, labelled_only=True):
    from pinnsoh.data import make_pointset
    S = make_pointset(cells, norm, 'global', labelled_only=labelled_only)
    return np.hstack([S.X, S.T]), S.Y.ravel(), S


def fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed):
    mae = lambda m, X, y: float(np.abs(m.predict(X) - y).mean())
    if model == 'ridge':
        from sklearn.linear_model import Ridge
        best = None
        for a in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]:
            m = Ridge(alpha=a).fit(Xtr, ytr); v = mae(m, Xva, yva)
            if best is None or v < best[0]:
                best = (v, m, dict(alpha=a))
    elif model == 'gbdt':
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, max_iter=50,
                                          warm_start=True, random_state=seed)
        best, bad = None, 0
        for it in range(50, 1001, 50):
            m.set_params(max_iter=it); m.fit(Xtr, ytr); v = mae(m, Xva, yva)
            if best is None or v < best[0] - 1e-6:
                best, bad = (v, copy.deepcopy(m), dict(n_trees=it)), 0
            else:
                bad += 1
                if bad >= 4:
                    break
    elif model == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(Xtr), size=min(1500, len(Xtr)), replace=False)
        k = C(1.0) * RBF(length_scale=3.0) + WhiteKernel(1e-3)
        m = GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=seed).fit(Xtr[idx], ytr[idx])
        best = (mae(m, Xva, yva), m, dict(kernel=str(m.kernel_), n_train=int(len(idx))))
    else:
        raise ValueError(model)
    v, m, hp = best
    return m.predict(Xte), v, hp


def run_one(model, ds, frac, r, k):
    from pinnsoh.data import cv_split, choose_labelled, Normalizer
    from pinnsoh.metrics import evaluate
    path = f'{OUT}/{model}_{ds}_f{frac}_cv{r}-{k}of5.json'
    if os.path.exists(path):
        return path, None
    t0 = time.time()
    cells = _cells(ds); by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, 0.10)
    seed = 100 * r + k
    lab, unl = choose_labelled(sp['train'], frac, len(cells), seed, by)
    norm = Normalizer('global').fit([by[i] for i in lab + unl])           # cùng pool như E13
    Xtr, ytr, _ = _xy([by[i] for i in lab], norm)
    Xva, yva, _ = _xy([by[i] for i in sp['val']], norm)
    te = [by[i] for i in sp['test']]
    Xte, yte, S = _xy(te, norm)
    p, v, hp = fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed)
    p = p.astype(np.float64)
    res = dict(config=dict(tag='K', model=model, label_frac=frac, dataset=ds, repeat=r, fold=k, seed=seed,
                           split='cv', n_folds=5),
               n_cells=dict(train_labelled=len(lab), train_unlabelled=len(unl), val=len(sp['val']),
                            test=len(te), all=len(cells)),
               hyper=hp, val_MAE=v, train_time_s=time.time() - t0,
               test=evaluate(yte, p, S.cell, cycle=S.cyc),
               test_predictions=dict(y=yte.tolist(), p=p.tolist(), cell=S.cell.tolist(), cycle=S.cyc.tolist(),
                                     cell_ids=[c.cid for c in te]))
    json.dump(res, open(path, 'w'))
    return path, res['test']['MAE_cell']


def _init():
    os.environ.setdefault('OMP_NUM_THREADS', '1')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--models', nargs='+', default=MODELS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(m, ds, f, r, k) for m, ds, f, r, k in itertools.product(a.models, DATASETS, FRACS, range(a.repeats), range(5))
            if not os.path.exists(f'{OUT}/{m}_{ds}_f{f}_cv{r}-{k}of5.json')]
    print(f'E16: cần chạy {len(jobs)} lượt', flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=_init) as ex:
        futs = {ex.submit(run_one, *j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]; _, mae = fu.result()
            el = time.time() - t0
            print(f'[{i:4d}/{len(jobs)}] {j[0]:<5s} {j[1]:<4s} f{j[2]} cv{j[3]}-{j[4]} MAE_cell {mae:.4f} '
                  f'(đã {el/60:.1f} ph, còn ~{el/i*(len(jobs)-i)/60:.0f} ph)', flush=True)
    print('xong', flush=True)
