"""
E19 — GBDT và GPR trong thí nghiệm chuyển miền, CÙNG giao thức với E18.

    python run_transfer_baselines.py --workers 2      # -> runs/E14/samenorm_<nguồn>_to_<đích>_<model>_k<k>_s<seed>_global.json

Mỗi lượt dùng đúng cách chia của TransferTrainer (70/15/15 theo cell ở cả hai bộ, cùng hạt giống), đúng các cell
có nhãn (mọi cell train nguồn + k cell train đích), đúng normaliser (fit trên đặc trưng cell train của cả hai bộ).
Siêu tham số chọn trên validation của bộ NGUỒN; không dùng bất kỳ nhãn nào của bộ đích ngoài k cell có nhãn.
Khi k > 0, k cell đích được cân bằng để có tổng trọng số bằng dữ liệu nguồn (tương tự nửa batch của E18).
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baselines import _cells, _xy

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
MODELS = ['gbdt', 'gpr']
OUT = 'runs/E14'


def _fit(model, Xtr, ytr, wtr, Xva, yva, seed):
    import copy
    mae = lambda m, X, y: float(np.abs(m.predict(X) - y).mean())
    if model == 'gbdt':
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, max_iter=50,
                                          warm_start=True, random_state=seed)
        best, bad = None, 0
        for it in range(50, 1001, 50):
            m.set_params(max_iter=it); m.fit(Xtr, ytr, sample_weight=wtr); v = mae(m, Xva, yva)
            if best is None or v < best[0] - 1e-6:
                best, bad = (v, copy.deepcopy(m), dict(n_trees=it)), 0
            else:
                bad += 1
                if bad >= 4:
                    break
        return best
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=min(1500, len(Xtr)), replace=False, p=wtr / wtr.sum())
    k = C(1.0) * RBF(length_scale=3.0) + WhiteKernel(1e-3)
    m = GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=seed).fit(Xtr[idx], ytr[idx])
    return mae(m, Xva, yva), m, dict(kernel=str(m.kernel_), n_train=int(len(idx)))


def run_one(model, src_name, tgt_name, k, seed):
    from pinnsoh.data import split_cells, choose_labelled, Normalizer
    from pinnsoh.metrics import evaluate
    name = f'samenorm_{src_name}_to_{tgt_name}_{model}_k{k}_s{seed}_global'
    path = f'{OUT}/{name}.json'
    if os.path.exists(path):
        return path, None
    t0 = time.time()
    src, tgt = _cells(src_name), _cells(tgt_name)
    sby = {c.cid: c for c in src}; tby = {c.cid: c for c in tgt}
    ssplit = split_cells(src, seed); tsplit = split_cells(tgt, seed)
    s_lab, _ = choose_labelled(ssplit['train'], 0.7, len(src), seed, sby)
    rng = np.random.RandomState(seed + 7)
    t_train = list(tsplit['train']); rng.shuffle(t_train)
    t_lab, t_unl = t_train[:k], t_train[k:]
    S_all = [sby[i] for i in ssplit['train']]
    norm = Normalizer('global').fit(S_all + [tby[i] for i in t_lab + t_unl])
    Xs, ys, _ = _xy([sby[i] for i in s_lab], norm)
    if t_lab:
        Xt, yt, _ = _xy([tby[i] for i in t_lab], norm)
        Xtr, ytr = np.vstack([Xs, Xt]), np.concatenate([ys, yt])
        wtr = np.concatenate([np.ones(len(ys)), np.full(len(yt), len(ys) / len(yt))])
    else:
        Xtr, ytr, wtr = Xs, ys, np.ones(len(ys))
    Xva, yva, _ = _xy([sby[i] for i in ssplit['val']], norm)
    te = [tby[i] for i in tsplit['test']]
    Xte, yte, S = _xy(te, norm)
    v, m, hp = _fit(model, Xtr, ytr, wtr, Xva, yva, seed)
    p = m.predict(Xte).astype(np.float64)
    res = dict(config=dict(tag='samenorm', model=model, dataset=src_name, target=tgt_name, k_target=k, seed=seed,
                           norm='global', norm_pool='train', clean='causal'),
               n_cells=dict(source_labelled=len(s_lab), target_labelled=len(t_lab), target_unlabelled=len(t_unl),
                            target_test=len(te)),
               hyper=hp, source_val_MAE=v, train_time_s=time.time() - t0,
               target_test=evaluate(yte, p, S.cell, cycle=S.cyc))
    json.dump(res, open(path, 'w'))
    return path, res['target_test']['MAE_cell']


def _init():
    os.environ.setdefault('OMP_NUM_THREADS', '1')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--models', nargs='+', default=MODELS)
    a = ap.parse_args()
    jobs = [(m, s_, t_, k, sd) for m, s_, t_, k, sd in itertools.product(a.models, DATASETS, DATASETS, [0, 3], a.seeds)
            if s_ != t_ and not os.path.exists(f'{OUT}/samenorm_{s_}_to_{t_}_{m}_k{k}_s{sd}_global.json')]
    print(f'E19: cần chạy {len(jobs)} lượt', flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=_init) as ex:
        futs = {ex.submit(run_one, *j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]; _, mae = fu.result()
            el = time.time() - t0
            print(f'[{i:4d}/{len(jobs)}] {j[0]:<4s} {j[1]}->{j[2]:<4s} k{j[3]} s{j[4]} MAE_cell {mae:.4f} '
                  f'(đã {el/60:.1f} ph, còn ~{el/i*(len(jobs)-i)/60:.0f} ph)', flush=True)
    print('xong', flush=True)
