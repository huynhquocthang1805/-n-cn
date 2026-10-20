"""
Mô hình học có nhãn cổ điển — E16 (trong cùng bộ, kiểm định chéo) và E19 (chuyển miền).

    ridge : hồi quy tuyến tính có chuẩn hoá L2, alpha ∈ {1e-3 … 1e3} chọn theo MAE validation
    gbdt  : gradient boosting trên cây (HistGradientBoosting), số cây ∈ {50, 100, …, 1000} chọn theo MAE
            validation, dừng khi 4 mốc liên tiếp không cải thiện
    gpr   : hồi quy Gauss, nhân RBF + nhiễu trắng, 1 500 điểm có nhãn lấy ngẫu nhiên (chi phí O(n³))

Cùng giao thức với mạng nơ-ron: cùng cách chia, cùng tập cell có nhãn (seed 100·r + k), cùng normaliser
(fit trên đặc trưng mọi cell train), cùng đầu vào [x̃, t]. Siêu tham số chọn trên validation, không bao giờ
trên test. Khi chuyển miền, validation là của bộ NGUỒN; k cell đích có nhãn được cân để có tổng trọng số
bằng dữ liệu nguồn (tương tự nửa batch của MLP/PINN).

Tối ưu so với bản đầu: GBDT không còn deepcopy mô hình mỗi lần validation cải thiện — chỉ ghi lại số cây
tốt nhất rồi lấy dự đoán test bằng staged_predict tại đúng số cây đó. Cây thứ 1..n của mô hình huấn
luyện tiếp (warm start) giống hệt cây của mô hình dừng ở n cây, nên dự đoán không đổi.
"""
from __future__ import annotations
import json, os, time
from itertools import islice
import numpy as np

from .data import load_dataset, cv_split, choose_labelled, split_cells, make_pointset, Normalizer
from .metrics import evaluate
from .train import write_json

MODELS = ['ridge', 'gbdt', 'gpr']
_CELLS = {}


def cells_of(ds):
    if ds not in _CELLS:
        _CELLS[ds] = load_dataset('PINN4SOH/data', ds, clean='causal')
    return _CELLS[ds]


def xy(cells, norm, labelled_only=True):
    S = make_pointset(cells, norm, 'global', labelled_only=labelled_only)
    return np.hstack([S.X, S.T]), S.Y.ravel(), S


def fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed, wtr=None, return_val=False):
    """Huấn luyện, chọn siêu tham số theo validation, trả (dự đoán test, MAE validation, siêu tham số)
    và thêm dự đoán validation khi return_val=True."""
    out = lambda pte, v, hp, pva: (pte, v, hp, pva) if return_val else (pte, v, hp)
    mae = lambda pred, y: float(np.abs(pred - y).mean())
    if model == 'ridge':
        from sklearn.linear_model import Ridge
        best = None
        for a in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]:
            m = Ridge(alpha=a).fit(Xtr, ytr, sample_weight=wtr); v = mae(m.predict(Xva), yva)
            if best is None or v < best[0]:
                best = (v, m, dict(alpha=a))
        v, m, hp = best
        return out(m.predict(Xte), v, hp, m.predict(Xva))
    if model == 'gbdt':
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, max_iter=50,
                                          warm_start=True, random_state=seed)
        best, bad = None, 0
        for it in range(50, 1001, 50):
            m.set_params(max_iter=it); m.fit(Xtr, ytr, sample_weight=wtr); v = mae(m.predict(Xva), yva)
            if best is None or v < best[0] - 1e-6:
                best, bad = (v, it), 0
            else:
                bad += 1
                if bad >= 4:
                    break
        v, it = best
        at = lambda X: next(islice(m.staged_predict(X), it - 1, None))
        return out(at(Xte), v, dict(n_trees=it), at(Xva) if return_val else None)
    if model == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
        rng = np.random.RandomState(seed)
        n = min(1500, len(Xtr))
        idx = (rng.choice(len(Xtr), size=n, replace=False) if wtr is None
               else rng.choice(len(Xtr), size=n, replace=False, p=wtr / wtr.sum()))
        k = C(1.0) * RBF(length_scale=3.0) + WhiteKernel(1e-3)
        m = GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=seed).fit(Xtr[idx], ytr[idx])
        pva = m.predict(Xva)
        return out(m.predict(Xte), mae(pva, yva), dict(kernel=str(m.kernel_), n_train=int(n)), pva)
    raise ValueError(model)


def cv_path(model, ds, frac, r, k, out_dir='runs/E16', norm='global'):
    return f'{out_dir}/{model}{"" if norm == "global" else "-" + norm}_{ds}_f{frac}_cv{r}-{k}of5.json'


def run_cv(model, ds, frac, r, k, out_dir='runs/E16', save_val=False, norm='global'):
    """E16: một lượt (mô hình, bộ, mức nhãn, lần lặp r, fold k). Trả MAE theo cell trên test."""
    path = cv_path(model, ds, frac, r, k, out_dir, norm)
    if os.path.exists(path):
        return json.load(open(path))['test']['MAE_cell']
    t0 = time.time()
    cells = cells_of(ds); by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, 0.10)
    seed = 100 * r + k
    lab, unl = choose_labelled(sp['train'], frac, len(cells), seed, by)
    norm = Normalizer(norm).fit([by[i] for i in lab + unl])               # cùng pool như MLP/PINN
    Xtr, ytr, _ = xy([by[i] for i in lab], norm)
    Xva, yva, _ = xy([by[i] for i in sp['val']], norm)
    te = [by[i] for i in sp['test']]
    Xte, yte, S = xy(te, norm)
    p, v, hp, pva = fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed, return_val=True)
    p = np.asarray(p, dtype=np.float64)
    res = dict(config=dict(tag='K', model=model, label_frac=frac, dataset=ds, repeat=r, fold=k, seed=seed,
                           split='cv', n_folds=5, norm=norm.mode),
               n_cells=dict(train_labelled=len(lab), train_unlabelled=len(unl), val=len(sp['val']),
                            test=len(te), all=len(cells)),
               hyper=hp, val_MAE=v, train_time_s=time.time() - t0,
               test=evaluate(yte, p, S.cell, cycle=S.cyc),
               test_predictions=dict(y=yte.tolist(), p=p.tolist(), cell=S.cell.tolist(), cycle=S.cyc.tolist(),
                                     cell_ids=[c.cid for c in te]))
    if save_val:
        _, _, Sv = xy([by[i] for i in sp['val']], norm)
        res['val_predictions'] = dict(y=yva.tolist(), p=np.asarray(pva, dtype=np.float64).tolist(),
                                      cell=Sv.cell.tolist(), cycle=Sv.cyc.tolist(), cell_ids=list(sp['val']))
    write_json(res, path)
    return res['test']['MAE_cell']


def transfer_path(model, src, tgt, k, seed, out_dir='runs/E14'):
    return f'{out_dir}/samenorm_{src}_to_{tgt}_{model}_k{k}_s{seed}_global.json'


def run_transfer(model, src, tgt, k, seed, out_dir='runs/E14'):
    """E19: cùng cách chia, cùng cell có nhãn và cùng normaliser với TransferTrainer (E18)."""
    path = transfer_path(model, src, tgt, k, seed, out_dir)
    if os.path.exists(path):
        return json.load(open(path))['target_test']['MAE_cell']
    t0 = time.time()
    S_cells, T_cells = cells_of(src), cells_of(tgt)
    sby = {c.cid: c for c in S_cells}; tby = {c.cid: c for c in T_cells}
    ssplit = split_cells(S_cells, seed); tsplit = split_cells(T_cells, seed)
    s_lab, _ = choose_labelled(ssplit['train'], 0.7, len(S_cells), seed, sby)
    rng = np.random.RandomState(seed + 7)
    t_train = list(tsplit['train']); rng.shuffle(t_train)
    t_lab, t_unl = t_train[:k], t_train[k:]
    norm = Normalizer('global').fit([sby[i] for i in ssplit['train']] + [tby[i] for i in t_lab + t_unl])
    Xs, ys, _ = xy([sby[i] for i in s_lab], norm)
    if t_lab:
        Xt, yt, _ = xy([tby[i] for i in t_lab], norm)
        Xtr, ytr = np.vstack([Xs, Xt]), np.concatenate([ys, yt])
        wtr = np.concatenate([np.ones(len(ys)), np.full(len(yt), len(ys) / len(yt))])
    else:
        Xtr, ytr, wtr = Xs, ys, np.ones(len(ys))
    Xva, yva, _ = xy([sby[i] for i in ssplit['val']], norm)
    te = [tby[i] for i in tsplit['test']]
    Xte, yte, S = xy(te, norm)
    p, v, hp = fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed, wtr=wtr)
    p = np.asarray(p, dtype=np.float64)
    res = dict(config=dict(tag='samenorm', model=model, dataset=src, target=tgt, k_target=k, seed=seed,
                           norm='global', norm_pool='train', clean='causal'),
               n_cells=dict(source_labelled=len(s_lab), target_labelled=len(t_lab), target_unlabelled=len(t_unl),
                            target_test=len(te)),
               hyper=hp, source_val_MAE=v, train_time_s=time.time() - t0,
               target_test=evaluate(yte, p, S.cell, cycle=S.cyc))
    write_json(res, path)
    return res['target_test']['MAE_cell']

