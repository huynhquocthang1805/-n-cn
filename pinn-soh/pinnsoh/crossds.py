"""
E21 — huấn luyện trên một bộ dữ liệu, kiểm tra trên các bộ còn lại; mô hình kết hợp và tự huấn luyện nhiều vòng.

Hai giao thức trên bộ NGUỒN (chia theo cell, phân tầng theo nhóm thí nghiệm, 3 hạt giống):
    A  ('a7030')   70 % cell train, 30 % cell validation             -> test trên MỌI cell của 3 bộ ĐÍCH
    B  ('b701515') 70 % train, 15 % validation, 15 % test trong bộ   -> test trong bộ + test trên 3 bộ đích
Mọi cell train của bộ nguồn đều có nhãn. Dừng sớm và số cây GBDT chọn trên validation của bộ NGUỒN; không
nhãn nào của bộ đích được dùng để huấn luyện hay chọn mô hình.

Giai đoạn nền (base, không thích nghi): MLP, GBDT, PINN (3 thành viên). Mọi tập dùng normaliser fit trên đặc trưng
của cell train nguồn. Khi phân tích, mọi dự đoán được giới hạn trong miền SOH hợp lý [0,40; 1,15] (cùng miền với
L_range), vì mạng nơ-ron ngoại suy không giới hạn khi đặc trưng bộ đích nằm xa miền đã học.

Giai đoạn thích nghi + tự huấn luyện (chain, chỉ giao thức A): cell của bộ đích chia đôi theo nhóm thí
nghiệm (nửa 0, nửa 1). Với mỗi nửa h làm tập thích nghi (chỉ đặc trưng), nửa còn lại là tập test:
    vòng 0  chuẩn hoá theo từng bộ: bộ nguồn dùng thống kê cell train nguồn, bộ đích dùng thống kê (chỉ đặc trưng)
            của cell thích nghi — cách thích nghi không nhãn đơn giản nhất (như AdaBN); PINN-semi (3 thành viên) có loss vật lý trên
            cả cell thích nghi; GBDT học trên nguồn. Mô hình kết hợp = ½·trung bình PINN + ½·GBDT, qua bộ lọc
            quỹ đạo nhân quả (postproc.causal_trend, w = 10).
    vòng r  nhãn giả trên cell thích nghi = dự đoán của mô hình kết hợp ở vòng r−1 (đã lọc). PINN học thêm
            số hạng khớp nhãn giả (distill_w = 1, như chưng cất), GBDT học trên nguồn + nhãn giả (tổng trọng số
            nhãn giả bằng tổng trọng số nguồn). Lặp R vòng.
Mỗi lượt ghi dự đoán trên mọi tập cần cho phân tích (test nguồn, cell đích, cell thích nghi).
"""
from __future__ import annotations
import json, os, time
from dataclasses import asdict
from typing import Dict, List, Optional
import numpy as np
import torch

from .data import Normalizer, make_pointset
from .train import Config, Trainer, _load, write_json
from .postproc import by_cell, causal_trend

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
PROTOCOLS = {'a7030': (0.7, 0.3, 0.0), 'b701515': (0.7, 0.15, 0.15)}
NET = dict(hidden=(128, 128, 64), beta_mono=0.5, residual='autograd', alpha_ode=0.02)
W_FILTER = 10
ROOT = 'runs/E21'


def split_frac(cells, seed: int, frac) -> Dict[str, List[str]]:
    """Chia theo cell, phân tầng theo nhóm thí nghiệm, tỉ lệ (train, val, test); test có thể bằng 0."""
    rng = np.random.RandomState(50_000 + seed)
    by: Dict[str, List[str]] = {}
    for c in cells:
        by.setdefault(c.batch, []).append(c.cid)
    out = {'train': [], 'val': [], 'test': []}
    carry = {'val': 0.0, 'test': 0.0}                      # phần dư làm tròn chuyển sang nhóm sau
    for b in sorted(by):
        ids = list(by[b]); rng.shuffle(ids); n = len(ids)
        want = {k: frac[i] * n + carry[k] for k, i in (('val', 1), ('test', 2))}
        nv, nt = int(round(want['val'])), int(round(want['test']))
        nv, nt = min(nv, n - 1), min(nt, max(0, n - 1 - nv))  # mỗi nhóm giữ ít nhất 1 cell train
        carry['val'], carry['test'] = want['val'] - nv, want['test'] - nt
        out['test'] += ids[:nt]; out['val'] += ids[nt:nt + nv]; out['train'] += ids[nt + nv:]
    return out


def halves(cells, seed: int) -> List[List[str]]:
    """Chia cell của bộ đích làm hai nửa xen kẽ theo nhóm thí nghiệm."""
    rng = np.random.RandomState(60_000 + seed)
    by: Dict[str, List[str]] = {}
    for c in cells:
        by.setdefault(c.batch, []).append(c.cid)
    out, k = [[], []], int(rng.randint(2))
    for b in sorted(by):
        ids = list(by[b]); rng.shuffle(ids)
        for cid in ids:
            out[k % 2].append(cid); k += 1
    return out


def _pred(tr: Trainer, model, S, ids):
    p = tr._forward(model, S).ravel()
    return dict(y=S.Y.ravel().tolist(), p=p.tolist(), cell=S.cell.tolist(), cycle=S.cyc.tolist(), cell_ids=ids)


class CrossTrainer(Trainer):
    """Trainer với tập huấn luyện/đánh giá lấy từ nhiều bộ dữ liệu."""

    def setup(self, src_train, src_val, src_test, evals: Dict[str, list], adapt: Optional[list],
              tgt_stats: Optional[list], pseudo: Optional[np.ndarray] = None):
        """tgt_stats: cell đích dùng để tính thống kê chuẩn hoá RIÊNG cho bộ đích (chỉ đặc trưng). None: mọi tập
        dùng normaliser của cell train nguồn (không thích nghi)."""
        self.norm = Normalizer('global').fit(src_train)
        self.tnorm = Normalizer('global').fit(tgt_stats) if tgt_stats else self.norm
        mk = lambda cs, lab=True, n=None: make_pointset(cs, n or self.norm, 'global', labelled_only=lab)
        self.S_lab = mk(src_train)
        self.S_unl = mk(adapt, lab=False, n=self.tnorm) if adapt else None
        self.val = mk(src_val)
        self.test = mk(src_test) if src_test else self.val
        self.evals = {k: (mk(cs, n=self.tnorm), [c.cid for c in cs]) for k, cs in evals.items()}
        if adapt:
            self.evals['adapt'] = (self.S_unl, [c.cid for c in adapt])
        if src_test:
            self.evals['src_test'] = (self.test, [c.cid for c in src_test])
        self.n_cells = dict(train_labelled=len(src_train), train_unlabelled=len(adapt or []), val=len(src_val),
                            test=len(src_test or []))
        self.split = {}; self.transfer = {}
        self.test_ids = [c.cid for c in (src_test or src_val)]; self.val_ids = [c.cid for c in src_val]
        if pseudo is not None:
            self.teacher_unl = torch.from_numpy(np.asarray(pseudo, dtype=np.float32).reshape(-1, 1)).to(self.dev)
        return self


def _config(src, model, seed, member, tag, pseudo):
    kw = dict(dataset=src, model=model, seed=seed, member=member, tag=tag, out_dir=ROOT, save_predictions=False,
              **(NET if model != 'mlp' else dict(hidden=NET['hidden'])))
    if pseudo:
        kw['distill_w'] = 1.0
    return Config(**kw).v2()


def _cells(ds):
    return _load(Config().v2(), ds)


def _split_src(src, proto, seed):
    cells = _cells(src); by = {c.cid: c for c in cells}
    sp = split_frac(cells, seed, PROTOCOLS[proto])
    return [by[i] for i in sp['train']], [by[i] for i in sp['val']], [by[i] for i in sp['test']]


# ───────────────────────────────────────────────────────────── giai đoạn nền
def base_path(proto, src, seed, model, member=0):
    return f'{ROOT}/base_{proto}_{src}_s{seed}_{model}' + (f'_m{member}' if member else '') + '.json'


def run_base(proto, src, seed, model, member=0):
    path = base_path(proto, src, seed, model, member)
    if os.path.exists(path):
        return json.load(open(path))['val_MAE']
    t0 = time.time()
    tr_c, va_c, te_c = _split_src(src, proto, seed)
    evals = {t: _cells(t) for t in DATASETS if t != src}
    if model == 'gbdt':
        res = _gbdt(tr_c, va_c, te_c, evals, None, None, None, seed)
    else:
        cfg = _config(src, model, seed, member, f'E21{proto}', False)
        tr = CrossTrainer(cfg).setup(tr_c, va_c, te_c, evals, None, None)
        fit = tr.fit(verbose=False)
        res = dict(config=asdict(cfg), steps=fit['steps'], val_MAE=fit['val']['MAE'],
                   preds={k: _pred(tr, tr.model, S, ids) for k, (S, ids) in tr.evals.items()})
    res.update(kind='base', proto=proto, src=src, seed=seed, model=model, member=member, time_s=time.time() - t0)
    write_json(res, path)
    return res['val_MAE']


def _gbdt(tr_c, va_c, te_c, evals, adapt, tgt_stats, pseudo, seed):
    from .baselines import fit_predict
    norm = Normalizer('global').fit(tr_c)
    tnorm = Normalizer('global').fit(tgt_stats) if tgt_stats else norm
    mk = lambda cs, lab=True, n=None: make_pointset(cs, n or norm, 'global', labelled_only=lab)
    xy = lambda S: np.hstack([S.X, S.T])
    S_tr, S_va = mk(tr_c), mk(va_c)
    Xtr, ytr, wtr = xy(S_tr), S_tr.Y.ravel(), np.ones(len(S_tr))
    sets = {k: (mk(cs, n=tnorm), [c.cid for c in cs]) for k, cs in evals.items()}
    if adapt:
        S_ad = mk(adapt, lab=False, n=tnorm)
        sets['adapt'] = (S_ad, [c.cid for c in adapt])
        if pseudo is not None:
            Xtr = np.vstack([Xtr, xy(S_ad)]); ytr = np.concatenate([ytr, pseudo])
            wtr = np.concatenate([wtr, np.full(len(S_ad), len(S_tr) / len(S_ad))])
    if te_c:
        sets['src_test'] = (mk(te_c), [c.cid for c in te_c])
    keys = list(sets)
    Xall = np.vstack([xy(sets[k][0]) for k in keys])
    p, v, hp = fit_predict('gbdt', Xtr, ytr, xy(S_va), S_va.Y.ravel(), Xall, seed, wtr=wtr)
    p = np.asarray(p, dtype=np.float64); out, o = {}, 0
    for k in keys:
        S, ids = sets[k]; n = len(S)
        out[k] = dict(y=S.Y.ravel().tolist(), p=p[o:o + n].tolist(), cell=S.cell.tolist(), cycle=S.cyc.tolist(),
                      cell_ids=ids)
        o += n
    return dict(val_MAE=v, hyper=hp, preds=out)


# ───────────────────────────────────────────────────────────── thích nghi + tự huấn luyện
def chain_path(src, tgt, seed, half, rnd, model, member=0):
    return f'{ROOT}/chain_{src}_to_{tgt}_s{seed}_h{half}_r{rnd}_{model}' + (f'_m{member}' if member else '') + '.json'


def _hybrid_pseudo(recs, key='adapt'):
    """½·trung bình PINN + ½·GBDT trên tập thích nghi, qua bộ lọc quỹ đạo -> nhãn giả theo thứ tự điểm."""
    pin = np.mean([np.asarray(r['preds'][key]['p']) for r in recs['pinn']], axis=0)
    g = np.asarray(recs['gbdt']['preds'][key]['p'])
    d = recs['gbdt']['preds'][key]
    return by_cell(causal_trend, 0.5 * pin + 0.5 * g, np.asarray(d['cell']), np.asarray(d['cycle']), w=W_FILTER)


def run_chain(src, tgt, seed, half, rounds=2, members=(0, 1, 2)):
    """Một chuỗi: vòng 0 (thích nghi không nhãn) rồi `rounds` vòng tự huấn luyện. Bỏ qua lượt đã có."""
    proto = 'a7030'
    tr_c, va_c, _ = _split_src(src, proto, seed)
    tcells = _cells(tgt); tby = {c.cid: c for c in tcells}
    hv = halves(tcells, seed)
    adapt = [tby[i] for i in hv[half]]; test = [tby[i] for i in hv[1 - half]]
    pseudo = None
    for rnd in range(rounds + 1):
        recs = {'pinn': []}
        for m in members:
            path = chain_path(src, tgt, seed, half, rnd, 'pinn', m)
            if not os.path.exists(path):
                t0 = time.time()
                cfg = _config(src, 'pinn_semi', seed, m, f'E21c{rnd}', pseudo is not None)
                trn = CrossTrainer(cfg).setup(tr_c, va_c, None, {'test': test}, adapt, adapt, pseudo)
                fit = trn.fit(verbose=False)
                res = dict(kind='chain', src=src, tgt=tgt, seed=seed, half=half, round=rnd, model='pinn', member=m,
                           config=asdict(cfg), steps=fit['steps'], val_MAE=fit['val']['MAE'], time_s=time.time() - t0,
                           preds={k: _pred(trn, trn.model, S, ids) for k, (S, ids) in trn.evals.items()})
                write_json(res, path)
            recs['pinn'].append(json.load(open(path)))
        path = chain_path(src, tgt, seed, half, rnd, 'gbdt')
        if not os.path.exists(path):
            t0 = time.time()
            res = _gbdt(tr_c, va_c, None, {'test': test}, adapt, adapt, pseudo, seed)
            res.update(kind='chain', src=src, tgt=tgt, seed=seed, half=half, round=rnd, model='gbdt',
                       time_s=time.time() - t0)
            write_json(res, path)
        recs['gbdt'] = json.load(open(path))
        pseudo = _hybrid_pseudo(recs)
    return float(np.mean([r['val_MAE'] for r in recs['pinn']]))
