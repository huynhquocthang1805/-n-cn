"""
E33 — dự báo SOH kết hợp: GBDT lịch sử (pinnsoh/forecast_hist.py, mã của repo G-doan) và PINN-dự báo (E22), cùng một
giao thức, sửa các điểm còn rò rỉ của E22.

Giao thức (theo G-doan): cùng fold theo cell của E13/E20/E22 (5 fold, 70/10/20 % cell, lần lặp 0–2). Cell validation
chia đôi theo thứ tự tên: nửa 'chọn' (chọn cấu hình, số cây, trọng số, dừng sớm) và nửa 'hiệu chỉnh' (khoảng dự báo).
Thang đời L và bộ chuẩn hoá chỉ tính trên cell train (E22 tính L trên mọi cell của bộ). Gốc dự báo SOH là các chu kỳ
0,2L; 0,4L; 0,6L (không phụ thuộc đời thật của cell test); chấm trên mọi hàng đo sau gốc.

Hai ngân sách nhãn cho dữ liệu học của bộ dự báo:
    lab30  chỉ cell train có nhãn, 30 % tổng số cell (như G-doan)
    all    mọi cell train (như E22)
Hai kịch bản lịch sử SOH của cell test tới gốc: 'do' (dung lượng đo) và, với ngân sách 'all', 'uoc' (ước lượng
PINN-lai của E20 ở 70 % nhãn, như E22).

Mô hình:
    giu, tuyen   giữ mức l_k; ngoại suy tuyến tính theo độ dốc 100 hàng
    gbdt         GBDT theo tầm của E22 (dừng sớm trên nửa 'chọn')
    gbdt_ls      GBDT lịch sử của G-doan: cấu hình tốt nhất trộn với gbdt theo trọng số chọn trên validation
    pinn         PINN-dự báo của E22 (3 mạng)
    ket_hop      kết hợp của E22: ½ gbdt + ½ pinn
    ket_hop_ls   kết hợp mới: w·gbdt_ls + (1 − w)·pinn, w ∈ {0; 0,25; 0,5; 0,75; 1} chọn trên validation
Khoảng 90 % (theo G-doan, dự đoán phù hợp): q = phân vị 90 % của |y − p| / sqrt(1 + h/L) trên nửa 'hiệu chỉnh';
khoảng p ± q·sqrt(1 + h/L), tính cho gbdt_ls và ket_hop_ls.
RUL: như E22 (gốc ở 25, 50, 75 % chu kỳ cuối đời thật, lưới tầm tới 3L, lúc dự báo chạm ngưỡng tau).
"""
from __future__ import annotations
import json, os, time
import numpy as np

from .data import Normalizer, cv_split, choose_labelled, load_dataset
from .forecast import (TAU, CLIP, life_scale, eol_cycle, first_cross, train_samples, fit_gbdt, gbdt_predict,
                       PINNForecaster, drop_np, e20_estimates, RUL_ORIGINS)
from . import forecast_hist as H

ROOT = 'runs/E33'
BUDGETS = ('lab30', 'all')
ORIGINS_REL = (0.2, 0.4, 0.6)
MODELS = ['giu', 'tuyen', 'gbdt', 'gbdt_ls', 'pinn', 'ket_hop', 'ket_hop_ls']
TCOL = ['scen', 'cell', 'origin', 'h', 'y'] + MODELS + ['cand', 'lo_ls', 'hi_ls', 'lo_kls', 'hi_kls']
RCOL = ['scen', 'cell', 'q', 'k', 'true'] + MODELS + ['lo_kls', 'hi_kls']
SCEN = ['do', 'uoc']


def out_path(ds, budget, r, k, root=ROOT):
    return f'{root}/fc_{ds}_{budget}_cv{r}-{k}of5.npz'


class Combo:
    """Các mô hình đã học của một fold; `predict(d)` trả dự báo mọi mô hình cho một bộ mẫu (z, rich, lev, h)."""

    def __init__(self, gb, m_ls, cfg_ls, a_ls, nets, w):
        self.gb, self.m_ls, self.cfg_ls, self.a_ls, self.nets, self.w = gb, m_ls, cfg_ls, a_ls, nets, w

    def predict(self, d):
        gb_raw = gbdt_predict(self.gb, d['z'], d['h'], d['lev'])
        cand = H.predict(self.m_ls, self.cfg_ls, d)
        gbdt = np.clip(gb_raw, *CLIP)
        gbdt_ls = self.a_ls * cand + (1 - self.a_ls) * gbdt
        if self.nets:
            pr = []
            for net in self.nets:
                a, lam = net.params(d['z'])
                pr.append(d['lev'] - drop_np(a, lam, d['h'] / 1000.0))
            pinn_raw = np.mean(pr, axis=0)
        else:
            pinn_raw = np.full(len(d['h']), np.nan)
        P = dict(giu=np.clip(d['lev'], *CLIP),
                 tuyen=np.clip(d['lev'] + np.minimum(d['z'][:, 2], 0.) * d['h'] / 1000., *CLIP),
                 gbdt=gbdt, gbdt_ls=gbdt_ls, pinn=np.clip(pinn_raw, *CLIP),
                 ket_hop=np.clip(0.5 * gb_raw + 0.5 * pinn_raw, *CLIP), cand=cand)
        P['ket_hop_ls'] = self.w * P['gbdt_ls'] + (1 - self.w) * P['pinn']
        return P


def _rows_at(z, rich, lev, i, hgrid):
    n = len(hgrid)
    return dict(z=np.repeat(z[i:i + 1], n, 0), rich=np.repeat(rich[i:i + 1], n, 0), lev=np.full(n, lev[i]),
                h=np.asarray(hgrid, float))


def run_fold(ds: str, budget: str, r: int, k: int, root: str = ROOT, members=(0, 1, 2)) -> float:
    t0 = time.time()
    path = out_path(ds, budget, r, k, root)
    cells = load_dataset('PINN4SOH/data', ds, clean='causal')
    by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, 0.10)
    seed = 100 * r + k
    lab = (choose_labelled(sp['train'], 0.3, len(cells), seed, by)[0] if budget == 'lab30' else list(sp['train']))
    sel, cal = sorted(sp['val'])[::2], sorted(sp['val'])[1::2]
    tr_cells = [by[i] for i in sp['train']]
    norm = Normalizer('global').fit(tr_cells)
    L, tau = life_scale(tr_cells), TAU[ds]
    origins = np.maximum(1, np.rint(np.array(ORIGINS_REL) * L)).astype(int).tolist()
    xn = {i: norm.transform(c) for i, c in by.items()}
    feat = {i: H.features(c, xn[i]) for i, c in by.items()}
    # ---- học: GBDT gốc (E22), GBDT lịch sử (G-doan), PINN-dự báo (E22), trên cell 'lab'; chọn trên nửa 'chọn'
    tr = H.sample(lab, by, feat)
    va = H.sample(sel, by, feat, origins)
    gb, ginfo = fit_gbdt(*train_samples([by[i] for i in lab], xn), *train_samples([by[i] for i in sel], xn), seed=seed)
    old_va = np.clip(gbdt_predict(gb, va['z'], va['h'], va['lev']), *CLIP)
    m_ls, cfg_ls, a_ls, trace, score = H.search(tr, va, seed, old_va)
    Ztr, Htr, Ltr, Ytr = train_samples([by[i] for i in lab], xn)
    Zs, Hs, Ls, Ys = train_samples([by[i] for i in sel], xn)
    nets = [PINNForecaster(seed=1000 * seed + m).fit(Ztr, Htr, Ltr, Ytr, Zs, Hs, Ls, Ys) for m in members]
    model = Combo(gb, m_ls, cfg_ls, a_ls, nets, w=1.0)
    Pva = model.predict(va)
    w = min(H.ALPHAS, key=lambda a: score(a * Pva['gbdt_ls'] + (1 - a) * Pva['pinn']))
    model.w = w
    Pva['ket_hop_ls'] = w * Pva['gbdt_ls'] + (1 - w) * Pva['pinn']
    val_mae = {m: score(Pva[m]) for m in MODELS}
    # ---- khoảng dự báo: nửa 'hiệu chỉnh'
    ca = H.sample(cal, by, feat, origins); Pca = model.predict(ca)
    norm_err = lambda p: np.abs(ca['y'] - p) / np.sqrt(1 + ca['h'] / L)
    q_ls = float(np.quantile(norm_err(Pca['gbdt_ls']), .9, method='higher'))
    q_kls = float(np.quantile(norm_err(Pca['ket_hop_ls']), .9, method='higher'))
    # ---- chấm trên cell test
    te_ids = list(sp['test'])
    scens = ['do'] + (['uoc'] if budget == 'all' else [])
    est = e20_estimates(ds, r, k, 'test') if 'uoc' in scens else None
    step = max(1, int(round(L / 200)))
    grid = np.arange(step, int(3 * L) + step, step, dtype=float)
    cross = lambda p: first_cross(p, grid, tau, grid[-1])
    T, RUL = [], []
    for si, sc in enumerate(scens):
        if sc == 'do':
            fs = {i: feat[i] for i in te_ids}
        else:
            fs = {}
            for i in te_ids:
                cyc, s = est[i]
                assert np.array_equal(cyc, by[i].cycle), i
                fs[i] = H.features(by[i], xn[i], soh=s)
        te = H.sample(te_ids, by, fs, origins)
        P = model.predict(te)
        wd_ls = q_ls * np.sqrt(1 + te['h'] / L); wd_kls = q_kls * np.sqrt(1 + te['h'] / L)
        T.append(np.column_stack([np.full(len(te['h']), si), te['cell'], te['origin'], te['h'], te['y'],
                                  *[P[m] for m in MODELS], P['cand'], P['gbdt_ls'] - wd_ls, P['gbdt_ls'] + wd_ls,
                                  P['ket_hop_ls'] - wd_kls, P['ket_hop_ls'] + wd_kls]))
        for ci, cid in enumerate(te_ids):
            c = by[cid]; neol = eol_cycle(c, tau)
            if neol is None:
                continue
            z, rich, lev = fs[cid]
            for q in RUL_ORIGINS:
                i = int(np.searchsorted(c.cycle, q * neol, side='right') - 1)
                if not (i >= 5 and c.cycle[i] < neol):
                    continue
                G = model.predict(_rows_at(z, rich, lev, i, grid))
                band = q_kls * np.sqrt(1 + grid / L)
                RUL.append([si, ci, q, c.cycle[i], neol - c.cycle[i], *[cross(G[m]) for m in MODELS],
                            cross(G['ket_hop_ls'] - band), cross(G['ket_hop_ls'] + band)])
    info = dict(ds=ds, budget=budget, repeat=r, fold=k, L=L, tau=tau, origins=origins, grid_step=step, scen=scens,
                split=dict(train=list(sp['train']), lab=list(lab), sel=sel, cal=cal, test=te_ids),
                gbdt=ginfo, history=dict(config=cfg_ls, weight_new=a_ls, trace=trace),
                pinn=[n.info for n in nets], w_combo=w, val_MAE=val_mae, q_ls=q_ls, q_kls=q_kls,
                n_train_samples=int(len(tr['y'])), time_s=time.time() - t0)
    os.makedirs(root, exist_ok=True)
    tmp = path[:-4] + f'.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp, traj=np.vstack(T).astype(np.float64), rul=np.asarray(RUL, np.float64),
                        info=np.asarray(json.dumps(info)))
    os.replace(tmp, path)
    t = np.vstack(T)
    return float(np.abs(t[:, 5 + MODELS.index('ket_hop_ls')] - t[:, 4])[t[:, 0] == 0].mean())
