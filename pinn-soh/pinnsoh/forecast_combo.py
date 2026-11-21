"""
E33 — dự báo SOH có đặc trưng lịch sử: GBDT lịch sử (pinnsoh/forecast_hist.py), PINN-dự báo có lịch sử, nhánh theo đời
(pinnsoh/forecast_life.py) và mô hình HiPINN ghép ba nhánh, cùng một giao thức, sửa các điểm còn rò rỉ của E22.

Giao thức: cùng fold theo cell của E13/E20/E22 (5 fold, 70/10/20 % cell, lần lặp 0–2). Cell validation
chia đôi theo thứ tự tên: nửa 'chọn' (chọn cấu hình, số cây, trọng số, dừng sớm) và nửa 'hiệu chỉnh' (khoảng dự báo).
Thang đời L và bộ chuẩn hoá chỉ tính trên cell train (E22 tính L trên mọi cell của bộ). Gốc dự báo SOH là các chu kỳ
0,2L; 0,4L; 0,6L (không phụ thuộc đời thật của cell test); chấm trên mọi hàng đo sau gốc.

Hai ngân sách nhãn cho dữ liệu học của bộ dự báo:
    lab30  chỉ cell train có nhãn, 30 % tổng số cell
    all    mọi cell train (như E22)
Hai kịch bản lịch sử SOH của cell test tới gốc: 'do' (dung lượng đo) và, với ngân sách 'all', 'uoc' (ước lượng
PINN-lai của E20 ở 70 % nhãn, như E22).

Mô hình:
    giu, tuyen   giữ mức l_k; ngoại suy tuyến tính theo độ dốc 100 hàng
    gbdt         GBDT theo tầm của E22 (dừng sớm trên nửa 'chọn')
    gbdt_ls      GBDT lịch sử: cấu hình tốt nhất trộn với gbdt theo trọng số chọn trên validation
    pinn         PINN-dự báo của E22 (3 mạng, đầu vào 23 đặc trưng gốc)
    pinn_ls      PINN-dự báo có lịch sử: như pinn nhưng đầu vào là 23 đặc trưng gốc và đặc trưng lịch sử SOH
    ket_hop      kết hợp của E22: ½ gbdt + ½ pinn
    ket_hop_ls   HiPINN không có nhánh theo đời: w·gbdt_ls + (1 − w)·p, p là nhánh PINN chọn trên validation
                 (pinn hoặc pinn_ls), w ∈ {0; 0,25; 0,5; 0,75; 1}
    doi          nhánh theo đời: đời còn lại dự đoán + khuôn quỹ đạo của cell train
    hipinn       HiPINN: w1·gbdt_ls + w2·p + w3·doi, (w1, w2, w3) trên lưới bước 0,25 của đơn hình, chọn trên validation
Khoảng 90 % (dự đoán phù hợp): q = phân vị 90 % của |y − p| / sqrt(1 + h/L) trên nửa 'hiệu chỉnh';
khoảng p ± q·sqrt(1 + h/L), tính cho gbdt_ls, ket_hop_ls và hipinn. MAE trên nửa 'hiệu chỉnh' (cal_MAE) được lưu để
so các mô hình trên dữ liệu không dùng để chọn.
RUL: như E22 (gốc ở 25, 50, 75 % chu kỳ cuối đời thật, lưới tầm tới 3L, lúc dự báo chạm ngưỡng tau).
"""
from __future__ import annotations
import json, os, time
import numpy as np

from .data import Normalizer, cv_split, choose_labelled, load_dataset
from .forecast import (TAU, CLIP, life_scale, eol_cycle, first_cross, train_samples, fit_gbdt, gbdt_predict,
                       PINNForecaster, drop_np, e20_estimates, RUL_ORIGINS)
from . import forecast_hist as H
from .forecast_life import LifeBranch

ROOT = 'runs/E33'
BUDGETS = ('lab30', 'all')
ORIGINS_REL = (0.2, 0.4, 0.6)
MODELS = ['giu', 'tuyen', 'gbdt', 'gbdt_ls', 'pinn', 'pinn_ls', 'ket_hop', 'ket_hop_ls', 'doi', 'hipinn']
TCOL = ['scen', 'cell', 'origin', 'h', 'y'] + MODELS + ['cand', 'lo_ls', 'hi_ls', 'lo_kls', 'hi_kls', 'lo_h', 'hi_h']
RCOL = ['scen', 'cell', 'q', 'k', 'true'] + MODELS + ['lo_kls', 'hi_kls', 'lo_h', 'hi_h']
SIMPLEX = [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]   # (gbdt_ls, nhánh PINN, doi)
SCEN = ['do', 'uoc']


def rich_samples(cells, feat, n_origin=40, n_h=40):
    """Đúng các mẫu (gốc, tầm) của forecast.train_samples, nhưng đặc trưng gốc là `rich` (23 đặc trưng gốc và đặc trưng
    lịch sử SOH của forecast_hist.features)."""
    R, Hh, Lv, Y = [], [], [], []
    for c in cells:
        s, cyc = c.soh.astype(float), c.cycle
        n = len(s)
        if n < 8:
            continue
        _, rich, lev = feat[c.cid]
        for i in np.unique(np.linspace(5, n - 2, n_origin).round().astype(int)):
            js = np.unique(np.linspace(i + 1, n - 1, n_h).round().astype(int))
            R.append(np.repeat(rich[i:i + 1], len(js), 0)); Hh.append(cyc[js] - cyc[i])
            Lv.append(np.full(len(js), lev[i])); Y.append(s[js])
    cat = np.concatenate
    return cat(R), cat(Hh).astype(np.float32), cat(Lv).astype(np.float32), cat(Y).astype(np.float32)


def out_path(ds, budget, r, k, root=ROOT):
    return f'{root}/fc_{ds}_{budget}_cv{r}-{k}of5.npz'


class Combo:
    """Các mô hình đã học của một fold; `predict(d)` trả dự báo mọi mô hình cho một bộ mẫu (z, rich, lev, h)."""

    def __init__(self, gb, m_ls, cfg_ls, a_ls, nets, w, life=None, wh=(1.0, 0.0, 0.0), nets_ls=(), pb='pinn'):
        self.gb, self.m_ls, self.cfg_ls, self.a_ls, self.nets, self.w = gb, m_ls, cfg_ls, a_ls, nets, w
        self.life, self.wh, self.nets_ls, self.pb = life, wh, list(nets_ls), pb

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
        if self.nets_ls:
            pr = [d['lev'] - drop_np(*net.params(d['rich']), d['h'] / 1000.0) for net in self.nets_ls]
            P['pinn_ls'] = np.clip(np.mean(pr, axis=0), *CLIP)
        else:
            P['pinn_ls'] = np.full(len(d['h']), np.nan)
        P['ket_hop_ls'] = self.w * P['gbdt_ls'] + (1 - self.w) * P[self.pb]
        P['doi'] = (np.clip(self.life.predict(d), *CLIP) if self.life is not None
                    else np.full(len(d['h']), np.nan))
        P['hipinn'] = self.wh[0] * P['gbdt_ls'] + self.wh[1] * P[self.pb] + self.wh[2] * P['doi']
        return P


def _rows_at(z, rich, lev, i, hgrid, origin):
    n = len(hgrid)
    return dict(z=np.repeat(z[i:i + 1], n, 0), rich=np.repeat(rich[i:i + 1], n, 0), lev=np.full(n, lev[i]),
                h=np.asarray(hgrid, float), origin=np.full(n, float(origin)))


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
    # ---- các nhánh có lịch sử, học sau các PINN của E22 nên không đổi trạng thái ngẫu nhiên của chúng
    Rtr, h_, l_, y_ = rich_samples([by[i] for i in lab], feat)
    assert np.array_equal(h_, Htr) and np.array_equal(l_, Ltr) and np.array_equal(y_, Ytr)
    Rs = rich_samples([by[i] for i in sel], feat)[0]
    nets_ls = [PINNForecaster(seed=1000 * seed + 500 + m).fit(Rtr, Htr, Ltr, Ytr, Rs, Hs, Ls, Ys) for m in members]
    life = LifeBranch.fit(lab, by, feat, tau, seed)
    life_info = life.choose(va, score)
    model = Combo(gb, m_ls, cfg_ls, a_ls, nets, w=1.0, life=life, nets_ls=nets_ls)
    Pva = model.predict(va)
    pb = min(('pinn', 'pinn_ls'), key=lambda m: score(Pva[m]))              # nhánh PINN của HiPINN
    model.pb = pb
    w = min(H.ALPHAS, key=lambda a: score(a * Pva['gbdt_ls'] + (1 - a) * Pva[pb]))
    model.w = w
    Pva['ket_hop_ls'] = w * Pva['gbdt_ls'] + (1 - w) * Pva[pb]
    wh = min(SIMPLEX, key=lambda t: score(t[0] * Pva['gbdt_ls'] + t[1] * Pva[pb] + t[2] * Pva['doi']))
    model.wh = wh
    Pva['hipinn'] = wh[0] * Pva['gbdt_ls'] + wh[1] * Pva[pb] + wh[2] * Pva['doi']
    val_mae = {m: score(Pva[m]) for m in MODELS}
    # ---- khoảng dự báo: nửa 'hiệu chỉnh'
    ca = H.sample(cal, by, feat, origins); Pca = model.predict(ca)
    norm_err = lambda p: np.abs(ca['y'] - p) / np.sqrt(1 + ca['h'] / L)
    q_ls = float(np.quantile(norm_err(Pca['gbdt_ls']), .9, method='higher'))
    q_kls = float(np.quantile(norm_err(Pca['ket_hop_ls']), .9, method='higher'))
    q_h = float(np.quantile(norm_err(Pca['hipinn']), .9, method='higher'))
    wc = H.weights(ca['cell'], ca['origin'])
    cal_mae = {m: float(wc @ np.abs(Pca[m] - ca['y'])) for m in MODELS}
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
        wd_h = q_h * np.sqrt(1 + te['h'] / L)
        T.append(np.column_stack([np.full(len(te['h']), si), te['cell'], te['origin'], te['h'], te['y'],
                                  *[P[m] for m in MODELS], P['cand'], P['gbdt_ls'] - wd_ls, P['gbdt_ls'] + wd_ls,
                                  P['ket_hop_ls'] - wd_kls, P['ket_hop_ls'] + wd_kls,
                                  P['hipinn'] - wd_h, P['hipinn'] + wd_h]))
        for ci, cid in enumerate(te_ids):
            c = by[cid]; neol = eol_cycle(c, tau)
            if neol is None:
                continue
            z, rich, lev = fs[cid]
            for q in RUL_ORIGINS:
                i = int(np.searchsorted(c.cycle, q * neol, side='right') - 1)
                if not (i >= 5 and c.cycle[i] < neol):
                    continue
                G = model.predict(_rows_at(z, rich, lev, i, grid, c.cycle[i]))
                band = q_kls * np.sqrt(1 + grid / L); band_h = q_h * np.sqrt(1 + grid / L)
                RUL.append([si, ci, q, c.cycle[i], neol - c.cycle[i], *[cross(G[m]) for m in MODELS],
                            cross(G['ket_hop_ls'] - band), cross(G['ket_hop_ls'] + band),
                            cross(G['hipinn'] - band_h), cross(G['hipinn'] + band_h)])
    info = dict(ds=ds, budget=budget, repeat=r, fold=k, L=L, tau=tau, origins=origins, grid_step=step, scen=scens,
                split=dict(train=list(sp['train']), lab=list(lab), sel=sel, cal=cal, test=te_ids),
                gbdt=ginfo, history=dict(config=cfg_ls, weight_new=a_ls, trace=trace),
                pinn=[n.info for n in nets], pinn_ls=[n.info for n in nets_ls], pinn_branch=pb, w_combo=w,
                life=life_info, w_hipinn=list(wh), val_MAE=val_mae,
                cal_MAE=cal_mae, q_ls=q_ls, q_kls=q_kls, q_h=q_h,
                n_train_samples=int(len(tr['y'])), time_s=time.time() - t0)
    os.makedirs(root, exist_ok=True)
    tmp = path[:-4] + f'.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp, traj=np.vstack(T).astype(np.float64), rul=np.asarray(RUL, np.float64),
                        info=np.asarray(json.dumps(info)))
    os.replace(tmp, path)
    t = np.vstack(T)
    return float(np.abs(t[:, 5 + MODELS.index('hipinn')] - t[:, 4])[t[:, 0] == 0].mean())
