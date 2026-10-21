"""
E23 — dự báo SOH có xét hồi phục dung lượng.

Dung lượng đo được không giảm đều: sau một khoảng nghỉ hoặc khi đổi chế độ sạc, cell lấy lại một phần dung lượng rồi
mất dần trong vài chục chu kỳ (phần dung lượng thuận nghịch); một số giao thức còn cho dao động đều đặn theo chu kỳ
của chính giao thức (XJTU, nhóm R2.5: chu kỳ 5). Bộ dự báo của E22 lấy mức hiện tại và độ dốc từ lịch sử thô, nên một
lần hồi phục ở gần gốc làm lệch cả mức lẫn độ dốc, và dao động tương lai bị bỏ qua.

Tách lịch sử tại gốc k (chỉ dùng các chu kỳ trong W_DEC = 60 chu kỳ gần nhất, <= k):
    1. xu thế: đường thẳng khớp bằng hồi quy Huber (lặp trọng số), không bị kéo bởi vài điểm hồi phục;
    2. dao động: trên phần dư, dò chu kỳ P ∈ [2,2; 15] (bước 0,02, ít nhất 4 chu kỳ trọn trong cửa sổ) với hai hài; chỉ
       giữ khi giải thích được ít nhất R2_PER = 60 % phương sai của phần dư trên cả cửa sổ, ít nhất 50 % trên mỗi nửa
       cửa sổ, và biên độ (độ lệch chuẩn) >= 1e-3; rồi khớp lại xu thế trên chuỗi đã trừ dao động;
    3. phần hồi phục hiện tại e_k: trung vị phần dư của 3 hàng cuối sau khi trừ xu thế và dao động.
Dự báo
    SOH(k+h) = l_k + e_k·exp(−h/tau) − dv(h; a, lam) + p_k(h),
trong đó l_k là giá trị xu thế tại k, dv là nghiệm đóng của động học xám (phần mất thuận nghịch được tách riêng), tau là
thời gian tắt của phần hồi phục, học chung với mạng (một tham số chung), p_k(h) là dao động ngoại suy.
    pinn_r   mạng (đặc trưng gốc bền) -> (a, lam), cộng tau học được; 3 thành viên
    gbdt_r   GBDT học phần giảm của xu thế l_k + e_k·exp(−h/tau) + p_k(h) − SOH(k+h) (tau lấy từ pinn_r), đơn điệu theo h;
             dưới tầm h1 (bề rộng ngăn đầu tiên của trục h trong cây) phần giảm nội suy tuyến tính từ 0, để đường dự báo
             xuất phát đúng từ mức hiện tại
    ket_hop_r ½ + ½
Cùng lượt còn học lại đúng các mô hình E22 (cùng hạt giống, kết quả trùng E22) để so ghép cặp trên mọi gốc, kể cả gốc
đặt ngay sau một lần hồi phục (sự kiện: SOH đo vượt trung vị 5 hàng trước đó quá 0,004).
Kịch bản 'uoc' dùng ước lượng ½ + ½ CHƯA lọc (bộ lọc quỹ đạo xoá mất dao động), xu thế bền thay cho bộ lọc.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import torch
import torch.nn.functional as Fnn

from .data import Normalizer, cv_split, load_dataset
from .forecast import (CLIP, ORIGINS, RUL_ORIGINS, SCEN, TAU, Forecaster, RateNet, band, calibrate, drop_np, drop_t,
                       e20_estimates, eol_cycle, first_cross, fit_gbdt, gbdt_predict, life_scale, nominal_ah,
                       origin_features, _history, W_FEAT)

ROOT = 'runs/E23'
W_DEC, W_LONG, N_MIN, R2_PER, AMP_MIN = 60, 150, 20, 0.6, 1e-3
PER_GRID = np.arange(2.2, W_DEC / 4 + 1e-9, 0.02)     # ít nhất 4 chu kỳ trọn trong cửa sổ: một bước nhảy đơn lẻ
                                                    # trông như nửa chu kỳ của một sóng dài, không được coi là dao động
EV_JUMP, EV_MAX = 0.004, 3
N_NET = 23                      # số cột đầu của đặc trưng gốc bền đưa vào mạng PINN (không gồm e, biên độ, độ khớp)
MODELS_R = ['giu', 'gbdt', 'pinn', 'ket_hop', 'gbdt_r', 'pinn_r', 'ket_hop_r']


# --------------------------------------------------------------------------- tách lịch sử
def huber_line(t, y, it=10):
    """Đường thẳng y = b0 + b1·t khớp bằng hồi quy Huber (lặp trọng số, ngưỡng 1,345·MAD)."""
    w = np.ones_like(y)
    A = np.column_stack([np.ones_like(t), t])
    b = np.zeros(2)
    for _ in range(it):
        sw = np.sqrt(w)
        b = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = y - A @ b
        c = 1.345 * (1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6)
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-12))
    return b


def _harm(t, P):
    ph = 2 * np.pi * np.asarray(t, float) / P
    return np.column_stack([np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)])


def periodic_fit(t, r):
    """Dò chu kỳ trên lưới PER_GRID (hai hài) bằng bình phương tối thiểu theo lô; trả (P, hệ số, R²)."""
    ph = (2 * np.pi / PER_GRID)[:, None] * t[None, :]
    X = np.stack([np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)], axis=2)
    XtX = np.einsum('pni,pnj->pij', X, X) + 1e-9 * np.eye(4)
    b = np.linalg.solve(XtX, np.einsum('pni,n->pi', X, r)[..., None])[..., 0]
    sse = np.sum((np.einsum('pni,pi->pn', X, b) - r) ** 2, axis=1)
    j = int(np.argmin(sse))
    return float(PER_GRID[j]), b[j], 1.0 - sse[j] / max(float(np.sum((r - r.mean()) ** 2)), 1e-12)


def decompose(s, cyc, i):
    """Tách lịch sử tới hàng i. Trả dict: level, slope (trên 1000 chu kỳ), slope_long, e (hồi phục hiện tại),
    P, coef (dao động, None nếu không có), amp, r2."""
    cyc = np.asarray(cyc, float); s = np.asarray(s, float)
    c0 = cyc[i]
    m = (cyc > c0 - W_DEC) & (np.arange(len(s)) <= i)
    t, y = cyc[m] - c0, s[m]
    out = dict(P=0.0, coef=None, amp=0.0, r2=0.0)
    if len(y) < 3:
        return dict(out, level=float(s[i]), slope=0.0, slope_long=0.0, e=0.0)
    b = huber_line(t, y)
    y2 = y
    if len(y) >= N_MIN:
        res0 = y - (b[0] + b[1] * t)
        P, coef, r2 = periodic_fit(t, res0)
        fit = _harm(t, P) @ coef
        out['r2'] = float(r2)
        # dao động phải lặp lại: cùng chu kỳ và hệ số giải thích được phần lớn phương sai ở cả hai nửa cửa sổ
        half = t <= np.median(t)
        r2h = [1 - np.sum((res0[m] - fit[m]) ** 2) / max(float(np.sum((res0[m] - res0[m].mean()) ** 2)), 1e-12)
               for m in (half, ~half)]
        if r2 >= R2_PER and min(r2h) >= R2_PER - 0.1 and fit.std() >= AMP_MIN:
            out.update(P=P, coef=coef, amp=float(fit.std()))
            y2 = y - fit
            b = huber_line(t, y2)
    res = y2 - (b[0] + b[1] * t)
    ml = (cyc > c0 - W_LONG) & (np.arange(len(s)) <= i)
    tl, yl = cyc[ml] - c0, s[ml]
    if out['coef'] is not None:
        yl = yl - _harm(tl, out['P']) @ out['coef']
    bl = huber_line(tl, yl) if len(yl) >= 3 else b
    return dict(out, level=float(b[0]), slope=float(b[1] * 1000), slope_long=float(bl[1] * 1000),
                e=float(np.median(res[-3:])))


def periodic_at(dec, h):
    if dec['coef'] is None:
        return np.zeros(len(np.atleast_1d(h)))
    return _harm(np.atleast_1d(h), dec['P']) @ dec['coef']


def features_r(s, cyc, Xn, temp_k, i, dec):
    """Đặc trưng gốc bền (25 chiều): mức và độ dốc của xu thế, mức ban đầu, độ giảm, k, 1/T, 16 đặc trưng sạc trung bình
    10 hàng gần nhất, phần hồi phục hiện tại, biên độ và độ khớp của dao động."""
    l0 = float(np.mean(np.asarray(s[:5], float)))
    a = max(0, i - W_FEAT + 1)
    fm = Xn[a:i + 1].astype(float).mean(0)
    return np.concatenate([[dec['level'], dec['slope'], dec['slope_long'], l0, l0 - dec['level'], cyc[i] / 1000.0,
                            1000.0 / temp_k], fm, [dec['e'], dec['amp'], dec['r2']]]).astype(np.float32)


def samples_r(cells, Xn_of, n_origin=40, n_h=40):
    """Mẫu (gốc, tầm) từ lịch sử đo: Z, h, l_k, e_k, p_k(h), SOH thật tại k+h."""
    Z, H, L, E, Pv, Y = [], [], [], [], [], []
    for c in cells:
        s, cyc = c.soh.astype(float), c.cycle
        n = len(s)
        if n < 8:
            continue
        for i in np.unique(np.linspace(5, n - 2, n_origin).round().astype(int)):
            dec = decompose(s, cyc, i)
            z = features_r(s, cyc, Xn_of[c.cid], c.temp_k, i, dec)
            js = np.unique(np.linspace(i + 1, n - 1, n_h).round().astype(int))
            h = (cyc[js] - cyc[i]).astype(float)
            Z.append(np.repeat(z[None], len(js), 0)); H.append(h); L.append(np.full(len(js), dec['level']))
            E.append(np.full(len(js), dec['e'])); Pv.append(periodic_at(dec, h)); Y.append(s[js])
    cat = lambda x: np.concatenate(x).astype(np.float32)
    return np.concatenate(Z), cat(H), cat(L), cat(E), cat(Pv), cat(Y)


# --------------------------------------------------------------------------- mô hình
class PINNForecasterR:
    """Như PINN-dự báo, cộng phần hồi phục e_k·exp(−h/tau) (tau học được, chung) và dao động ngoại suy."""
    def __init__(self, seed=0, steps=4000, batch=1024, lr=3e-3, eval_every=100, patience=10):
        self.seed, self.steps, self.batch, self.lr, self.ev, self.pat = seed, steps, batch, lr, eval_every, patience

    def tau(self):
        return float(1.0 + Fnn.softplus(self.tau_raw).item())

    def _norm(self, Z):
        # mạng chỉ nhận phần đặc trưng xu thế (bỏ e, biên độ và độ khớp dao động — các phần này đã cộng thẳng vào dự báo),
        # z-score chặn ở ±5 để một gốc lạ (bước tụt do lỗi đo) không đẩy mạng ngoại suy
        return np.clip((Z[:, :N_NET] - self.mu) / self.sd, -5, 5)

    def _pred_t(self, z, h, l, e, pv):
        a, lam = self.net(z)
        tau = 1.0 + Fnn.softplus(self.tau_raw)
        u = l + e * torch.exp(-h / tau) - drop_t(a, lam, h / 1000.0) + pv
        return torch.clamp(u, CLIP[0], CLIP[1])

    def fit(self, tr, va):
        Ztr, Htr, Ltr, Etr, Ptr, Ytr = tr
        Zva, Hva, Lva, Eva, Pva, Yva = va
        torch.manual_seed(self.seed); rng = np.random.RandomState(self.seed)
        self.mu, self.sd = Ztr[:, :N_NET].mean(0), Ztr[:, :N_NET].std(0) + 1e-6
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        col = lambda x: T(x[:, None])
        z, h, l, e, pv, y = T(self._norm(Ztr)), col(Htr), col(Ltr), col(Etr), col(Ptr), col(Ytr)
        zv, hv, lv, ev, pvv = T(self._norm(Zva)), col(Hva), col(Lva), col(Eva), col(Pva)
        slope = np.median(np.maximum((Ltr - Ytr) / np.maximum(Htr, 1) * 1000.0, 1e-3))
        self.net = RateNet(N_NET, a0=float(slope))
        self.tau_raw = torch.nn.Parameter(torch.tensor(float(np.log(np.expm1(9.0)))))       # tau ban đầu 10 chu kỳ
        opt = torch.optim.AdamW(list(self.net.parameters()) + [self.tau_raw], lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps, eta_min=self.lr * 0.05)
        best, state, bad = np.inf, None, 0
        for step in range(1, self.steps + 1):
            self.net.train()
            idx = torch.from_numpy(rng.randint(0, len(z), size=min(self.batch, len(z))))
            loss = Fnn.mse_loss(self._pred_t(z[idx], h[idx], l[idx], e[idx], pv[idx]), y[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.net.parameters()) + [self.tau_raw], 5.0)
            opt.step(); sched.step()
            if step % self.ev == 0:
                self.net.eval()
                with torch.no_grad():
                    v = float(np.abs(self._pred_t(zv, hv, lv, ev, pvv).numpy().ravel() - Yva).mean())
                if v < best - 1e-7:
                    best, bad = v, 0
                    state = ({k: t.detach().clone() for k, t in self.net.state_dict().items()}, self.tau_raw.detach().clone())
                else:
                    bad += 1
                    if bad >= self.pat:
                        break
        self.net.load_state_dict(state[0]); self.tau_raw = torch.nn.Parameter(state[1]); self.net.eval()
        self.info = dict(val_MAE=best, steps=step, tau=self.tau())
        return self

    @torch.no_grad()
    def params(self, Z):
        a, lam = self.net(torch.from_numpy(self._norm(Z).astype(np.float32)))
        return a.numpy().ravel().astype(float), lam.numpy().ravel().astype(float)


def fit_gbdt_rate(Z, H, B, Y, Zv, Hv, Bv, Yv, seed):
    """GBDT học tốc độ giảm trung bình của xu thế trên tầm h, r = (B − SOH(k+h))/(h/1000), với trọng số (h/1000)²:
    tổng bình phương có trọng số đúng bằng tổng bình phương của phần giảm, nhưng phần giảm h·r tự bằng 0 tại h = 0 nên
    tầm ngắn không bị gộp chung ngăn với tầm dài. Số cây chọn theo MAE validation như E16."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from itertools import islice
    hk, hkv = H / 1000.0, Hv / 1000.0
    X, Xv = np.column_stack([Z, hk]), np.column_stack([Zv, hkv])
    kw = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
              random_state=seed)
    m = HistGradientBoostingRegressor(max_iter=50, warm_start=True, **kw)
    best, bad = None, 0
    for it in range(50, 1001, 50):
        m.set_params(max_iter=it); m.fit(X, (B - Y) / hk, sample_weight=hk ** 2)
        v = float(np.abs(np.clip(Bv - hkv * m.predict(Xv), *CLIP) - Yv).mean())
        if best is None or v < best[0] - 1e-6:
            best, bad = (v, it), 0
        else:
            bad += 1
            if bad >= 4:
                break
    if best[1] != m.max_iter:
        m = HistGradientBoostingRegressor(max_iter=best[1], **kw).fit(X, (B - Y) / hk, sample_weight=hk ** 2)
    return m, dict(val_MAE=best[0], n_trees=best[1])


class ForecasterR:
    def __init__(self, gb, nets, tau, info, h1=3.0, h_mix=0.0):
        self.gb, self.nets, self.tau, self.info, self.h1, self.h_mix = gb, nets, tau, info, h1, h_mix

    @classmethod
    def fit(cls, tr, va, Xn, seed: int, members=(0, 1, 2)):
        Str, Sva = samples_r(tr, Xn), samples_r(va, Xn)
        nets = [PINNForecasterR(seed=1000 * seed + m).fit(Str, Sva) for m in members]
        tau = float(np.mean([n.tau() for n in nets]))
        Ztr, Htr, Ltr, Etr, Ptr, Ytr = Str
        Zva, Hva, Lva, Eva, Pva, Yva = Sva
        dec = lambda L, E, H, P: L + E * np.exp(-H / tau) + P
        gb, ginfo = fit_gbdt(Ztr, Htr, dec(Ltr, Etr, Htr, Ptr), Ytr, Zva, Hva, dec(Lva, Eva, Hva, Pva), Yva, seed=seed)
        h1 = float(max(3.0, np.quantile(Htr, 1 / 255)))          # bề rộng ngăn đầu tiên của trục h trong GBDT
        h_mix = 0.1 * life_scale(list(tr) + list(va))             # GBDT đạt trọng số ½ từ tầm 0,1L
        return cls(gb, nets, tau, dict(n_train_samples=int(len(Ytr)), gbdt=ginfo, pinn=[n.info for n in nets], tau=tau,
                                       h1=h1, h_mix=h_mix), h1, h_mix)

    def predict(self, z, dec, hh):
        n = len(hh)
        base = dec['level'] + dec['e'] * np.exp(-hh / self.tau) + periodic_at(dec, hh)
        # phần giảm của xu thế bằng 0 tại h = 0; cây gộp các tầm rất ngắn vào cùng một ngăn nên không tự bảo đảm điều này.
        # Dưới tầm h1 thay bằng đoạn thẳng từ 0 tới giá trị tại h1; từ h1 trở đi giữ nguyên dự đoán của cây.
        hc = np.maximum(hh, self.h1)
        d = -gbdt_predict(self.gb, np.repeat(z[None], n, 0), hc, np.zeros(n))
        d = np.where(hh < self.h1, d * hh / self.h1, d)
        P = {'gbdt_r': base - d}
        ab = [net.params(z[None]) for net in self.nets]
        P['pinn_r'] = np.mean([dec['level'] + dec['e'] * np.exp(-hh / net.tau()) + periodic_at(dec, hh)
                               - drop_np(a[0], lam[0], hh / 1000.0) for net, (a, lam) in zip(self.nets, ab)], axis=0)
        P['ket_hop_r0'] = 0.5 * P['gbdt_r'] + 0.5 * P['pinn_r']
        w = 0.5 * np.minimum(1.0, hh / self.h_mix) if self.h_mix > 0 else np.full(n, 0.5)
        P['ket_hop_r'] = w * P['gbdt_r'] + (1 - w) * P['pinn_r']
        return {m: np.clip(v, *CLIP) for m, v in P.items()}


# --------------------------------------------------------------------------- một lượt
def out_path(ds, r, k, root=ROOT):
    return f'{root}/rc_{ds}_cv{r}-{k}of5.npz'


def event_rows(c):
    """Hàng ngay sau một lần hồi phục: SOH đo vượt trung vị 5 hàng trước quá EV_JUMP; gộp sự kiện cách nhau <= 5 hàng,
    chỉ xét sau đỉnh của giai đoạn kích hoạt đầu đời (SOH của nhiều cell tăng trong vài chục chu kỳ đầu, không phải hồi
    phục) và trước 10 hàng cuối; lấy tối đa EV_MAX sự kiện rải đều."""
    import pandas as pd
    s = c.soh.astype(float); n = len(s)
    pre = pd.Series(s).shift(1).rolling(5, min_periods=3).median().values
    idx = np.flatnonzero((s - pre) > EV_JUMP)
    start = max(10, int(np.argmax(s[:max(1, int(0.3 * n))])) + 5)     # sau đỉnh của giai đoạn kích hoạt đầu đời
    ev = []
    for i in idx:
        if start <= i <= n - 11 and (not ev or i - ev[-1] > 5):
            ev.append(int(i))
    if len(ev) > EV_MAX:
        ev = [ev[j] for j in np.unique(np.linspace(0, len(ev) - 1, EV_MAX).round().astype(int))]
    return ev


def run_fold(ds: str, r: int, k: int, root: str = ROOT, members=(0, 1, 2)) -> float:
    t0 = time.time()
    path = out_path(ds, r, k, root)
    cells = load_dataset('PINN4SOH/data', ds, clean='causal')
    by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, 0.10)
    tr, va, te = ([by[i] for i in sp[p]] for p in ('train', 'val', 'test'))
    norm = Normalizer('global').fit(tr)
    Xn = {c.cid: norm.transform(c) for c in cells}
    L, tau = life_scale(cells), TAU[ds]
    old = Forecaster.fit(tr, va, Xn, seed=100 * r + k, members=members)        # đúng mô hình E22
    new = ForecasterR.fit(tr, va, Xn, seed=100 * r + k, members=members)
    est = {p: e20_estimates(ds, r, k, p) for p in ('test', 'val')}
    est_raw = {p: e20_estimates(ds, r, k, p, raw=True) for p in ('test', 'val')}
    step = max(1, int(round(L / 200)))
    grid = np.arange(step, int(3 * L) + step, step, dtype=float)

    def forecast(c, sc, part, origins):
        s_old = _history(c, sc, est[part]) if sc == 'uoc' else c.soh.astype(float)
        s_new = _history(c, sc, est_raw[part]) if sc == 'uoc' else c.soh.astype(float)
        Z, lev = origin_features(s_old, c.cycle, Xn[c.cid], c.temp_k)
        res = {}
        for i in origins:
            hj = (c.cycle[i + 1:] - c.cycle[i]).astype(float)
            hh = np.concatenate([hj, grid])
            P, _, _ = old.predict(Z, lev, i, hh, need_phys=False)
            P = {m: P[m] for m in ('giu', 'gbdt', 'pinn', 'ket_hop')}
            dec = decompose(s_new, c.cycle, i)
            P.update(new.predict(features_r(s_new, c.cycle, Xn[c.cid], c.temp_k, i, dec), dec, hh))
            res[i] = (hj, {m: v[:len(hj)] for m, v in P.items()}, {m: v[len(hj):] for m, v in P.items()}, dec)
        return res

    def soh_origins(c):
        n = len(c.soh)
        return sorted({max(5, int(round(q * (n - 1)))) for q in ORIGINS if int(round(q * (n - 1))) < n - 1})

    resid = {(sc, m): [] for sc in SCEN for m in ('ket_hop', 'ket_hop_r')}
    for c in va:
        if len(c.soh) < 8:
            continue
        for sc in SCEN:
            for i, (hj, P, _, _) in forecast(c, sc, 'val', soh_origins(c)).items():
                for m in ('ket_hop', 'ket_hop_r'):
                    resid[(sc, m)].append(np.column_stack([hj / L, c.soh[i + 1:] - P[m]]))
    qtab = {key: calibrate(v) for key, v in resid.items()}
    cross = lambda p: first_cross(p, grid, tau, grid[-1])

    T, RUL, DEC = [], [], []
    for ci, c in enumerate(te):
        n = len(c.soh)
        if n < 8:
            continue
        neol = eol_cycle(c, tau)
        rorig = []
        if neol is not None:
            for q in RUL_ORIGINS:
                i = int(np.searchsorted(c.cycle, q * neol, side='right') - 1)
                if i >= 5 and c.cycle[i] < neol:
                    rorig.append((q, i))
        so, ev = soh_origins(c), event_rows(c)
        for si, sc in enumerate(SCEN):
            out = forecast(c, sc, 'test', sorted(set(so) | set(ev) | {i for _, i in rorig}))
            for otype, lst in ((0, so), (1, ev)):
                for oq, i in enumerate(lst):
                    hj, P, _, dec = out[i]
                    lo, hi = band(qtab[(sc, 'ket_hop')], hj, L)
                    lo_r, hi_r = band(qtab[(sc, 'ket_hop_r')], hj, L)
                    T.append(np.column_stack([np.full(len(hj), si), np.full(len(hj), ci), np.full(len(hj), otype),
                                              np.full(len(hj), oq), np.full(len(hj), c.cycle[i]), hj, c.soh[i + 1:],
                                              *[P[m] for m in MODELS_R], P['ket_hop'] + lo, P['ket_hop'] + hi,
                                              P['ket_hop_r'] + lo_r, P['ket_hop_r'] + hi_r]))
                    DEC.append([si, ci, otype, oq, c.cycle[i], dec['level'], dec['e'], dec['P'], dec['amp'], dec['r2'],
                                dec['slope']])
            for q, i in rorig:
                hj, P, G, dec = out[i]
                # chu kỳ cuối đời thật tính trên trung vị trượt (đã làm trơn dao động), nên RUL của các mô hình _r
                # cũng tính trên dự báo đã bỏ phần dao động, tránh cắt ngưỡng sớm ở một đáy dao động
                per = periodic_at(dec, grid)
                G = {m: (np.clip(v - per, *CLIP) if m.endswith('_r') else v) for m, v in G.items()}
                lo_r, hi_r = band(qtab[(sc, 'ket_hop_r')], grid, L)
                RUL.append([si, ci, q, c.cycle[i], neol - c.cycle[i], *[cross(G[m]) for m in MODELS_R],
                            cross(G['ket_hop_r'] + lo_r), cross(G['ket_hop_r'] + hi_r)])
    info = dict(ds=ds, repeat=r, fold=k, L=L, tau=tau, grid_step=step, old=old.info, new=new.info,
                time_s=time.time() - t0, nominal_ah=[nominal_ah(c) for c in te],
                q={f'{a}|{b}': v.tolist() for (a, b), v in qtab.items()})
    os.makedirs(root, exist_ok=True)
    tmp = path[:-4] + f'.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp, traj=np.vstack(T).astype(np.float32), rul=np.asarray(RUL, np.float64),
                        dec=np.asarray(DEC, np.float64), cells=np.asarray([c.cid for c in te]),
                        info=np.asarray(json.dumps(info)))
    os.replace(tmp, path)
    tt = np.vstack(T)
    j = 7 + MODELS_R.index('ket_hop_r')
    return float(np.abs(tt[:, j] - tt[:, 6])[(tt[:, 0] == 1) & (tt[:, 2] == 0)].mean())
