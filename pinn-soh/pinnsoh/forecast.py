"""
E22 — dự báo SOH ở các chu kỳ tương lai, dung lượng còn lại (Ah) và số chu kỳ còn lại tới cuối đời (RUL).

Bài toán. Tại chu kỳ gốc k của một cell, biết đặc trưng sạc của mọi chu kỳ <= k và một chuỗi SOH lịch sử
s(0..k); dự báo SOH(k+h) với mọi h > 0. Từ đó dung lượng Q(k+h) = SOH(k+h)·C_danh_định và RUL = số chu kỳ
tới khi SOH xuống dưới ngưỡng cuối đời tau của bộ (TAU, chọn trước: mức thấp nhất, bước 0,01, mà ít nhất 80 % cell
của bộ chạm tới trong dữ liệu).

Hai kịch bản lịch sử cho cell test:
    'do'   lịch sử là dung lượng đo thật (BMS đo dung lượng định kỳ tới chu kỳ k)
    'uoc'  lịch sử là ước lượng PINN-lai của E20 (mức nhãn 70 %, ngoài fold) — không đo dung lượng cell test
Bộ dự báo học trên cell train với lịch sử đo thật, nên kịch bản 'uoc' đo thêm phần sai số ước lượng truyền sang.

Chia dữ liệu: đúng các fold của E13/E20 (5 fold theo cell, 10 % cell validation, lần lặp 0–2), nên mỗi cell được
dự báo đúng một lần mỗi lần lặp và ước lượng 'uoc' của nó đến từ mô hình chưa thấy cell đó.

Đặc trưng tại gốc k (chỉ dùng hàng có chu kỳ <= k):
    mức hiện tại l_k = causal_trend(s, w = 10) tại k; độ dốc OLS trên 30 và 100 hàng gần nhất (trên 1000 chu kỳ);
    mức ban đầu l_0 (trung bình 5 hàng đầu) và độ giảm l_0 − l_k; k/1000; 1000/T;
    trung bình 16 đặc trưng sạc đã chuẩn hoá trên 10 hàng gần nhất.

Mô hình dự báo:
    giu      giữ nguyên l_k
    tuyen    ngoại suy tuyến tính: l_k + min(độ dốc 100 hàng, 0)·h
    vatly    khớp nghiệm đóng của động học xám lên nửa sau lịch sử của chính cell (3 tham số, phạt lam) rồi ngoại suy
    gbdt     GBDT trực tiếp: (đặc trưng gốc, h) -> độ giảm l_k − SOH(k+h), ràng buộc đơn điệu tăng theo h
    pinn     PINN-dự báo: mạng (đặc trưng gốc) -> (a, lam); SOH(k+h) = l_k − dv(h; a, lam); 3 thành viên
    ket_hop  ½ gbdt + ½ pinn

Nghiệm đóng. Động học xám của mô hình ước lượng, dv/dN = a_0·e^{lam·v} với v = 1 − SOH (hệ số Arrhenius gộp vào a_0),
tích phân từ gốc k cho
    dv(h) = −ln(1 − lam·a·h)/lam,     a = tốc độ suy giảm tại k (trên 1000 chu kỳ), h tính bằng nghìn chu kỳ.
lam > 0: suy giảm tăng tốc (điểm gãy); lam < 0: chậm dần; lam -> 0: tuyến tính a·h. Dạng này không âm và
không giảm theo h với mọi (a >= 0, lam), nên dự báo luôn đơn điệu mà không cần số hạng phạt.

Khoảng dự báo 90 %: phân vị 5 % và 95 % của sai số có dấu (thật − dự báo) của mô hình kết hợp trên cell validation,
theo nhóm tầm dự báo h/L (L = trung vị số chu kỳ của bộ); RUL dưới/trên là lúc biên dưới/trên cắt ngưỡng.
"""
from __future__ import annotations
import json, os, time
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .data import Normalizer, cv_split, load_dataset, DATASET_INFO, TJU_NOMINAL
from .postproc import by_cell, causal_trend

ROOT = 'runs/E22'
DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
TAU = {'XJTU': 0.84, 'TJU': 0.81, 'MIT': 0.87, 'HUST': 0.81}
ORIGINS = (0.2, 0.4, 0.6)            # gốc dự báo SOH: tỉ lệ số hàng đã qua của cell (chỉ để chọn điểm đánh giá)
RUL_ORIGINS = (0.25, 0.5, 0.75)      # gốc dự báo RUL: tỉ lệ của chu kỳ cuối đời thật
H_REL = (0.1, 0.25, 0.5)             # tầm dự báo báo cáo trong bảng, theo L của bộ
BINS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, np.inf)
LMAX = 80.0
W_LEVEL, W_FEAT = 10, 10
MODELS = ['giu', 'tuyen', 'vatly', 'gbdt', 'pinn', 'ket_hop']
SCEN = ['do', 'uoc']
CLIP = (0.40, 1.15)
NOMINAL = {'XJTU': 2.0, 'MIT': 1.1, 'HUST': 1.1}


def nominal_ah(c) -> float:
    if c.dataset == 'TJU':
        return TJU_NOMINAL[c.cid.split('/')[1]]
    return NOMINAL[c.dataset]


def life_scale(cells) -> float:
    """L của bộ: trung vị số chu kỳ ghi được của các cell."""
    return float(np.median([c.cycle[-1] + 1 for c in cells]))


def eol_cycle(c, tau: float):
    """Chu kỳ cuối đời thật: chu kỳ đầu tiên mà trung vị trượt (cửa sổ 9, căn giữa) của SOH đo <= tau; None nếu chưa tới."""
    import pandas as pd
    m = pd.Series(c.soh.astype(float)).rolling(9, center=True, min_periods=1).median().values
    hit = np.flatnonzero(m <= tau)
    return int(c.cycle[hit[0]]) if len(hit) else None


# --------------------------------------------------------------------------- nghiệm đóng
def drop_np(a, lam, h):
    """dv(h) = −ln(1 − lam·a·h)/lam, an toàn số khi lam·a·h -> 0 và chặn trước điểm vô hạn."""
    a, lam, h = np.broadcast_arrays(np.asarray(a, float), np.asarray(lam, float), np.asarray(h, float))
    z = np.minimum(lam * a * h, 1 - 1e-6)
    small = np.abs(z) < 1e-6
    g = np.where(small, 1 + z / 2, -np.log1p(-z) / np.where(small, 1.0, z))
    return a * h * g


def drop_t(a, lam, h):
    z = torch.clamp(lam * a * h, max=1 - 1e-4)
    small = z.abs() < 1e-4
    g = torch.where(small, 1 + z / 2, -torch.log1p(-z) / torch.where(small, torch.ones_like(z), z))
    return a * h * g


def rul_closed(a, lam, d):
    """h tới khi dv = d (nghìn chu kỳ): (1 − e^{−lam·d})/(lam·a); d <= 0 -> 0."""
    a, lam, d = (np.asarray(x, float) for x in (a, lam, d))
    small = np.abs(lam * d) < 1e-6
    num = np.where(small, d, -np.expm1(-lam * d) / np.where(small, 1.0, lam))
    return np.where(d <= 0, 0.0, num / np.maximum(a, 1e-9))


# --------------------------------------------------------------------------- đặc trưng tại gốc
def _seg(x, i, a):
    S = np.concatenate([[0.0], np.cumsum(x, axis=0)]) if x.ndim == 1 else \
        np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    return S[i + 1] - S[a]


def causal_slope(p, t, w):
    """Độ dốc OLS của p theo t trên w điểm gần nhất (kể cả hiện tại), 0 khi chưa đủ 3 điểm."""
    p = np.asarray(p, float); t = np.asarray(t, float) - float(t[0])
    n = len(p); i = np.arange(n); a = np.maximum(0, i - w + 1); cnt = (i - a + 1).astype(float)
    st, sy, stt, sty = _seg(t, i, a), _seg(p, i, a), _seg(t * t, i, a), _seg(t * p, i, a)
    sxx = stt - st * st / cnt; sxy = sty - st * sy / cnt
    ok = (sxx > 1e-12) & (cnt >= 3)
    return np.where(ok, sxy / np.where(ok, sxx, 1.0), 0.0)


def origin_features(s, cyc, Xn, temp_k):
    """Ma trận đặc trưng gốc Z cho MỌI hàng của một cell (hàng i chỉ dùng hàng <= i) và mức l."""
    cyc = np.asarray(cyc, float)
    lev = causal_trend(s, cyc, W_LEVEL)
    s30 = causal_slope(s, cyc, 30) * 1000.0
    s100 = causal_slope(s, cyc, 100) * 1000.0
    l0 = float(np.mean(s[:5]))
    n = len(s); i = np.arange(n); a = np.maximum(0, i - W_FEAT + 1)
    fm = _seg(Xn.astype(float), i, a) / (i - a + 1)[:, None]
    Z = np.column_stack([lev, s30, s100, np.full(n, l0), l0 - lev, cyc / 1000.0, np.full(n, 1000.0 / temp_k), fm])
    return Z.astype(np.float32), lev


def train_samples(cells, Xn_of, n_origin=40, n_h=40, rng_seed=0):
    """Mẫu (gốc, tầm) từ lịch sử đo thật của các cell: Z gốc, h (chu kỳ), mức l_k, SOH thật tại k+h."""
    Zs, H, L, Y = [], [], [], []
    for c in cells:
        s, cyc = c.soh.astype(float), c.cycle
        n = len(s)
        if n < 8:
            continue
        Z, lev = origin_features(s, cyc, Xn_of[c.cid], c.temp_k)
        for i in np.unique(np.linspace(5, n - 2, n_origin).round().astype(int)):
            js = np.unique(np.linspace(i + 1, n - 1, n_h).round().astype(int))
            Zs.append(np.repeat(Z[i:i + 1], len(js), 0)); H.append(cyc[js] - cyc[i])
            L.append(np.full(len(js), lev[i])); Y.append(s[js])
    cat = np.concatenate
    return cat(Zs), cat(H).astype(np.float32), cat(L).astype(np.float32), cat(Y).astype(np.float32)


# --------------------------------------------------------------------------- mô hình
def fit_gbdt(Ztr, Htr, Ltr, Ytr, Zva, Hva, Lva, Yva, seed):
    """GBDT dự báo độ giảm l_k − SOH(k+h), đơn điệu tăng theo h; số cây chọn theo MAE validation như E16."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.column_stack([Ztr, Htr / 1000.0]); Xv = np.column_stack([Zva, Hva / 1000.0])
    cst = [0] * Ztr.shape[1] + [1]
    kw = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
              early_stopping=False, monotonic_cst=cst, random_state=seed)
    m = HistGradientBoostingRegressor(max_iter=50, warm_start=True, **kw)
    best, bad = None, 0
    for it in range(50, 1001, 50):
        m.set_params(max_iter=it); m.fit(X, Ltr - Ytr)
        v = float(np.abs(np.clip(Lva - m.predict(Xv), *CLIP) - Yva).mean())
        if best is None or v < best[0] - 1e-6:
            best, bad = (v, it), 0
        else:
            bad += 1
            if bad >= 4:
                break
    if best[1] != m.max_iter:                         # cây 1..n của bản warm start trùng bản dừng ở n cây
        m = HistGradientBoostingRegressor(max_iter=best[1], **kw).fit(X, Ltr - Ytr)
    return m, dict(val_MAE=best[0], n_trees=best[1])


def gbdt_predict(m, Z, H, L):
    return L - m.predict(np.column_stack([Z, H / 1000.0]))


class RateNet(nn.Module):
    """Đặc trưng gốc -> (a >= 0, |lam| <= LMAX) của nghiệm đóng."""
    def __init__(self, d, hidden=(64, 64), a0=0.2):
        super().__init__()
        layers, k = [], d
        for h in hidden:
            layers += [nn.Linear(k, h), nn.SiLU()]; k = h
        self.body = nn.Sequential(*layers); self.out = nn.Linear(k, 2)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([float(np.log(np.expm1(a0))), 0.0]))

    def forward(self, z):
        o = self.out(self.body(z))
        return Fnn.softplus(o[:, :1]), LMAX * torch.tanh(o[:, 1:] / LMAX)


class PINNForecaster:
    def __init__(self, seed=0, steps=4000, batch=1024, lr=3e-3, eval_every=100, patience=10):
        self.seed, self.steps, self.batch, self.lr, self.ev, self.pat = seed, steps, batch, lr, eval_every, patience

    def _pred_t(self, z, h, l):
        a, lam = self.net(z)
        return torch.clamp(l - drop_t(a, lam, h / 1000.0), CLIP[0], CLIP[1])

    def fit(self, Ztr, Htr, Ltr, Ytr, Zva, Hva, Lva, Yva):
        torch.manual_seed(self.seed); rng = np.random.RandomState(self.seed)
        self.mu, self.sd = Ztr.mean(0), Ztr.std(0) + 1e-6
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        z, h, l, y = T((Ztr - self.mu) / self.sd), T(Htr[:, None]), T(Ltr[:, None]), T(Ytr[:, None])
        zv, hv, lv = T((Zva - self.mu) / self.sd), T(Hva[:, None]), T(Lva[:, None])
        slope = np.median(np.maximum((Ltr - Ytr) / np.maximum(Htr, 1) * 1000.0, 1e-3))
        self.net = RateNet(Ztr.shape[1], a0=float(slope))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps, eta_min=self.lr * 0.05)
        best, state, bad = np.inf, None, 0
        for step in range(1, self.steps + 1):
            self.net.train()
            idx = torch.from_numpy(rng.randint(0, len(z), size=min(self.batch, len(z))))
            loss = Fnn.mse_loss(self._pred_t(z[idx], h[idx], l[idx]), y[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            opt.step(); sched.step()
            if step % self.ev == 0:
                self.net.eval()
                with torch.no_grad():
                    v = float(np.abs(self._pred_t(zv, hv, lv).numpy().ravel() - Yva).mean())
                if v < best - 1e-7:
                    best, bad, state = v, 0, {k: t.detach().clone() for k, t in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.pat:
                        break
        self.net.load_state_dict(state); self.net.eval()
        self.info = dict(val_MAE=best, steps=step)
        return self

    @torch.no_grad()
    def params(self, Z):
        a, lam = self.net(torch.from_numpy(((Z - self.mu) / self.sd).astype(np.float32)))
        return a.numpy().ravel().astype(float), lam.numpy().ravel().astype(float)


def fit_physics(cyc, s, frac=0.5, pen=0.005 * np.sqrt(50)):
    """Khớp SOH(N) = u0 − dv(N − N0; a0, lam) lên nửa sau của lịch sử, có phạt lam (lam/20 nhân pen, ngang một
    phần dư cỡ 5·10⁻³ khi lam = 20 trên 50 điểm); trả (a_k, lam): tốc độ tại hàng cuối và hệ số gãy.
    Khớp trên cả lịch sử, không phạt, cho dự báo bùng nổ ngay sau gốc; cấu hình này là tốt nhất trong bốn biến thể
    có điểm gãy khi thử trên cell validation của fold đầu (lần lặp 0, fold 0)."""
    from scipy.optimize import least_squares
    m = max(5, int(round(frac * len(s))))
    cyc, s = np.asarray(cyc[-m:], float), np.asarray(s[-m:], float)
    N = (cyc - cyc[0]) / 1000.0
    if len(s) < 5:
        return 1e-4, 0.0
    sl = max(1e-3, -float(np.polyfit(N, s, 1)[0])) if N[-1] > 0 else 1e-3
    lo, hi = [0.4, np.log(1e-5), -LMAX], [1.3, np.log(50.0), LMAX]
    best = None
    for lam0 in (0.0, 10.0, -10.0):
        x0 = [float(np.clip(s[:5].mean(), 0.41, 1.29)), float(np.clip(np.log(sl), lo[1] + 1e-3, hi[1] - 1e-3)), lam0]
        f = lambda p: np.concatenate([p[0] - drop_np(np.exp(p[1]), p[2], N) - s, [pen * p[2] / 20.0]])
        try:
            r = least_squares(f, x0, bounds=(lo, hi), loss='soft_l1', f_scale=0.005, max_nfev=200)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return sl, 0.0
    a0, lam = float(np.exp(best.x[1])), float(best.x[2])
    den = 1 - lam * a0 * N[-1]
    return (a0 / den if den > 1e-3 else a0 / 1e-3), lam


# --------------------------------------------------------------------------- bộ dự báo đầy đủ
class Forecaster:
    """GBDT trực tiếp + 3 PINN-dự báo học trên lịch sử đo của cell train, kèm các mô hình đối chứng; dùng chung cho
    E22 và demo. `predict` trả dự báo của mọi mô hình từ một hàng gốc cho các tầm `hh` (chu kỳ)."""

    def __init__(self, gb, nets, info):
        self.gb, self.nets, self.info = gb, nets, info

    @classmethod
    def fit(cls, tr, va, Xn, seed: int, members=(0, 1, 2)):
        Ztr, Htr, Ltr, Ytr = train_samples(tr, Xn)
        Zva, Hva, Lva, Yva = train_samples(va, Xn)
        gb, ginfo = fit_gbdt(Ztr, Htr, Ltr, Ytr, Zva, Hva, Lva, Yva, seed=seed)
        nets = [PINNForecaster(seed=1000 * seed + m).fit(Ztr, Htr, Ltr, Ytr, Zva, Hva, Lva, Yva) for m in members]
        return cls(gb, nets, dict(n_train_samples=int(len(Ytr)), gbdt=ginfo, pinn=[n.info for n in nets]))

    def predict(self, Z, lev, i, hh, cyc=None, s=None, need_phys=True):
        """Z, lev: đặc trưng gốc và mức của mọi hàng (origin_features); i: hàng gốc; cyc, s: lịch sử cho 'vatly'."""
        n = len(hh)
        Zi = np.repeat(Z[i:i + 1], n, 0); li = np.full(n, lev[i])
        P = {'giu': li.copy(), 'tuyen': li + min(float(Z[i, 2]), 0.0) * hh / 1000.0}
        if need_phys:
            ak, lk = fit_physics(cyc[:i + 1], s[:i + 1])
            P['vatly'] = li - drop_np(ak, lk, hh / 1000.0)
        else:
            P['vatly'] = np.full(n, np.nan)
        P['gbdt'] = gbdt_predict(self.gb, Zi, hh, li)
        ab = [net.params(Z[i:i + 1]) for net in self.nets]
        P['pinn'] = np.mean([li - drop_np(a[0], lam[0], hh / 1000.0) for a, lam in ab], axis=0)
        P['ket_hop'] = 0.5 * P['gbdt'] + 0.5 * P['pinn']
        P = {m: np.clip(v, *CLIP) for m, v in P.items()}
        return P, float(np.mean([a[0] for a, _ in ab])), float(np.mean([lam[0] for _, lam in ab]))


def calibrate(resid):
    """resid: mảng (h/L, thật − dự báo) -> bảng phân vị 5 % và 95 % theo nhóm BINS (nhóm thiếu dữ liệu lấy nhóm trước)."""
    R = np.vstack(resid); b = np.digitize(R[:, 0], BINS) - 1
    q = []
    for j in range(len(BINS) - 1):
        e = R[b == j, 1]
        q.append((np.quantile(e, 0.05), np.quantile(e, 0.95)) if len(e) >= 20 else q[-1] if q else (-0.02, 0.02))
    return np.asarray(q)


def band(qtab, h, L):
    j = np.clip(np.digitize(np.asarray(h, float) / L, BINS) - 1, 0, len(BINS) - 2)
    return qtab[j, 0], qtab[j, 1]


def first_cross(p, hgrid, tau, cap):
    hit = np.flatnonzero(np.asarray(p) <= tau)
    return float(hgrid[hit[0]]) if len(hit) else float(cap)


# --------------------------------------------------------------------------- lịch sử ước lượng (E20)
def e20_estimates(ds, r, k, part='test'):
    """PINN-lai (½ trung bình 3 PINN-semi + ½ GBDT, lọc quỹ đạo w = 10) của fold (r, k), mức nhãn 70 %:
    dict cell_id -> (chu kỳ, SOH ước lượng). Đúng công thức của analysis/analyze_e20.py."""
    s = 100 * r + k
    P = [f'runs/E20/P_{ds}_pinn_semi_f0.7_s{s}_global_global_cv{r}-{k}of5' + ('' if m == 0 else f'_m{m}') + '.json'
         for m in range(3)]
    G = f'runs/E20/gbdt_{ds}_f0.7_cv{r}-{k}of5.json'
    key = 'test_predictions' if part == 'test' else 'val_predictions'
    ld = lambda p: json.load(open(p))[key]
    ps = [ld(p) for p in P]; g = ld(G)
    ids = ps[0]['cell_ids']; cell = np.asarray([ids[i] for i in ps[0]['cell']]); cyc = np.asarray(ps[0]['cycle'], float)
    for o in ps[1:] + [g]:
        assert o['cycle'] == ps[0]['cycle'] and o['cell'] == ps[0]['cell'], (ds, r, k)
    lai = 0.5 * np.mean([np.asarray(o['p'], float) for o in ps], axis=0) + 0.5 * np.asarray(g['p'], float)
    lai = by_cell(causal_trend, lai, cell, cyc, w=W_LEVEL)
    out = {}
    for c in np.unique(cell):
        i = np.flatnonzero(cell == c); i = i[np.argsort(cyc[i])]
        out[c] = (cyc[i].astype(np.int64), lai[i])
    return out


# --------------------------------------------------------------------------- một lượt
def out_path(ds, r, k, root=ROOT):
    return f'{root}/fc_{ds}_cv{r}-{k}of5.npz'


def _history(c, scen, est):
    if scen == 'do':
        return c.soh.astype(float)
    cyc, p = est[c.cid]
    assert np.array_equal(cyc, c.cycle), c.cid
    return p


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
    fc = Forecaster.fit(tr, va, Xn, seed=100 * r + k, members=members)
    est = {'test': e20_estimates(ds, r, k, 'test'), 'val': e20_estimates(ds, r, k, 'val')}
    step = max(1, int(round(L / 200)))
    grid = np.arange(step, int(3 * L) + step, step, dtype=float)       # tầm (chu kỳ) để tìm lúc cắt ngưỡng

    def forecast(c, scen, part, origins, need_phys=True):
        """Dự báo mọi mô hình từ các hàng gốc `origins`; trả list (i, h thật, dự báo thật, dự báo lưới, a, lam)."""
        s = _history(c, scen, est[part]) if scen == 'uoc' else c.soh.astype(float)
        Z, lev = origin_features(s, c.cycle, Xn[c.cid], c.temp_k)
        res = []
        for i in origins:
            hj = (c.cycle[i + 1:] - c.cycle[i]).astype(float)
            P, a_m, lam_m = fc.predict(Z, lev, i, np.concatenate([hj, grid]), c.cycle, s, need_phys)
            res.append((i, hj, {m: v[:len(hj)] for m, v in P.items()}, {m: v[len(hj):] for m, v in P.items()},
                        a_m, lam_m))
        return res

    def soh_origins(c):
        n = len(c.soh)
        return sorted({max(5, int(round(q * (n - 1)))) for q in ORIGINS if int(round(q * (n - 1))) < n - 1})

    # ---- hiệu chỉnh khoảng dự báo trên cell validation (mô hình kết hợp, theo kịch bản)
    resid = {sc: [] for sc in SCEN}
    for c in va:
        if len(c.soh) < 8:
            continue
        for sc in SCEN:
            for i, hj, P, _, _, _ in forecast(c, sc, 'val', soh_origins(c), need_phys=False):
                resid[sc].append(np.column_stack([hj / L, c.soh[i + 1:] - P['ket_hop']]))
    qtab = {sc: calibrate(resid[sc]) for sc in SCEN}
    cross = lambda p: first_cross(p, grid, tau, grid[-1])

    T, RUL, PAR = [], [], []
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
        so = soh_origins(c)
        for si, sc in enumerate(SCEN):
            allo = sorted(set(so) | {i for _, i in rorig})
            out = dict(zip(allo, forecast(c, sc, 'test', allo)))
            for oq, i in enumerate(so):
                _, hj, P, _, a_m, lam_m = out[i]
                lo, hi = band(qtab[sc], hj, L)
                T.append(np.column_stack([np.full(len(hj), si), np.full(len(hj), ci), np.full(len(hj), oq),
                                          np.full(len(hj), c.cycle[i]), hj, c.soh[i + 1:],
                                          *[P[m] for m in MODELS], P['ket_hop'] + lo, P['ket_hop'] + hi]))
                PAR.append([si, ci, oq, c.cycle[i], a_m, lam_m])
            for q, i in rorig:
                _, hj, P, G, a_m, lam_m = out[i]
                lo, hi = band(qtab[sc], grid, L)
                RUL.append([si, ci, q, c.cycle[i], neol - c.cycle[i], *[cross(G[m]) for m in MODELS],
                            cross(G['ket_hop'] + lo), cross(G['ket_hop'] + hi), a_m, lam_m])
    info = dict(ds=ds, repeat=r, fold=k, L=L, tau=tau, grid_step=step, **fc.info, time_s=time.time() - t0,
                nominal_ah=[nominal_ah(c) for c in te], q=dict((sc, qtab[sc].tolist()) for sc in SCEN))
    os.makedirs(root, exist_ok=True)
    tmp = path[:-4] + f'.{os.getpid()}.tmp.npz'
    np.savez_compressed(tmp, traj=np.vstack(T).astype(np.float32), rul=np.asarray(RUL, np.float64),
                        par=np.asarray(PAR, np.float64), cells=np.asarray([c.cid for c in te]),
                        info=np.asarray(json.dumps(info)))
    os.replace(tmp, path)
    tt = np.vstack(T)
    return float(np.abs(tt[:, 6 + MODELS.index('ket_hop')] - tt[:, 5])[tt[:, 0] == 1].mean())
