"""
Nhánh dự báo theo đời cell (E33): đặt hình dạng suy giảm của các cell train lên thang đời dự đoán của cell cần dự báo.

Lý do: dự báo trực tiếp theo tầm (GBDT, PINN-dự báo) cho đường gần thẳng: ở giữa đời dự báo thấp hơn thực tế, ở 10 %
cuối đời cao hơn, vì thời điểm gãy (knee) của từng cell khác nhau và mẫu học ở tầm dài chỉ đến từ các cell sống lâu.
Nhánh này tách hai việc:

1. Đời còn lại: GBDT học log(N_j − c) từ đặc trưng tại gốc (23 đặc trưng gốc và lịch sử SOH, như GBDT lịch sử), với
   N_j là chu kỳ cuối đời (SOH chạm ngưỡng tau) của cell train j; cell train chưa chạm ngưỡng không dùng.
2. Hình dạng: mỗi cell train j cho một khuôn f_j(u) = SOH_j(u·N_j) theo thời gian chuẩn hoá u (trung vị trượt 9 hàng;
   sau hàng cuối giữ giá trị cuối: trên validation lần lặp 0, kéo dài theo độ dốc cuối đời làm dự báo sụp xuống giới hạn
   dưới mỗi khi đời bị đoán ngắn đi nhiều, ví dụ MIT 28 so với 13 ×10⁻³). Tại gốc c_k của cell cần dự báo, với
   N̂ = c_k + đời còn lại dự đoán:  ŷ(c_k + h) = l_k + trung bình_j [f_j((c_k + h)/N̂) − f_j(c_k/N̂)],
   lấy trên K khuôn có đặc trưng gốc (chuẩn hoá) gần nhất ở cùng thời điểm chuẩn hoá, hoặc mọi khuôn (K = 0).

Chỉ dùng hàng <= gốc của cell cần dự báo; số cây và K chọn trên validation như các mô hình khác của E33.
"""
from __future__ import annotations
import copy
import numpy as np

UG = np.linspace(0.0, 4.0, 4001)        # lưới thời gian chuẩn hoá của khuôn
STAGES = (100, 250, 450)
KS = (5, 10, 20, 0)                     # 0: mọi khuôn


def _template(c, N):
    import pandas as pd
    u = c.cycle / float(N)
    f = pd.Series(c.soh.astype(float)).rolling(9, center=True, min_periods=1).median().values
    return np.interp(UG, u, f)                  # ngoài [u_0, u_cuối]: giữ giá trị đầu/cuối


class LifeBranch:
    def __init__(self, models, tpl, tz, tN, tcyc, mu, sd):
        self.models, self.tpl, self.tz, self.tN, self.tcyc, self.mu, self.sd = models, tpl, tz, tN, tcyc, mu, sd
        self.n_trees, self.K = STAGES[-1], 0

    @staticmethod
    def fit(ids, by, feat, tau, seed):
        """ids: cell học (có nhãn); feat[cid] = (z, rich, l) từ forecast_hist.features."""
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        from .forecast import eol_cycle
        from .forecast_hist import weights
        X, Y, G, tpl, tz, tN, tcyc = [], [], [], [], [], [], []
        for j, cid in enumerate(ids):
            c = by[cid]; N = eol_cycle(c, tau)
            if N is None:
                continue
            iE = int(np.searchsorted(c.cycle, N))
            rows = np.unique(np.linspace(5, max(5, iE - 1), 32).round().astype(int))
            rows = rows[rows < iE]
            z, rich, _ = feat[cid]
            if len(rows):
                X.append(rich[rows]); Y.append(np.log(N - c.cycle[rows])); G.append(np.full(len(rows), j))
            tpl.append(_template(c, N)); tz.append(z); tN.append(float(N)); tcyc.append(c.cycle.astype(float))
        X, Y, G = np.vstack(X), np.concatenate(Y), np.concatenate(G)
        w = weights(G)
        m = HGB(loss='squared_error', learning_rate=.05, max_leaf_nodes=15, l2_regularization=1., min_samples_leaf=20,
                early_stopping=False, warm_start=True, random_state=seed)
        models = {}
        for n in STAGES:
            m.set_params(max_iter=n); m.fit(X, Y, sample_weight=w * len(w))
            models[n] = copy.deepcopy(m)
        zs = np.vstack([z[5:] for z in tz])
        return LifeBranch(models, np.vstack(tpl), tz, np.asarray(tN), tcyc, zs.mean(0), zs.std(0) + 1e-9)

    def predict(self, d, n_trees=None, K=None):
        """d: mẫu (z, rich, lev, h, origin = chu kỳ tại gốc). Trả dự báo SOH theo hàng."""
        n_trees = self.n_trees if n_trees is None else n_trees
        K = self.K if K is None else K
        key = np.column_stack([d['origin'], d['rich']])
        uk, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        R = np.exp(self.models[n_trees].predict(d['rich'][first]))
        out = np.empty(len(d['h']))
        for g in range(len(uk)):
            ix = np.flatnonzero(inv == g); i0 = first[g]
            c0 = float(d['origin'][i0]); Nh = c0 + float(R[g])
            u0 = c0 / Nh; uu = (c0 + d['h'][ix]) / Nh
            if K and K < len(self.tN):
                zt = (d['z'][i0] - self.mu) / self.sd
                rows = [min(int(np.searchsorted(cy, u0 * N)), len(cy) - 1) for cy, N in zip(self.tcyc, self.tN)]
                zz = np.vstack([(z[r] - self.mu) / self.sd for z, r in zip(self.tz, rows)])
                nb = np.argsort(((zz - zt) ** 2).sum(1), kind='stable')[:K]
            else:
                nb = np.arange(len(self.tN))
            T = self.tpl[nb]
            F0 = np.array([np.interp(u0, UG, t) for t in T])
            Fu = np.vstack([np.interp(uu, UG, t) for t in T])
            out[ix] = d['lev'][ix] + (Fu - F0[:, None]).mean(0)
        return out

    def choose(self, va, score):
        """Chọn số cây và K theo `score` (MAE có trọng số) trên mẫu validation."""
        best = min(((score(self.predict(va, n, K)), n, K) for n in STAGES for K in KS), key=lambda x: x[0])
        self.n_trees, self.K = best[1], best[2]
        return dict(n_trees=self.n_trees, K=self.K, val_MAE=best[0], n_templates=len(self.tN))
