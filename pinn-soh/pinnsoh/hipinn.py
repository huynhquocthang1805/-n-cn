"""
HiPINN — mô hình dự báo SOH có đặc trưng lịch sử của đồ án, ở dạng học một lần rồi dự báo cho một cell bất kỳ.

HiPINN lấy trung vị của ba nhánh tại từng (gốc, tầm): HiGBDT, PINN-dự báo, nhánh theo đời. Các nhánh học và chọn
cấu hình như E33 (pinnsoh/forecast_combo.py): cấu hình của từng nhánh chọn trên một nửa cell validation, khoảng 90 %
hiệu chỉnh trên nửa còn lại. Với cùng cách chia, dự báo trùng với cột hipinn của lượt E33 tương ứng.

    from pinnsoh.hipinn import HiPINN
    m = HiPINN.fit('MIT', lab=..., val=..., train=..., seed=0)
    P = m.forecast(cell, i, hh)                 # i: hàng gốc; hh: các tầm (chu kỳ); soh=...: lịch sử ước lượng
"""
from __future__ import annotations
import numpy as np

from .data import Normalizer, load_dataset
from .forecast import TAU, CLIP, life_scale, train_samples, fit_gbdt, gbdt_predict, PINNForecaster, first_cross
from . import forecast_hist as H
from .forecast_life import LifeBranch
from .forecast_combo import Combo, ORIGINS_REL, _rows_at


class HiPINN:
    VERSION = 2

    def __init__(self, model, norm, L, tau, q, info):
        self.model, self.norm, self.L, self.tau, self.info = model, norm, L, tau, info
        self.q = dict(q)                        # khoảng 90 % theo kịch bản lịch sử: 'do' (đo), có thể thêm 'uoc'

    @classmethod
    def fit(cls, dataset, lab, val, train, seed=0, members=(0, 1, 2), root='PINN4SOH/data'):
        """lab: cell học (có nhãn); val: cell validation (chia đôi theo tên: chọn / hiệu chỉnh); train: mọi cell train
        (để chuẩn hoá và tính thang đời L)."""
        cells = load_dataset(root, dataset, clean='causal'); by = {c.cid: c for c in cells}
        sel, cal = sorted(val)[::2], sorted(val)[1::2]
        norm = Normalizer('global').fit([by[i] for i in train])
        L, tau = life_scale([by[i] for i in train]), TAU[dataset]
        origins = np.maximum(1, np.rint(np.array(ORIGINS_REL) * L)).astype(int).tolist()
        used = sorted(set(lab) | set(val))
        xn = {i: norm.transform(by[i]) for i in used}
        feat = {i: H.features(by[i], xn[i]) for i in used}
        tr = H.sample(lab, by, feat)
        va = H.sample(sel, by, feat, origins)
        gb, ginfo = fit_gbdt(*train_samples([by[i] for i in lab], xn), *train_samples([by[i] for i in sel], xn),
                             seed=seed)
        old_va = np.clip(gbdt_predict(gb, va['z'], va['h'], va['lev']), *CLIP)
        m_ls, cfg_ls, a_ls, _, score = H.search(tr, va, seed, old_va)
        Ztr, Htr, Ltr, Ytr = train_samples([by[i] for i in lab], xn)
        Zs, Hs, Ls, Ys = train_samples([by[i] for i in sel], xn)
        nets = [PINNForecaster(seed=1000 * seed + m).fit(Ztr, Htr, Ltr, Ytr, Zs, Hs, Ls, Ys) for m in members]
        life = LifeBranch.fit(lab, by, feat, tau, seed)
        life_info = life.choose(va, score)
        model = Combo(gb, m_ls, cfg_ls, a_ls, nets, w=1.0, life=life)
        obj = cls(model, norm, L, tau, {}, dict(dataset=dataset, seed=seed, life=life_info,
                                                history=dict(cfg_ls, weight_new=a_ls), gbdt=ginfo,
                                                n_lab=len(lab), n_sel=len(sel), n_cal=len(cal), sel=sel, cal=cal,
                                                origins=origins))
        obj.q['do'] = obj.calibrate([by[i] for i in cal])
        return obj

    def calibrate(self, cells, soh_of=None):
        """q: phân vị 90 % của |y − ŷ| / sqrt(1 + h/L) trên các cell cho trước (mọi chu kỳ sau các gốc của E33);
        soh_of(cell) trả lịch sử SOH dùng làm đầu vào (mặc định: dung lượng đo)."""
        by = {c.cid: c for c in cells}
        feat = {c.cid: H.features(c, self.norm.transform(c), soh=None if soh_of is None else soh_of(c)) for c in cells}
        d = H.sample(list(by), by, feat, self.info['origins'])
        p = self.model.predict(d)['hipinn']
        return float(np.quantile(np.abs(d['y'] - p) / np.sqrt(1 + d['h'] / self.L), .9, method='higher'))

    def forecast(self, cell, i, hh, soh=None, scen='do'):
        """Dự báo từ hàng gốc i của `cell` cho các tầm hh (chu kỳ). soh: lịch sử SOH dùng làm đầu vào (mặc định đo).
        Trả dict: dự báo của HiPINN và ba nhánh, biên dưới/trên của khoảng 90 %, mức SOH tại gốc, RUL và khoảng RUL."""
        hh = np.asarray(hh, float)
        z, rich, lev = H.features(cell, self.norm.transform(cell), soh=soh)
        P = self.model.predict(_rows_at(z, rich, lev, i, hh, cell.cycle[i]))
        band = self.q.get(scen, self.q['do']) * np.sqrt(1 + hh / self.L)
        lo, hi = np.clip(P['hipinn'] - band, *CLIP), np.clip(P['hipinn'] + band, *CLIP)
        cap = float(hh[-1])
        return dict(hipinn=P['hipinn'], gbdt_ls=P['gbdt_ls'], pinn=P['pinn'], doi=P['doi'], lo=lo, hi=hi,
                    level=float(lev[i]), rul=first_cross(P['hipinn'], hh, self.tau, cap),
                    rul_lo=first_cross(lo, hh, self.tau, cap), rul_hi=first_cross(hi, hh, self.tau, cap))
