"""
E26: dữ liệu có xác định được hai tham số vật lý của động học xám không — hệ số đầu gối λ và năng lượng hoạt hoá Ea.

    python analysis/analyze_e26.py      # -> results/e26_params.csv, results/e26_mae.csv

Ở cấu hình chuẩn (trọng số ODE 0,02), λ và Ea học được nằm sát giá trị khởi tạo (1 và 30 kJ/mol) trên mọi bộ. Hai cách
giải thích: dữ liệu không chứa thông tin về chúng (không nhận dạng được), hoặc ràng buộc ODE quá yếu để kéo chúng đi. E26
phân biệt hai khả năng trên bộ có nhiều mức nhiệt độ môi trường (TJU: 25, 35, 45 °C), nơi Ea có cơ hội được đo.
PINN-semi, 70 % nhãn, cùng fold và hạt giống với nhánh chuẩn:
    A0   chuẩn: alpha_ode 0,02, khởi tạo λ = 1, Ea = 30 kJ/mol (lấy lại từ E13 nhánh G)
    A0i  như A0, khởi tạo λ = 3, Ea = 60 kJ/mol
    A1   alpha_ode 0,2
    A2   alpha_ode 1,0
    A2i  alpha_ode 1,0, khởi tạo λ = 3, Ea = 60 kJ/mol
    A3   alpha_ode 1,0 và ràng buộc Euler giữa hai chu kỳ (alpha_euler 2,0)
Một tham số được dữ liệu xác định nếu giá trị học được không phụ thuộc khởi tạo. Chỉ số: phần dịch theo khởi tạo
s = (θ_khởi tạo khác − θ_chuẩn) / (θ0_khác − θ0_chuẩn), ghép theo (lần lặp, fold); s ≈ 1 là tham số đứng yên ở khởi tạo,
s ≈ 0 là dữ liệu kéo về cùng một giá trị. Sai số: MAE theo cell test, trung bình qua lần lặp, Wilcoxon ghép cặp theo cell
so với A0.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import glob, json, os, re
import numpy as np
import pandas as pd
from dataclasses import asdict
from scipy import stats

from pinnsoh.train import Config

OUT = 'results'
BASE = {'TJU': 'runs/E13/G_TJU_pinn_semi_f0.7_*_cv0-*of5.json'}
ARMS = ['A0', 'A0i', 'A1', 'A2', 'A2i', 'A3']
PAIRS = [('A0i', 'A0'), ('A2i', 'A2')]           # (khởi tạo khác, khởi tạo chuẩn)
# khoá cấu hình được phép khác giữa các nhánh (các khoá save_* chỉ quyết định lưu gì); mọi khoá khác phải trùng nhánh chuẩn
FREE = {'tag', 'out_dir', 'lam_init', 'ea_init_kj', 'alpha_ode', 'alpha_euler', 'save_predictions', 'save_model',
        'save_val_predictions'}


def runs(ds, arm):
    pat = BASE[ds] if arm == 'A0' else f'runs/E26/E26{arm}_{ds}_pinn_semi_f0.7_*of5.json'
    out = []
    for p in sorted(glob.glob(pat)):
        if re.search(r'_m\d+\.json$', p):        # thành viên phụ của một nhóm mạng
            continue
        d = json.load(open(p)); c = d['config']
        tp = d['test_predictions']
        y, pr, ci = (np.asarray(tp[k]) for k in ('y', 'p', 'cell'))
        mae = {cid: float(np.abs(y[ci == i] - pr[ci == i]).mean()) for i, cid in enumerate(tp['cell_ids'])}
        out.append(dict(r=c['repeat'], k=c['fold'], cfg=c, lam=d['physics_params']['lam'],
                        ea=d['physics_params']['Ea_kJ_mol'], mae=mae,
                        lam0=c.get('lam_init', 1.0), ea0=c.get('ea_init_kj', 30.0),
                        a_ode=c['alpha_ode'], a_eu=c.get('alpha_euler', 0.0) or 0.0))
    return out


def check_same(ref, other, ds, arm):
    # khoá thêm vào Config sau khi chạy E13 không có trong JSON cũ: so bằng giá trị mặc định
    dflt = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(Config()).items()}
    get = lambda c, k: c.get(k, dflt.get(k))
    for a in other:
        b = next(x for x in ref if (x['r'], x['k']) == (a['r'], a['k']))
        diff = {k for k in set(a['cfg']) | set(b['cfg']) if k not in FREE and get(a['cfg'], k) != get(b['cfg'], k)}
        assert not diff, (ds, arm, sorted(diff))


def cell_mae(R):
    s = pd.DataFrame([dict(cell=c, r=x['r'], mae=v) for x in R for c, v in x['mae'].items()])
    return s.groupby('cell').mae.mean()


def main():
    P, M = [], []
    for ds in BASE:
        R = {arm: runs(ds, arm) for arm in ARMS}
        n0 = len(R['A0'])
        for arm in ARMS:
            assert len(R[arm]) == n0, (ds, arm, len(R[arm]), n0)
            check_same(R['A0'], R[arm], ds, arm)
        base = cell_mae(R['A0'])
        for arm in ARMS:
            X = R[arm]
            lam, ea = np.array([x['lam'] for x in X]), np.array([x['ea'] for x in X])
            row = dict(ds=ds, arm=arm, n=len(X), alpha_ode=X[0]['a_ode'], alpha_euler=X[0]['a_eu'],
                       lam0=X[0]['lam0'], ea0=X[0]['ea0'],
                       lam_med=np.median(lam), lam_lo=lam.min(), lam_hi=lam.max(),
                       ea_med=np.median(ea), ea_lo=ea.min(), ea_hi=ea.max(),
                       lam_move=np.median(np.abs(lam - X[0]['lam0'])), ea_move=np.median(np.abs(ea - X[0]['ea0'])))
            for a, b in PAIRS:
                if arm == a:
                    ref = {(x['r'], x['k']): x for x in R[b]}
                    sl = [(x['lam'] - ref[x['r'], x['k']]['lam']) / (x['lam0'] - ref[x['r'], x['k']]['lam0']) for x in X]
                    se = [(x['ea'] - ref[x['r'], x['k']]['ea']) / (x['ea0'] - ref[x['r'], x['k']]['ea0']) for x in X]
                    row.update(ref=b, s_lam=np.median(sl), s_lam_lo=np.min(sl), s_lam_hi=np.max(sl),
                               s_ea=np.median(se), s_ea_lo=np.min(se), s_ea_hi=np.max(se))
            P.append(row)
            m = cell_mae(X).reindex(base.index)
            d = m.values - base.values
            M.append(dict(ds=ds, arm=arm, n_cells=len(m), mae=m.mean(), mae_a0=base.mean(), ratio=m.mean() / base.mean(),
                          better=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    P, M = pd.DataFrame(P), pd.DataFrame(M)
    os.makedirs(OUT, exist_ok=True)
    P.to_csv(f'{OUT}/e26_params.csv', index=False)
    M.to_csv(f'{OUT}/e26_mae.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(P.round(3).to_string(index=False))
        print(M.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
