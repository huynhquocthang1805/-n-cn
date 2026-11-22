"""
Chi phí tính toán khi triển khai PINN-lai và HiPINN: số tham số, dung lượng lưu và độ trễ suy luận trên CPU một luồng.

    python analysis/compute_cost.py      # -> results/compute_cost.csv (bản sao ở report/generated/data/)

Mô hình đo là mô hình của công cụ dòng lệnh (tools/demo.py), học ở 30 % nhãn trên một fold của mỗi bộ:
  PINN-lai   ba mạng nghiệm 17-128-128-64-1 (khi suy luận chỉ cần mạng nghiệm, không cần mạng tốc độ), một GBDT học trên
             cùng cell có nhãn với cùng cách chọn số cây như các thí nghiệm chính, và bộ lọc xu thế trên 10 ước lượng
             gần nhất. Mạng của công cụ chỉ có một thành viên, nên ba thành viên được đo bằng ba bản sao của nó (cùng
             kích thước, cùng thời gian).
  HiPINN     demo/<bộ>_hipinn.pkl: HiGBDT, ba PINN-dự báo, nhánh theo đời (GBDT đời còn lại và khuôn quỹ đạo).
Độ trễ: trung vị của nhiều lần gọi sau khi làm nóng, một luồng, đo bằng time.perf_counter. PINN-lai: một chu kỳ mới (ba
lượt truyền thẳng, một lượt GBDT cho một hàng, bộ lọc trên 10 giá trị). HiPINN: một lần dự báo từ một gốc cho mọi tầm tới
3L, gồm cả tính đặc trưng lịch sử của cell.
"""
from __future__ import annotations
import os as _os, sys as _sys
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'tools'))
import copy, os, pickle, shutil, statistics, time
import numpy as np
import pandas as pd
import torch

OUT = 'results'
COPY = 'report/generated/data'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']


def timeit(fn, n=300, warm=20):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)


def nbytes(obj):
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def n_params(mod):
    return int(sum(p.numel() for p in mod.parameters()))


def main():
    import demo as D
    from pinnsoh.baselines import cells_of, xy, fit_predict
    from pinnsoh.data import Normalizer
    from pinnsoh.postproc import causal_trend
    torch.set_num_threads(1)
    rows = []
    for ds in DS:
        model, norm, c, split = D.load_model(f'demo/{ds}.pt')
        by = {x.cid: x for x in cells_of(ds)}
        # mạng nghiệm: mọi tham số trừ mạng tốc độ (chỉ dùng khi học)
        p_all = n_params(model); p_dyn = n_params(model.dyn) if getattr(model, 'dyn', None) is not None else 0
        p_sol = p_all - p_dyn
        sol = copy.deepcopy(model).eval()
        if getattr(sol, 'dyn', None) is not None:
            sol.dyn = None
        # GBDT cùng cách học với các thí nghiệm chính, trên cell có nhãn của fold của công cụ
        nrm = Normalizer('global').fit([by[i] for i in split['train']])
        Xtr, ytr, _ = xy([by[i] for i in split['labelled']], nrm)
        Xva, yva, _ = xy([by[i] for i in split['val']], nrm)
        from sklearn.ensemble import HistGradientBoostingRegressor
        _, _, hp = fit_predict('gbdt', Xtr, ytr, Xva, yva, Xva[:1], seed=0)
        gb = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                           l2_regularization=1.0, early_stopping=False, max_iter=hp['n_trees'],
                                           random_state=0).fit(Xtr, ytr)
        n_nodes = int(sum(len(p[0].nodes) for p in gb._predictors))
        cell = by[split['test'][0]]
        X = torch.from_numpy(norm.transform(cell)); t = torch.from_numpy(D.cycle_input(cell)[:, None])
        x1, t1, g1 = X[-1:], t[-1:], Xva[-1:]
        hist = np.linspace(0.95, 0.94, 10); tt = np.arange(10, dtype=float) / 1000
        nets = [sol, copy.deepcopy(sol), copy.deepcopy(sol)]

        def step():
            with torch.no_grad():
                for m in nets:
                    m.u(x1, t1)
            gb.predict(g1)
            causal_trend(hist, tt, 10)
        with torch.no_grad():
            ms_nn = timeit(lambda: [m.u(x1, t1) for m in nets])
        ms_gb = timeit(lambda: gb.predict(g1))
        ms_flt = timeit(lambda: causal_trend(hist, tt, 10))
        ms_lai = timeit(step)
        b_nets = 3 * nbytes(sol.state_dict()); b_gb = nbytes(gb)
        # HiPINN của công cụ
        M = pickle.load(open(f'demo/{ds}_hipinn.pkl', 'rb'))
        co = M.model
        comp = dict(higbdt=nbytes((co.__dict__.get('m_ls'), co.__dict__.get('gb'))), pinn=nbytes(co.__dict__.get('nets')),
                    life=nbytes(co.__dict__.get('life')))
        p_fc = int(sum(n_params(m) for net in (co.__dict__.get('nets') or []) for m in [getattr(net, 'net', None)]
                       if isinstance(m, torch.nn.Module)))
        L = M.L; step_h = max(1, int(round(L / 200)))
        hh = np.arange(step_h, int(3 * L) + step_h, step_h, dtype=float)
        fc_cell = max((by[i] for i in split['test']), key=lambda x: len(x.cycle))
        i0 = int(0.4 * len(fc_cell.cycle))
        ms_hip = timeit(lambda: M.forecast(fc_cell, i0, hh), n=40, warm=3)
        rows.append(dict(ds=ds, sol_params=p_sol, dyn_params=p_dyn, gbdt_trees=int(hp['n_trees']), gbdt_nodes=n_nodes,
                         lai_bytes=b_nets + b_gb, nets_bytes=b_nets, gbdt_bytes=b_gb, ms_nn3=ms_nn, ms_gbdt=ms_gb,
                         ms_filter=ms_flt, ms_lai=ms_lai, hipinn_bytes=nbytes(M), hip_higbdt_bytes=comp['higbdt'],
                         hip_pinn_bytes=comp['pinn'], hip_life_bytes=comp['life'], hip_pinn_params=p_fc,
                         n_horizons=len(hh), hist_rows=i0 + 1, ms_hipinn=ms_hip))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True); os.makedirs(COPY, exist_ok=True)
    R.to_csv(f'{OUT}/compute_cost.csv', index=False)
    shutil.copy(f'{OUT}/compute_cost.csv', f'{COPY}/compute_cost.csv')
    print(R.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
