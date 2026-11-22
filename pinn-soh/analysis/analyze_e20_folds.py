"""
PINN-lai và HiPINN-lai với mọi lựa chọn làm riêng trong từng fold, chỉ bằng cell xác thực của fold đó.

    python analysis/analyze_e20_folds.py     # -> results/e20f_*.csv (bản sao ở report/generated/data/)

Đọc dự đoán đã lưu (tập xác thực và tập kiểm tra) của E20 (3 PINN-semi + GBDT, 4 mức nhãn, 3 lần lặp x 5 fold) và E29
(cùng các mô hình có đặc trưng lịch sử của đặc trưng sạc). Không huấn luyện thêm gì.

  pinn          PINN-semi đơn lẻ (thành viên 0)
  gbdt          GBDT
  pinn_lai      PINN-lai cố định: 1/2 trung bình 3 PINN-semi + 1/2 GBDT, bộ lọc xu thế nhân quả w = 10. Cấu hình chốt ở giai
                đoạn phát triển (E20dev), trước khi đọc dự đoán trên tập kiểm tra
  pinn_lai_tc   PINN-lai tinh chỉnh: (omega, w) chọn trên lưới RIÊNG cho từng (bộ, mức nhãn, lần lặp, fold) bằng MAE theo
                cell trên cell xác thực của chính fold đó. Ý tưởng tinh chỉnh có sau khi xem kết quả của bản cố định, nên
                đây là phân tích hậu kiểm, chỉ để đối chiếu
  pinn_h        PINN-semi thành viên 0 có đặc trưng lịch sử (E29)
  lai_h         PINN-lai cố định dựng từ 3 PINN-semi có lịch sử và GBDT (có lịch sử nếu E29 chọn cho GBDT của bộ đó)
  hipinn_lai    trong từng fold: lai_h nếu MAE trên cell xác thực của fold đó thấp hơn pinn_lai ít nhất 2 %, ngược lại
                pinn_lai. Bộ không có dạng lịch sử nào được chọn ở E29dev (MIT) trùng pinn_lai

Sai số: MAE theo cell trên tập kiểm tra, mỗi cell lấy trung bình qua 3 lần lặp. So ghép cặp: Wilcoxon theo cell, hiệu
chỉnh Holm qua các bộ có mặt trong phép so. RMSE theo cell, MAPE (tính chung mọi chu kỳ) và R^2 (bỏ 10 chu kỳ đầu mỗi
cell, ghép 5 fold của một lần lặp) tính như analysis/analyze_metrics.py.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import json, os, shutil
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh import experiments as E
from pinnsoh.postproc import by_cell, causal_trend

OUT = 'results'
COPY = 'report/generated/data'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = (0.1, 0.3, 0.5, 0.7)
REPEATS = (0, 1, 2)
OMEGAS, WINDOWS = (0.3, 0.4, 0.5, 0.6, 0.7), (5, 10, 20, 40)
W_FIXED, OM_FIXED = 10, 0.5
HIST_GAIN = 0.98                    # chỉ dùng lịch sử khi MAE xác thực giảm ít nhất 2 %
MODELS = ['pinn', 'gbdt', 'pinn_lai', 'pinn_lai_tc', 'pinn_h', 'lai_h', 'hipinn_lai']


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def _read(path):
    d = json.load(open(path))
    out = {}
    for part, key in (('test', 'test_predictions'), ('val', 'val_predictions')):
        tp = d[key]
        ids = np.asarray(tp['cell_ids'], dtype=object)
        out[part] = dict(y=np.asarray(tp['y'], float), p=np.asarray(tp['p'], float),
                         cell=ids[np.asarray(tp['cell'], int)], cyc=np.asarray(tp['cycle'], float))
    return out


def units():
    """Đường dẫn tệp kết quả cho mỗi (bộ, mức nhãn, lần lặp, fold), lấy thẳng từ định nghĩa thí nghiệm."""
    U = {}
    key = lambda a: (a.get('dataset', a.get('ds')), a.get('label_frac', a.get('frac')), a.get('repeat', a.get('r')),
                     a.get('fold', a.get('k')))
    for j in E.E20():
        a = j.args
        if j.kind == 'train' and a['tag'] == 'P':
            U.setdefault(key(a), {}).setdefault('P', {})[a['member']] = j.out
        elif j.kind == 'baseline':
            U.setdefault(key(a), {})['G'] = j.out
    for j in E.E29():
        a = j.args
        if j.kind == 'train' and a['tag'] == 'H':
            U.setdefault(key(a), {}).setdefault('H', {})[a['member']] = j.out
        elif j.kind == 'baseline':
            U.setdefault(key(a), {})['GH'] = j.out
    return U


def cell_mae(y, p, cell):
    return {c: float(np.abs(p[cell == c] - y[cell == c]).mean()) for c in np.unique(cell)}


def mean_cell_mae(y, p, cell):
    return float(np.mean(list(cell_mae(y, p, cell).values())))


def lai(t, p3, g, om=OM_FIXED, w=W_FIXED):
    return by_cell(causal_trend, om * p3 + (1 - om) * g, t['cell'], t['cyc'], w=int(w))


def one_unit(key, u):
    """Dự đoán của mọi mô hình trên tập kiểm tra của một đơn vị, cùng các lựa chọn làm trên tập xác thực."""
    need = [u['P'][m] for m in (0, 1, 2)] + [u['G']]
    if not all(os.path.exists(p) for p in need):
        return None
    P = [_read(u['P'][m]) for m in (0, 1, 2)]; G = _read(u['G'])
    for o in P[1:] + [G]:
        for part in ('test', 'val'):
            assert np.array_equal(o[part]['cell'], P[0][part]['cell']) and np.array_equal(o[part]['cyc'], P[0][part]['cyc'])
    pred, info = {}, dict(zip(('ds', 'f', 'r', 'k'), key))
    p3 = {pt: np.mean([m[pt]['p'] for m in P], axis=0) for pt in ('test', 'val')}
    g = {pt: G[pt]['p'] for pt in ('test', 'val')}
    T, V = P[0]['test'], P[0]['val']
    pred['pinn'] = T['p']; pred['gbdt'] = g['test']
    pred['pinn_lai'] = lai(T, p3['test'], g['test'])
    v_fixed = mean_cell_mae(V['y'], lai(V, p3['val'], g['val']), V['cell'])
    # tinh chỉnh riêng trong fold này, chỉ bằng cell xác thực của fold
    best = min(((mean_cell_mae(V['y'], lai(V, p3['val'], g['val'], om, w), V['cell']), om, w)
                for om in OMEGAS for w in WINDOWS))
    info.update(om_tc=best[1], w_tc=best[2], val_fixed=v_fixed, val_tc=best[0], n_val=len(np.unique(V['cell'])))
    pred['pinn_lai_tc'] = lai(T, p3['test'], g['test'], best[1], best[2])
    # lịch sử (E29): có ở các bộ mà E29dev chọn một dạng lịch sử
    pred['hipinn_lai'] = pred['pinn_lai']; info.update(hist_avail=False, hist_used=False)
    if 'H' in u and all(os.path.exists(p) for p in u['H'].values()) and len(u['H']) == 3:
        H = [_read(u['H'][m]) for m in (0, 1, 2)]
        for o in H:
            for part in ('test', 'val'):
                assert np.array_equal(o[part]['cell'], P[0][part]['cell']) and np.array_equal(o[part]['cyc'], P[0][part]['cyc'])
        gh = g
        if 'GH' in u:
            if not os.path.exists(u['GH']):
                return None
            GH = _read(u['GH']); gh = {pt: GH[pt]['p'] for pt in ('test', 'val')}
        p3h = {pt: np.mean([m[pt]['p'] for m in H], axis=0) for pt in ('test', 'val')}
        pred['pinn_h'] = H[0]['test']['p']
        pred['lai_h'] = lai(T, p3h['test'], gh['test'])
        v_hist = mean_cell_mae(V['y'], lai(V, p3h['val'], gh['val']), V['cell'])
        use = v_hist <= HIST_GAIN * v_fixed
        info.update(hist_avail=True, hist_used=bool(use), val_hist=v_hist)
        if use:
            pred['hipinn_lai'] = pred['lai_h']
    return T, pred, info


def metrics_rows(key, T, pred):
    """Theo cell: MAE, SSE, n; theo lần lặp (ghép 5 fold): SSE, n, MAPE, cho R^2."""
    rows = []
    for m, p in pred.items():
        e = p - T['y']
        for c in np.unique(T['cell']):
            i = T['cell'] == c
            first10 = T['cyc'][i] >= np.sort(T['cyc'][i])[min(10, i.sum() - 1)]
            rows.append(dict(ds=key[0], f=key[1], r=key[2], k=key[3], model=m, cell=c, mae=float(np.abs(e[i]).mean()),
                             sse=float((e[i] ** 2).sum()), n=int(i.sum()), ape=float((np.abs(e[i]) / T['y'][i]).sum()),
                             sse10=float((e[i][first10] ** 2).sum()), sy10=float(T['y'][i][first10].sum()),
                             syy10=float((T['y'][i][first10] ** 2).sum()), n10=int(first10.sum())))
    return rows


def wilcox(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


def main():
    U = units()
    rows, infos, miss = [], [], 0
    for key in sorted(U):
        if key[0] not in DS or key[1] not in FRACS or key[2] not in REPEATS:
            continue
        res = one_unit(key, U[key])
        if res is None:
            miss += 1
            continue
        T, pred, info = res
        rows += metrics_rows(key, T, pred); infos.append(info)
    if miss:
        print(f'cảnh báo: thiếu tệp ở {miss} đơn vị (bỏ qua)')
    C = pd.DataFrame(rows); I = pd.DataFrame(infos)
    # theo cell (trung bình qua lần lặp)
    pc = C.groupby(['ds', 'f', 'model', 'cell']).agg(mae=('mae', 'mean'), sse=('sse', 'sum'), n=('n', 'sum')).reset_index()
    pc['rmse'] = np.sqrt(pc.sse / pc.n)
    # RMSE theo cell như analyze_metrics: căn của từng (cell, lần lặp), rồi trung bình theo lần lặp và theo cell
    pr = C.groupby(['ds', 'f', 'model', 'cell', 'r']).agg(sse=('sse', 'sum'), n=('n', 'sum')).reset_index()
    pr['rmse'] = np.sqrt(pr.sse / pr.n)
    rmse_cell = pr.groupby(['ds', 'f', 'model', 'cell']).rmse.mean().groupby(['ds', 'f', 'model']).mean()
    rep = C.groupby(['ds', 'f', 'model', 'r']).agg(ape=('ape', 'sum'), n=('n', 'sum'), sse10=('sse10', 'sum'),
                                                   sy10=('sy10', 'sum'), syy10=('syy10', 'sum'), n10=('n10', 'sum'))
    rep['mape'] = 100 * rep.ape / rep.n
    rep['r2_10'] = 1 - rep.sse10 / (rep.syy10 - rep.sy10 ** 2 / rep.n10)
    rep = rep.groupby(['ds', 'f', 'model'])[['mape', 'r2_10']].mean()
    S = pc.groupby(['ds', 'f', 'model']).mae.mean().rename('mae').to_frame().join(rmse_cell.rename('rmse_cell')).join(rep)
    S = S.reset_index()
    # so ghép cặp
    pairs = [('pinn_lai', 'pinn'), ('pinn_lai', 'gbdt'), ('pinn_lai_tc', 'pinn_lai'), ('pinn_lai_tc', 'gbdt'),
             ('hipinn_lai', 'pinn_lai'), ('hipinn_lai', 'gbdt'), ('lai_h', 'pinn_lai'), ('pinn_h', 'pinn')]
    tests = []
    for a, b in pairs:
        for f in FRACS:
            part = []
            for ds in DS:
                A = pc[(pc.ds == ds) & (pc.f == f) & (pc.model == a)].set_index('cell').mae
                B = pc[(pc.ds == ds) & (pc.f == f) & (pc.model == b)].set_index('cell').mae
                cells = sorted(set(A.index) & set(B.index))
                if not cells or (a in ('pinn_h', 'lai_h') and not I[(I.ds == ds) & (I.f == f)].hist_avail.any()):
                    continue
                x, y = A.reindex(cells).values, B.reindex(cells).values
                part.append(dict(a=a, b=b, f=f, ds=ds, n=len(cells), ratio=x.mean() / y.mean(),
                                 a_better=int((x < y).sum()), p=wilcox(x, y)))
            if part:
                ph = holm([q['p'] for q in part])
                for q, h in zip(part, ph):
                    q['p_holm'] = float(h)
                tests += part
    Tt = pd.DataFrame(tests)
    choice = I.groupby(['ds', 'f']).agg(n_units=('k', 'size'), hist_avail=('hist_avail', 'mean'),
                                        hist_used=('hist_used', 'mean'), om_tc_med=('om_tc', 'median'),
                                        w_tc_med=('w_tc', 'median')).reset_index()
    os.makedirs(OUT, exist_ok=True); os.makedirs(COPY, exist_ok=True)
    for name, D in (('summary', S), ('tests', Tt), ('choices', choice), ('units', I)):
        D.to_csv(f'{OUT}/e20f_{name}.csv', index=False)
        shutil.copy(f'{OUT}/e20f_{name}.csv', f'{COPY}/e20f_{name}.csv')
    with pd.option_context('display.width', 220, 'display.max_rows', 300):
        W = S.pivot_table(index=['ds', 'f'], columns='model', values='mae') * 1e3
        print(W.round(2).to_string())
        print(choice.to_string(index=False))
        print(Tt.assign(ratio=Tt.ratio.round(3)).to_string(index=False))


if __name__ == '__main__':
    main()
