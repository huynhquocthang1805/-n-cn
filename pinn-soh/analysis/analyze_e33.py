"""
E33: dự báo kết hợp GBDT lịch sử (repo G-doan) với PINN-dự báo — bảng, kiểm định, hình.

    python analysis/analyze_e33.py      # -> results/e33_*.csv, results/fig_e33_*.pdf|png

Sai số của một cell (như G-doan): với mỗi gốc, MAE trên mọi chu kỳ đo được sau gốc; lấy trung bình qua các gốc. Bảng
tổng hợp: trung bình qua cell trong mỗi lần lặp, rồi trung bình ± độ lệch chuẩn qua 3 lần lặp. Kiểm định: Wilcoxon ghép
cặp theo cell (sai số trung bình qua lần lặp), Holm qua bốn bộ trong mỗi (ngân sách, kịch bản, phép so).
Kèm số liệu đã lưu của G-doan (improvements/forecasts trong repo G-doan, nếu có) tính cùng cách để đối chiếu.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.forecast_combo import TCOL, RCOL, MODELS, SCEN, ROOT

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST']
GDOAN = os.environ.get('GDOAN', '/home/user/huynhquocthang1805/g-doan')
LABEL = {'giu': 'Giữ nguyên', 'tuyen': 'Tuyến tính', 'gbdt': 'GBDT (E22)', 'gbdt_ls': 'GBDT lịch sử',
         'pinn': 'PINN-dự báo', 'ket_hop': 'Kết hợp E22', 'ket_hop_ls': 'Kết hợp mới'}
HBINS = [0, .1, .25, .5, 1, np.inf]
HLAB = ['0-0.1', '0.1-0.25', '0.25-0.5', '0.5-1', '>1']


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); out[j] = min(1.0, run)
    return out


def a_wins(d):
    """1 nếu tổng hạng |d| của các cell có d < 0 (a sai số nhỏ hơn) lớn hơn tổng hạng của các cell có d > 0: hướng
    của kiểm định Wilcoxon, không bị vài cell ngoại lai chi phối như hiệu của trung bình."""
    d = np.asarray(d, float); d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    return int(r[d < 0].sum() > r[d > 0].sum())


def flip_k(d):
    """Số cell có hiệu lớn nhất ngược hướng của kiểm định cần bỏ để hiệu của hai trung bình đổi chiều theo hướng đó, và
    vị trí của cell có hiệu ngược hướng lớn nhất; (0, -1) khi trung bình và kiểm định cùng hướng."""
    d = np.asarray(d, float)
    t = d if a_wins(d) else -d                       # t > 0: cell đi ngược hướng của kiểm định
    if t.sum() <= 0:
        return 0, -1
    v = np.sort(t)[::-1]
    return int(np.argmax(t.sum() - np.cumsum(v) <= 0) + 1), int(np.argmax(t))


def load():
    cells, ruls, choices, horiz = [], [], [], []
    for f in sorted(glob.glob(f'{ROOT}/fc_*.npz')):
        q = np.load(f); i = json.loads(str(q['info']))
        meta = dict(budget=i['budget'], ds=i['ds'], r=i['repeat'], fold=i['fold'])
        T = pd.DataFrame(q['traj'], columns=TCOL); te = i['split']['test']
        choices.append(dict(**meta, w=i['w_combo'], hist=i['history']['config']['name'],
                            n_trees=i['history']['config']['n_trees'], a=i['history']['weight_new'], L=i['L'],
                            q_ls=i['q_ls'], q_kls=i['q_kls'], time_s=i['time_s'],
                            **{f'val_{m}': v for m, v in i['val_MAE'].items()}))
        T['hb'] = pd.cut(T.h / i['L'], HBINS, labels=HLAB)
        for (sc, ci), g in T.groupby(['scen', 'cell']):
            row = dict(**meta, scen=SCEN[int(sc)], cid=te[int(ci)])
            for m in MODELS:
                row[m] = (g[m] - g.y).abs().groupby(g.origin).mean().mean()
            for b in ('ls', 'kls'):
                cov = ((g.y >= g[f'lo_{b}']) & (g.y <= g[f'hi_{b}'])).astype(float)
                row[f'cov_{b}'] = cov.groupby(g.origin).mean().mean()
                row[f'wid_{b}'] = (g[f'hi_{b}'] - g[f'lo_{b}']).groupby(g.origin).mean().mean()
            cells.append(row)
            for hb, h in g.groupby('hb', observed=True):
                horiz.append(dict(**meta, scen=SCEN[int(sc)], cid=te[int(ci)], hb=hb,
                                  **{m: (h[m] - h.y).abs().groupby(h.origin).mean().mean() for m in MODELS}))
        if len(q['rul']):
            R = pd.DataFrame(q['rul'], columns=RCOL)
            R['scen'] = np.asarray(SCEN)[R.scen.astype(int)]; R['cid'] = np.asarray(te)[R.cell.astype(int)]
            for k, v in meta.items():
                R[k] = v
            ruls.append(R)
    return (pd.DataFrame(cells), pd.concat(ruls, ignore_index=True), pd.DataFrame(choices), pd.DataFrame(horiz))


def gdoan_cells():
    """Số đã lưu của G-doan (60 fold, ngân sách 30 %, lịch sử đo), cùng cách tính sai số theo cell."""
    rows = []
    for f in sorted(glob.glob(f'{GDOAN}/improvements/forecasts/*_r[0-2]_f[0-4].npz')):
        q = np.load(f); i = json.loads(str(q['info']))
        d = pd.DataFrame(q['data'], columns=i['columns'])
        for ci, g in d.groupby('cell'):
            rows.append(dict(ds=i['dataset'], r=i['repeat'], cid=i['split']['test'][int(ci)],
                             **{m: (g[m] - g.y).abs().groupby(g.origin).mean().mean()
                                for m in ('persistence', 'linear', 'gbdt', 'selected')}))
    return pd.DataFrame(rows)


def gdoan_anchor():
    """Bài toán cập nhật của G-doan (improvements/anchors): ước lượng SOH ở chu kỳ không đo từ lần đo gần nhất, lịch đo
    q20/q10/q05 (mỗi 0,2L/0,1L/0,05L chu kỳ) và M2 (biết mọi lần đo trước). Sai số theo cell như analyze.py của G-doan:
    MAE trên các hàng được chấm; trung bình qua cell mỗi lần lặp, rồi trung bình ± SD qua lần lặp."""
    tiers = ['q20', 'q10', 'q05', 'M2']; models = ['hold', 'linear', 'e32_style', 'history_delta', 'history_residual',
                                                  'selected']
    rows = []
    for f in sorted(glob.glob(f'{GDOAN}/improvements/anchors/*_r[0-2]_f[0-4].npz')):
        q = np.load(f); i = json.loads(str(q['info']))
        d = pd.DataFrame(q['data'], columns=i['columns'])
        for (ci, ti), g in d.groupby(['cell', 'tier']):
            rows.append(dict(ds=i['dataset'], r=i['repeat'], tier=tiers[int(ti)], cid=i['split']['test'][int(ci)],
                             **{m: float(np.abs(g[m] - g.y).mean()) for m in models}))
    if not rows:
        return None
    A = pd.DataFrame(rows)
    rep = A.groupby(['ds', 'tier', 'r'])[models].mean()
    S = rep.groupby(['ds', 'tier']).agg(['mean', 'std'])
    S.columns = [f'{a}' if b == 'mean' else f'{a}_sd' for a, b in S.columns]
    S = S.reset_index()
    pc = A.groupby(['ds', 'tier', 'cid'])[models].mean().reset_index()
    for c in ('history_delta', 'history_residual', 'selected'):
        S[f'p_{c}'] = [stats.wilcoxon(g[c] - g['e32_style']).pvalue for _, g in pc.groupby(['ds', 'tier'])]
        S[f'better_{c}'] = [int((g[c] < g['e32_style']).sum()) for _, g in pc.groupby(['ds', 'tier'])]
        S[f'wins_{c}'] = [a_wins(g[c] - g['e32_style']) for _, g in pc.groupby(['ds', 'tier'])]
        S[f'flip_{c}'] = [flip_k(g[c] - g['e32_style'])[0] for _, g in pc.groupby(['ds', 'tier'])]
    S['n_cell'] = [len(g) for _, g in pc.groupby(['ds', 'tier'])]
    S.to_csv(f'{OUT}/e33_gdoan_anchor.csv', index=False)
    return S


def main():
    os.makedirs(OUT, exist_ok=True)
    if not glob.glob(f'{ROOT}/fc_*.npz'):
        print(f'chưa có lượt nào trong {ROOT}: chạy python run.py E33 trước')
        return
    C, R, CH, H = load()
    print(f'{len(CH)} lượt; {len(C)} dòng (cell × kịch bản × lần lặp)')
    C.to_csv(f'{OUT}/e33_cells.csv', index=False); CH.to_csv(f'{OUT}/e33_choices.csv', index=False)
    keys = ['budget', 'scen', 'ds']
    # ---- tổng hợp: trung bình qua cell mỗi lần lặp, rồi trung bình ± SD qua lần lặp
    cols = MODELS + ['cov_ls', 'cov_kls', 'wid_ls', 'wid_kls']
    rep = C.groupby(keys + ['r'])[cols].mean().reset_index()
    S = rep.groupby(keys)[cols].agg(['mean', 'std'])
    S.columns = [f'{a}' if b == 'mean' else f'{a}_sd' for a, b in S.columns]
    S = S.reset_index()
    S['n_cell'] = C.groupby(keys).cid.nunique().values
    S['n_runs'] = CH.groupby(['budget', 'ds']).size().reindex(list(zip(S.budget, S.ds))).values
    S.to_csv(f'{OUT}/e33_summary.csv', index=False)
    # ---- kiểm định ghép cặp theo cell (trung bình qua lần lặp)
    pc = C.groupby(keys + ['cid'])[MODELS].mean().reset_index()
    pairs = [('ket_hop_ls', m) for m in MODELS if m != 'ket_hop_ls'] + [('gbdt_ls', 'gbdt'), ('gbdt_ls', 'ket_hop'),
                                                                        ('gbdt_ls', 'pinn')]
    rows = []
    for (b, sc), g in pc.groupby(['budget', 'scen']):
        for a, c in pairs:
            tmp = []
            for ds in DS:
                x = g[g.ds == ds]
                if len(x) < 3:
                    continue
                d = x[a].values - x[c].values
                k, top = flip_k(d)
                tmp.append(dict(budget=b, scen=sc, ds=ds, a=a, b=c, n=len(d), mae_a=x[a].mean(), mae_b=x[c].mean(),
                                ratio=x[a].mean() / x[c].mean(), a_better=int((d < 0).sum()), a_wins=a_wins(d),
                                p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0, flip_k=k,
                                top_cell=x.cid.values[top] if top >= 0 else '',
                                **{f'top_{m}': x[m].values[top] if top >= 0 else np.nan for m in MODELS}))
            for t, ph in zip(tmp, holm([t['p'] for t in tmp])):
                t['p_holm'] = ph; rows.append(t)
    pd.DataFrame(rows).to_csv(f'{OUT}/e33_tests.csv', index=False)
    # ---- RUL: sai số tuyệt đối (chu kỳ) theo cell, tỉ lệ trong ±20 %, độ phủ khoảng
    for m in MODELS:
        R[f's_{m}'] = R[m] - R.true                  # sai số có dấu: dương là dự báo RUL dài hơn thực tế
        R[f'e_{m}'] = (R[m] - R.true).abs()
        R[f'in20_{m}'] = (R[f'e_{m}'] / R.true.clip(lower=1) <= 0.2).astype(float)
    R['cov'] = ((R.true >= R.lo_kls) & (R.true <= R.hi_kls)).astype(float)
    rc = R.groupby(keys + ['cid'])[[f'e_{m}' for m in MODELS] + [f's_{m}' for m in MODELS] + [f'in20_{m}' for m in MODELS]
                                   + ['cov', 'true']].mean()
    RS = rc.groupby(keys).mean().reset_index()
    RS['n_cell'] = rc.groupby(keys).size().values
    RS.to_csv(f'{OUT}/e33_rul.csv', index=False)
    rows = []
    for (b, sc), g in rc.reset_index().groupby(['budget', 'scen']):
        for c in ('gbdt', 'gbdt_ls', 'pinn', 'ket_hop'):
            tmp = []
            for ds in DS:
                x = g[g.ds == ds]; d = x['e_ket_hop_ls'].values - x[f'e_{c}'].values
                tmp.append(dict(budget=b, scen=sc, ds=ds, a='ket_hop_ls', b=c, n=len(d),
                                ratio=x['e_ket_hop_ls'].mean() / x[f'e_{c}'].mean(), a_better=int((d < 0).sum()),
                                a_wins=a_wins(d), p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
                                flip_k=flip_k(d)[0]))
            for t, ph in zip(tmp, holm([t['p'] for t in tmp])):
                t['p_holm'] = ph; rows.append(t)
    pd.DataFrame(rows).to_csv(f'{OUT}/e33_rul_tests.csv', index=False)
    # ---- theo tầm
    hc = H.groupby(keys + ['hb', 'cid'])[MODELS].mean().groupby(keys + ['hb']).mean().reset_index()
    hc.to_csv(f'{OUT}/e33_horizon.csv', index=False)
    # ---- đối chiếu với số đã lưu của G-doan
    G = gdoan_cells()
    if len(G):
        gr = G.groupby(['ds', 'r'])[['persistence', 'linear', 'gbdt', 'selected']].mean()
        gs = gr.groupby('ds').agg(['mean', 'std'])
        gs.columns = [f'{a}' if b == 'mean' else f'{a}_sd' for a, b in gs.columns]
        gs.reset_index().to_csv(f'{OUT}/e33_gdoan.csv', index=False)
        mine = C[(C.budget == 'lab30') & (C.scen == 'do')]
        j = mine.merge(G, on=['ds', 'r', 'cid'], suffixes=('', '_g'))
        cmp_ = j.groupby('ds').apply(lambda x: pd.Series(dict(
            n=len(x), corr_gbdt=np.corrcoef(x.gbdt, x.gbdt_g)[0, 1], corr_ls=np.corrcoef(x.gbdt_ls, x.selected)[0, 1],
            mad_ls=np.abs(x.gbdt_ls - x.selected).mean(), mad_gbdt=np.abs(x.gbdt - x.gbdt_g).mean(),
            giu_equal=float(np.allclose(x.giu, x.persistence)), tuyen_equal=float(np.allclose(x.tuyen, x.linear)))),
            include_groups=False)
        cmp_.reset_index().to_csv(f'{OUT}/e33_gdoan_match.csv', index=False)
    GA = gdoan_anchor()
    pd.set_option('display.width', 220)
    show = S[keys + MODELS].copy()
    for m in MODELS:
        show[m] = (1e3 * show[m]).round(2)
    print(show.to_string(index=False))
    print(pd.read_csv(f'{OUT}/e33_tests.csv').query("a == 'ket_hop_ls'").round(4).to_string(index=False))
    if len(G):
        print(pd.read_csv(f'{OUT}/e33_gdoan.csv').round(5).to_string(index=False))
        print(pd.read_csv(f'{OUT}/e33_gdoan_match.csv').round(4).to_string(index=False))
    if GA is not None:
        print(GA[['ds', 'tier', 'hold', 'linear', 'e32_style', 'history_delta', 'history_residual', 'selected',
                  'p_selected', 'better_selected', 'n_cell']].assign(
            **{m: lambda x, m=m: (1e3 * x[m]).round(2) for m in ['hold', 'linear', 'e32_style', 'history_delta',
                                                                 'history_residual', 'selected']}).to_string(index=False))


if __name__ == '__main__':
    main()
