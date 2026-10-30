"""
So với PINN4SOH: số công bố, chạy lại mã gốc (hai bản), mô hình của đồ án (E27, hai cách tiền xử lý).

    python benchmark/compare.py     # -> benchmark/results/: cells.csv, summary.csv, tests.csv, table.md,
                                    #    logged_constant.json

Kết quả nhỏ (dưới 1 MB) nên được đưa lên git ở benchmark/results/: bản thảo bài báo (paper/build.py) đọc từ đây.

Sai số: MAE (và RMSE) theo cell test, trung bình qua các lần chạy (PINN4SOH chạy lại: hạt giống 0–9; đồ án: 3 hạt
giống), rồi qua các cell của nhóm, rồi qua các nhóm của bộ (batch của XJTU, TJU; cả bộ với MIT, HUST) — đúng cách bài
tính số công bố. Nguồn:
    published        số công bố (pinnsoh/paper.py: PAPER_MAE; theo cell ở chế độ đầy đủ: pinn4soh_reported.csv)
    rerun_published  mã gốc, bản đã sinh số công bố (loss đơn điệu không có gradient), siêu tham số trong nhật ký
    rerun_current    mã gốc, bản hiện tại (loss đơn điệu đã sửa), α, β mặc định mới
    paper_<mô hình>  mô hình của đồ án với bản sao tiền xử lý của bài (E27 'paper')
    strict_<mô hình> mô hình của đồ án với tiền xử lý không rò rỉ (E27 'strict')
với mô hình ∈ {pinn_lai, gbdt, pinn3}. Cùng điều kiện với PINN4SOH là các cột paper_*: cùng cách chia, cùng tiền xử lý
(kể cả phần rò rỉ của nó), cùng cách tính sai số. Cột strict_* trả lời câu hỏi khác: sai số khi bỏ rò rỉ.
Kiểm định: Wilcoxon ghép cặp theo cell test (trung bình qua lần chạy) cho mỗi (bộ, chế độ); p hiệu chỉnh Holm trong mỗi
cặp nguồn qua 12 (bộ, chế độ).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])
_sys.path.insert(0, _os.path.join(_sys.path[0], 'analysis'))
import glob, json, os, shutil
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.paper import PAPER_MAE
from pinnsoh.postproc import by_cell, causal_trend

OUT = 'benchmark/results'
RUNS = 'runs/pinn4soh'
REPORTED = 'pinnsoh/pinn4soh_reported.csv'
SPLITS = 'pinnsoh/pinn4soh_splits.json'
VERSIONS = ('published', 'current')
OURS = ('pinn_lai', 'gbdt', 'pinn3')
ROWS = [(ds, 'full') for ds in ('XJTU', 'TJU', 'MIT', 'HUST')] + [(ds, f'n{k}') for ds in ('XJTU', 'HUST')
                                                                   for k in (1, 2, 3, 4)]
PAIRS = [('paper_pinn_lai', 'rerun_published'), ('paper_pinn_lai', 'rerun_current'), ('paper_pinn_lai', 'published'),
         ('paper_gbdt', 'rerun_published'), ('strict_pinn_lai', 'rerun_published'),
         ('rerun_current', 'rerun_published'), ('rerun_published', 'published')]


def _cell_metrics(y, p, cell):
    return {c: (float(np.abs(p[cell == c] - y[cell == c]).mean()), float(np.sqrt(((p[cell == c] - y[cell == c]) ** 2).mean())))
            for c in np.unique(cell)}


def rerun_rows():
    """Mỗi lần chạy lại mã gốc: MAE, RMSE theo cell test."""
    rows = []
    for v in VERSIONS:
        for f in sorted(glob.glob(f'{RUNS}/{v}/*_run*.json')):
            d = json.load(open(f))
            ds, mode, grp = d['key'].split('|')
            for c in d['cells']:
                rows.append(dict(source=f'rerun_{v}', dataset=ds, mode=mode, group=grp, run=d['run'], cell=c['cell'],
                                 mae=c['MAE'], rmse=c['RMSE']))
    return rows


def ours_rows():
    """Mô hình của đồ án trên đúng cách chia của bài (E27): PINN-lai đúng công thức E20 (½ trung bình 3 PINN-semi + ½ GBDT,
    bộ lọc quỹ đạo w = 10), GBDT, trung bình 3 PINN-semi; chỉ các (nhóm, tiền xử lý, hạt giống) đủ file."""
    import analyze_e27 as A
    rows = []
    for (key, prep, seed), g in sorted(A.collect().items()):
        need = g['pinn'] + [g['gbdt']]
        if not all(p and os.path.exists(p) for p in need):
            continue
        P = [A._pred(p) for p in g['pinn']]; G = A._pred(g['gbdt'])
        y, cell, cyc = P[0]['y'], P[0]['cell'], P[0]['cyc']
        for o in P[1:] + [G]:
            assert np.array_equal(o['cell'], cell) and np.array_equal(o['cyc'], cyc), (key, prep, seed)
        p3 = np.mean([m['p'] for m in P], axis=0)
        pred = dict(pinn3=p3, gbdt=G['p'], pinn_lai=by_cell(causal_trend, 0.5 * p3 + 0.5 * G['p'], cell, cyc, w=A.W))
        ds, mode, grp = key.split('|')
        for m, p in pred.items():
            for c, (mae, rmse) in _cell_metrics(y, p, cell).items():
                rows.append(dict(source=f'{prep}_{m}', dataset=ds, mode=mode, group=grp, run=seed, cell=c, mae=mae,
                                 rmse=rmse))
    return rows


def published_rows():
    R = pd.read_csv(REPORTED)
    R = R[R.model == 'PINN4SOH']
    return [dict(source='published', dataset=r.dataset, mode=r.mode, group=r.group, run=0, cell=r.cell, mae=r.MAE,
                 rmse=r.RMSE) for r in R.itertuples()]


def summarize(C):
    """Bảng theo (bộ, chế độ): sai số của mỗi nguồn (cell -> nhóm -> bộ), độ lệch chuẩn qua lần chạy, số lần chạy ít
    nhất của một nhóm, số nhóm có kết quả (thiếu nhóm thì {nguồn}_complete = False và con số chưa so được với bài)."""
    groups = {}
    for key in json.load(open(SPLITS)):
        ds, mode, grp = key.split('|')
        groups.setdefault((ds, mode), set()).add(grp)
    out = []
    for (ds, mode) in ROWS:
        row = dict(dataset=ds, mode=mode, published_paper=PAPER_MAE[(ds, mode)]['PINN4SOH'])
        for src, g in C[(C.dataset == ds) & (C['mode'] == mode)].groupby('source'):
            per_cell = g.groupby(['group', 'cell'])[['mae', 'rmse']].mean()
            by_group = per_cell.groupby('group').mean()
            row[f'{src}'] = by_group.mae.mean()
            row[f'{src}_rmse'] = by_group.rmse.mean()
            per_run = g.groupby(['run', 'group']).mae.mean().groupby('run').mean()
            row[f'{src}_sd'] = per_run.std(ddof=1) if len(per_run) > 1 else np.nan
            row[f'{src}_runs'] = g.groupby('group').run.nunique().min()
            row[f'{src}_cells'] = len(per_cell)
            row[f'{src}_complete'] = set(by_group.index) == groups[(ds, mode)]
        out.append(row)
    return pd.DataFrame(out)


def tests(C):
    from analyze_e20 import holm
    pc = C.groupby(['source', 'dataset', 'mode', 'group', 'cell']).mae.mean().reset_index()
    out = []
    for a, b in PAIRS:
        for ds, mode in ROWS:
            g = pc[(pc.dataset == ds) & (pc['mode'] == mode)]
            A = g[g.source == a].set_index(['group', 'cell']).mae
            B = g[g.source == b].set_index(['group', 'cell']).mae
            j = A.index.intersection(B.index)
            if len(j) < 3:
                continue
            d = A[j].values - B[j].values
            out.append(dict(a=a, b=b, dataset=ds, mode=mode, n_cells=len(j), mae_a=A[j].mean(), mae_b=B[j].mean(),
                            ratio=A[j].mean() / B[j].mean(), a_better=int((d < 0).sum()),
                            p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    T = pd.DataFrame(out)
    if len(T):
        T['p_holm'] = T.groupby(['a', 'b']).p.transform(lambda p: holm(p.values))
    return T


def markdown(S, T):
    """Bảng đọc nhanh: MAE (×10⁻³), cùng điều kiện tiền xử lý của bài ở bốn cột đầu, không rò rỉ ở cột cuối."""
    cols = [('published_paper', 'PINN4SOH công bố'), ('rerun_published', 'chạy lại, bản công bố'),
            ('rerun_current', 'chạy lại, bản đã sửa'), ('paper_pinn_lai', 'PINN-lai, tiền xử lý của bài'),
            ('paper_gbdt', 'GBDT, tiền xử lý của bài'), ('strict_pinn_lai', 'PINN-lai, không rò rỉ')]
    p = {} if T.empty else {(r.dataset, r.mode, r.b): r.p_holm for r in T[T.a == 'paper_pinn_lai'].itertuples()}
    lines = ['| bộ | chế độ | ' + ' | '.join(l for _, l in cols) + ' | p (PINN-lai so với chạy lại bản công bố) |',
             '|---|---|' + '---:|' * (len(cols) + 1)]
    for r in S.itertuples():
        cells = []
        for c, _ in cols:
            v = getattr(r, c, np.nan)
            sd = getattr(r, f'{c}_sd', np.nan)
            part = '' if getattr(r, f'{c}_complete', True) is not False else '*'   # * : còn thiếu nhóm
            cells.append('–' if pd.isna(v) else (f'{1e3 * v:.2f}' + (f' ± {1e3 * sd:.2f}' if pd.notna(sd) else '') + part))
        pv = p.get((r.dataset, r.mode, 'rerun_published'), np.nan)
        lines.append(f'| {r.dataset} | {r.mode} | ' + ' | '.join(cells) + f" | {'–' if pd.isna(pv) else f'{pv:.3g}'} |")
    return '\n'.join(lines) + '\n\nMAE theo cell (×10⁻³) ± độ lệch chuẩn qua lần chạy; * là còn thiếu nhóm.\n'


def main():
    C = pd.DataFrame(rerun_rows() + ours_rows() + published_rows())
    os.makedirs(OUT, exist_ok=True)
    C.to_csv(f'{OUT}/cells.csv', index=False, float_format='%.6g')
    S = summarize(C); S.to_csv(f'{OUT}/summary.csv', index=False)
    T = tests(C); T.to_csv(f'{OUT}/tests.csv', index=False)
    md = markdown(S, T)
    open(f'{OUT}/table.md', 'w').write(md)
    if os.path.exists(f'{RUNS}/logged_constant.json'):
        shutil.copy(f'{RUNS}/logged_constant.json', f'{OUT}/logged_constant.json')
    print(md)
    with pd.option_context('display.width', 220, 'display.max_rows', 200):
        if len(T):
            print(T.assign(mae_a=1e3 * T.mae_a, mae_b=1e3 * T.mae_b).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
