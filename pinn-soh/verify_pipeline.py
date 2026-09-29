"""
Kiểm chứng bằng số năm sửa đổi của pipeline v2 (bản rà soát 29/09, slide 16) và giao thức CV.

    python verify_pipeline.py          # in bảng kết quả
    python verify_pipeline.py --csv    # kèm ghi results/verify_pipeline.csv

Mỗi nhóm phép kiểm ứng với một phát hiện của bản rà soát:
  P1  lọc ngoại lai nhân quả      — quyết định ở chu kỳ N không đổi khi sửa dữ liệu tương lai
  P2  nhãn tuỳ chọn               — cell thiếu dung lượng vẫn nạp được, hàng vẫn vào pool vật lý
  P3  cùng normaliser             — v2: MLP, PINN-sup, PINN-semi có normaliser giống hệt nhau
  P4  lấy cặp theo chu kỳ gốc     — cùng cell, j > i, khoảng cách chu kỳ >= h
  P5  EOL theo chu kỳ, có kiểm duyệt — khớp đáp án tính tay trên dữ liệu tổng hợp
  CV  kiểm định chéo theo cell    — mỗi cell test đúng 1 lần / lần lặp, không rò rỉ, fold cân bằng
  V1  tương thích ngược           — cấu hình mặc định cho đúng số hàng của báo cáo (304 028)
"""
from __future__ import annotations
import argparse, os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from pinnsoh.data import (FEATURES, _clean_causal_features, load_cell_csv, load_dataset, cv_split,
                          Normalizer, make_pointset, choose_labelled, CAUSAL_MIN_HIST)
from pinnsoh.metrics import eol_cycle
from pinnsoh.train import Config, Trainer

ROOT = 'PINN4SOH/data'
RESULTS: list[dict] = []


def record(group, name, ok, detail):
    RESULTS.append(dict(nhom=group, phep_kiem=name, dat=bool(ok), chi_tiet=detail))
    print(f'  [{"OK " if ok else "SAI"}] {name:<52s} {detail}')
    return ok


# ─────────────────────────────────────────────── P1 · lọc ngoại lai nhân quả
def test_causal_filter():
    print('\nP1 — bộ lọc chỉ dùng quá khứ của cell')
    rng = np.random.default_rng(0)
    n = 400
    base = np.linspace(0, 1, n)[:, None] * rng.normal(1, 0.2, (1, 16)) + rng.normal(0, 0.01, (n, 16)) + 5
    df = pd.DataFrame(base, columns=FEATURES); df['capacity'] = np.linspace(2.0, 1.6, n)
    df.loc[200, 'voltage mean'] += 3.0                         # ngoại lai thật ở chu kỳ 200
    keep = _clean_causal_features(df).index.values
    ok1 = record('P1', 'loại đúng ngoại lai cài sẵn', 200 not in keep and len(keep) >= n - 5,
                 f'giữ {len(keep)}/{n} hàng, hàng 200 bị loại')
    worst = 0
    for cut in [50, 150, 250, 350]:
        d2 = df.copy()
        d2.loc[cut:, FEATURES] = rng.normal(0, 50, (n - cut, 16))   # phá hoàn toàn TƯƠNG LAI
        k2 = _clean_causal_features(d2).index.values
        a = set(keep[keep < cut]); b = set(k2[k2 < cut])
        worst = max(worst, len(a ^ b))
    ok2 = record('P1', 'phá dữ liệu tương lai không đổi quyết định quá khứ', worst == 0,
                 f'số quyết định khác nhau trước điểm cắt = {worst} (4 điểm cắt)')
    d3 = df.copy(); d3.loc[:CAUSAL_MIN_HIST - 1, 'voltage std'] = 1e6
    k3 = _clean_causal_features(d3).index.values
    ok3 = record('P1', f'{CAUSAL_MIN_HIST} chu kỳ đầu (chưa đủ lịch sử) không bị phán xử',
                 all(i in k3 for i in range(CAUSAL_MIN_HIST)), 'giữ nguyên, chỉ loại khi không hữu hạn')
    d4 = df.copy(); d4.loc[300, 'CV Q'] = np.inf; d4['capacity'] = np.nan
    k4 = _clean_causal_features(d4).index.values
    ok4 = record('P1', 'inf trong đặc trưng bị loại, nhãn NaN không làm mất hàng',
                 300 not in k4 and len(k4) >= n - 6, f'giữ {len(k4)}/{n} hàng khi TOÀN BỘ nhãn là NaN')
    return ok1 and ok2 and ok3 and ok4


# ─────────────────────────────────────────────── P2 · nhãn tuỳ chọn
def test_optional_labels():
    print('\nP2 — cell chưa đo dung lượng')
    src = sorted(os.listdir(os.path.join(ROOT, 'XJTU data')))[0]
    df = pd.read_csv(os.path.join(ROOT, 'XJTU data', src))
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, 'co_nhan.csv'); df.to_csv(p1, index=False)
        p2 = os.path.join(d, 'khong_nhan.csv'); df.drop(columns=['capacity']).to_csv(p2, index=False)
        p3 = os.path.join(d, 'thieu_mot_phan.csv'); d3 = df.copy(); d3.loc[::3, 'capacity'] = np.nan
        d3.to_csv(p3, index=False)
        c1 = load_cell_csv(p1, 2.0); c2 = load_cell_csv(p2, 2.0); c3 = load_cell_csv(p3, 2.0)
    ok1 = record('P2', 'CSV không có cột capacity nạp được', len(c2.soh) == len(c1.soh) and np.isnan(c2.soh).all(),
                 f'{len(c2.soh)} hàng, toàn bộ soh = NaN, cùng số hàng với bản có nhãn')
    ok2 = record('P2', 'thiếu 1/3 nhãn không làm mất hàng', len(c3.soh) == len(c1.soh),
                 f'{np.isnan(c3.soh).sum()} hàng NaN vẫn giữ trong {len(c3.soh)}')
    norm = Normalizer('global').fit([c1])
    S_all = make_pointset([c3], norm); S_lab = make_pointset([c3], norm, labelled_only=True)
    ok3 = record('P2', 'pool vật lý giữ mọi hàng, pool dữ liệu chỉ hàng có nhãn',
                 len(S_all) == len(c3.soh) and len(S_lab) == int(np.isfinite(c3.soh).sum())
                 and np.isfinite(S_lab.Y).all(), f'vật lý {len(S_all)} · dữ liệu {len(S_lab)}')
    return ok1 and ok2 and ok3


# ─────────────────────────────────────────────── P3 · cùng normaliser
def test_same_normaliser():
    print('\nP3 — v2: mọi mô hình dùng cùng normaliser')
    stats = {}
    for m in ['mlp', 'pinn_sup', 'pinn_semi']:
        cfg = Config(dataset='XJTU', label_frac=0.3, model=m, seed=0, split='cv', fold=1, repeat=0).v2()
        tr = Trainer(cfg).prepare()
        stats[m] = (tr.norm.mu.copy(), tr.norm.sd.copy(), tr.test_ids)
    same = all(np.array_equal(stats['mlp'][0], stats[m][0]) and np.array_equal(stats['mlp'][1], stats[m][1])
               for m in stats)
    ok1 = record('P3', 'mu, sd của normaliser trùng từng bit (3 mô hình)', same, 'v2, XJTU, 30 % nhãn')
    ok2 = record('P3', 'cùng tập cell test, cùng thứ tự', all(stats[m][2] == stats['mlp'][2] for m in stats),
                 f'{len(stats["mlp"][2])} cell test')
    v1 = {}
    for m in ['mlp', 'pinn_semi']:
        tr = Trainer(Config(dataset='XJTU', label_frac=0.3, model=m, seed=0)).prepare()
        v1[m] = tr.norm.mu
    ok3 = record('P3', 'đối chứng: v1 cho normaliser KHÁC nhau', not np.array_equal(v1['mlp'], v1['pinn_semi']),
                 'đúng như bản rà soát chỉ ra')
    return ok1 and ok2 and ok3


# ─────────────────────────────────────────────── P4 · cặp theo chu kỳ
def test_pairs_by_cycle():
    print('\nP4 — lấy cặp (N, N+h) theo chu kỳ gốc')
    cells = load_dataset(ROOT, 'XJTU', clean='causal')[:12]
    S = make_pointset(cells, Normalizer('global').fit(cells))
    rng = np.random.RandomState(0)
    i, j = S.sample_pairs_cycle(rng, 200_000, 5, 50)
    ok1 = record('P4', 'cùng cell và j > i', bool((S.cell[i] == S.cell[j]).all() and (j > i).all()),
                 f'200 000 cặp, {int((S.cell[i] != S.cell[j]).sum())} cặp khác cell')
    rng = np.random.RandomState(0)
    i, j = S.sample_pairs_cycle(rng, 200_000, 5, 50)
    rng = np.random.RandomState(0); _ = rng.randint(0, len(S.src), size=200_000)
    h = rng.randint(5, 51, size=200_000)
    dc = S.cyc[j] - S.cyc[i]
    clipped = j == S.end[i] - 1
    ok2 = record('P4', 'khoảng cách chu kỳ >= h (trừ khi chạm cuối cell)', bool((dc[~clipped] >= h[~clipped]).all()),
                 f'trung vị Δchu kỳ = {np.median(dc):.0f}, h trung vị = {np.median(h):.0f}')
    # đối chứng v1: h theo hàng làm Δchu kỳ vượt h khi có hàng bị lọc
    i1, j1 = S.sample_pairs(np.random.RandomState(0), 200_000, 5, 50)
    rng = np.random.RandomState(0); _ = rng.randint(0, len(S.X), size=200_000); h1 = rng.randint(5, 51, size=200_000)
    over = ((S.cyc[j1] - S.cyc[i1]) > h1).mean()
    exact = (dc[~clipped] == h[~clipped]).mean()
    record('P4', 'đối chứng v1: tỉ lệ cặp có Δchu kỳ > h', True,
           f'v1 {100 * over:.2f} % · v2 khớp đúng h ở {100 * exact:.2f} % cặp (phần còn lại do hàng bị lọc)')
    return ok1 and ok2


# ─────────────────────────────────────────────── P5 · EOL theo chu kỳ
def test_eol():
    print('\nP5 — EOL theo chu kỳ, tách quan sát bị kiểm duyệt')
    cyc = np.arange(0, 200, 2)                                   # hàng cách nhau 2 chu kỳ
    y = np.concatenate([np.linspace(1.0, 0.80, 100), np.linspace(1.0, 0.80, 100),
                        np.linspace(1.0, 0.90, 100), np.linspace(1.0, 0.90, 100)])
    p = np.concatenate([np.linspace(1.0, 0.80, 100) - 0.01,     # cắt sớm hơn
                        np.linspace(1.0, 0.88, 100),            # không bao giờ cắt -> bỏ sót
                        np.linspace(1.0, 0.80, 100),            # báo động giả
                        np.linspace(1.0, 0.90, 100)])           # cả hai không cắt
    cell = np.repeat(np.arange(4), 100); cc = np.tile(cyc, 4)
    r = eol_cycle(y, p, cell, cc, thr=0.85)
    a = int(np.nonzero(np.linspace(1.0, 0.80, 100) <= 0.85)[0][0])
    b = int(np.nonzero(np.linspace(1.0, 0.80, 100) - 0.01 <= 0.85)[0][0])
    exp = 2 * (b - a)
    ok = (r['EOLc_n'] == 1 and r['EOLc_ntrue'] == 2 and r['EOLc_miss'] == 1 and r['EOLc_false'] == 1
          and abs(r['EOLc_bias'] - exp) < 1e-9)
    return record('P5', 'khớp đáp án tính tay (4 cell tổng hợp)', ok,
                  f'sai số {r["EOLc_bias"]:.0f} chu kỳ (đáp án {exp}), n=1, ntrue=2, miss=1, false=1')


# ─────────────────────────────────────────────── CV
def test_cv():
    print('\nCV — kiểm định chéo 5 fold theo cell')
    ok_all = True
    for ds in ['XJTU', 'TJU', 'MIT', 'HUST']:
        cells = load_dataset(ROOT, ds)
        by_id = {c.cid: c for c in cells}
        for rep in [0, 1]:
            tests, sizes = [], []
            for f in range(5):
                sp = cv_split(cells, f, rep)
                tests += sp['test']; sizes.append(len(sp['test']))
                lab30, _ = choose_labelled(sp['train'], 0.3, len(cells), 100 * rep + f, by_id)
                lab10, _ = choose_labelled(sp['train'], 0.1, len(cells), 100 * rep + f, by_id)
                ok_all &= set(lab10) <= set(lab30)
            ok_all &= sorted(tests) == sorted(c.cid for c in cells) and max(sizes) - min(sizes) <= 1
        sp = cv_split(cells, 0, 0)
        detail = f'{ds}: train/val/test = {len(sp["train"])}/{len(sp["val"])}/{len(sp["test"])} cell'
        record('CV', f'{ds}: mỗi cell test đúng 1 lần / lần lặp, fold ±1', ok_all, detail)
    a = cv_split(load_dataset(ROOT, 'XJTU'), 0, 0)['test']; b = cv_split(load_dataset(ROOT, 'XJTU'), 0, 1)['test']
    record('CV', 'hai lần lặp cho cách chia khác nhau', a != b, f'{len(set(a) & set(b))}/{len(a)} cell trùng ở fold 0')
    return ok_all


# ─────────────────────────────────────────────── V1
def test_v1_counts():
    print('\nV1 — tương thích ngược với 1 387 log cũ')
    n1 = {d: sum(len(c.soh) for c in load_dataset(ROOT, d)) for d in ['XJTU', 'TJU', 'MIT', 'HUST']}
    n2 = {d: sum(len(c.soh) for c in load_dataset(ROOT, d, clean='causal')) for d in ['XJTU', 'TJU', 'MIT', 'HUST']}
    ok = record('V1', 'mặc định v1: đúng 304 028 hàng như báo cáo', sum(n1.values()) == 304_028,
                ' · '.join(f'{k} {v}' for k, v in n1.items()))
    record('V1', 'v2 (lọc nhân quả): số hàng', True,
           ' · '.join(f'{k} {v}' for k, v in n2.items()) + f' · tổng {sum(n2.values())}')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', action='store_true'); a = ap.parse_args()
    oks = [test_causal_filter(), test_optional_labels(), test_same_normaliser(), test_pairs_by_cycle(),
           test_eol(), test_cv(), test_v1_counts()]
    n_ok = sum(r['dat'] for r in RESULTS)
    print(f'\n{n_ok}/{len(RESULTS)} phép kiểm đạt')
    if a.csv:
        os.makedirs('results', exist_ok=True)
        pd.DataFrame(RESULTS).to_csv('results/verify_pipeline.csv', index=False)
        print('đã ghi results/verify_pipeline.csv')
    sys.exit(0 if all(oks) else 1)
