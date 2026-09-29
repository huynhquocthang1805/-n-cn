"""
Kiểm định thống kê cho phép so chính: PINN-semi so với MLP ở 30 % nhãn.

    python stats_test.py

Thiết kế
--------
Đơn vị quan sát là SEED. Với cùng một seed, hai mô hình dùng ĐÚNG một cách chia cell
(split_cells gieo theo seed) và được đánh giá trên ĐÚNG một tập cell test, theo đúng
thứ tự hàng — đã kiểm. Nên mỗi seed cho một CẶP (MAE_mlp, MAE_pinn) ghép đôi hợp lệ,
và biến thiên do chia dữ liệu lẫn do khởi tạo đều nằm trong biến thiên giữa các seed.

Ba lớp phân tích:
  1. Wilcoxon signed-rank ghép cặp trên n seed  — không giả định phân phối chuẩn.
  2. Bootstrap ghép cặp trên seed               — khoảng tin cậy cho hiệu và cho tỉ số.
  3. Bootstrap CỤM theo cell trong từng seed    — các hàng trong một cell tương quan
     mạnh, nên phải lấy lại mẫu theo CELL chứ không theo hàng; bỏ qua điều này sẽ cho
     khoảng tin cậy hẹp giả tạo.

Hiệu chỉnh đa so sánh Holm trên bốn bộ dữ liệu.
Hai chỉ số: MAE gộp hàng (khớp phần còn lại của báo cáo) và MAE theo cell (chỉ số mà
mục 07-E11 đề xuất làm chính).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRAC = 0.3
RUNS, TAG, SUF = 'runs/E1', '', ''       # ghi đè bằng tham số dòng lệnh
B = 20_000
RNG = np.random.default_rng(12345)


# ───────────────────────────────────────────────────────────── nạp dự đoán
def load(ds, model, seed):
    pre = f'{TAG}_' if TAG else ''
    p = f'{RUNS}/{pre}{ds}_{model}_f{FRAC}_s{seed}_global_global.json'
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    tp = r['test_predictions']
    return (np.asarray(tp['y'], float), np.asarray(tp['p'], float), np.asarray(tp['cell']))


def per_cell_mae(y, p, cell):
    """Trả về (mảng MAE từng cell, mảng id cell) — theo thứ tự cell tăng dần."""
    cs = np.unique(cell)
    return np.array([np.abs(p[cell == c] - y[cell == c]).mean() for c in cs]), cs


# ──────────────────────────────────────────────── bootstrap và kiểm định
def boot_ci(diff, B=B, alpha=0.05):
    """Khoảng tin cậy percentile cho trung bình của mẫu ghép cặp."""
    n = len(diff)
    idx = RNG.integers(0, n, size=(B, n))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def boot_ratio_ci(a, b, B=B, alpha=0.05):
    """Khoảng tin cậy cho tỉ số trung bình(b)/trung bình(a), lấy lại mẫu GHÉP CẶP."""
    n = len(a)
    idx = RNG.integers(0, n, size=(B, n))
    r = b[idx].mean(axis=1) / a[idx].mean(axis=1)
    return float(np.percentile(r, 100 * alpha / 2)), float(np.percentile(r, 100 * (1 - alpha / 2)))


def holm(pvals):
    """Hiệu chỉnh Holm–Bonferroni, giữ nguyên thứ tự đầu vào."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        prev = max(prev, val)
        adj[i] = min(1.0, prev)
    return adj


# ─────────────────────────────────────────────────────────────── phân tích
def main():
    pre = f'{TAG}_' if TAG else ''
    seeds = sorted({int(f.split('_s')[1].split('_')[0])
                    for f in glob.glob(f'{RUNS}/{pre}*_mlp_f{FRAC}_s*_global_global.json')})
    print(f'seed tìm thấy: {seeds}\n')

    rows, cellrows = [], []
    for ds in DATASETS:
        for s in seeds:
            A, Bd = load(ds, 'mlp', s), load(ds, 'pinn_semi', s)
            if A is None or Bd is None:
                continue
            ya, pa, ca = A
            yb, pb, cb = Bd
            assert np.array_equal(ca, cb) and np.allclose(ya, yb), f'{ds} seed{s}: tập test lệch nhau'
            ma, cs = per_cell_mae(ya, pa, ca)
            mb, _ = per_cell_mae(yb, pb, cb)
            rows.append(dict(dataset=ds, seed=s,
                             mae_mlp=np.abs(pa - ya).mean(), mae_pinn=np.abs(pb - yb).mean(),
                             maecell_mlp=ma.mean(), maecell_pinn=mb.mean(), n_cell=len(cs)))
            for c, u, v in zip(cs, ma, mb):
                cellrows.append(dict(dataset=ds, seed=s, cell=c, mlp=u, pinn=v))

    df = pd.DataFrame(rows)
    cdf = pd.DataFrame(cellrows)
    df.to_csv(f'results/stats_perseed{SUF}.csv', index=False)
    cdf.to_csv(f'results/stats_percell{SUF}.csv', index=False)

    out = []
    for metric, (ca, cb) in [('MAE (gộp hàng)', ('mae_mlp', 'mae_pinn')),
                             ('MAE theo cell', ('maecell_mlp', 'maecell_pinn'))]:
        raw_p = []
        block = []
        for ds in DATASETS:
            g = df[df.dataset == ds]
            if len(g) < 5:
                continue
            a, b = g[ca].values, g[cb].values
            d = b - a                                   # âm = PINN tốt hơn
            W, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
            lo, hi = boot_ci(d)
            rlo, rhi = boot_ratio_ci(a, b)
            block.append(dict(chi_so=metric, bo=ds, n_seed=len(g),
                              mlp=a.mean(), pinn=b.mean(),
                              hieu=d.mean(), ci_lo=lo, ci_hi=hi,
                              ti_so=b.mean() / a.mean(), ti_so_lo=rlo, ti_so_hi=rhi,
                              p_tho=p, thang_seed=int((d < 0).sum())))
            raw_p.append(p)
        for r, pa_ in zip(block, holm(np.array(raw_p))):
            r['p_holm'] = pa_
            out.append(r)

    res = pd.DataFrame(out)
    res.to_csv(f'results/stats_table{SUF}.csv', index=False)

    for metric in res.chi_so.unique():
        print(f'=== {metric} · PINN-semi so với MLP, 30 % nhãn ===')
        t = res[res.chi_so == metric]
        print(f'{"bộ":<6}{"n":>3}{"MLP":>9}{"PINN":>9}{"hiệu":>10}{"KTC 95% của hiệu":>24}'
              f'{"tỉ số":>8}{"KTC 95% tỉ số":>20}{"p (Holm)":>11}  kết luận')
        for _, r in t.iterrows():
            sig = 'PINN tốt hơn' if (r.p_holm < 0.05 and r.hieu < 0) else \
                  ('MLP tốt hơn' if (r.p_holm < 0.05 and r.hieu > 0) else 'chưa đủ bằng chứng')
            print(f'{r.bo:<6}{int(r.n_seed):>3}{r.mlp:>9.4f}{r.pinn:>9.4f}{r.hieu:>+10.4f}'
                  f'  [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]{r.ti_so:>8.3f}'
                  f'  [{r.ti_so_lo:.3f}, {r.ti_so_hi:.3f}]{r.p_holm:>11.4f}  {sig}')
        print()

    # bootstrap CỤM theo cell, trong từng seed rồi lấy trung bình qua seed
    print('=== Bootstrap CỤM theo cell (trong từng seed, rồi trung bình qua seed) ===')
    print(f'{"bộ":<6}{"tỉ số TB":>10}{"KTC 95%":>22}{"tỉ lệ seed có KTC < 1":>26}')
    clu = []
    for ds in DATASETS:
        g = cdf[cdf.dataset == ds]
        if g.empty: continue
        ratios, below = [], 0
        for s in sorted(g.seed.unique()):
            h = g[g.seed == s]
            a, b = h.mlp.values, h.pinn.values
            n = len(a)
            idx = RNG.integers(0, n, size=(4000, n))       # lấy lại mẫu theo CELL
            r = b[idx].mean(axis=1) / a[idx].mean(axis=1)
            ratios.append(r)
            if np.percentile(r, 97.5) < 1.0: below += 1
        R = np.concatenate(ratios)
        clu.append(dict(bo=ds, ti_so=float(R.mean()),
                        lo=float(np.percentile(R, 2.5)), hi=float(np.percentile(R, 97.5)),
                        seed_ro_rang=below, n_seed=len(ratios)))
        print(f'{ds:<6}{R.mean():>10.3f}   [{np.percentile(R,2.5):.3f}, {np.percentile(R,97.5):.3f}]'
              f'{below:>18}/{len(ratios)}')
    pd.DataFrame(clu).to_csv(f'results/stats_cluster{SUF}.csv', index=False)

    # 3 seed so với 10 seed: độ rộng khoảng tin cậy
    print('\n=== Vì sao 3 seed là không đủ ===')
    print(f'{"bộ":<6}{"tỉ số n=3":>12}{"tỉ số n=10":>13}{"biên độ MAE của MLP qua seed":>32}'
          f'{"rộng KTC n=3":>15}{"rộng KTC n=10":>15}')
    for ds in DATASETS:
        g = df[df.dataset == ds].sort_values('seed')
        if len(g) < 10: continue
        a, b = g.maecell_mlp.values, g.maecell_pinn.values
        d_all = b - a
        w3 = abs(np.subtract(*boot_ci(d_all[:3])))
        w10 = abs(np.subtract(*boot_ci(d_all)))
        spread = f'{a.min():.4f}–{a.max():.4f} ({a.max()/a.min():.1f}×)'
        print(f'{ds:<6}{b[:3].mean()/a[:3].mean():>12.3f}{b.mean()/a.mean():>13.3f}'
              f'{spread:>32}{w3:>15.5f}{w10:>15.5f}')
    print('KTC rộng RA khi thêm seed nghĩa là ba seed đầu đã che mất biến thiên thật.')

    print('\nĐã ghi results/stats_table.csv, stats_cluster.csv, stats_perseed.csv, stats_percell.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs/E1')
    ap.add_argument('--tag', default='')
    ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    RUNS, TAG, SUF = a.runs, a.tag, a.suffix
    print(f'nguồn: {RUNS}  tag={TAG or "(không)"}\n')
    main()
