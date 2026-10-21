"""
E25: cập nhật ước lượng và dự báo khi có thêm vài lần đo dung lượng của chính cell (bộ lọc Kalman cho độ lệch theo cell).

    python analysis/analyze_e25.py      # -> results/e25_*.csv, results/fig_e25_*.png|pdf

Không huấn luyện thêm gì: dùng dự đoán ngoài fold của PINN-lai (E20; NASA từ E24) trên cell test và cell validation.

Mô hình. Gọi p_n là SOH do PINN-lai ước lượng ở hàng n của một cell, y_n là SOH thật. Sai số y_n − p_n = b_n + e_n, trong
đó b_n là độ lệch riêng của cell, biến đổi chậm và hồi quy về trung bình của tổng thể (quá trình Ornstein–Uhlenbeck:
tương quan giữa hai chu kỳ cách nhau Δ là exp(−Δ/ell), phương sai dừng P0), e_n là nhiễu nhanh, Var e = R. Trước lần đo đầu
tiên b ~ N(mu0, P0). Mỗi lần đo dung lượng cập nhật (b, P) theo Kalman; giữa hai lần đo, b trôi về mu0 và P về P0, nên thông
tin của một lần đo phai dần theo khoảng cách chu kỳ và khoảng dự đoán không rộng vô hạn. Ước lượng đã hiệu chỉnh ở hàng m là
p_m + b(m), với b(m) chỉ dùng các lần đo ở chu kỳ <= chu kỳ của m (nhân quả). Khoảng 90 %: p_m + b(m) ± 1,645·sqrt(P(m) + R).

Tham số trên cell VALIDATION của chính fold đó: mu0 = trung bình sai số; R = ½ phương sai của sai phân bậc một (phần nhiễu
nhanh); P0 = phương sai sai số quanh mu0 trừ R; ell (theo đơn vị L) chọn trên lưới bằng MAE, riêng cho từng lịch đo.
Bản thử đầu dùng bước ngẫu nhiên (b trôi tự do): một lần đo lúc nhận làm kém đi vì độ lệch ở chu kỳ đầu không đại diện cho
về sau (XJTU có giai đoạn kích hoạt), và phương sai giữa hai lần đo tăng không giới hạn.

Lịch đo (L = trung vị số chu kỳ của bộ), lần đo rơi vào hàng đầu tiên có chu kỳ >= mốc:
    dau      một lần đo ở chu kỳ đầu (nghiệm thu khi nhận pin)
    q25      chu kỳ đầu và mỗi 0,25L
    q10      chu kỳ đầu và mỗi 0,1L
Đối chứng: 'goc' (PINN-lai không hiệu chỉnh) và 'gan_nhat' (dịch theo sai số của lần đo gần nhất, không dùng tiên nghiệm).
Sai số tính trên các hàng KHÔNG được đo, theo cell (trung bình qua lần lặp), Wilcoxon ghép cặp theo cell so với 'goc'.

Dự báo: với lịch sử ước lượng (kịch bản 'uoc' của E23) cộng độ lệch hậu nghiệm tại gốc vào toàn bộ đường dự báo của mô hình
kết hợp có xét hồi phục, so với không hiệu chỉnh và với lịch sử đo thật ('do').
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

from pinnsoh.data import load_dataset
from pinnsoh.forecast import life_scale
from pinnsoh.postproc import by_cell, causal_trend

OUT = 'results'
DS = ['XJTU', 'TJU', 'MIT', 'HUST', 'NASA']
FRACS = {ds: (0.1, 0.3, 0.5, 0.7) for ds in DS}; FRACS['NASA'] = (0.3, 0.7)
REPEATS = (0, 1, 2)
SCHED = {'dau': None, 'q25': 0.25, 'q10': 0.10}
ELL_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 3.0, np.inf]      # độ dài tương quan, theo đơn vị L
Z90 = 1.645


def _src(ds):
    return 'runs/E24' if ds == 'NASA' else 'runs/E20'


def _read(path, key):
    tp = json.load(open(path))[key]
    ids = tp['cell_ids']
    return (np.asarray([ids[i] for i in tp['cell']]), np.asarray(tp['cycle'], float), np.asarray(tp['y'], float),
            np.asarray(tp['p'], float))


def pinn_lai(ds, f, r, k, part):
    """PINN-lai ngoài fold: ½ trung bình 3 PINN-semi + ½ GBDT, qua bộ lọc quỹ đạo w = 10. Trả DataFrame theo hàng."""
    s = 100 * r + k; src = _src(ds)
    key = 'test_predictions' if part == 'test' else 'val_predictions'
    P = [f'{src}/P_{ds}_pinn_semi_f{f}_s{s}_global_global_cv{r}-{k}of5' + ('' if m == 0 else f'_m{m}') + '.json'
         for m in range(3)]
    G = f'{src}/gbdt_{ds}_f{f}_cv{r}-{k}of5.json'
    if not all(os.path.exists(p) for p in P + [G]):
        return None
    ps = [_read(p, key) for p in P]; g = _read(G, key)
    cell, cyc, y = ps[0][0], ps[0][1], ps[0][2]
    for o in ps[1:] + [g]:
        assert np.array_equal(o[0], cell) and np.array_equal(o[1], cyc)
    raw = 0.5 * np.mean([o[3] for o in ps], axis=0) + 0.5 * g[3]
    return pd.DataFrame(dict(cell=cell, cyc=cyc, y=y, p=by_cell(causal_trend, raw, cell, cyc, w=10),
                             gbdt=g[3])).sort_values(['cell', 'cyc'], kind='stable').reset_index(drop=True)


# --------------------------------------------------------------------------- bộ lọc
def measured_rows(cyc, frac, L):
    """Chỉ số các hàng được đo dung lượng theo lịch (hàng đầu tiên có chu kỳ >= mốc)."""
    marks = [cyc[0]] if frac is None else np.arange(cyc[0], cyc[-1] + 1e-9, frac * L)
    rows = sorted({int(np.searchsorted(cyc, m, side='left')) for m in marks if m <= cyc[-1]})
    return np.asarray(rows, int)


def _decay(dc, ell):
    return 1.0 if not np.isfinite(ell) else float(np.exp(-dc / ell))


def kalman(cyc, r, meas, mu0, P0, R, ell):
    """Độ lệch tiên nghiệm b(m) và phương sai P(m) ở mỗi hàng m, chỉ dùng các lần đo ở hàng < m (hàng được đo không tính
    vào sai số). Bước dự báo của Ornstein–Uhlenbeck: b <- mu0 + a(b − mu0), P <- a²P + (1 − a²)P0, a = exp(−Δ/ell)."""
    n = len(cyc); b, P = mu0, P0
    B, V = np.empty(n), np.empty(n)
    is_m = np.zeros(n, bool); is_m[meas] = True
    last = cyc[0]
    for m in range(n):
        a = _decay(cyc[m] - last, ell); last = cyc[m]
        b = mu0 + a * (b - mu0); P = a * a * P + (1 - a * a) * P0
        B[m], V[m] = b, P
        if is_m[m]:
            K = P / (P + R)
            b += K * (r[m] - b); P = (1 - K) * P
    return B, V, is_m


def nearest(cyc, r, meas):
    n = len(cyc); B = np.zeros(n); is_m = np.zeros(n, bool); is_m[meas] = True; cur = 0.0
    for m in range(n):
        B[m] = cur
        if is_m[m]:
            cur = r[m]
    return B, is_m


def fit_params(V, L):
    """mu0, R, P0 trên cell validation; ell chọn trên lưới bằng MAE riêng cho từng lịch đo (lịch đo biết trước)."""
    V = V.copy(); V['r'] = V.y - V.p
    mu0 = float(V.r.mean())
    dr = V.groupby('cell').r.diff().dropna()
    R = max(float(dr.var()) / 2, 1e-8)
    P0 = max(float(((V.r - mu0) ** 2).mean()) - R, 1e-8)
    out = dict(mu0=mu0, P0=P0, R=R)
    for sc, frac in SCHED.items():
        best = None
        for ell in ELL_GRID:
            err = []
            for c, g in V.groupby('cell', sort=False):
                cyc, r = g.cyc.values, g.r.values
                meas = measured_rows(cyc, frac, L)
                B, _, is_m = kalman(cyc, r, meas, mu0, P0, R, ell * L)
                if (~is_m).any():
                    err.append(np.abs(r - B)[~is_m].mean())
            v = float(np.mean(err))
            if best is None or v < best[0]:
                best = (v, ell)
        out[f'ell_{sc}'] = best[1]; out[f'ellc_{sc}'] = best[1] * L
    return out


def run_estimation():
    rows, params, cover = [], [], []
    for ds in DS:
        L = life_scale(load_dataset('PINN4SOH/data', ds, clean='causal'))
        for f in FRACS[ds]:
            for r in REPEATS:
                for k in range(5):
                    Te, Va = pinn_lai(ds, f, r, k, 'test'), pinn_lai(ds, f, r, k, 'val')
                    if Te is None or Va is None:
                        continue
                    th = fit_params(Va, L); params.append(dict(ds=ds, frac=f, r=r, k=k, **th))
                    qcf = float(np.quantile(np.abs(Va.y - Va.p), 0.90))          # khoảng phù hợp cố định như E20
                    for c, g in Te.groupby('cell', sort=False):
                        cyc, y, p = g.cyc.values, g.y.values, g.p.values
                        res = y - p
                        for sc, frac in SCHED.items():
                            meas = measured_rows(cyc, frac, L)
                            B, Vb, is_m = kalman(cyc, res, meas, th['mu0'], th['P0'], th['R'], th[f'ellc_{sc}'])
                            Bn, _ = nearest(cyc, res, meas)
                            ev = ~is_m
                            if not ev.any():
                                continue
                            e_goc = np.abs(res)[ev].mean()
                            e_kal = np.abs(res - B)[ev].mean()
                            e_gan = np.abs(res - Bn)[ev].mean()
                            half = Z90 * np.sqrt(Vb + th['R'])
                            cov_k = float((np.abs(res - B) <= half)[ev].mean())
                            cov_c = float((np.abs(res) <= qcf)[ev].mean())
                            rows.append(dict(ds=ds, frac=f, r=r, cid=c, sched=sc, n_meas=len(meas), goc=e_goc, kalman=e_kal,
                                             gan_nhat=e_gan, cov_kalman=cov_k, cov_conf=cov_c,
                                             wid_kalman=2 * float(half[ev].mean()), wid_conf=2 * qcf))
    return pd.DataFrame(rows), pd.DataFrame(params)


def summarize(E):
    c = E.groupby(['ds', 'frac', 'sched', 'cid'])[['goc', 'kalman', 'gan_nhat', 'cov_kalman', 'cov_conf', 'wid_kalman',
                                                     'wid_conf', 'n_meas']].mean().reset_index()
    rows = []
    for (ds, f, sc), g in c.groupby(['ds', 'frac', 'sched']):
        d = g.kalman.values - g.goc.values
        dn = g.kalman.values - g.gan_nhat.values
        rows.append(dict(ds=ds, frac=f, sched=sc, n_cell=len(g), n_meas=g.n_meas.mean(), goc=g.goc.mean(),
                         kalman=g.kalman.mean(), gan_nhat=g.gan_nhat.mean(),
                         ratio=float(np.exp(np.mean(np.log(g.kalman / g.goc)))),
                         better=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
                         p_vs_gan=stats.wilcoxon(dn).pvalue if np.any(dn != 0) else 1.0,
                         cov_kalman=g.cov_kalman.mean(), cov_conf=g.cov_conf.mean(),
                         c80_kalman=float((g.cov_kalman < 0.8).mean()), c80_conf=float((g.cov_conf < 0.8).mean()),
                         wid_kalman=g.wid_kalman.mean(), wid_conf=g.wid_conf.mean()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- dự báo
def posterior_end(cyc, r, meas, mu0, P0, R, ell):
    """Độ lệch hậu nghiệm sau mọi lần đo ở các hàng đã cho (kể cả hàng cuối nếu được đo), tại chu kỳ của hàng cuối."""
    b, P, last = mu0, P0, cyc[0]
    is_m = np.zeros(len(cyc), bool); is_m[meas] = True
    for m in range(len(cyc)):
        a = _decay(cyc[m] - last, ell); last = cyc[m]
        b = mu0 + a * (b - mu0); P = a * a * P + (1 - a * a) * P0
        if is_m[m]:
            K = P / (P + R); b += K * (r[m] - b); P = (1 - K) * P
    return b


def forecast_shift(params):
    """Cộng độ lệch hậu nghiệm tại gốc vào dự báo của mô hình kết hợp có xét hồi phục (E23, kịch bản 'uoc')."""
    from pinnsoh.recovery import MODELS_R
    TCOL = ['scen', 'cell', 'otype', 'oq', 'k', 'h', 'y'] + MODELS_R + ['lo', 'hi', 'lo_r', 'hi_r']
    P = params.set_index(['ds', 'frac', 'r', 'k'])
    rows = []
    for f in sorted(glob.glob('runs/E23/rc_*.npz')):
        d = np.load(f); inf = json.loads(str(d['info'])); ds, r, k, L = inf['ds'], inf['repeat'], inf['fold'], inf['L']
        if (ds, 0.7, r, k) not in P.index:
            continue
        th = P.loc[(ds, 0.7, r, k)]
        Te = pinn_lai(ds, 0.7, r, k, 'test')
        cells = np.asarray(d['cells'])
        T = pd.DataFrame(d['traj'].astype(np.float64), columns=TCOL)
        T = T[T.otype == 0]
        for (sc, ci, oq), g in T.groupby(['scen', 'cell', 'oq']):
            cid = cells[int(ci)]; kc = g.k.iloc[0]
            e = Te[Te.cell == cid]
            cyc, res = e.cyc.values, (e.y - e.p).values
            err = lambda pred: float(np.abs(pred - g.y.values).mean())
            if sc == 0:                                   # lịch sử đo thật: mức tham chiếu
                rows.append(dict(ds=ds, r=r, cid=cid, oq=oq, sched='lich_su_do', err=err(g.ket_hop_r.values)))
                continue
            rows.append(dict(ds=ds, r=r, cid=cid, oq=oq, sched='goc', err=err(g.ket_hop_r.values)))
            upto = cyc <= kc
            for name, frac in SCHED.items():
                meas = measured_rows(cyc, frac, L); meas = meas[upto[meas]]
                ellc = th[f'ellc_{name}']
                bk = posterior_end(cyc[upto], res[upto], meas, th.mu0, th.P0, th.R, ellc)
                # đường dự báo dịch theo độ lệch dự kiến ở từng chu kỳ tương lai (độ lệch trôi dần về mu0)
                a = np.exp(-g.h.values / ellc) if np.isfinite(ellc) else np.ones(len(g))
                bk = th.mu0 + a * (bk - th.mu0)
                rows.append(dict(ds=ds, r=r, cid=cid, oq=oq, sched=name, err=err(g.ket_hop_r.values + bk)))
    F = pd.DataFrame(rows)
    c = F.groupby(['ds', 'sched', 'cid']).err.mean().reset_index()
    base = c[c.sched == 'goc'].set_index(['ds', 'cid']).err
    out = []
    for (ds, sc), g in c.groupby(['ds', 'sched']):
        b = base.loc[ds].reindex(g.cid).values
        dd = g.err.values - b
        out.append(dict(ds=ds, sched=sc, n_cell=len(g), mae=g.err.mean(),
                        ratio=float(np.exp(np.mean(np.log(g.err.values / b)))),
                        p=stats.wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0))
    return pd.DataFrame(out)


def figure(E):
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, GRID, panel_title, save
    apply_theme()
    c = E.groupby(['ds', 'frac', 'sched', 'cid'])[['goc', 'kalman']].mean().reset_index()
    fig, axes = plt.subplots(1, len(DS), figsize=(7.2, 2.3), sharey=False)
    col = {'goc': '#9aa5ad', 'dau': '#9cc3e6', 'q25': '#2a78d6', 'q10': '#0b3d73'}
    for ax, ds in zip(axes, DS):
        g = c[c.ds == ds]
        fr = sorted(g.frac.unique())
        base = g[g.sched == 'dau'].groupby('frac').goc.mean()
        ax.plot(range(len(fr)), 1e3 * base.reindex(fr).values, color=col['goc'], marker='o', ms=3, lw=1.2)
        for sc in SCHED:
            v = g[g.sched == sc].groupby('frac').kalman.mean().reindex(fr)
            ax.plot(range(len(fr)), 1e3 * v.values, color=col[sc], marker='o', ms=3, lw=1.3)
        ax.set_xticks(range(len(fr))); ax.set_xticklabels([f'{int(100 * x)}' for x in fr], fontsize=6.5)
        ax.set_yscale('log'); lo_, hi_ = ax.get_ylim()
        ax.set_yticks([t for t in (2, 3, 5, 8, 10, 15, 20, 30, 50) if lo_ <= t <= hi_])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(False); ax.yaxis.grid(True, color=GRID, lw=.5); ax.tick_params(labelsize=6.5)
        panel_title(ax, ds, pad=4)
    axes[0].set_ylabel('MAE theo cell (×10$^{-3}$)')
    hd = [plt.Line2D([], [], color=col[k], marker='o', ms=3, lw=1.3, label=l) for k, l in
          (('goc', 'không đo'), ('dau', 'đo 1 lần lúc nhận'), ('q25', 'đo mỗi 0,25L'), ('q10', 'đo mỗi 0,1L'))]
    fig.legend(handles=hd, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.text(0.5, 0.1, '% cell có nhãn khi huấn luyện', ha='center', fontsize=8)
    save(fig, f'{OUT}/fig_e25_calib')


def main():
    os.makedirs(OUT, exist_ok=True)
    E, params = run_estimation()
    params.to_csv(f'{OUT}/e25_params.csv', index=False)
    S = summarize(E); S.to_csv(f'{OUT}/e25_estimation.csv', index=False)
    F = forecast_shift(params); F.to_csv(f'{OUT}/e25_forecast.csv', index=False)
    pd.set_option('display.width', 220)
    show = S.copy()
    for col in ('goc', 'kalman', 'gan_nhat', 'wid_kalman', 'wid_conf'):
        show[col] = (1e3 * show[col]).round(2)
    print(show.round(3).to_string(index=False))
    print(params.groupby('ds')[['mu0', 'P0', 'R'] + [f'ell_{sc}' for sc in SCHED]].median())
    print(F.assign(mae=lambda x: (1e3 * x.mae).round(2)).round(3).to_string(index=False))
    figure(E)


if __name__ == '__main__':
    main()
