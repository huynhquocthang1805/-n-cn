"""
Demo: ước lượng SOH theo từng chu kỳ cho MỘT cell, chỉ từ 16 đặc trưng đường sạc.

    # 1) huấn luyện mô hình demo (PINN-semi, 30 % cell có nhãn, pipeline v2) — ~1 phút
    python tools/demo.py train --dataset XJTU

    # 2) ước lượng cho một cell. Cell này KHÔNG được dùng khi huấn luyện (nằm trong fold test).
    python tools/demo.py predict --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-1.csv"

    # 3) giả lập cell CHƯA ĐO dung lượng: bỏ cột capacity trước khi đưa vào mô hình
    python tools/demo.py predict --model demo/XJTU.pt --csv "..." --an-nhan

    # 4) dự báo SOH tương lai, dung lượng còn lại (Ah) và số chu kỳ còn lại tới ngưỡng cuối đời,
    #    đứng ở chu kỳ 150 (bỏ --den-chu-ky để đứng ở chu kỳ cuối cùng của file)
    python tools/demo.py forecast --model demo/XJTU.pt --csv "..." --den-chu-ky 150 [--an-nhan]

Mô hình chạy nhân quả: ước lượng ở chu kỳ N chỉ dùng đặc trưng của chu kỳ N và chỉ số N; bộ lọc
ngoại lai chỉ dùng các chu kỳ trước N. Vì vậy cùng một mô hình dùng được trực tuyến trong BMS.

Đầu ra: demo/<tên cell>_soh.csv (chu kỳ, SOH ước lượng, SOH thực đo nếu có) và .png;
lệnh forecast ghi thêm demo/<tên cell>_dubao.csv và .png.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])                                    # đường dẫn runs/, results/ tính từ gốc
import argparse, os, sys, tempfile
import numpy as np
import pandas as pd
import torch

from pinnsoh.data import Normalizer, load_cell_csv, cycle_input, DATASET_INFO, TJU_NOMINAL
from pinnsoh.models import SOHModel
from pinnsoh.metrics import EOL_THRESHOLD
from pinnsoh.train import Config, Trainer

DEMO_DIR = 'demo'
PHYS = dict(beta_mono=0.5, residual='autograd', alpha_ode=0.02)      # cấu hình E13, nhánh C


def cmd_train(a):
    os.makedirs(DEMO_DIR, exist_ok=True)
    torch.set_num_threads(1)
    cfg = Config(dataset=a.dataset, label_frac=0.3, model='pinn_semi', seed=0, split='cv', fold=0, repeat=0,
                 hidden=(128, 128, 64), **PHYS).v2()
    tr = Trainer(cfg).prepare()
    res = tr.fit(verbose=True)
    path = os.path.join(DEMO_DIR, f'{a.dataset}.pt')
    tr.save_checkpoint(path)
    ck = torch.load(path, weights_only=False)
    ck['split'] = tr.split
    torch.save(ck, path)
    t = res['test']
    print(f'\nđã lưu {path}   ({tr.n_cells["train_labelled"]} cell có nhãn, '
          f'{tr.n_cells["train_unlabelled"]} cell không nhãn)')
    print(f'MAE theo cell trên {tr.n_cells["test"]} cell test: {t["MAE_cell"]:.4f}')
    print('cell test (chưa từng thấy khi huấn luyện):')
    for cid in tr.split['test']:
        print('   ', cid)


def load_model(path):
    ck = torch.load(path, weights_only=False)
    c = ck['config']
    model = SOHModel(dyn='greybox' if c['model'] != 'mlp' else 'none', hidden=tuple(c['hidden']),
                     dropout=c['dropout'], use_cycle=c['use_cycle'], use_knee=c['use_knee'],
                     use_arrhenius=c['use_arrhenius'], act=c['act'], arch=c['arch'])
    model.load_state_dict(ck['state']); model.eval()
    norm = Normalizer(ck['norm_mode']); norm.mu, norm.sd = ck['mu'], ck['sd']
    return model, norm, c, ck.get('split')


def _nominal(dataset, csv):
    if dataset == 'TJU':
        for b, v in TJU_NOMINAL.items():
            if b in csv:
                return v
    return DATASET_INFO[dataset]['nominal']


def cmd_predict(a):
    model, norm, c, split = load_model(a.model)
    nominal = a.nominal or _nominal(c['dataset'], a.csv)
    src = a.csv
    if a.an_nhan:                                            # giả lập cell chưa đo dung lượng
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        pd.read_csv(a.csv).drop(columns=['capacity'], errors='ignore').to_csv(tmp, index=False)
        src = tmp
    cell = load_cell_csv(src, nominal, temp_c=a.temp)
    X = torch.from_numpy(norm.transform(cell)); t = torch.from_numpy(cycle_input(cell)[:, None])
    with torch.no_grad():
        soh = model.u(X, t).numpy().ravel()
    truth = None
    if not a.an_nhan:
        truth = load_cell_csv(a.csv, nominal, temp_c=a.temp).soh
    name = os.path.splitext(os.path.basename(a.csv))[0]
    seen = None
    if split is not None:
        cid_guess = [s for s in split['train'] + split['val'] + split['test'] if s.endswith('/' + name)]
        if cid_guess:
            seen = 'TEST (chưa từng thấy)' if cid_guess[0] in split['test'] else \
                   ('VALIDATION' if cid_guess[0] in split['val'] else 'TRAIN (đã thấy khi huấn luyện)')

    os.makedirs(DEMO_DIR, exist_ok=True)
    out = pd.DataFrame(dict(chu_ky=cell.cycle, soh_uoc_luong=soh))
    if truth is not None:
        out['soh_thuc_do'] = truth
    out.to_csv(os.path.join(DEMO_DIR, f'{name}_soh.csv'), index=False)

    cross = np.nonzero(soh <= EOL_THRESHOLD)[0]
    print(f'cell {name} · {len(soh)} chu kỳ sau lọc · danh định {nominal} Ah'
          + (f' · vai trò trong mô hình: {seen}' if seen else ''))
    print(f'nhãn dung lượng: {"KHÔNG dùng (--an-nhan)" if a.an_nhan else "có, chỉ để đối chiếu"}')
    print(f'SOH ước lượng: đầu {soh[:5].mean():.3f} → hiện tại {soh[-5:].mean():.3f}')
    if len(cross):
        print(f'cảnh báo: SOH ước lượng chạm {EOL_THRESHOLD:.2f} lần đầu ở chu kỳ {cell.cycle[cross[0]]}')
    if truth is not None and np.isfinite(truth).any():
        m = np.isfinite(truth)
        tc = np.nonzero(truth[m] <= EOL_THRESHOLD)[0]
        print(f'MAE so với thực đo: {np.abs(soh[m] - truth[m]).mean():.4f}'
              + (f' · thực đo chạm {EOL_THRESHOLD:.2f} ở chu kỳ {cell.cycle[m][tc[0]]}' if len(tc) else ''))

    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, MODEL_COLOR, save
    apply_theme()
    fig, ax = plt.subplots(figsize=(4.6, 2.4))                # không vẽ chú thích lên ảnh; tên cell nằm ở tên file
    if truth is not None:
        ax.plot(cell.cycle, truth, color=MODEL_COLOR['truth'], lw=1.6, label='thực đo')
    ax.plot(cell.cycle, soh, color=MODEL_COLOR['pinn_semi'], lw=1.1, label='PINN-semi ước lượng')
    ax.axhline(EOL_THRESHOLD, color='#c4ccc8', lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel('chu kỳ'); ax.set_ylabel('SOH')
    ax.legend(loc='lower left')
    fig.tight_layout()
    save(fig, os.path.join(DEMO_DIR, f'{name}_soh'), also_pdf=False, dpi=200)
    print(f'đã ghi {DEMO_DIR}/{name}_soh.csv và .png')


# ───────────────────────────────────────────────────────────── dự báo
def _estimate(model, norm, cell):
    """SOH ước lượng từ đặc trưng sạc, qua bộ lọc quỹ đạo nhân quả (w = 10) như PINN-lai."""
    from pinnsoh.postproc import causal_trend
    X = torch.from_numpy(norm.transform(cell)); t = torch.from_numpy(cycle_input(cell)[:, None])
    with torch.no_grad():
        p = model.u(X, t).numpy().ravel().astype(float)
    return causal_trend(p, cell.cycle, 10)


def _forecaster(path, model, norm, c, split):
    """Bộ dự báo của demo: học trên cell CÓ NHÃN của cùng fold với mô hình ước lượng (cell test không bao giờ được
    dùng), hiệu chỉnh khoảng 90 % trên cell validation cho cả hai kịch bản lịch sử; lưu ở demo/<bộ>_dubao.pkl."""
    import pickle
    from pinnsoh.data import load_dataset
    from pinnsoh import forecast as FC
    fpath = os.path.join(DEMO_DIR, f'{c["dataset"]}_dubao.pkl')
    if os.path.exists(fpath):
        return pickle.load(open(fpath, 'rb'))
    print('lần đầu: huấn luyện bộ dự báo (GBDT + 3 PINN-dự báo) — vài phút ...', flush=True)
    torch.set_num_threads(1)
    cells = load_dataset('PINN4SOH/data', c['dataset'], clean='causal'); by = {x.cid: x for x in cells}
    lab = [by[i] for i in split['labelled']]; va = [by[i] for i in split['val']]
    fnorm = Normalizer('global').fit([by[i] for i in split['train']])
    Xn = {x.cid: fnorm.transform(x) for x in cells}
    fcm = FC.Forecaster.fit(lab, va, Xn, seed=0)
    L = FC.life_scale(cells)
    resid = {'do': [], 'uoc': []}
    for x in va:
        n = len(x.soh)
        if n < 8:
            continue
        hist = {'do': x.soh.astype(float), 'uoc': _estimate(model, norm, x)}
        for sc, sh in hist.items():
            Z, lev = FC.origin_features(sh, x.cycle, Xn[x.cid], x.temp_k)
            for q in FC.ORIGINS:
                i = max(5, int(round(q * (n - 1))))
                if i >= n - 1:
                    continue
                hj = (x.cycle[i + 1:] - x.cycle[i]).astype(float)
                P, _, _ = fcm.predict(Z, lev, i, hj, need_phys=False)
                resid[sc].append(np.column_stack([hj / L, x.soh[i + 1:] - P['ket_hop']]))
    obj = dict(fc=fcm, norm=fnorm, L=L, tau=FC.TAU[c['dataset']], q={k: FC.calibrate(v) for k, v in resid.items()})
    pickle.dump(obj, open(fpath, 'wb'))
    return obj


def cmd_forecast(a):
    from pinnsoh import forecast as FC
    model, norm, c, split = load_model(a.model)
    F = _forecaster(a.model, model, norm, c, split)
    nominal = a.nominal or _nominal(c['dataset'], a.csv)
    cell = load_cell_csv(a.csv, nominal, temp_c=a.temp)
    measured = np.isfinite(cell.soh).all() and not a.an_nhan
    sc = 'do' if measured else 'uoc'
    hist = cell.soh.astype(float) if measured else _estimate(model, norm, cell)
    k = cell.cycle[-1] if a.den_chu_ky is None else a.den_chu_ky
    i = int(np.searchsorted(cell.cycle, k, side='right') - 1)
    if i < 5:
        sys.exit('cần ít nhất 6 chu kỳ lịch sử trước gốc dự báo')
    L, tau = F['L'], F['tau']
    Z, lev = FC.origin_features(hist, cell.cycle, F['norm'].transform(cell), cell.temp_k)
    step = max(1, int(round(L / 200)))
    hh = np.arange(1, int(3 * L) + 1, step, dtype=float)
    P, a_, lam = F['fc'].predict(Z, lev, i, hh, cell.cycle, hist)
    lo, hi = FC.band(F['q'][sc], hh, L)
    p, plo, phi = P['ket_hop'], np.clip(P['ket_hop'] + lo, *FC.CLIP), np.clip(P['ket_hop'] + hi, *FC.CLIP)
    k0 = int(cell.cycle[i])
    rul = FC.first_cross(p, hh, tau, hh[-1]); rlo = FC.first_cross(plo, hh, tau, hh[-1])
    rhi = FC.first_cross(phi, hh, tau, hh[-1])
    upto = hh <= rul
    ah_left = float(np.sum(p[upto] * nominal) * step)          # điện lượng còn xả được tới ngưỡng, xấp xỉ Σ Q(n)
    name = os.path.splitext(os.path.basename(a.csv))[0]
    print(f'cell {name} · gốc dự báo: chu kỳ {k0} · danh định {nominal} Ah · ngưỡng cuối đời SOH {tau:.2f} '
          f'({tau * nominal:.3f} Ah)')
    print(f'lịch sử SOH: {"dung lượng đo" if measured else "ước lượng từ đặc trưng sạc (không đo dung lượng)"}')
    print(f'hiện tại: SOH {lev[i]:.3f} · dung lượng {lev[i] * nominal:.3f} Ah · '
          f'suy giảm hiện tại {10 * a_:.2f} % SOH mỗi 100 chu kỳ · hệ số gãy λ = {lam:.1f}')
    for hr in FC.H_REL:
        h = int(round(hr * L)); j = min(np.searchsorted(hh, h), len(hh) - 1)
        print(f'  sau {h:4d} chu kỳ (chu kỳ {k0 + h}): SOH {p[j]:.3f} [{plo[j]:.3f}; {phi[j]:.3f}] · '
              f'dung lượng {p[j] * nominal:.3f} Ah [{plo[j] * nominal:.3f}; {phi[j] * nominal:.3f}]')
    cap = rul >= hh[-1]
    print(f'số chu kỳ còn lại tới ngưỡng (RUL): ' + (f'> {int(hh[-1])}' if cap else
          f'{int(rul)} [{int(rlo)}; {int(rhi)}]') + f' · điện lượng còn xả được trước ngưỡng ≈ {ah_left:.0f} Ah')
    truth = cell.soh if np.isfinite(cell.soh).all() else None
    if truth is not None and i < len(cell.soh) - 1:
        fut = cell.cycle[i + 1:] - k0
        pp = np.interp(fut, hh, p)
        cross = np.flatnonzero(cell.soh[i + 1:] <= tau)
        print(f'đối chiếu với thực đo sau gốc: MAE SOH {np.abs(pp - cell.soh[i + 1:]).mean():.4f}'
              + (f' · RUL thực {int(cell.cycle[i + 1:][cross[0]] - k0)}' if len(cross) else ''))
    os.makedirs(DEMO_DIR, exist_ok=True)
    pd.DataFrame(dict(chu_ky=k0 + hh.astype(int), soh_du_bao=p, soh_can_duoi=plo, soh_can_tren=phi,
                      dung_luong_Ah=p * nominal)).to_csv(os.path.join(DEMO_DIR, f'{name}_dubao.csv'), index=False)
    import matplotlib.pyplot as plt
    from pinnsoh.plotstyle import apply_theme, MODEL_COLOR, save
    apply_theme()
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    if truth is not None:
        ax.plot(cell.cycle, truth, color=MODEL_COLOR['truth'], lw=1.0, label='thực đo')
    ax.plot(cell.cycle[:i + 1], hist[:i + 1], color='#7b5cd6', lw=1.3, label='lịch sử dùng để dự báo')
    seen = (cell.cycle[-1] - k0) if truth is not None else 0
    show = hh <= max(rul, seen) * 1.15 + L * 0.05                # tới quá ngưỡng một đoạn, không vẽ phần đuôi xa
    ax.fill_between(k0 + hh[show], plo[show], phi[show], color='#1baf7a', alpha=0.2, lw=0, label='khoảng 90 %')
    ax.plot(k0 + hh[show], p[show], color='#0b4f3c', lw=1.6, label='dự báo')
    ax.axhline(tau, color='#c0392b', lw=0.7, ls=(0, (4, 2)))
    ax.axvline(k0, color='#9aa5ad', lw=0.7, ls=(0, (2, 2)))
    top = max(float(np.max(hist[:i + 1])), float(np.nanmax(truth)) if truth is not None else 0.0) + 0.03
    ax.set_ylim(min(tau - 0.04, float(np.nanmin(truth)) - 0.01 if truth is not None else tau - 0.04), top)
    ax.set_xlabel('chu kỳ'); ax.set_ylabel('SOH')
    sec = ax.secondary_yaxis('right', functions=(lambda v: v * nominal, lambda q: q / nominal)); sec.set_ylabel('Ah')
    ax.legend(loc='lower left', fontsize=6.5, framealpha=0.9)
    fig.tight_layout()
    save(fig, os.path.join(DEMO_DIR, f'{name}_dubao'), also_pdf=False, dpi=200)
    print(f'đã ghi {DEMO_DIR}/{name}_dubao.csv và .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('train'); t.add_argument('--dataset', default='XJTU', choices=['XJTU', 'TJU', 'MIT', 'HUST'])
    p = sp.add_parser('predict')
    p.add_argument('--model', required=True); p.add_argument('--csv', required=True)
    p.add_argument('--nominal', type=float, default=None, help='dung lượng danh định (Ah); mặc định theo bộ')
    p.add_argument('--temp', type=float, default=25.0, help='nhiệt độ môi trường (°C); chỉ ảnh hưởng mạng tốc độ, không ảnh hưởng suy luận')
    p.add_argument('--an-nhan', action='store_true', help='bỏ cột capacity: giả lập cell chưa đo dung lượng')
    f = sp.add_parser('forecast')
    f.add_argument('--model', required=True); f.add_argument('--csv', required=True)
    f.add_argument('--den-chu-ky', type=int, default=None, help='chu kỳ hiện tại (gốc dự báo); mặc định chu kỳ cuối của file')
    f.add_argument('--nominal', type=float, default=None, help='dung lượng danh định (Ah); mặc định theo bộ')
    f.add_argument('--temp', type=float, default=25.0, help='nhiệt độ môi trường (°C)')
    f.add_argument('--an-nhan', action='store_true', help='không dùng dung lượng đo: lịch sử SOH lấy từ mô hình ước lượng')
    a = ap.parse_args()
    {'train': cmd_train, 'predict': cmd_predict, 'forecast': cmd_forecast}[a.cmd](a)
