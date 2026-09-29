"""
Experimental protocol.

E1  label-efficiency  : within-dataset, cell-wise 70/15/15 split; labelled fraction of ALL cells
                        in {0.1, 0.3, 0.5, 0.7}; val/test cells FIXED for every fraction.
                        models: mlp (data only) | pinn_sup (physics on labelled cells) |
                                pinn_semi (physics also on the unlabelled training cells)
E2  cross-dataset     : train on source (70 % labelled), evaluate ZERO-SHOT on the entire target dataset.
                        same-chemistry pairs HUST<->MIT (LFP, A123 1.1 Ah), XJTU<->TJU (NCM/NCA).
                        norm in {global, first_cycle}.
E3  ablation          : XJTU & TJU at 30 % labels: remove ODE / mono / knee / Arrhenius, black-box dynamics,
                        derivative-form residuals.
E4  leakage audit     : PINN4SOH-style per-cell min-max normalisation of features and cycle index vs. causal
                        normalisation; and no-cycle-input.
E5  adaptation        : source (70 % labelled) + target with k in {0,3} labelled cells; PINN-semi applies the
                        label-free physics losses to ALL target training cells; evaluated on the target TEST cells.
E13 confirmatory      : pipeline v2, 5-fold × 5-repeat cell-wise CV, 11 arms (tai-lieu/de-cuong-E13.md).
E14 same-normaliser   : E5 with ONE normaliser for both models (source + target train features), pipeline v2.

usage:  python experiments.py E1 [--seeds 0 1 2] [--datasets XJTU TJU MIT HUST] [--threads 2]
"""
import argparse, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from pinnsoh.train import Config, run
from pinnsoh.transfer import TransferConfig, run_transfer

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.3, 0.5, 0.7]
PAIRS = {'HUST': ('MIT',), 'MIT': ('HUST',), 'XJTU': ('TJU',), 'TJU': ('XJTU',)}


def E1(seeds, datasets, out='runs/E1'):
    for ds, s in itertools.product(datasets, seeds):
        for f in FRACS:
            for m in ['mlp', 'pinn_sup', 'pinn_semi']:
                run(Config(dataset=ds, label_frac=f, model=m, seed=s, out_dir=out), verbose=False)


def E2(seeds, datasets, out='runs/E2'):
    for ds, s in itertools.product(datasets, seeds):
        for norm in ['first_cycle', 'global']:
            for m in ['mlp', 'pinn_semi']:
                run(Config(dataset=ds, label_frac=0.7, model=m, seed=s, norm=norm, transfer_to=PAIRS[ds],
                           out_dir=out, save_predictions=False), verbose=False)


def E3(seeds, datasets, out='runs/E3'):
    variants = {
        'full':        dict(),
        'no_ode':      dict(alpha_ode=0.0),
        'no_mono':     dict(beta_mono=0.0),
        'no_range':    dict(gamma_range=0.0),
        'no_knee':     dict(use_knee=False),
        'no_arrh':     dict(use_arrhenius=False),
        'only_mono':   dict(alpha_ode=0.0, gamma_range=0.0),
        'res_fd':      dict(residual='fd', alpha_ode=0.02),
        'res_autograd': dict(residual='autograd', alpha_ode=0.02),
    }
    for ds, s in itertools.product([d for d in datasets if d in ('XJTU', 'TJU')], seeds):
        for name, kw in variants.items():
            if name == 'no_arrh' and ds != 'TJU':
                continue
            run(Config(dataset=ds, label_frac=0.3, model='pinn_semi', seed=s, tag=name, out_dir=out,
                       save_predictions=False, **kw), verbose=False)
        run(Config(dataset=ds, label_frac=0.3, model='pinn_bb', seed=s, tag='blackbox', out_dir=out,
                   save_predictions=False), verbose=False)


def E4(seeds, datasets, out='runs/E4'):
    settings = {
        'causal':          dict(norm='global', cyc='global'),
        'leak_cycle':      dict(norm='global', cyc='per_cell_minmax'),
        'leak_feat':       dict(norm='per_cell_minmax', cyc='global'),
        'leak_both':       dict(norm='per_cell_minmax', cyc='per_cell_minmax'),   # PINN4SOH replica
        'no_cycle_input':  dict(norm='global', cyc='global', use_cycle=False),
    }
    for ds, s in itertools.product([d for d in datasets if d in ('XJTU', 'MIT')], seeds):
        for name, kw in settings.items():
            run(Config(dataset=ds, label_frac=0.7, model='mlp', seed=s, tag=name, out_dir=out,
                       save_predictions=False, **kw), verbose=False)


def E5(seeds, datasets, out='runs/E5'):
    """cross-dataset adaptation: k in {0, 3} labelled target cells; physics on unlabelled target cells."""
    for ds, s in itertools.product(datasets, seeds):
        for k in [0, 3]:
            for m in ['mlp', 'pinn_semi']:
                run_transfer(TransferConfig(dataset=ds, target=PAIRS[ds][0], model=m, k_target=k, seed=s,
                                            norm='global', out_dir=out, save_predictions=False))


E7_VARIANTS = {
    'b0.5':     dict(beta_mono=0.5),
    'b1.5':     dict(beta_mono=1.5),
    'b0.5_ag':  dict(beta_mono=0.5, residual='autograd', alpha_ode=0.02),
    'b5':       dict(),                                            # E1 default, for reference
    'adapt':    dict(beta_mono=2.0, alpha_ode=1.0, adapt_w=True, residual='autograd'),
}


def E7(seeds, datasets, out='runs/E7'):
    """validation-selected physics weight at EVERY label fraction (the E6 grid, swept over fractions)."""
    for ds, s in itertools.product(datasets, seeds):
        for f in FRACS:
            for name, kw in E7_VARIANTS.items():
                run(Config(dataset=ds, label_frac=f, model='pinn_semi', seed=s, tag=name, out_dir=out,
                           save_predictions=False, **kw), verbose=False)


def E6(seeds, datasets, out='runs/E6'):
    """validation-selected physics weight at 30 % labels (XJTU, TJU): beta / eps variants of pinn_semi."""
    variants = {'b0.5': dict(beta_mono=0.5), 'b1.5': dict(beta_mono=1.5), 'b5_eps0.005': dict(eps_mono=0.005),
                'b0.5_ag': dict(beta_mono=0.5, residual='autograd', alpha_ode=0.02)}
    for ds, s in itertools.product([d for d in datasets if d in ('XJTU', 'TJU', 'MIT', 'HUST')], seeds):
        for name, kw in variants.items():
            run(Config(dataset=ds, label_frac=0.3, model='pinn_semi', seed=s, tag=name, out_dir=out,
                       save_predictions=False, **kw), verbose=False)


# --------------------------------------------------------------------------- E9 network / activation
E9_ARCHS = {
    'silu-mlp':     dict(act='silu',  arch='mlp'),      # cơ sở
    'tanh-mlp':     dict(act='tanh',  arch='mlp'),      # kích hoạt PINN kinh điển
    'sin-mlp':      dict(act='sin',   arch='mlp'),      # PINN4SOH dùng
    'snake-mlp':    dict(act='snake', arch='mlp'),      # xu thế + dao động
    'silu-res':     dict(act='silu',  arch='res'),      # khối residual
    'silu-fourier': dict(act='silu',  arch='fourier'),  # mã hoá Fourier cho t
    'silu-mono':    dict(act='silu',  arch='mono'),     # đơn điệu theo cấu trúc (tắt L_mono)
    'silu-mono+L':  dict(act='silu',  arch='mono'),     # đơn điệu theo cấu trúc VÀ giữ L_mono
}


def E9(seeds, datasets, out='runs/E9'):
    """lưới kiến trúc × hàm kích hoạt, ở 30 % nhãn, cho CẢ baseline dữ liệu lẫn PINN.
    'mono' tách bạch: tiên nghiệm đơn điệu đặt vào KIẾN TRÚC so với đặt vào HÀM MẤT MÁT."""
    for ds, s in itertools.product(datasets, seeds):
        for name, kw in E9_ARCHS.items():
            for m in ['mlp', 'pinn_semi']:
                # với kiến trúc mono, tính đơn điệu đã có sẵn -> tắt số hạng phạt
                extra = dict(beta_mono=0.0) if (name == 'silu-mono' and m == 'pinn_semi') else {}
                if name == 'silu-mono+L' and m == 'mlp':
                    continue                       # biến thể này chỉ có nghĩa với PINN
                run(Config(dataset=ds, label_frac=0.3, model=m, seed=s, tag=name, out_dir=out,
                           save_predictions=(s == 0), **kw, **extra), verbose=False)


# --------------------------------------------------------------------------- E10 full transfer matrix
def E10(seeds, datasets, out='runs/E10'):
    """ma trận chuyển miền ĐẦY ĐỦ 4x4: mỗi bộ nguồn -> đánh giá zero-shot trên tập TEST
    của cả bốn bộ (đường chéo = kết quả trong miền, nên so sánh trực tiếp được)."""
    for ds, s in itertools.product(datasets, seeds):
        for norm in ['global', 'first_cycle']:
            for name, kw in [('mlp', dict(model='mlp')),
                             ('pinn', dict(model='pinn_semi', beta_mono=0.5, residual='autograd', alpha_ode=0.02))]:
                run(Config(dataset=ds, label_frac=0.7, seed=s, norm=norm, tag=name,
                           transfer_to=tuple(DATASETS), out_dir=out, save_predictions=False, **kw), verbose=False)


# --------------------------------------------------------------------------- E12 fair tuning budget
# Lưới siêu tham số TỔNG QUÁT (không liên quan vật lý), áp dụng y hệt cho cả baseline và PINN.
# Trước E12, PINN được quét beta/residual/arch còn MLP chạy cấu hình cố định -> so sánh thiên vị.
E12_GRID = {
    'base':      dict(),
    'wd1e-3':    dict(wd=1e-3),
    'wd1e-2':    dict(wd=1e-2),
    'drop0.1':   dict(dropout=0.1),
    'wide':      dict(hidden=(128, 128, 64)),
    'small':     dict(hidden=(32, 32, 16)),
    'mono':      dict(arch='mono'),                      # kiến trúc tốt nhất theo E9
    'mono_wd':   dict(arch='mono', wd=1e-3),
}
E12_FRACS = [0.3, 0.7]
# PINN dùng cấu hình vật lý tốt nhất theo E7 làm nền, rồi quét đúng lưới tổng quát bên trên
E12_PINN_BASE = dict(beta_mono=0.5, residual='autograd', alpha_ode=0.02)


def E12(seeds, datasets, out='runs/E12'):
    for ds, s in itertools.product(datasets, seeds):
        for f in E12_FRACS:
            for name, kw in E12_GRID.items():
                for m in ['mlp', 'pinn_semi']:
                    extra = dict(E12_PINN_BASE) if m == 'pinn_semi' else {}
                    if kw.get('arch') == 'mono' and m == 'pinn_semi':
                        extra['beta_mono'] = 0.0       # đơn điệu đã có trong kiến trúc
                    run(Config(dataset=ds, label_frac=f, model=m, seed=s, tag=name, out_dir=out,
                               save_predictions=False, **kw, **extra), verbose=False)


# --------------------------------------------------------------------------- E13 confirmatory (pipeline v2, CV)
# Đề cương ấn định trước: tai-lieu/de-cuong-E13.md. KHÔNG sửa cấu hình dưới đây sau khi đã chạy.
E13_NET = dict(hidden=(128, 128, 64))                                   # TUNED 'wide'
E13_PHYS = dict(beta_mono=0.5, residual='autograd', alpha_ode=0.02)     # E7, chọn theo validation
E13_ARMS = [   # (tag, model, label_frac, extra) — thứ tự = thứ tự ưu tiên chạy
    ('A', 'mlp', 0.3, {}),
    ('C', 'pinn_semi', 0.3, E13_PHYS),
    ('F', 'mlp', 0.7, {}),
    ('B', 'pinn_sup', 0.3, E13_PHYS),
    ('D', 'pinn_semi', 0.3, dict(E13_PHYS, alpha_ode=0.0, gamma_range=0.0)),
    ('E', 'pinn_semi', 0.3, dict(beta_mono=0.5, residual='euler', alpha_ode=2.0)),
    ('G', 'pinn_semi', 0.7, E13_PHYS),
    ('L', 'mlp', 0.1, {}), ('L', 'pinn_semi', 0.1, E13_PHYS),
    ('L', 'mlp', 0.5, {}), ('L', 'pinn_semi', 0.5, E13_PHYS),
]
E13_FOLDS = 5


def _worker_init():
    torch.set_num_threads(1)


def _run_job(kind, kw):
    import time as _t
    t0 = _t.time()
    if kind == 'train':
        cfg = Config(**kw).v2()
        r = run(cfg, verbose=False)
        return kw.get('tag', ''), cfg.dataset, cfg.model, r['test'].get('MAE_cell'), _t.time() - t0
    cfg = TransferConfig(**kw)
    cfg.clean, cfg.pair_by = 'causal', 'cycle'                           # v2; norm_pool đặt trong kw
    r = run_transfer(cfg)
    return kw.get('tag', ''), f'{cfg.dataset}->{cfg.target}', cfg.model, r['target_test'].get('MAE_cell'), _t.time() - t0


def _pool_run(jobs, workers):
    import time as _t
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    t0 = _t.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=_worker_init) as ex:
        futs = [ex.submit(_run_job, k, kw) for k, kw in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            tag, ds, m, mae, dt = f.result()
            el = _t.time() - t0
            print(f'[{i:4d}/{len(jobs)}] {tag:>2s} {ds:<11s} {m:<10s} MAE_cell {mae:.4f}  {dt:4.0f}s  '
                  f'(đã {el/60:.1f} ph, còn ~{el/i*(len(jobs)-i)/60:.0f} ph)', flush=True)
    print('xong', flush=True)


def E13(seeds, datasets, out='runs/E13', workers=4, repeats=5):
    """Đối chứng xác nhận, ấn định trước (tai-lieu/de-cuong-E13.md). `seeds` không dùng:
    seed = 100 * lần_lặp + fold, giống nhau cho mọi nhánh -> ghép cặp."""
    from pinnsoh.train import run_name
    jobs = []
    for (tag, m, f, extra), r, ds, k in itertools.product(E13_ARMS, range(repeats), datasets, range(E13_FOLDS)):
        kw = dict(dataset=ds, label_frac=f, model=m, seed=100 * r + k, split='cv', fold=k, repeat=r,
                  n_folds=E13_FOLDS, tag=tag, out_dir=out, **E13_NET, **extra)
        if not os.path.exists(os.path.join(out, run_name(Config(**kw)) + '.json')):
            jobs.append(('train', kw))
    print(f'E13: cần chạy {len(jobs)} lượt', flush=True)
    _pool_run(jobs, workers)


def E14(seeds, datasets, out='runs/E14', workers=4, repeats=None):
    """E5 lặp lại với CÙNG normaliser cho MLP và PINN (norm_pool='train'), pipeline v2."""
    jobs = []
    for ds, s, k, m in itertools.product(datasets, seeds, [0, 3], ['mlp', 'pinn_semi']):
        kw = dict(dataset=ds, target=PAIRS[ds][0], model=m, k_target=k, seed=s, norm='global',
                  norm_pool='train', tag='samenorm', out_dir=out, save_predictions=False)
        name = f'samenorm_{ds}_to_{PAIRS[ds][0]}_{m}_k{k}_s{s}_global'
        if not os.path.exists(os.path.join(out, name + '.json')):
            jobs.append(('transfer', kw))
    print(f'E14: cần chạy {len(jobs)} lượt', flush=True)
    _pool_run(jobs, workers)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', choices=['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E12', 'E13', 'E14'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4, help='E13/E14: số tiến trình song song (mỗi tiến trình 1 luồng)')
    ap.add_argument('--repeats', type=int, default=5, help='E13: số lần lặp kiểm định chéo')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.exp in ('E13', 'E14'):
        globals()[a.exp](a.seeds, a.datasets, workers=a.workers, repeats=a.repeats)
    else:
        globals()[a.exp](a.seeds, a.datasets)
