"""
Demo: ước lượng SOH theo từng chu kỳ cho MỘT cell, chỉ từ 16 đặc trưng đường sạc.

    # 1) huấn luyện mô hình demo (PINN-semi, 30 % cell có nhãn, pipeline v2) — ~1 phút
    python demo.py train --dataset XJTU

    # 2) ước lượng cho một cell. Cell này KHÔNG được dùng khi huấn luyện (nằm trong fold test).
    python demo.py predict --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-1.csv"

    # 3) giả lập cell CHƯA ĐO dung lượng: bỏ cột capacity trước khi đưa vào mô hình
    python demo.py predict --model demo/XJTU.pt --csv "..." --an-nhan

Mô hình chạy nhân quả: ước lượng ở chu kỳ N chỉ dùng đặc trưng của chu kỳ N và chỉ số N; bộ lọc
ngoại lai chỉ dùng các chu kỳ trước N. Vì vậy cùng một mô hình dùng được trực tuyến trong BMS.

Đầu ra: demo/<tên cell>_soh.csv (chu kỳ, SOH ước lượng, SOH thực đo nếu có) và .png.
"""
from __future__ import annotations
import argparse, os, sys, tempfile
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    from pinnsoh.plotstyle import apply_theme, MODEL_COLOR, MUTED, save
    apply_theme()
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    if truth is not None:
        ax.plot(cell.cycle, truth, color=MODEL_COLOR['truth'], lw=1.6, label='thực đo')
    ax.plot(cell.cycle, soh, color=MODEL_COLOR['pinn_semi'], lw=1.1, label='PINN-semi ước lượng')
    ax.axhline(EOL_THRESHOLD, color='#c4ccc8', lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel('chu kỳ'); ax.set_ylabel('SOH')
    ax.set_title(f'{name}' + (' — không dùng nhãn dung lượng' if a.an_nhan else ''), loc='left', fontweight='bold')
    ax.legend(loc='lower left')
    fig.tight_layout()
    save(fig, os.path.join(DEMO_DIR, f'{name}_soh'), also_pdf=False, dpi=200)
    print(f'đã ghi {DEMO_DIR}/{name}_soh.csv và .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('train'); t.add_argument('--dataset', default='XJTU', choices=['XJTU', 'TJU', 'MIT', 'HUST'])
    p = sp.add_parser('predict')
    p.add_argument('--model', required=True); p.add_argument('--csv', required=True)
    p.add_argument('--nominal', type=float, default=None, help='dung lượng danh định (Ah); mặc định theo bộ')
    p.add_argument('--temp', type=float, default=25.0, help='nhiệt độ môi trường (°C); chỉ ảnh hưởng mạng tốc độ, không ảnh hưởng suy luận')
    p.add_argument('--an-nhan', action='store_true', help='bỏ cột capacity: giả lập cell chưa đo dung lượng')
    a = ap.parse_args()
    {'train': cmd_train, 'predict': cmd_predict}[a.cmd](a)
