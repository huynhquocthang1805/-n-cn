"""
Chạy lại mã gốc của PINN4SOH (Wang và cộng sự, Nature Communications 2024) trên đúng 39 nhóm chia của bài.

    python benchmark/rerun_pinn4soh.py --version published      # bản đã sinh ra số liệu công bố
    python benchmark/rerun_pinn4soh.py --version current        # bản hiện tại của repo (đã sửa lỗi)
    python benchmark/rerun_pinn4soh.py --version published --runs 3 --keys 'XJTU|full|0-0' --workers 2

Mã gốc: github.com/wang-fujin/PINN4SOH, tự tải về .cache/PINN4SOH-src nếu chưa có. Lớp PINN (Model/Model.py) và các
lớp đọc dữ liệu (dataloader/dataloader.py) được import nguyên trạng, không sửa dòng nào: lọc 3σ cả vòng đời, min–max
theo từng cell, validation là 20 % hàng ngẫu nhiên (random_state 420), chọn epoch theo MSE validation, dừng sớm sau
20 epoch không cải thiện. Dataloader không đổi suốt lịch sử repo; Model.py chỉ khác nhau ở loss đơn điệu:

    published  commit 5d57a9d (bản tải lên đầu tiên): loss3 = relu(y2 − y1).sum(), chỉ tính từ nhãn nên không có
               gradient — số hạng β không tác động lên mô hình. Nhật ký công bố (results/Ours/*.zip, 2023) ghi
               "physics loss" là một hằng số qua mọi epoch ở cả 39 nhóm × 10 lần chạy, đúng bằng giá trị của biểu
               thức này trên các hàng train dựng từ dữ liệu ở đây (check_logs, ví dụ 0.043214 ở XJTU 2C); siêu tham
               số lấy từ các dòng CRITICAL của nhật ký đó (HYPER['published']). Hai lần sửa sau đó: relu(u2 − u1).sum()
               (07/2024) rồi bản hiện tại.
    current    commit cc3cc50 (2026-09): loss3 = relu((u2 − u1)(y1 − y2)).sum(); α, β lấy từ giá trị mặc định
               trong main_*.py ở commit này (fix.md: sau khi sửa lỗi, α và β khác Supplementary).

Cách chia: pinnsoh/pinn4soh_splits.json (dựng lại từ dòng "data path:" trong nhật ký công bố, đúng thứ tự tệp).
Bài không đặt hạt giống; ở đây lần chạy k đặt hạt giống k cho torch, numpy, random để chạy lại được.

Môi trường: mã gốc viết cho pandas 1.3.5. Từ pandas 3, phép gán df.iloc[:, :-1] = f_df trong read_one_csv báo lỗi
kiểu dữ liệu (cột chỉ số chu kỳ kiểu int nhận giá trị thực), nên chạy trong môi trường có pandas 2.x, ví dụ
    python -m venv --system-site-packages .cache/venv-pinn4soh
    .cache/venv-pinn4soh/bin/python -m pip install "pandas==2.3.3"
    .cache/venv-pinn4soh/bin/python benchmark/rerun_pinn4soh.py --version published

Kết quả mỗi lần chạy: runs/pinn4soh/<bản>/<nhóm>/run<k>/ (nhật ký, pred_label.npy, true_label.npy, model.pth của mã
gốc) và runs/pinn4soh/<bản>/<nhóm>_run<k>.json (dự đoán theo cell test và sai số từng cell).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # gốc dự án
_os.chdir(_sys.path[0])
import argparse, contextlib, glob, json, os, re, shutil, subprocess, sys, time, warnings, zipfile
from types import SimpleNamespace

REPO = 'https://github.com/wang-fujin/PINN4SOH'
COMMIT = 'cc3cc5053caf38f04e0665f7f88cb109144d035e'            # dữ liệu, dataloader, utils, nhật ký công bố
MODEL_COMMIT = dict(published='5d57a9d1bc0afcc6c2f5ea9fecbea5d6d11b8e14', current=COMMIT)
SRC = '.cache/PINN4SOH-src'
DATA = 'PINN4SOH/data'
OUT = 'runs/pinn4soh'
SPLITS = 'pinnsoh/pinn4soh_splits.json'

COMMON = dict(normalization_method='min-max', epochs=200, early_stop=20, warmup_epochs=30, warmup_lr=0.002, lr=0.01,
              final_lr=0.0002, F_layers_num=3, F_hidden_dim=60, log_dir='logging.txt')
# (bộ, chế độ) -> siêu tham số; 'nK' là thí nghiệm ít mẫu K = 1..4 cell có nhãn
HYPER = {
    'published': {('XJTU', 'full'): dict(batch_size=256, alpha=0.7, beta=20, lr_F=0.001),
                  ('XJTU', 'nK'): dict(batch_size=128, alpha=0.5, beta=10, lr_F=0.001),
                  ('TJU', 'full'): dict(batch_size=512, alpha=1, beta=50, lr_F=0.001),
                  ('MIT', 'full'): dict(batch_size=512, alpha=1, beta=50, lr_F=0.001),
                  ('HUST', 'full'): dict(batch_size=512, alpha=0.5, beta=80, lr_F=0.0005),
                  ('HUST', 'nK'): dict(batch_size=128, alpha=0.5, beta=80, lr_F=0.0005)},
    'current': {('XJTU', 'full'): dict(batch_size=256, alpha=0.7, beta=0.2, lr_F=0.001),
                ('XJTU', 'nK'): dict(batch_size=128, alpha=0.5, beta=10, lr_F=0.001),
                ('TJU', 'full'): dict(batch_size=512, alpha=0.1, beta=0.1, lr_F=0.001),
                ('MIT', 'full'): dict(batch_size=512, alpha=0.5, beta=0.01, lr_F=0.001),
                ('HUST', 'full'): dict(batch_size=512, alpha=0.6, beta=0.1, lr_F=0.0005),
                ('HUST', 'nK'): dict(batch_size=128, alpha=0.6, beta=0.1, lr_F=0.0005)},
}
# thời gian tương đối một lần chạy, để xếp việc lớn lên trước
COST = {'XJTU': 1, 'TJU': 12, 'MIT': 30, 'HUST': 25}


def hyper(key, version):
    ds, mode, _ = key.split('|')
    return dict(COMMON, data=ds, **HYPER[version][(ds, 'full' if mode == 'full' else 'nK')])


def fetch_src(dest=SRC):
    """Tải mã ở các commit đã ghim (cần git): dataloader/, utils/, nhật ký công bố ở COMMIT; Model/ của từng bản
    ở <dest>/<bản>/Model."""
    if all(os.path.exists(os.path.join(dest, v, 'Model', 'Model.py')) for v in MODEL_COMMIT):
        return dest
    tmp = dest + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(['git', 'clone', '--quiet', '--no-checkout', REPO + '.git', tmp], check=True)
    subprocess.run(['git', '-C', tmp, 'checkout', '--quiet', COMMIT, '--', 'dataloader', 'utils', 'results/Ours'],
                   check=True)
    for v, c in MODEL_COMMIT.items():
        os.makedirs(os.path.join(tmp, v, 'Model'), exist_ok=True)
        for f in ('__init__.py', 'Model.py'):
            blob = subprocess.run(['git', '-C', tmp, 'show', f'{c}:Model/{f}'], check=True, capture_output=True).stdout
            open(os.path.join(tmp, v, 'Model', f), 'wb').write(blob)
    shutil.rmtree(os.path.join(tmp, '.git'))
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return dest


def check_logged_args(src):
    """So bảng HYPER['published'] với các dòng CRITICAL trong nhật ký công bố (bỏ qua nếu không có tệp zip)."""
    fields = ('batch_size', 'alpha', 'beta', 'lr_F', 'lr', 'epochs', 'early_stop')
    seen = {}
    for zp in glob.glob(os.path.join(src, 'results', 'Ours', '*.zip')):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.endswith('logging.txt'):
                    continue
                args = dict(line.split('\t')[-1].strip().split(':', 1)
                            for line in z.read(name).decode('utf-8', 'replace').splitlines() if 'CRITICAL' in line)
                mode = 'nK' if re.search(r'small sample \d', name) else 'full'
                seen.setdefault((args['data'], mode), set()).add(tuple((k, float(args[k])) for k in fields))
    for k, v in seen.items():
        want = dict(COMMON, **HYPER['published'][k])
        assert v == {tuple((a, float(want[a])) for a in fields)}, (k, v)
    return sorted(seen)


def log_member(key, e):
    """(tệp zip, đường dẫn trong zip) của nhật ký công bố, lần chạy thứ e (1..10) của nhóm key."""
    ds, mode, grp = key.split('|')
    n = '' if mode == 'full' else f' (small sample {mode[1]})'
    zipname = {'XJTU': 'XJTU.zip', 'TJU': 'TJU results.zip', 'MIT': 'MIT results.zip', 'HUST': 'HUST.zip'}[ds]
    folder = {'XJTU': f'XJTU results{n}/{grp}', 'TJU': f'TJU results/{grp}', 'MIT': 'MIT results',
              'HUST': f'HUST results{n}'}[ds]
    return zipname, f'{folder}/Experiment{e}/logging.txt'


def check_logs(src, out=os.path.join(OUT, 'logged_constant.json')):
    """Nhật ký công bố ghi 'physics loss' trung bình mỗi epoch. Kiểm tra (i) đó là một hằng số qua mọi epoch của cả 10
    lần chạy; (ii) hằng số đó bằng relu(y2 − y1).sum() lấy trung bình theo batch trên các hàng train mà dataloader gốc
    dựng từ bản dữ liệu ở đây (với batch xáo trộn, trung bình theo batch là tổng chia số batch, không phụ thuộc thứ tự).
    (i) nghĩa là loss đơn điệu của bản công bố không phụ thuộc mô hình; (ii) nghĩa là các hàng train (sau lọc 3σ và chia
    80/20 với random_state 420) trùng với lần chạy công bố."""
    import numpy as np
    import torch
    sys.path[:0] = [os.path.abspath(src)]
    from dataloader.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata
    rows = []
    for key, sp in json.load(open(SPLITS)).items():
        ds = key.split('|')[0]
        logged = set()
        for e in range(1, 11):
            zp, member = log_member(key, e)
            with zipfile.ZipFile(os.path.join(src, 'results', 'Ours', zp)) as z:
                text = z.read(member).decode('utf-8', 'replace')
            logged |= {float(v) for v in re.findall(r'\[Train\] epoch:\d+.*?physics loss:([\d.]+)', text)}
        args = SimpleNamespace(**hyper(key, 'published'), save_folder=None)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            data = dict(XJTU=XJTUdata, TJU=TJUdata, MIT=MITdata, HUST=HUSTdata)[ds](
                root=os.path.join(DATA, f'{ds} data'), args=args)
            loader = data.read_all(specific_path_list=[csv_path(c) for c in sp['train']])['train_2']
        ours = sum(torch.relu(y2 - y1).sum().item() for _, _, y1, y2 in loader) / len(loader)
        rows.append(dict(key=key, logged=sorted(logged), ours=ours,
                         constant=len(logged) == 1,
                         equal=len(logged) == 1 and bool(np.isclose(next(iter(logged)), ours, rtol=2e-6, atol=6e-7))))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, 'w'), indent=1)
    return sum(r['constant'] for r in rows), sum(r['equal'] for r in rows), len(rows)


def csv_path(cid):
    ds, rest = cid.split('/', 1)
    if ds == 'MIT':
        return os.path.join(DATA, 'MIT data', rest.split('_battery')[0], rest + '.csv')
    return os.path.join(DATA, f'{ds} data', rest + '.csv')


def run_path(version, key, k):
    return os.path.join(OUT, version, f'{key.replace("|", "_")}_run{k}.json')


def run_one(version, key, k, src):
    """Một lần chạy mã gốc (bản version) trên nhóm key với hạt giống k."""
    import random
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path[:0] = [os.path.abspath(os.path.join(src, version)), os.path.abspath(src)]
    from Model.Model import PINN
    from dataloader.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata
    split = json.load(open(SPLITS))[key]
    ds = key.split('|')[0]
    folder = os.path.join(OUT, version, key.replace('|', '_'), f'run{k}')
    os.makedirs(folder, exist_ok=True)
    for f in glob.glob(os.path.join(folder, '*')):
        os.remove(f)                                   # chạy lại từ đầu nếu lần trước dở dang
    args = SimpleNamespace(**hyper(key, version), save_folder=folder)
    random.seed(k); np.random.seed(k); torch.manual_seed(k)
    t0 = time.time()
    train_paths = [csv_path(c) for c in split['train']]
    test_paths = [csv_path(c) for c in split['test']]
    with open(os.path.join(folder, 'stdout.txt'), 'w') as fo, contextlib.redirect_stdout(fo), \
            contextlib.redirect_stderr(fo), warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)   # pandas 2 cảnh báo phép gán khác kiểu trong read_one_csv
        cls = dict(XJTU=XJTUdata, TJU=TJUdata, MIT=MITdata, HUST=HUSTdata)[ds]
        data = cls(root=os.path.join(DATA, f'{ds} data'), args=args)
        trainloader = data.read_all(specific_path_list=train_paths)
        testloader = data.read_all(specific_path_list=test_paths)
        pinn = PINN(args)
        pinn.Train(trainloader=trainloader['train_2'], validloader=trainloader['valid_2'],
                   testloader=testloader['test_3'])
        # mỗi cell test góp (số hàng sau lọc − 1) hàng vào test_3, đúng thứ tự test_paths
        if ds == 'TJU':
            nominal = next(c for b, c in zip(data.batchs, data.nominal_capacities) if b in test_paths[0])
        else:
            nominal = data.nominal_capacity
        ys = [data.load_one_battery(q, nominal)[0][1] for q in test_paths]
    y = np.load(os.path.join(folder, 'true_label.npy')).reshape(-1).astype(np.float64)
    p = np.load(os.path.join(folder, 'pred_label.npy')).reshape(-1).astype(np.float64)
    n = [len(v) for v in ys]
    assert sum(n) == len(y) and np.array_equal(np.concatenate(ys).astype(np.float32), y.astype(np.float32)), key
    cell = np.repeat(np.arange(len(n)), n)
    cyc = np.concatenate([np.arange(m) for m in n])
    err = p - y
    per = [dict(cell=split['test'][i], n=int(n[i]), MAE=float(np.abs(err[cell == i]).mean()),
                RMSE=float(np.sqrt((err[cell == i] ** 2).mean())),
                MAPE=float((np.abs(err[cell == i]) / np.abs(y[cell == i])).mean())) for i in range(len(n))]
    log = open(os.path.join(folder, 'logging.txt')).read()
    res = dict(key=key, version=version, run=k, seed=k, hyper=hyper(key, version), model_commit=MODEL_COMMIT[version],
               epochs=log.count('[Train]'), time_s=time.time() - t0, cells=per,
               MAE_cell=float(np.mean([c['MAE'] for c in per])), RMSE_cell=float(np.mean([c['RMSE'] for c in per])),
               pooled=dict(MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean()))),
               test_predictions=dict(y=y.tolist(), p=p.tolist(), cell=cell.tolist(), cycle=cyc.tolist(),
                                     cell_ids=split['test']))
    out = run_path(version, key, k)
    json.dump(res, open(out + '.tmp', 'w'))
    os.replace(out + '.tmp', out)
    return key, k, res['MAE_cell'], res['epochs'], res['time_s']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=sorted(MODEL_COMMIT), default='published')
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--keys', nargs='*', help="mặc định: cả 39 nhóm, ví dụ 'XJTU|full|0-0'")
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--src', default=SRC, help='thư mục mã gốc (xem fetch_src)')
    ap.add_argument('--check-logs-only', action='store_true', help='chỉ đối chiếu nhật ký công bố (check_logs)')
    a = ap.parse_args()
    import pandas
    if int(pandas.__version__.split('.')[0]) >= 3:
        sys.exit(f'pandas {pandas.__version__}: mã gốc cần pandas < 3 (xem đầu tệp này)')
    src = fetch_src(a.src)
    print('siêu tham số của bản published khớp nhật ký công bố:', check_logged_args(src))
    if not os.path.exists(os.path.join(OUT, 'logged_constant.json')) or a.check_logs_only:
        c, e, n = check_logs(src)
        print(f'nhật ký công bố: physics loss là hằng số ở {c}/{n} nhóm, bằng relu(y2 − y1).sum() trên dữ liệu ở đây '
              f'ở {e}/{n} nhóm', flush=True)
    if a.check_logs_only:
        return
    keys = a.keys or list(json.load(open(SPLITS)))
    todo = [(key, k) for key in keys for k in range(a.runs) if not os.path.exists(run_path(a.version, key, k))]
    todo.sort(key=lambda t: -COST[t[0].split('|')[0]] * (1 if '|full|' in t[0] else 0.3))
    print(f'{a.version}: {len(todo)} lần chạy cần làm ({len(keys)} nhóm × {a.runs} lần)', flush=True)
    os.makedirs(os.path.join(OUT, a.version), exist_ok=True)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as ex:
        futs = {ex.submit(run_one, a.version, key, k, src): (key, k) for key, k in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                key, k, mae, ep, dt = f.result()
                print(f'[{i}/{len(todo)}] {key} run{k}: MAE_cell {mae:.5f}, {ep} epoch ({dt:.0f} s; '
                      f'{time.time() - t0:.0f} s)', flush=True)
            except Exception as e:
                print(f'[{i}/{len(todo)}] {futs[f]} LỖI: {e!r}', flush=True)


if __name__ == '__main__':
    main()
