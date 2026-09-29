"""
Nâng số seed của phép so chính từ 3 lên 10, để có đủ lực cho kiểm định thống kê.

    python seed_sweep.py

Phạm vi: E1 ở mức 30 % nhãn — đúng cấu hình phát biểu giả thuyết — trên cả bốn bộ,
hai mô hình MLP và PINN-semi, seed 0..9.  Ghi vào runs/E1 nên ba seed đã chạy được
dùng lại, chỉ chạy phần thiếu.  Bắt buộc save_predictions=True vì kiểm định ghép cặp
cần dự đoán theo từng cell.

Với cùng một seed, hai mô hình dùng ĐÚNG một cách chia cell (split_cells gieo theo
seed), nên cặp (MLP, PINN) trên mỗi cell test là cặp hợp lệ để ghép.
"""
from __future__ import annotations
import functools, os, sys, time
import pinnsoh.data as D
from pinnsoh.train import Config, run, run_name

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
MODELS = ['mlp', 'pinn_semi']
SEEDS = list(range(10))
FRAC = 0.3
OUT = 'runs/E1'

# nạp mỗi bộ dữ liệu đúng một lần cho cả 80 lượt
_orig = D.load_dataset
D.load_dataset = functools.lru_cache(maxsize=8)(lambda root, name: tuple(_orig(root, name)))
import pinnsoh.train as T
T.load_dataset = lambda root, name: list(D.load_dataset(root, name))


def main():
    todo = []
    for ds in DATASETS:
        for m in MODELS:
            for s in SEEDS:
                cfg = Config(dataset=ds, label_frac=FRAC, model=m, seed=s, out_dir=OUT)
                if not os.path.exists(os.path.join(OUT, run_name(cfg) + '.json')):
                    todo.append(cfg)
    print(f'cần chạy {len(todo)} / {len(DATASETS)*len(MODELS)*len(SEEDS)} lượt', flush=True)
    t0 = time.time()
    for i, cfg in enumerate(todo, 1):
        r = run(cfg, verbose=False)
        el = time.time() - t0
        print(f'[{i:3d}/{len(todo)}] {cfg.dataset:5s} {cfg.model:10s} seed{cfg.seed}  '
              f'test MAE {r["test"]["MAE"]:.4f}   ({el/60:.1f} ph, còn ~{el/i*(len(todo)-i)/60:.0f} ph)',
              flush=True)
    print('xong', flush=True)


if __name__ == '__main__':
    main()
