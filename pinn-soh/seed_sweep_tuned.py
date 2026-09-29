"""
Lặp lại phép so chính với cấu hình ĐÃ TINH CHỈNH, 10 seed.

    python seed_sweep_tuned.py

Vì sao cần: seed_sweep.py chạy cấu hình E1 (β = 5 cố định) — chính là cấu hình mà báo
cáo đã nói là yếu. Kiểm định trên nó chỉ bác bỏ được phát biểu của E1, không nói gì về
kết luận của E12. Ở đây dùng cấu hình mạng `wide` (128,128,64) — biến thể mà E12 chọn
theo validation ở 6/8 ô — cho CẢ HAI mô hình, cộng cấu hình vật lý tốt nhất theo E7 cho
PINN. Cấu hình được ẤN ĐỊNH TRƯỚC, không chọn lại theo 10 seed, nên không có thiên lệch
do chọn mô hình.
"""
from __future__ import annotations
import functools, os, time
import pinnsoh.data as D
from pinnsoh.train import Config, run, run_name

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
SEEDS = list(range(10))
FRAC = 0.3
OUT = 'runs/TUNED'
WIDE = dict(hidden=(128, 128, 64))
PINN_PHYS = dict(beta_mono=0.5, residual='autograd', alpha_ode=0.02)   # E7 chọn theo validation

_orig = D.load_dataset
D.load_dataset = functools.lru_cache(maxsize=8)(lambda root, name: tuple(_orig(root, name)))
import pinnsoh.train as T
T.load_dataset = lambda root, name: list(D.load_dataset(root, name))


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for ds in DATASETS:
        for s in SEEDS:
            jobs.append(Config(dataset=ds, label_frac=FRAC, model='mlp', seed=s,
                               tag='wide', out_dir=OUT, **WIDE))
            jobs.append(Config(dataset=ds, label_frac=FRAC, model='pinn_semi', seed=s,
                               tag='wide', out_dir=OUT, **WIDE, **PINN_PHYS))
    todo = [c for c in jobs if not os.path.exists(os.path.join(OUT, run_name(c) + '.json'))]
    print(f'cần chạy {len(todo)} / {len(jobs)} lượt', flush=True)
    t0 = time.time()
    for i, cfg in enumerate(todo, 1):
        r = run(cfg, verbose=False)
        el = time.time() - t0
        print(f'[{i:3d}/{len(todo)}] {cfg.dataset:5s} {cfg.model:10s} seed{cfg.seed}  '
              f'MAE {r["test"]["MAE"]:.4f}  ({el/60:.1f} ph, còn ~{el/i*(len(todo)-i)/60:.0f} ph)',
              flush=True)
    print('xong', flush=True)


if __name__ == '__main__':
    main()
