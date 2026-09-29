"""
Vì sao chia chỉ số chu kỳ cho 1000? Quét thử để trả lời bằng số.

    python scale_sweep.py

Hằng số này KHÔNG phải siêu tham số được tinh chỉnh — nó là ĐƠN VỊ ĐO thời gian.
Điều kiện duy nhất bắt buộc là nó phải TOÀN CỤC (giống nhau cho mọi cell) để không
rò rỉ tuổi thọ. Giá trị cụ thể chỉ ảnh hưởng tới điều kiện số học:

    t = N / S            -> miền đầu vào của mạng nghiệm
    r = -du/dt           -> biên độ đầu ra của mạng động học (softplus)

Quét S ∈ {1, 100, 1000, 10000} để kiểm chứng: trong vùng hợp lý kết quả gần như
không đổi, còn ở hai đầu cực đoan thì điều kiện số học hỏng.
"""
from __future__ import annotations
import functools, json, os, sys
import numpy as np, pandas as pd
import pinnsoh.data as D
from pinnsoh.train import Config, run

SCALES = [1.0, 100.0, 1000.0, 10000.0]
DATASETS = ['XJTU', 'HUST']
SEEDS = [0, 1, 2]
OUT = 'runs/SCALE'

# nạp dữ liệu một lần rồi dùng lại — quét 24 lượt mà đọc đĩa 2 lần
_orig = D.load_dataset
D.load_dataset = functools.lru_cache(maxsize=8)(lambda root, name: tuple(_orig(root, name)))
import pinnsoh.train as T
T.load_dataset = lambda root, name: list(D.load_dataset(root, name))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for S in SCALES:
        D.CYCLE_SCALE = S                       # cycle_input() đọc hằng số này lúc gọi
        for ds in DATASETS:
            for sd in SEEDS:
                cfg = Config(dataset=ds, label_frac=0.3, model='pinn_semi', seed=sd,
                             tag=f'S{int(S)}', out_dir=OUT, save_predictions=False)
                from pinnsoh.train import run_name
                jp = os.path.join(OUT, run_name(cfg) + '.json')
                if os.path.exists(jp):                    # chạy lại được, không làm lại việc đã xong
                    r = json.load(open(jp))
                else:
                    r = run(cfg, verbose=False)
                cells = list(D.load_dataset('PINN4SOH/data', ds))
                life = np.array([c.cycle.max() - c.cycle.min() + 1 for c in cells])
                rows.append(dict(scale=S, dataset=ds, seed=sd,
                                 test_MAE=r['test']['MAE'], val_MAE=r['val']['MAE'],
                                 t_max=float(life.max() / S),
                                 lam=r.get('physics_params', {}).get('lam', np.nan)))
                print(f'S={S:>7.0f} {ds} seed{sd}  test MAE {r["test"]["MAE"]:.4f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/scale_raw.csv', index=False)
    t = (df.groupby(['dataset', 'scale'])
           .agg(MAE=('test_MAE', 'mean'), sd=('test_MAE', 'std'), t_max=('t_max', 'first'))
           .reset_index().round(4))
    t.to_csv('results/scale_table.csv', index=False)
    print('\n=== MAE trên tập test theo hằng số chia, 30 % nhãn, trung bình 3 seed ===')
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
