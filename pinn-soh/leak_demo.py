"""
Chứng minh bằng số vì sao chuẩn hoá chỉ số chu kỳ THEO TỪNG CELL là rò rỉ.

    python leak_demo.py

Ba phép thử, không dùng mạng nơ-ron nào cho hai phép đầu:

  A. Chỉ số chu kỳ sau min-max theo cell BẰNG ĐÚNG "phần trăm tuổi thọ đã đi qua".
     Chứng minh bằng đẳng thức đại số trên dữ liệu thật, sai số máy.

  B. Chỉ riêng con số đó — MỘT đầu vào duy nhất, KHÔNG dùng chút đặc trưng sạc nào —
     đã dự đoán được SOH tốt tới mức nào. Nếu tốt, nghĩa là đáp án đã nằm sẵn trong
     đầu vào chứ không phải mô hình học được gì từ đường sạc.

  C. Cùng mô hình đó khi triển khai thật: tại chu kỳ N ta KHÔNG biết n_max, nên phải
     thay bằng ước lượng. Đo xem sai số bung ra bao nhiêu.
"""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from pinnsoh.data import load_dataset

ROOT = 'PINN4SOH/data'


def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def phan_a(cells, ds):
    """Đẳng thức: minmax(N) = 2·(N − N_min)/(N_max − N_min) − 1 = 2·(% tuổi thọ) − 1."""
    errs = []
    for c in cells:
        n = c.cycle.astype(float)
        minmax = 2 * (n - n.min()) / (n.max() - n.min() + 1e-12) - 1
        pct_life = (n - n.min()) / (n.max() - n.min() + 1e-12)      # phần trăm vòng đời đã đi
        errs.append(np.max(np.abs(minmax - (2 * pct_life - 1))))
    print(f'  A. {ds}: sai lệch lớn nhất giữa "chỉ số đã chuẩn hoá" và "2·%tuổi thọ − 1" '
          f'trên {len(cells)} cell = {max(errs):.2e}  → là CÙNG MỘT ĐẠI LƯỢNG')


def phan_b(cells, ds, seed=0):
    """Dự đoán SOH CHỈ từ chỉ số chu kỳ đã chuẩn hoá theo cell. Không đặc trưng sạc nào."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cells))
    ntr = int(0.7 * len(cells))
    tr = [cells[i] for i in idx[:ntr]]
    te = [cells[i] for i in idx[ntr:]]

    def feat(c, leak: bool):
        n = c.cycle.astype(float)
        if leak:                                   # cách của PINN4SOH — cần biết n_max
            return 2 * (n - n.min()) / (n.max() - n.min() + 1e-12) - 1
        return n / 1000.0                          # nhân quả — chỉ cần chu kỳ hiện tại

    out = {}
    for leak in (True, False):
        xtr = np.concatenate([feat(c, leak) for c in tr])
        ytr = np.concatenate([c.soh for c in tr])
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip').fit(xtr, ytr)
        xte = np.concatenate([feat(c, leak) for c in te])
        yte = np.concatenate([c.soh for c in te])
        out[leak] = mae(iso.predict(xte), yte)
    print(f'  B. {ds}: MAE khi CHỈ dùng chỉ số chu kỳ, không dùng 16 đặc trưng sạc'
          f'\n        · chuẩn hoá theo cell (rò rỉ) : {out[True]:.4f}'
          f'\n        · chia hằng số 1000 (nhân quả): {out[False]:.4f}'
          f'   →  rò rỉ tốt hơn {out[False] / out[True]:.1f} lần')
    return out


def phan_c(cells, ds, seed=0):
    """Triển khai thật: tại chu kỳ N chưa biết n_max. Thay bằng ước lượng và đo sai số."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cells))
    ntr = int(0.7 * len(cells))
    tr = [cells[i] for i in idx[:ntr]]
    te = [cells[i] for i in idx[ntr:]]

    leak = lambda c: 2 * (c.cycle.astype(float) - c.cycle.min()) / (c.cycle.max() - c.cycle.min() + 1e-12) - 1
    xtr = np.concatenate([leak(c) for c in tr]); ytr = np.concatenate([c.soh for c in tr])
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip').fit(xtr, ytr)

    n_tb = float(np.mean([c.cycle.max() - c.cycle.min() + 1 for c in tr]))   # tuổi thọ TRUNG BÌNH của tập train
    p, q = [], []
    for c in te:
        n = c.cycle.astype(float)
        xhat = 2 * (n - n.min()) / n_tb - 1           # chỉ dùng thứ biết được lúc chạy thật
        p.append(iso.predict(np.clip(xhat, -1, 1))); q.append(c.soh)
    print(f'  C. {ds}: cùng mô hình đó khi chạy online (không biết trước tuổi thọ, '
          f'lấy trung bình {n_tb:.0f} chu kỳ của tập train)\n        MAE = {mae(np.concatenate(p), np.concatenate(q)):.4f}')


if __name__ == '__main__':
    for ds in ['XJTU', 'MIT', 'HUST']:
        cells = load_dataset(ROOT, ds)
        life = np.array([c.cycle.max() - c.cycle.min() + 1 for c in cells])
        print(f'\n### {ds} — {len(cells)} cell, tuổi thọ {life.min()}–{life.max()} chu kỳ '
              f'(trung vị {int(np.median(life))}, lệch {life.max()/life.min():.1f} lần)')
        phan_a(cells, ds)
        phan_b(cells, ds)
        phan_c(cells, ds)
