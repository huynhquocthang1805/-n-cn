"""
Bộ chỉ số đánh giá cho ước lượng SOH.

MAE/RMSE/MAPE gộp mọi chu kỳ của mọi cell thành một con số, nên bị chi phối bởi
cell có vòng đời dài và bởi vùng SOH cao (dễ đoán). Bộ dưới đây bổ sung các chỉ số
nói đúng thứ người dùng pin quan tâm:

  Độ chính xác
    MAE, RMSE, MAPE, R2            : gộp toàn bộ điểm (giữ để so với literature)
    MAE_cell / RMSE_cell           : trung bình theo CELL (mỗi cell một phiếu, không thiên vị cell dài)
    MAE_cell_max                   : cell tệ nhất — chỉ số an toàn, không phải trung bình
    MAE_late                       : chỉ tính ở vùng SOH <= 0.90, nơi quyết định thay pin
    MaxAE                          : sai số tuyệt đối lớn nhất trên toàn bộ

  Tính hợp lý vật lý (không cần nhãn — đo được cả trên cell không nhãn)
    MonoViol                       : tổng bước tăng ngược của quỹ đạo dự đoán, trung bình theo cell
    Jitter                         : trung bình |u(N+1) - u(N)| — độ nhiễu của quỹ đạo
                                     (quỹ đạo thật rất trơn, nên jitter lớn = mô hình đang đoán theo nhiễu)

  Hữu dụng vận hành
    EOL_MAE / EOL_bias             : sai số (chu kỳ) khi dự báo thời điểm SOH cắt ngưỡng
    EOL_cov                        : tỉ lệ cell test thực sự cắt ngưỡng (mẫu số của hai chỉ số trên)
    RateErr                        : sai số tương đối của tốc độ suy giảm ước lượng trên cửa sổ 100 chu kỳ

Quy ước: mọi chỉ số đều "càng nhỏ càng tốt" trừ R2 và EOL_cov.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

EOL_THRESHOLD = 0.85          # ngưỡng cắt; 0.85 để đủ cell test cắt qua (0.80 quá ít trên MIT)
LATE_SOH = 0.90


def _first_crossing(v: np.ndarray, thr: float) -> Optional[int]:
    """Chỉ số đầu tiên mà v <= thr; None nếu không bao giờ cắt."""
    idx = np.nonzero(v <= thr)[0]
    return int(idx[0]) if len(idx) else None


def evaluate(y: np.ndarray, p: np.ndarray, cell: Optional[np.ndarray] = None,
             thr: float = EOL_THRESHOLD) -> Dict[str, float]:
    y = np.asarray(y, dtype=float).ravel(); p = np.asarray(p, dtype=float).ravel()
    e = p - y
    out = dict(
        MAE=float(np.abs(e).mean()),
        RMSE=float(np.sqrt((e ** 2).mean())),
        MAPE=float(100 * np.abs(e / np.clip(np.abs(y), 1e-6, None)).mean()),
        R2=float(1 - (e ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)),
        MaxAE=float(np.abs(e).max()),
    )
    late = y <= LATE_SOH
    out['MAE_late'] = float(np.abs(e[late]).mean()) if late.sum() > 10 else float('nan')
    out['n_late'] = int(late.sum())
    if cell is None:
        return out

    cell = np.asarray(cell)
    per_mae, per_rmse, mono, jit, eol_err, rate_err = [], [], [], [], [], []
    for k in np.unique(cell):
        m = cell == k
        yy, pp, ee = y[m], p[m], e[m]
        per_mae.append(np.abs(ee).mean()); per_rmse.append(np.sqrt((ee ** 2).mean()))
        d = np.diff(pp)
        mono.append(d[d > 0].sum())                       # tổng bước tăng ngược (0 nếu hoàn toàn đơn điệu)
        jit.append(np.abs(d).mean() if len(d) else 0.0)
        # EOL: chu kỳ đầu tiên cắt ngưỡng
        a, b = _first_crossing(yy, thr), _first_crossing(pp, thr)
        if a is not None:
            eol_err.append((b if b is not None else len(pp)) - a)
        # tốc độ suy giảm trên cửa sổ 100 chu kỳ (so sánh độ dốc, không so mức)
        w = 100
        if len(yy) > w:
            ry = (yy[:-w] - yy[w:]); rp = (pp[:-w] - pp[w:])
            denom = np.clip(np.abs(ry), 1e-4, None)
            rate_err.append(float(np.abs(rp - ry).mean() / denom.mean()))

    out['MAE_cell'] = float(np.mean(per_mae))
    out['MAE_cell_std'] = float(np.std(per_mae))
    out['MAE_cell_max'] = float(np.max(per_mae))
    out['RMSE_cell'] = float(np.mean(per_rmse))
    out['MonoViol'] = float(np.mean(mono))
    out['Jitter'] = float(np.mean(jit))
    out['RateErr'] = float(np.mean(rate_err)) if rate_err else float('nan')
    if eol_err:
        ee = np.asarray(eol_err, dtype=float)
        out['EOL_MAE'] = float(np.abs(ee).mean()); out['EOL_bias'] = float(ee.mean())
    else:
        out['EOL_MAE'] = float('nan'); out['EOL_bias'] = float('nan')
    out['EOL_cov'] = float(len(eol_err) / max(1, len(np.unique(cell))))
    out['n_cells'] = int(len(np.unique(cell)))
    return out


# giữ tên cũ để phần code còn lại không phải đổi
def metrics(y, p, cell_idx=None):
    return evaluate(y, p, cell_idx)
