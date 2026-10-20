"""
Hậu xử lý dự đoán theo quỹ đạo của từng cell (E20) — chỉ dùng dự đoán của chính cell ở các chu kỳ đã qua,
không dùng nhãn, không dùng chu kỳ tương lai, nên áp được khi chạy trực tuyến trong BMS.

    causal_trend(p, t, w)   hồi quy tuyến tính trên w điểm gần nhất (kể cả điểm hiện tại), độ dốc bị chặn
                            trên bởi 0 (SOH không tăng), lấy giá trị đường thẳng tại điểm hiện tại.
                            Khi độ dốc bình phương tối thiểu dương, nghiệm có ràng buộc là độ dốc 0, tức
                            trung bình cửa sổ.
    causal_iso(p)           giá trị cuối của hồi quy đơn điệu giảm (PAVA) trên tiền tố — để so sánh;
                            luôn <= p hiện tại nên lệch xuống khi dự đoán có nhiễu.
    by_cell(fn, p, cell, cyc, ...)  áp một bộ lọc cho từng cell theo thứ tự chu kỳ, trả mảng cùng thứ tự gốc.
"""
from __future__ import annotations
import numpy as np


def causal_trend(p: np.ndarray, t: np.ndarray, w: int) -> np.ndarray:
    """Dạng vector hoá bằng tổng tích luỹ (O(n)); cùng kết quả với vòng lặp tới sai số làm tròn (< 1e-9 SOH)."""
    p = np.asarray(p, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p.copy()
    t = t - t[0]                                    # giảm mất chính xác khi trừ hai tổng lớn
    i = np.arange(n); a = np.maximum(0, i - w + 1); cnt = (i - a + 1).astype(np.float64)
    seg = lambda x: (lambda S: S[i + 1] - S[a])(np.concatenate([[0.0], np.cumsum(x)]))
    st, sy, stt, sty = seg(t), seg(p), seg(t * t), seg(t * p)
    tm, ym = st / cnt, sy / cnt
    sxx = stt - cnt * tm * tm
    sxy = sty - cnt * tm * ym
    ok = (sxx > 1e-12) & (cnt >= 3)
    slope = np.where(ok, sxy / np.where(ok, sxx, 1.0), 0.0)
    return ym + np.minimum(slope, 0.0) * (t - tm)


def _causal_trend_loop(p, t, w):
    """Bản vòng lặp, giữ để kiểm tra bản vector hoá."""
    p = np.asarray(p, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(p)
    for n in range(len(p)):
        a = max(0, n - w + 1)
        pp, tt = p[a:n + 1], t[a:n + 1]
        if len(pp) < 3:
            out[n] = pp.mean()
            continue
        tm, pm = tt.mean(), pp.mean()
        dt = tt - tm
        sxx = float(dt @ dt)
        slope = float(dt @ (pp - pm)) / sxx if sxx > 0 else 0.0
        out[n] = pm + min(slope, 0.0) * (t[n] - tm)
    return out


def causal_iso(p: np.ndarray, t=None, w=None) -> np.ndarray:
    """Giá trị tại điểm cuối của hồi quy đơn điệu giảm trên p[0..n], với mọi n (PAVA dạng ngăn xếp)."""
    out = np.empty(len(p))
    means, wts = [], []
    for n, v in enumerate(np.asarray(p, dtype=np.float64)):
        means.append(v); wts.append(1.0)
        while len(means) > 1 and means[-1] > means[-2]:
            m2, w2 = means.pop(), wts.pop()
            m1, w1 = means.pop(), wts.pop()
            means.append((m1 * w1 + m2 * w2) / (w1 + w2)); wts.append(w1 + w2)
        out[n] = means[-1]
    return out


FILTERS = dict(trend=causal_trend, iso=causal_iso)


def by_cell(fn, p, cell, cyc, **kw) -> np.ndarray:
    p, cell, cyc = np.asarray(p, dtype=np.float64), np.asarray(cell), np.asarray(cyc, dtype=np.float64)
    out = np.empty_like(p)
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        idx = idx[np.argsort(cyc[idx], kind='stable')]
        out[idx] = fn(p[idx], cyc[idx], **kw)
    return out
