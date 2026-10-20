"""
Nhớ kết quả đọc file kết quả (runs/**/*.json) giữa các lần phân tích.

Mỗi file JSON có kèm dự đoán test (tới hàng trăm KB), nên đọc lại cả ~3 000 file mỗi lần phân tích mất vài
phút. cached_map(files, fn, name) chỉ gọi fn cho file mới hoặc đã đổi (theo kích thước + thời điểm sửa) và
lấy phần còn lại từ .cache/<name>.pkl. Đổi cách đọc thì đổi `name` (ví dụ thêm _v2) để bộ nhớ cũ bị bỏ.
"""
from __future__ import annotations
import os, pickle
from typing import Callable, List

from .data import CACHE_DIR


def cached_map(files: List[str], fn: Callable, name: str) -> list:
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    try:
        with open(path, 'rb') as fh:
            memo = pickle.load(fh)
    except Exception:
        memo = {}
    out, fresh, n_new = [], {}, 0
    for f in files:
        st = os.stat(f)
        key = (st.st_size, st.st_mtime_ns)
        hit = memo.get(f)
        if hit is None or hit[0] != key:
            hit = (key, fn(f)); n_new += 1
        fresh[f] = hit
        out.append(hit[1])
    if n_new or len(fresh) != len(memo):
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(fresh, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    print(f'{name}: {len(files)} file, đọc mới {n_new}, lấy từ bộ nhớ đệm {len(files) - n_new}')
    return out
