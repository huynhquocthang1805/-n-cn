"""
Bộ dữ liệu NASA PCoE (Li-ion Battery Aging, 34 cell 18650 NCA, danh định 2,0 Ah) -> cùng định dạng 16 đặc trưng sạc
của bốn bộ PINN4SOH, để dùng nguyên các mô hình và giao thức của đồ án.

    python -m pinnsoh.nasa NASA/raw            # đọc các file B00xx.mat, ghi PINN4SOH/data/NASA data/<nhóm>/B00xx.csv

Mỗi hàng là một chu kỳ xả có dung lượng hợp lệ, ghép với lần sạc ngay trước nó (cùng cell, sau lần xả trước):
    * nhãn: dung lượng của lần xả (Ah); SOH = dung lượng / 2,0 Ah như các công trình trên bộ này;
    * đặc trưng: 16 đặc trưng của đoạn sạc, cùng định nghĩa với PINN4SOH (suy ra từ giá trị trong các CSV của bốn bộ):
        đoạn CC: các mẫu có dòng >= 1,4 A và điện áp trong [4,0; 4,2) V -> trung bình, độ lệch chuẩn, độ nhọn (Fisher),
                 độ lệch của điện áp; điện lượng (Ah) và thời gian (s) của đoạn; độ dốc (V_cuối − V_đầu)/thời gian;
                 entropy −Σ p log p với p = V/ΣV;
        đoạn CV: các mẫu sau khi điện áp chạm 4,195 V có dòng trong [0,1; 0,5] A -> các đại lượng tương tự cho dòng điện;
      đoạn nào có ít hơn 5 mẫu, hoặc đoạn CC không bắt đầu từ dưới 4,02 V (lần sạc bắt đầu từ trạng thái đã đầy một phần),
      cho đặc trưng NaN và hàng bị bộ lọc nhân quả loại;
    * cột phụ: rest_h (giờ giữa thời điểm bắt đầu hai lần xả liên tiếp, như SambaMixer), ambient_c, t_start (giờ kể từ
      lần xả đầu tiên).
Chỉ dùng 19 cell có chế độ xả cố định như SambaMixer (CELLS). Lọc lần xả hỏng: dung lượng < 0,2 Ah hoặc thấp hơn trung
vị 7 lần xả lân cận quá 10 %; giữ cell có ít nhất 20 hàng có đủ đặc trưng.
"""
from __future__ import annotations
import datetime as _dt
import os, sys
import numpy as np
import pandas as pd

FEATURES = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy']
V_LO, V_HI, V_CV, I_CC = 4.0, 4.2, 4.195, 1.4
I_HI, I_LO = 0.5, 0.1
NOMINAL = 2.0
# 19 cell có chế độ xả cố định mà SambaMixer (Olalde-Verano và cộng sự, 2024) dùng trong các tập NASA-S/M/L; các nhóm
# 38–44 đổi dòng xả giữa các lần xả (dung lượng đo không so được với nhau), 49–52 chỉ có vài lần xả.
CELLS = (5, 6, 7, 18, 25, 26, 27, 28, 29, 31, 34, 36, 45, 46, 47, 48, 54, 55, 56)
# nhóm thí nghiệm theo README của NASA (giao thức xả, nhiệt độ môi trường) — dùng để phân tầng khi chia
GROUPS = {(5, 6, 7, 18): 'G1_24C_2A', (25, 26, 27, 28): 'G2_24C_PWM4A', (29, 30, 31, 32): 'G3_43C_4A',
          (33, 34, 36): 'G4_24C_4A2A', (38, 39, 40): 'G5_24-44C_multi', (41, 42, 43, 44): 'G6_4C_4A1A',
          (45, 46, 47, 48): 'G7_4C_1A', (49, 50, 51, 52): 'G8_4C_2A', (53, 54, 55, 56): 'G9_4C_2A'}


def group_of(num: int) -> str:
    for k, v in GROUPS.items():
        if num in k:
            return v
    raise KeyError(num)


_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')      # numpy 2 đổi tên trapz -> trapezoid


def _stats(x, t, I=None):
    from scipy.stats import kurtosis, skew
    x = np.asarray(x, float); t = np.asarray(t, float)
    dt = t[-1] - t[0]
    p = x / x.sum()
    q = _trapz(np.abs(I), t) / 3600.0 if I is not None else _trapz(np.abs(x), t) / 3600.0
    return [x.mean(), x.std(), kurtosis(x), skew(x), q, dt, (x[-1] - x[0]) / dt if dt > 0 else np.nan,
            float(-(p * np.log(p)).sum())]


def charge_features(d) -> list:
    V = np.asarray(d['Voltage_measured'], float); I = np.asarray(d['Current_measured'], float)
    t = np.asarray(d['Time'], float)
    nan = [np.nan] * 16
    if len(V) < 20:
        return nan
    cc = (I >= I_CC) & (V >= V_LO) & (V < V_HI)
    reach = np.flatnonzero(V >= V_CV)
    if cc.sum() < 5 or not len(reach):
        return nan
    first_cc = np.flatnonzero(I >= I_CC)
    if not len(first_cc) or V[first_cc[0]] > V_LO + 0.02:      # lần sạc bắt đầu khi cell đã gần đầy
        return nan
    i_cc = np.flatnonzero(cc)
    seg = slice(i_cc[0], i_cc[-1] + 1)
    fv = _stats(V[seg], t[seg], I[seg])
    after = np.arange(len(V)) >= reach[0]
    cv = after & (I <= I_HI) & (I >= I_LO)
    if cv.sum() < 5:
        return nan
    i_cv = np.flatnonzero(cv)
    seg = slice(i_cv[0], i_cv[-1] + 1)
    fi = _stats(I[seg], t[seg], I[seg])
    return fv + fi


def _time(v) -> _dt.datetime:
    v = np.asarray(v, float)
    sec = float(v[5]); s = int(sec)
    return _dt.datetime(int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]), s) + _dt.timedelta(seconds=sec - s)


def convert_battery(path: str) -> pd.DataFrame:
    import scipy.io as sio
    m = sio.loadmat(path, simplify_cells=True)
    key = [k for k in m if not k.startswith('__')][0]
    rows, last_charge = [], None
    for c in m[key]['cycle']:
        if c['type'] == 'charge':
            last_charge = c
        elif c['type'] == 'discharge':
            cap = c['data'].get('Capacity', np.nan)
            cap = float(np.atleast_1d(cap)[0]) if np.size(cap) else np.nan
            feats = charge_features(last_charge['data']) if last_charge is not None else [np.nan] * 16
            rows.append(dict(zip(FEATURES, feats), capacity=cap, ambient_c=float(c['ambient_temperature']),
                             t=_time(c['time'])))
            last_charge = None                                   # mỗi lần sạc chỉ ghép với một lần xả
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    t0 = df.t.iloc[0]
    df['t_start'] = [(x - t0).total_seconds() / 3600 for x in df.t]
    df['rest_h'] = np.concatenate([[np.nan], np.diff(df.t_start.values)])
    # lọc lần xả hỏng (README của NASA: một số lần xả cho dung lượng rất thấp): bỏ dung lượng < 0,2 Ah hoặc thấp hơn
    # trung vị của 7 lần xả lân cận quá 10 %
    q = df.capacity.values.astype(float)
    med = pd.Series(np.where(q >= 0.2, q, np.nan)).rolling(7, center=True, min_periods=3).median().values
    keep = np.isfinite(q) & (q >= 0.2) & (q >= 0.9 * med)
    return df[keep].drop(columns=['t']).reset_index(drop=True)


def main(raw_dir: str, out_root: str = 'PINN4SOH/data/NASA data', min_rows: int = 20):
    import glob
    files = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, '**', 'B00*.mat'), recursive=True)):
        files[os.path.basename(f)] = f          # bản trong gói 25–44 thay bản P1 cùng tên (đầy đủ hơn)
    rows = []
    for name, f in sorted(files.items()):
        num = int(name[1:5])
        if num not in CELLS:
            continue
        df = convert_battery(f)
        n_ok = int(df[FEATURES].notna().all(axis=1).sum()) if len(df) else 0
        rows.append(dict(cell=name[:-4], group=group_of(num), n_discharge=len(df), n_feat=n_ok,
                         ambient=df.ambient_c.median() if len(df) else np.nan,
                         soh0=df.capacity.iloc[:3].mean() / NOMINAL if len(df) else np.nan,
                         soh1=df.capacity.iloc[-3:].mean() / NOMINAL if len(df) else np.nan))
        if n_ok < min_rows:
            continue
        d = os.path.join(out_root, group_of(num)); os.makedirs(d, exist_ok=True)
        df.to_csv(os.path.join(d, name[:-4] + '.csv'), index=False)
    summ = pd.DataFrame(rows)
    print(summ.round(3).to_string(index=False))
    print(f'giữ {int((summ.n_feat >= min_rows).sum())}/{len(summ)} cell')
    return summ


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'NASA/raw')
