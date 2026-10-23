"""
E28: ngân sách đo dung lượng — đo thưa trên MỌI cell train thay vì đo đầy đủ trên một phần cell.

Cách đánh giá chính của đồ án đếm nhãn theo số CELL: 30 % cell có nhãn nghĩa là mỗi chu kỳ của 30 % số cell đều được đo
dung lượng. Trong thực tế chi phí nằm ở số LẦN ĐO (mỗi lần là một lần xả đầy ở điều kiện chuẩn). E28 đặt cùng câu hỏi
theo đơn vị đó: mọi cell train được đo dung lượng ở chu kỳ đầu, sau mỗi q·L chu kỳ (L = trung vị số chu kỳ của các cell
train; lần đo rơi vào hàng đầu tiên có chu kỳ >= mốc) và ở chu kỳ cuối của thử nghiệm; các chu kỳ còn lại chỉ có đặc
trưng sạc.

    meas    mục tiêu dữ liệu chỉ ở các hàng được đo; PINN-semi áp các hàm mất mát vật lý lên MỌI hàng của mọi cell,
            nên ràng buộc đơn điệu và động học lan thông tin của lần đo sang các chu kỳ giữa hai lần đo
    interp  lan nhãn trước khi học: SOH ở các chu kỳ giữa hai lần đo lấy theo nội suy tuyến tính, sau lần đo cuối
            ngoại suy theo độ dốc của hai lần đo cuối nếu độ dốc âm (SOH không tăng), rồi học như nhãn đầy đủ

Cell validation và cell test giữ nguyên như E13/E20 (validation có nhãn đầy đủ, test không dùng nhãn), cùng fold, cùng
hạt giống, nên so ghép cặp được với các mức nhãn theo cell của E20.
"""
from __future__ import annotations
import dataclasses, json, os, time
from typing import List
import numpy as np

QS = (0.5, 0.2, 0.1, 0.05)


def measured_rows(cyc: np.ndarray, step: int, final: bool = True) -> np.ndarray:
    """Chỉ số các hàng được đo: hàng đầu tiên có chu kỳ >= m·step, m = 0, 1, 2, …, và (final=True) hàng cuối cùng —
    cell train được lão hoá tới khi kết thúc thử nghiệm nên lần đo kiểm tra cuối luôn có. Không có lần đo cuối, cell
    ngắn hơn một khoảng đo (ví dụ nhóm 1C của TJU, 23–35 chu kỳ) chỉ có đúng một lần đo ở chu kỳ đầu."""
    cyc = np.asarray(cyc)
    if len(cyc) == 0:
        return np.zeros(0, dtype=np.int64)
    marks = np.arange(0, int(cyc.max()) + 1, max(1, int(step)))
    idx = np.searchsorted(cyc, marks, side='left')
    if final:
        idx = np.append(idx, len(cyc) - 1)
    idx = np.unique(idx)
    return idx[idx < len(cyc)]


def interp_labels(cyc: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    xc, yc = np.asarray(cyc, float)[idx], np.asarray(y, float)[idx]
    out = np.interp(np.asarray(cyc, float), xc, yc)
    if len(xc) >= 2:
        slope = min(0.0, (yc[-1] - yc[-2]) / max(1.0, xc[-1] - xc[-2]))
        after = cyc > xc[-1]
        out[after] = yc[-1] + slope * (np.asarray(cyc, float)[after] - xc[-1])
    return out


def sparse_cells(cells, step: int, interp: bool = False):
    """Bản sao các cell chỉ còn nhãn ở hàng được đo (interp=False) hoặc nhãn nội suy ở mọi hàng (interp=True)."""
    out = []
    for c in cells:
        ok = np.flatnonzero(np.isfinite(c.soh))
        y = np.full(len(c.soh), np.nan, dtype=np.float64)
        if len(ok):
            sub = measured_rows(c.cycle[ok], step)
            idx = ok[sub]
            if interp and len(idx):
                y[ok] = interp_labels(c.cycle[ok], c.soh[ok], sub)
            else:
                y[idx] = c.soh[idx]
        out.append(dataclasses.replace(c, soh=y.astype(np.float32)))
    return out


def life_steps(cells, q: float) -> int:
    return max(1, int(round(q * float(np.median([len(c.soh) for c in cells])))))


def gbdt_path(ds, q, mode, r, k, out_dir='runs/E28'):
    return f'{out_dir}/gbdt-{mode}_{ds}_q{q}_cv{r}-{k}of5.json'


def run_gbdt(ds: str, q: float, mode: str, r: int, k: int, out_dir='runs/E28'):
    """GBDT (cấu hình E16) học trên các hàng được đo ('meas') hoặc trên nhãn nội suy ('interp') của mọi cell train."""
    from .baselines import cells_of, fit_predict, xy
    from .data import cv_split, choose_labelled, Normalizer
    from .metrics import evaluate
    from .train import write_json
    path = gbdt_path(ds, q, mode, r, k, out_dir)
    if os.path.exists(path):
        return json.load(open(path))['test']['MAE_cell']
    t0 = time.time()
    cells = cells_of(ds); by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, 0.10)
    seed = 100 * r + k
    lab, unl = choose_labelled(sp['train'], 0.7, len(cells), seed, by)
    tr = [by[i] for i in lab + unl]
    norm = Normalizer('global').fit(tr)
    step = life_steps(tr, q)
    Xtr, ytr, Str = xy(sparse_cells(tr, step, mode == 'interp'), norm)
    Xva, yva, Sv = xy([by[i] for i in sp['val']], norm)
    te = [by[i] for i in sp['test']]
    Xte, yte, S = xy(te, norm)
    p, v, hp, pva = fit_predict('gbdt', Xtr, ytr, Xva, yva, Xte, seed, return_val=True)
    p = np.asarray(p, dtype=np.float64)
    n_meas = sum(len(measured_rows(c.cycle[np.isfinite(c.soh)], step)) for c in tr)
    res = dict(config=dict(model='gbdt', mode=mode, q=q, dataset=ds, repeat=r, fold=k, seed=seed, step=step),
               n_meas=int(n_meas), n_rows=int(sum(np.isfinite(c.soh).sum() for c in tr)), n_train_cells=len(tr),
               hyper=hp, val_MAE=v, train_time_s=time.time() - t0, test=evaluate(yte, p, S.cell, cycle=S.cyc),
               test_predictions=dict(y=yte.tolist(), p=p.tolist(), cell=S.cell.tolist(), cycle=S.cyc.tolist(),
                                     cell_ids=[c.cid for c in te]),
               val_predictions=dict(y=yva.tolist(), p=np.asarray(pva, dtype=np.float64).tolist(), cell=Sv.cell.tolist(),
                                    cycle=Sv.cyc.tolist(), cell_ids=list(sp['val'])))
    write_json(res, path)
    return res['test']['MAE_cell']


def smooth_cells(cells, w: int):
    """Khử nhiễu nhãn dọc quỹ đạo: trung vị trượt có tâm rồi trung bình trượt có tâm, cùng cửa sổ w chu kỳ, trên các hàng
    có nhãn của từng cell. Chỉ dùng cho cell TRAIN (học ngoại tuyến trên toàn bộ quỹ đạo đã đo của chúng); nhãn validation
    và test không bao giờ bị làm trơn."""
    import pandas as pd
    out = []
    for c in cells:
        ok = np.flatnonzero(np.isfinite(c.soh))
        y = np.full(len(c.soh), np.nan, dtype=np.float64)
        if len(ok):
            s = pd.Series(c.soh[ok].astype(np.float64))
            y[ok] = s.rolling(w, center=True, min_periods=1).median().rolling(w, center=True, min_periods=1).mean().values
        out.append(dataclasses.replace(c, soh=y.astype(np.float32)))
    return out
