"""
E27: so với PINN4SOH (Wang và cộng sự, Nature Communications 2024) trên đúng cách chia của bài.

Cách chia lấy từ nhật ký huấn luyện mà nhóm tác giả công bố kèm mã nguồn (results/Ours/*.zip, dòng "data path:"),
lưu ở pinn4soh_splits.json với tên cell theo dạng của đồ án. Khoá '<bộ>|<chế độ>|<nhóm>':
    XJTU|full|i-i   mỗi batch một mô hình; cell test là cell số 4, 8 (thêm 14 ở batch 3C) của batch
    TJU|full|i-i    mỗi batch một mô hình; cell test chọn theo thứ tự tệp như trong mã của bài
    MIT|full|all    một mô hình; cell test có số hiệu chia hết cho 5
    HUST|full|all   một mô hình; 20 cell test cố định
    XJTU|nK|i-i, HUST|nK|all   thí nghiệm ít mẫu của bài: chỉ K = 1..4 cell train đầu tiên có nhãn
Số liệu của PINN4SOH (bảng 2 và thí nghiệm ít mẫu) lấy từ các tệp kết quả công bố cùng mã nguồn (PAPER_MAE).

Khác bài ở tiền xử lý, như mọi thí nghiệm của đồ án: lọc ngoại lai nhân quả chỉ trên đặc trưng, chuẩn hoá z theo
đặc trưng của các cell train, chỉ số chu kỳ chia 1000. PINN4SOH lọc 3σ trên cả vòng đời và trên cả cột dung lượng,
chuẩn hoá min–max theo từng cell, nên con số của bài có lợi thế (Mục kiểm toán rò rỉ). Validation như bài: 20 % hàng
ngẫu nhiên của các cell có nhãn. Ở chế độ ít mẫu, đặc trưng sạc của các cell train còn lại trong cùng nhóm (không
dùng nhãn của chúng) đi vào các hàm mất mát vật lý — đúng ý học bán giám sát; cell test không được dùng dưới dạng nào.
Đánh giá: MAE theo cell trên cell test (bài tính MAE từng cell rồi lấy trung bình, cùng định nghĩa).
"""
from __future__ import annotations
import dataclasses, json, os, time
from typing import Dict, List
import numpy as np

SPLITS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pinn4soh_splits.json')
VAL_ROWS = 0.2
# MAE (SOH, trung bình qua 10 lần chạy, rồi qua các nhóm) của PINN4SOH, MLP và CNN theo các tệp kết quả công bố
PAPER_MAE = {
    ('XJTU', 'full'): dict(PINN4SOH=0.00785, MLP=0.02102, CNN=0.02119),
    ('TJU', 'full'): dict(PINN4SOH=0.00979, MLP=0.01361, CNN=0.01266),
    ('MIT', 'full'): dict(PINN4SOH=0.00618, MLP=0.00745, CNN=0.00615),
    ('HUST', 'full'): dict(PINN4SOH=0.00757, MLP=0.00777, CNN=0.00721),
    ('XJTU', 'n1'): dict(PINN4SOH=0.03154, MLP=0.04914, CNN=0.13116),
    ('XJTU', 'n2'): dict(PINN4SOH=0.01006, MLP=0.02488, CNN=0.06280),
    ('XJTU', 'n3'): dict(PINN4SOH=0.00798, MLP=0.02359, CNN=0.04996),
    ('XJTU', 'n4'): dict(PINN4SOH=0.00762, MLP=0.02260, CNN=0.03862),
    ('HUST', 'n1'): dict(PINN4SOH=0.03942, MLP=0.05037, CNN=0.11278),
    ('HUST', 'n2'): dict(PINN4SOH=0.01694, MLP=0.03600, CNN=0.06504),
    ('HUST', 'n3'): dict(PINN4SOH=0.01459, MLP=0.02299, CNN=0.04371),
    ('HUST', 'n4'): dict(PINN4SOH=0.01362, MLP=0.02404, CNN=0.03674),
}


def keys() -> List[str]:
    return list(json.load(open(SPLITS_FILE)))


def paper_split(key: str) -> Dict[str, List[str]]:
    """{'labelled', 'unlabelled', 'test'}: ở chế độ ít mẫu, cell không nhãn là phần còn lại của tập train đầy đủ."""
    S = json.load(open(SPLITS_FILE))
    ds, mode, grp = key.split('|')
    lab, test = list(S[key]['train']), list(S[key]['test'])
    full = S[f'{ds}|full|{grp}']['train']
    unl = [c for c in full if c not in set(lab)] if mode != 'full' else []
    assert not (set(lab) | set(unl)) & set(test), key
    return dict(labelled=lab, unlabelled=unl, test=test)


def row_split(cells, seed: int, frac: float = VAL_ROWS):
    """Chia các hàng có nhãn của các cell có nhãn thành train / validation (ngẫu nhiên, như train_test_split của bài).
    Trả hai danh sách Cell cùng đặc trưng; nhãn của hàng không thuộc phần đó là NaN (make_pointset bỏ qua)."""
    rng = np.random.RandomState(10_000 + seed)
    idx = [(i, j) for i, c in enumerate(cells) for j in np.flatnonzero(np.isfinite(c.soh))]
    pick = rng.permutation(len(idx))[:int(round(frac * len(idx)))]
    val = np.zeros(len(idx), bool); val[pick] = True
    tr_m = [np.zeros(len(c.soh), bool) for c in cells]; va_m = [np.zeros(len(c.soh), bool) for c in cells]
    for (i, j), v in zip(idx, val):
        (va_m if v else tr_m)[i][j] = True
    mask = lambda c, m: dataclasses.replace(c, soh=np.where(m, c.soh, np.nan).astype(np.float32))
    return [mask(c, m) for c, m in zip(cells, tr_m)], [mask(c, m) for c, m in zip(cells, va_m)]


# tiền xử lý: (bộ lọc, chuẩn hoá đặc trưng, chuẩn hoá chỉ số chu kỳ)
PREPS = {'strict': ('causal', 'global', 'global'), 'paper': ('paper3s', 'per_cell_minmax', 'per_cell_minmax')}


def gbdt_path(key, seed, out_dir='runs/E27', prep='strict'):
    return f'{out_dir}/gbdt-{prep}_{key.replace("|", "_")}_s{seed}.json'


def run_gbdt(key: str, seed: int, out_dir='runs/E27', prep: str = 'strict'):
    from .baselines import fit_predict
    from .data import Normalizer, make_pointset, load_dataset
    from .metrics import evaluate
    from .train import write_json
    path = gbdt_path(key, seed, out_dir, prep)
    if os.path.exists(path):
        return json.load(open(path))['test']['MAE_cell']
    t0 = time.time()
    ds = key.split('|')[0]
    clean, nmode, cmode = PREPS[prep]
    by = {c.cid: c for c in load_dataset('PINN4SOH/data', ds, clean=clean)}
    sp = paper_split(key)
    lab = [by[i] for i in sp['labelled']]
    norm = Normalizer(nmode).fit(lab + [by[i] for i in sp['unlabelled']])
    tr, va = row_split(lab, seed)
    xy = lambda cs: (lambda S: (np.hstack([S.X, S.T]), S.Y.ravel(), S))(make_pointset(cs, norm, cmode, labelled_only=True))
    Xtr, ytr, _ = xy(tr); Xva, yva, _ = xy(va)
    te = [by[i] for i in sp['test']]
    Xte, yte, S = xy(te)
    p, v, hp = fit_predict('gbdt', Xtr, ytr, Xva, yva, Xte, seed)
    p = np.asarray(p, dtype=np.float64)
    res = dict(config=dict(model='gbdt', key=key, seed=seed, prep=prep), hyper=hp, val_MAE=v, train_time_s=time.time() - t0,
               test=evaluate(yte, p, S.cell, cycle=S.cyc),
               test_predictions=dict(y=yte.tolist(), p=p.tolist(), cell=S.cell.tolist(), cycle=S.cyc.tolist(),
                                     cell_ids=[c.cid for c in te]))
    write_json(res, path)
    return res['test']['MAE_cell']
