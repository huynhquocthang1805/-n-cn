"""
E30: mô hình chuỗi thời gian theo chu kỳ — CNN, LSTM, Transformer, theo đề cương đồ án (so với ít nhất bốn baseline
MLP, CNN, LSTM, Transformer).

Đầu vào ở chu kỳ n là cửa sổ nhân quả gồm L chu kỳ gần nhất của chính cell (n − L + 1, …, n); mỗi chu kỳ có 16 đặc trưng
sạc đã chuẩn hoá và chỉ số chu kỳ. Khi cell chưa đủ L chu kỳ, đầu cửa sổ lặp lại chu kỳ đầu. Đầu ra là SOH ở chu kỳ n, nên
mô hình chạy trực tuyến được như các mô hình khác của đồ án. Chỉ học có nhãn (không có loss vật lý); cùng cách chia, cùng
tập cell có nhãn (hạt giống 100·r + k), cùng normaliser với MLP và GBDT; dừng sớm theo MAE trên cell validation.

    cnn          hai lớp tích chập nhân quả (32 kênh, kernel 5, dilation 1 và 2), lấy bước cuối, MLP đầu ra
    lstm         LSTM một lớp 32 nút ẩn, trạng thái ẩn ở bước cuối -> tuyến tính
    transformer  chiếu tuyến tính lên 32 chiều + mã hoá vị trí sin theo vị trí trong cửa sổ, 1 lớp encoder (4 đầu,
                 FFN 64), token cuối -> tuyến tính
Huấn luyện: AdamW (lr 1e-3, weight decay 1e-4), batch 128, cắt gradient ở chuẩn 1, tối đa 3 000 bước, đánh giá validation
mỗi 150 bước và dừng khi 6 lần liên tiếp không cải thiện. Độ dài cửa sổ L chọn trên validation ở E30dev.
"""
from __future__ import annotations
import json, math, os, time
import numpy as np

from .data import cv_split, choose_labelled, make_pointset, Normalizer, SAMBA_SPLIT
from .metrics import evaluate
from .train import write_json

MODELS = ('cnn', 'lstm', 'transformer')
HIDDEN = 32
BATCH = 128
MAX_STEPS = 3000
EVAL_EVERY = 150
PATIENCE = 6                   # số lần đánh giá liên tiếp không cải thiện thì dừng
LR = 1e-3


def _net(model: str, d_in: int, L: int):
    import torch
    from torch import nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv1d(d_in, HIDDEN, 5, dilation=1)
            self.c2 = nn.Conv1d(HIDDEN, HIDDEN, 5, dilation=2)
            self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

        def forward(self, x):                                   # x: (B, L, d)
            h = x.transpose(1, 2)
            h = torch.relu(self.c1(nn.functional.pad(h, (4, 0))))          # đệm bên trái: nhân quả
            h = torch.relu(self.c2(nn.functional.pad(h, (8, 0))))
            return self.head(h[:, :, -1])

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(d_in, HIDDEN, batch_first=True)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1])

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d_in, HIDDEN)
            pos = torch.arange(L, dtype=torch.float32)[:, None]
            div = torch.exp(torch.arange(0, HIDDEN, 2, dtype=torch.float32) * (-math.log(10000.0) / HIDDEN))
            pe = torch.zeros(L, HIDDEN)
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe[None])
            layer = nn.TransformerEncoderLayer(HIDDEN, 4, 2 * HIDDEN, dropout=0.0, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            h = self.enc(self.inp(x) + self.pe)
            return self.head(h[:, -1])

    return dict(cnn=CNN, lstm=LSTM, transformer=Transformer)[model]()


def _windows(S, L: int) -> np.ndarray:
    """Chỉ số hàng của cửa sổ nhân quả L chu kỳ cho mọi hàng của PointSet (đầu cửa sổ lặp hàng đầu của cell)."""
    i = np.arange(len(S.X))[:, None]
    idx = i - (L - 1) + np.arange(L)[None, :]
    return np.maximum(idx, S.start[:, None]).astype(np.int64)


def _feats(S) -> np.ndarray:
    return np.hstack([S.X, S.T]).astype(np.float32)


def _per_cell_mae(y, p, cell):
    return float(np.mean([np.abs(p[cell == c] - y[cell == c]).mean() for c in np.unique(cell)]))


def fit_predict(model: str, L: int, Str, Sva, Ste, seed: int):
    """Huấn luyện trên các hàng có nhãn của Str, dừng sớm theo MAE theo cell trên Sva; trả dự đoán test, validation."""
    import torch
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    F = {k: torch.from_numpy(_feats(S)) for k, S in (('tr', Str), ('va', Sva), ('te', Ste))}
    W = {k: torch.from_numpy(_windows(S, L)) for k, S in (('tr', Str), ('va', Sva), ('te', Ste))}
    ytr = torch.from_numpy(Str.Y.ravel().astype(np.float32))
    net = _net(model, F['tr'].shape[1], L)
    with torch.no_grad():                                           # bắt đầu từ SOH trung bình của tập train
        last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.bias.fill_(float(ytr.mean()))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)

    def predict(part):
        net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(W[part]), 2048):
                out.append(net(F[part][W[part][s:s + 2048]]).squeeze(-1))
        net.train()
        return torch.cat(out).numpy().astype(np.float64)

    best, bad, best_step, state = np.inf, 0, 0, None
    yva = Sva.Y.ravel()
    for step in range(1, MAX_STEPS + 1):
        b = torch.from_numpy(rng.randint(0, len(ytr), size=BATCH))
        loss = torch.mean((net(F['tr'][W['tr'][b]]).squeeze(-1) - ytr[b]) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            v = _per_cell_mae(yva, predict('va'), Sva.cell)
            if v < best - 1e-6:
                best, bad, best_step = v, 0, step
                state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    net.load_state_dict(state)
    return predict('te'), predict('va'), best, dict(window=L, hidden=HIDDEN, best_step=best_step, steps=step)


def cv_path(model, L, ds, frac, r, k, out_dir='runs/E30'):
    return f'{out_dir}/{model}-L{L}_{ds}_f{frac}_cv{r}-{k}of5.json'


def _result(cfg, n_cells, hp, v, t0, te, yte, p, S, Sva, val_ids, pva):
    return dict(config=cfg, n_cells=n_cells, hyper=hp, val_MAE=v, train_time_s=time.time() - t0,
                test=evaluate(yte, p, S.cell, cycle=S.cyc),
                test_predictions=dict(y=yte.tolist(), p=p.tolist(), cell=S.cell.tolist(), cycle=S.cyc.tolist(),
                                      cell_ids=[c.cid for c in te]),
                val_predictions=dict(y=Sva.Y.ravel().tolist(), p=pva.tolist(), cell=Sva.cell.tolist(),
                                     cycle=Sva.cyc.tolist(), cell_ids=list(val_ids)))


def run_cv(model, L, ds, frac, r, k, out_dir='runs/E30'):
    """Một lượt (mô hình, cửa sổ, bộ, mức nhãn, lần lặp r, fold k), cùng cách chia và cùng cell có nhãn với E13/E16/E20."""
    from .baselines import cells_of
    path = cv_path(model, L, ds, frac, r, k, out_dir)
    if os.path.exists(path):
        return json.load(open(path))['test']['MAE_cell']
    t0 = time.time()
    cells = cells_of(ds); by = {c.cid: c for c in cells}
    sp = cv_split(cells, k, r, 5, 0.10)
    seed = 100 * r + k
    lab, unl = choose_labelled(sp['train'], frac, len(cells), seed, by)
    norm = Normalizer('global').fit([by[i] for i in lab + unl])
    Str = make_pointset([by[i] for i in lab], norm, 'global', labelled_only=True)
    Sva = make_pointset([by[i] for i in sp['val']], norm, 'global', labelled_only=True)
    te = [by[i] for i in sp['test']]
    Ste = make_pointset(te, norm, 'global', labelled_only=True)
    p, pva, v, hp = fit_predict(model, L, Str, Sva, Ste, seed)
    cfg = dict(tag='Q', model=model, label_frac=frac, dataset=ds, repeat=r, fold=k, seed=seed, split='cv', n_folds=5,
               norm='global', window=L)
    n_cells = dict(train_labelled=len(lab), train_unlabelled=len(unl), val=len(sp['val']), test=len(te), all=len(cells))
    write_json(_result(cfg, n_cells, hp, v, t0, te, Ste.Y.ravel(), p, Ste, Sva, sp['val'], pva), path)
    return json.load(open(path))['test']['MAE_cell']


def samba_path(model, L, ds, seed, out_dir='runs/E30'):
    return f'{out_dir}/{model}-L{L}_{ds}_samba_s{seed}.json'


def run_samba(model, L, ds, seed, out_dir='runs/E30'):
    """Cách chia NASA-L của SambaMixer: 8 cell train, B0018 và B0055 validation, B0006, B0007, B0047 đánh giá."""
    from .baselines import cells_of
    path = samba_path(model, L, ds, seed, out_dir)
    if os.path.exists(path):
        return json.load(open(path))['test']['MAE_cell']
    t0 = time.time()
    cells = cells_of(ds); by = {c.cid: c for c in cells}
    sp = {k: [f'NASA/{c}' for c in v] for k, v in SAMBA_SPLIT.items()}
    norm = Normalizer('global').fit([by[i] for i in sp['train']])
    Str = make_pointset([by[i] for i in sp['train']], norm, 'global', labelled_only=True)
    Sva = make_pointset([by[i] for i in sp['val']], norm, 'global', labelled_only=True)
    te = [by[i] for i in sp['test']]
    Ste = make_pointset(te, norm, 'global', labelled_only=True)
    p, pva, v, hp = fit_predict(model, L, Str, Sva, Ste, seed)
    cfg = dict(tag='QS', model=model, dataset=ds, seed=seed, split='samba', norm='global', window=L)
    n_cells = dict(train_labelled=len(sp['train']), val=len(sp['val']), test=len(te), all=len(cells))
    write_json(_result(cfg, n_cells, hp, v, t0, te, Ste.Y.ravel(), p, Ste, Sva, sp['val'], pva), path)
    return json.load(open(path))['test']['MAE_cell']
