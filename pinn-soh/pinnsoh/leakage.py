"""
E15 — kiểm toán rò rỉ thông tin: cùng mô hình (MLP, 70 % nhãn), chỉ đổi cách chia / chuẩn hoá / lọc.
Kết quả: runs/E15/<thiết_lập>_<bộ>_cv<r>-<k>.json. Chạy qua `python run.py E15`.

Thiết lập (so với giao thức chuẩn = nhánh F của E13, cùng lần lặp và fold):
    row        chia theo HÀNG: mỗi chu kỳ được gán ngẫu nhiên vào train / val / test, nên các chu kỳ kề
               nhau của cùng một cell nằm ở cả train lẫn test
    minmax     chia theo cell, nhưng đặc trưng và chỉ số chu kỳ được min–max theo TOÀN BỘ vòng đời của
               từng cell (cách làm của PINN4SOH) — chỉ số chu kỳ sau chuẩn hoá chính là "% tuổi thọ đã qua"
    life3s     chia theo cell, nhưng lọc ngoại lai bằng mean/std của cả vòng đời (dùng chu kỳ tương lai)
    normtest   chia theo cell, nhưng normaliser fit trên cả đặc trưng của cell validation và test
"""
from __future__ import annotations
import dataclasses, json, os, time
import numpy as np

SETTINGS = ['row', 'minmax', 'life3s', 'normtest']
OUT = 'runs/E15'
NET = dict(hidden=(128, 128, 64))


def _make_trainer(setting, cfg):
    from .train import Trainer, _load
    from .data import Normalizer, make_pointset, cv_split, choose_labelled

    class RowSplit(Trainer):
        def prepare(self):
            c = self.cfg
            cells = _load(c, c.dataset)
            rng = np.random.RandomState(50_000 + 100 * c.repeat + c.fold)
            tr, va, te = [], [], []
            for cell in cells:
                n = len(cell.soh)
                fold = rng.randint(0, c.n_folds, size=n)
                is_te = fold == c.fold
                is_va = (~is_te) & (rng.rand(n) < 0.125)          # 10 % tổng số hàng
                for lst, m in [(tr, ~is_te & ~is_va), (va, is_va), (te, is_te)]:
                    if m.sum() >= 2:
                        lst.append(dataclasses.replace(cell, X=cell.X[m], cycle=cell.cycle[m], soh=cell.soh[m]))
            self.norm = Normalizer(c.norm).fit(tr)
            self.S_lab = make_pointset(tr, self.norm, c.cyc, labelled_only=True)
            self.S_unl = None
            self.val = make_pointset(va, self.norm, c.cyc, labelled_only=True)
            self.test = make_pointset(te, self.norm, c.cyc, labelled_only=True)
            self.test_ids = [x.cid for x in te]
            self.split = dict(mode='row')
            self.n_cells = dict(train_labelled=len(tr), train_unlabelled=0, val=len(va), test=len(te), all=len(cells))
            self.transfer = {}
            return self

    class NormTest(Trainer):
        def prepare(self):
            super().prepare()
            c = self.cfg
            cells = _load(c, c.dataset); by = {x.cid: x for x in cells}
            sp = cv_split(cells, c.fold, c.repeat, c.n_folds, c.val_frac_cv)
            lab, unl = choose_labelled(sp['train'], c.label_frac, len(cells), c.seed, by)
            allc = [by[i] for i in lab + unl + sp['val'] + sp['test']]
            self.norm = Normalizer(c.norm).fit(allc)                     # RÒ RỈ: có cả val + test
            lo = True
            self.S_lab = make_pointset([by[i] for i in lab], self.norm, c.cyc, labelled_only=lo)
            self.val = make_pointset([by[i] for i in sp['val']], self.norm, c.cyc, labelled_only=lo)
            te = [by[i] for i in sp['test']]
            self.test = make_pointset(te, self.norm, c.cyc, labelled_only=lo)
            self.test_ids = [x.cid for x in te]
            return self

    return {'row': RowSplit, 'normtest': NormTest}.get(setting, Trainer)(cfg)


def leak_path(setting, ds, r, k, out_dir=OUT):
    return f'{out_dir}/{setting}_{ds}_cv{r}-{k}.json'


def run_one(setting, ds, r, k, out_dir=OUT):
    from .train import Config, write_json
    path = leak_path(setting, ds, r, k, out_dir)
    if os.path.exists(path):
        return json.load(open(path))['test']['MAE_cell']
    kw = dict(dataset=ds, label_frac=0.7, model='mlp', seed=100 * r + k, split='cv', fold=k, repeat=r, n_folds=5,
              tag=setting, out_dir=out_dir, **NET)
    cfg = Config(**kw).v2()
    if setting == 'minmax':
        cfg.norm, cfg.cyc = 'per_cell_minmax', 'per_cell_minmax'
    if setting == 'life3s':
        cfg.clean = 'life3sigma'
    t0 = time.time()
    tr = _make_trainer(setting, cfg).prepare()
    res = tr.fit(verbose=False)
    res['train_time_s'] = time.time() - t0
    write_json(res, path)
    return res['test']['MAE_cell']
