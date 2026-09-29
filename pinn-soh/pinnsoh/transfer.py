"""
E5 — cross-dataset adaptation with label-free physics.

Source: 70 % labelled cells of the source dataset (its usual split).
Target: cell-wise 70/15/15 split of the target dataset (same seed);
        k labelled target cells taken from the target TRAIN split (k = 0 or 3),
        the remaining target train cells are UNLABELLED (features only),
        evaluation on the target TEST split (never seen, never used for selection).
Model selection: none on target labels — fixed number of steps, final model is evaluated
(strict; identical for every method).

Methods
  mlp        : data loss on source + k target cells
  pinn_semi  : + physics losses on ALL source-train and target-train cells (unlabelled included)
"""
from __future__ import annotations
import json, os, time
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn.functional as Fnn

from .data import Normalizer, load_dataset, split_cells, choose_labelled, make_pointset
from .models import SOHModel
from .train import Config, Trainer, metrics, set_seed


@dataclass
class TransferConfig(Config):
    target: str = 'MIT'
    k_target: int = 0          # labelled target cells
    steps: int = 2500


class TransferTrainer(Trainer):
    def prepare(self):
        cfg = self.cfg
        src = load_dataset(cfg.root, cfg.dataset); tgt = load_dataset(cfg.root, cfg.target)
        sby = {c.cid: c for c in src}; tby = {c.cid: c for c in tgt}
        ssplit = split_cells(src, cfg.seed); tsplit = split_cells(tgt, cfg.seed)
        s_lab, _ = choose_labelled(ssplit['train'], 0.7, len(src), cfg.seed, sby)
        rng = np.random.RandomState(cfg.seed + 7)
        t_train = list(tsplit['train']); rng.shuffle(t_train)
        t_lab, t_unl = t_train[:cfg.k_target], t_train[cfg.k_target:]
        semi = cfg.model == 'pinn_semi'
        S_lab = [sby[i] for i in s_lab]; T_lab = [tby[i] for i in t_lab]; T_unl = [tby[i] for i in t_unl]
        S_all = [sby[i] for i in ssplit['train']]
        fit = (S_all + T_lab + T_unl) if semi else (S_lab + T_lab)
        self.norm = Normalizer(cfg.norm).fit(fit)
        self.S_lab = make_pointset(S_lab, self.norm, cfg.cyc)
        self.T_lab = make_pointset(T_lab, self.norm, cfg.cyc) if T_lab else None
        self.S_unl = make_pointset(S_all + T_lab + T_unl, self.norm, cfg.cyc) if semi else None   # physics pool
        self.val = make_pointset([sby[i] for i in ssplit['val']], self.norm, cfg.cyc)              # source val (logging only)
        self.test = make_pointset([tby[i] for i in tsplit['test']], self.norm, cfg.cyc)            # TARGET test
        self.split = dict(source_labelled=s_lab, target_labelled=t_lab, target_unlabelled=t_unl,
                          target_test=tsplit['test'])
        self.n_cells = dict(source_labelled=len(S_lab), target_labelled=len(T_lab), target_unlabelled=len(T_unl),
                            target_test=len(tsplit['test']))
        self.transfer = {}
        return self

    def fit(self, verbose=False):
        cfg = self.cfg
        dyn = 'greybox' if cfg.model == 'pinn_semi' else 'none'
        model = SOHModel(dyn=dyn, hidden=cfg.hidden, dropout=cfg.dropout, use_cycle=cfg.use_cycle,
                         use_knee=cfg.use_knee, use_arrhenius=cfg.use_arrhenius,
                         act=cfg.act, arch=cfg.arch).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr * 0.05)
        rng = np.random.RandomState(cfg.seed)
        t0 = time.time()
        for step in range(1, cfg.steps + 1):
            model.train()
            nb = cfg.batch // 2 if self.T_lab is not None else cfg.batch
            idx = rng.randint(0, len(self.S_lab), size=nb)
            xb = [self.S_lab.X[idx]]; tb = [self.S_lab.T[idx]]; yb = [self.S_lab.Y[idx]]
            if self.T_lab is not None:                       # oversample the few labelled target cells
                j = rng.randint(0, len(self.T_lab), size=cfg.batch - nb)
                xb.append(self.T_lab.X[j]); tb.append(self.T_lab.T[j]); yb.append(self.T_lab.Y[j])
            cat = lambda l: torch.from_numpy(np.concatenate(l, 0)).to(self.dev)
            loss = Fnn.mse_loss(model.u(cat(xb), cat(tb)), cat(yb))
            if dyn == 'greybox':
                w = min(1.0, step / max(1, cfg.warmup_frac * cfg.steps))
                ph = self.physics(model, self._pairs(self.S_unl, rng, cfg.batch))
                loss = loss + w * (cfg.alpha_ode * ph['ode'] + cfg.beta_mono * ph['mono'] + cfg.gamma_range * ph['range'])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        self.model = model
        res = dict(config=asdict(cfg), n_cells=self.n_cells, steps=cfg.steps, train_time_s=time.time() - t0,
                   n_params=model.n_params(), source_val=self.evaluate(model, self.val),
                   target_test=self.evaluate(model, self.test), split=self.split)
        if dyn == 'greybox':
            res['physics_params'] = dict(lam=float(model.dyn.lam), Ea_kJ_mol=float(model.dyn.Ea) / 1e3)
        return res


def run_transfer(cfg: TransferConfig, skip_existing=True):
    os.makedirs(cfg.out_dir, exist_ok=True)
    name = f'{cfg.dataset}_to_{cfg.target}_{cfg.model}_k{cfg.k_target}_s{cfg.seed}_{cfg.norm}'
    path = os.path.join(cfg.out_dir, name + '.json')
    if skip_existing and os.path.exists(path):
        return json.load(open(path))
    set_seed(cfg.seed)
    tr = TransferTrainer(cfg).prepare()
    res = tr.fit()
    json.dump(res, open(path, 'w'))
    t = res['target_test']
    print(f'[{name}] target-test MAE {t["MAE"]:.4f} MAPE {t["MAPE"]:.1f}% (source val {res["source_val"]["MAE"]:.4f}, {res["train_time_s"]:.0f}s)', flush=True)
    return res
