"""
Định nghĩa các thí nghiệm của báo cáo đồ án (pipeline v2) dưới dạng danh sách lượt chạy (Job).

    E13  kiểm định chéo 5 fold × 5 lần lặp theo cell: MLP, PINN-semi ở 4 mức nhãn; PINN-sup, 2 biến thể
         cắt bỏ ở 30 % nhãn (đề cương ấn định trước: tai-lieu/de-cuong-E13.md)          1 100 lượt
    E14  chuyển miền, 4 cặp cùng hoá học, MLP và PINN-semi cùng normaliser (tập con của E18)  80
    E15  kiểm toán rò rỉ: 4 cách chia/chuẩn hoá sai, MLP 70 % nhãn, 2 lần lặp                 160
    E16  Ridge, GPR, GBDT, cùng cách chia và cùng tập nhãn với E13                           1 200
    E17  PINN-sup ở 10 và 50 %, PINN hộp đen ở 4 mức nhãn, cùng giao thức E13                600
    E18  chuyển miền đủ 12 cặp nguồn–đích, MLP và PINN-semi, k ∈ {0, 3}, 5 hạt giống          240
    E19  GBDT trên đúng 12 cặp của E18                                                       120
    E20  PINN-lai: 3 PINN-semi + GBDT + bộ lọc quỹ đạo, 3 lần lặp (tai-lieu/de-cuong-E20.md)  1 440
    E21  huấn luyện một bộ, kiểm tra bộ khác (2 giao thức) + thích nghi, tự huấn luyện       168
    E22  dự báo SOH tương lai, dung lượng còn lại, RUL; cùng fold E20, dùng ước lượng E20      60

Mỗi Job biết file kết quả của nó (`out`), nên chạy lại sẽ bỏ qua lượt đã xong; tên file giữ nguyên như các
lần chạy trước để kết quả cũ dùng lại được. Cấu hình của E13 đã ấn định trước — KHÔNG sửa.
"""
from __future__ import annotations
import itertools, os
from dataclasses import dataclass, field
from typing import Callable, Dict, List

DATASETS = ['XJTU', 'TJU', 'MIT', 'HUST']
FRACS = [0.1, 0.3, 0.5, 0.7]
SAME_CHEM = {'HUST': 'MIT', 'MIT': 'HUST', 'XJTU': 'TJU', 'TJU': 'XJTU'}

E13_NET = dict(hidden=(128, 128, 64))
E13_PHYS = dict(beta_mono=0.5, residual='autograd', alpha_ode=0.02)
E13_FOLDS = 5
E13_ARMS = [   # (nhánh, mô hình, mức nhãn, cấu hình riêng)
    ('A', 'mlp', 0.3, {}),
    ('C', 'pinn_semi', 0.3, E13_PHYS),
    ('F', 'mlp', 0.7, {}),
    ('B', 'pinn_sup', 0.3, E13_PHYS),
    ('D', 'pinn_semi', 0.3, dict(E13_PHYS, alpha_ode=0.0, gamma_range=0.0)),       # chỉ đơn điệu
    ('E', 'pinn_semi', 0.3, dict(beta_mono=0.5, residual='euler', alpha_ode=2.0)),  # phần dư Euler
    ('G', 'pinn_semi', 0.7, E13_PHYS),
    ('L', 'mlp', 0.1, {}), ('L', 'pinn_semi', 0.1, E13_PHYS),
    ('L', 'mlp', 0.5, {}), ('L', 'pinn_semi', 0.5, E13_PHYS),
]
# ở 70 % nhãn mọi cell train đều có nhãn nên PINN-sup trùng từng bit với PINN-semi (nhánh G) — không chạy lại
E17_ARMS = [('B', 'pinn_sup', 0.1, E13_PHYS), ('B', 'pinn_sup', 0.5, E13_PHYS),
            ('H', 'pinn_bb', 0.1, E13_PHYS), ('H', 'pinn_bb', 0.3, E13_PHYS),
            ('H', 'pinn_bb', 0.5, E13_PHYS), ('H', 'pinn_bb', 0.7, E13_PHYS)]

# số bước trung bình tới lúc dừng sớm (đo trên E13/E17) × mili giây mỗi bước trên CPU 1 luồng của máy
# chạy thử -> chi phí tương đối để xếp lịch (lượt dài chạy trước) và ước lượng thời gian
MEAN_STEPS = {'mlp': 3600, 'pinn_sup': 3300, 'pinn_semi': 3330, 'pinn_bb': 3380}
MS_PER_STEP = {'mlp': 2.6, 'pinn_sup': 8.2, 'pinn_semi': 8.2, 'pinn_bb': 12.3}
BASELINE_S = {('ridge', 'XJTU'): 0.2, ('ridge', 'TJU'): 0.3, ('ridge', 'MIT'): 0.4, ('ridge', 'HUST'): 0.8,
              ('gbdt', 'XJTU'): 6, ('gbdt', 'TJU'): 11, ('gbdt', 'MIT'): 18, ('gbdt', 'HUST'): 20,
              ('gpr', 'XJTU'): 12, ('gpr', 'TJU'): 8, ('gpr', 'MIT'): 8, ('gpr', 'HUST'): 12}
TRANSFER_STEPS = 2500


@dataclass
class Job:
    kind: str                     # train | transfer | leak | baseline | baseline_transfer
    args: Dict
    out: str                      # file kết quả; có rồi thì bỏ qua
    cost: float                   # giây ước lượng trên CPU 1 luồng của máy chạy thử
    label: str = ''
    datasets: tuple = field(default_factory=tuple)

    def done(self) -> bool:
        return os.path.exists(self.out)


def _train_cost(model):
    return MEAN_STEPS[model] * MS_PER_STEP[model] / 1000 + 2.0


def _train_job(tag, model, frac, extra, r, ds, k, out):
    from .train import Config, run_name
    kw = dict(dataset=ds, label_frac=frac, model=model, seed=100 * r + k, split='cv', fold=k, repeat=r,
              n_folds=E13_FOLDS, tag=tag, out_dir=out, **E13_NET, **extra)
    return Job('train', kw, os.path.join(out, run_name(Config(**kw)) + '.json'), _train_cost(model),
               f'{tag} {ds:<4s} {model:<9s} f{frac} cv{r}-{k}', (ds,))


def E13(repeats=5, datasets=DATASETS, seeds=None, out='runs/E13') -> List[Job]:
    """seed = 100·lần_lặp + fold, giống nhau cho mọi nhánh -> so sánh ghép cặp theo cell."""
    return [_train_job(tag, m, f, extra, r, ds, k, out)
            for (tag, m, f, extra), r, ds, k in itertools.product(E13_ARMS, range(repeats), datasets, range(E13_FOLDS))]


def E17(repeats=5, datasets=DATASETS, seeds=None, out='runs/E17') -> List[Job]:
    return [_train_job(tag, m, f, extra, r, ds, k, out)
            for (tag, m, f, extra), r, ds, k in itertools.product(E17_ARMS, range(repeats), datasets, range(E13_FOLDS))]


def _transfer_job(src, tgt, model, k, s, out):
    kw = dict(dataset=src, target=tgt, model=model, k_target=k, seed=s, norm='global', norm_pool='train',
              tag='samenorm', out_dir=out, save_predictions=False)
    cost = TRANSFER_STEPS * MS_PER_STEP[model] / 1000 + 2.0
    return Job('transfer', kw, f'{out}/samenorm_{src}_to_{tgt}_{model}_k{k}_s{s}_global.json', cost,
               f'{src}->{tgt} {model} k{k} s{s}', (src, tgt))


def E18(repeats=None, datasets=DATASETS, seeds=(0, 1, 2, 3, 4), out='runs/E14') -> List[Job]:
    return [_transfer_job(src, tgt, m, k, s, out)
            for src, tgt, s, k, m in itertools.product(datasets, DATASETS, seeds, [0, 3], ['mlp', 'pinn_semi'])
            if src != tgt]


def E14(repeats=None, datasets=DATASETS, seeds=(0, 1, 2, 3, 4), out='runs/E14') -> List[Job]:
    """Bốn cặp cùng hoá học — tập con của E18, giữ lại để tái lập bảng E14 cũ."""
    return [_transfer_job(src, SAME_CHEM[src], m, k, s, out)
            for src, s, k, m in itertools.product(datasets, seeds, [0, 3], ['mlp', 'pinn_semi'])]


def E15(repeats=2, datasets=DATASETS, seeds=None, out='runs/E15') -> List[Job]:
    from .leakage import SETTINGS, leak_path
    return [Job('leak', dict(setting=st, ds=ds, r=r, k=k, out_dir=out), leak_path(st, ds, r, k, out),
                _train_cost('mlp'), f'{st} {ds} cv{r}-{k}', (ds,))
            for st, ds, r, k in itertools.product(SETTINGS, datasets, range(repeats), range(E13_FOLDS))]


def E16(repeats=5, datasets=DATASETS, seeds=None, out='runs/E16', models=('ridge', 'gbdt', 'gpr')) -> List[Job]:
    from .baselines import cv_path
    return [Job('baseline', dict(model=m, ds=ds, frac=f, r=r, k=k, out_dir=out), cv_path(m, ds, f, r, k, out),
                BASELINE_S[(m, ds)], f'{m} {ds} f{f} cv{r}-{k}', (ds,))
            for m, ds, f, r, k in itertools.product(models, datasets, FRACS, range(repeats), range(E13_FOLDS))]


def E19(repeats=None, datasets=DATASETS, seeds=(0, 1, 2, 3, 4), out='runs/E14', models=('gbdt',)) -> List[Job]:
    from .baselines import transfer_path
    return [Job('baseline_transfer', dict(model=m, src=src, tgt=tgt, k=k, seed=s, out_dir=out),
                transfer_path(m, src, tgt, k, s, out), 5 * BASELINE_S[(m, src)], f'{m} {src}->{tgt} k{k} s{s}',
                (src, tgt))
            for m, src, tgt, k, s in itertools.product(models, datasets, DATASETS, [0, 3], seeds) if src != tgt]


# ---------------------------------------------------------------- E20: cải tiến mô hình
# Giai đoạn phát triển: chỉ lần lặp 0, mức nhãn 10 % và 50 %, mọi quyết định dựa trên MAE theo cell của
# tập VALIDATION (dự đoán validation được lưu kèm). Tập test của các lượt này không được dùng để chọn.
E20_DEV_FRACS = [0.1, 0.5]
E20_DEV_ARMS = [   # (nhãn, mô hình, cấu hình riêng, các thành viên)
    ('P', 'pinn_semi', E13_PHYS, (0, 1, 2)),
    ('KD03', 'pinn_semi', dict(E13_PHYS, distill_w=0.3), (0,)),
    ('KD1', 'pinn_semi', dict(E13_PHYS, distill_w=1.0), (0,)),
    ('EU', 'pinn_semi', dict(E13_PHYS, alpha_euler=2.0), (0,)),
    ('M2', 'pinn_semi', dict(E13_PHYS, beta_mono=2.0), (0,)),
    ('REL', 'pinn_semi', dict(E13_PHYS, norm='both'), (0,)),
]


def _member_job(tag, model, frac, extra, r, ds, k, out, member):
    j = _train_job(tag, model, frac, dict(extra, member=member, save_val_predictions=True), r, ds, k, out)
    j.label += f' m{member}'
    return j


def E20dev(repeats=1, datasets=DATASETS, seeds=None, out='runs/E20dev') -> List[Job]:
    from .baselines import cv_path
    jobs = [_member_job(tag, m, f, extra, r, ds, k, out, mem)
            for (tag, m, extra, mems), f, r, ds, k in itertools.product(E20_DEV_ARMS, E20_DEV_FRACS, range(repeats),
                                                                          datasets, range(E13_FOLDS))
            for mem in mems]
    jobs += [Job('baseline', dict(model='gbdt', ds=ds, frac=f, r=r, k=k, out_dir=out, save_val=True),
                 cv_path('gbdt', ds, f, r, k, out), BASELINE_S[('gbdt', ds)], f'gbdt {ds} f{f} cv{r}-{k}', (ds,))
             for f, r, ds, k in itertools.product(E20_DEV_FRACS, range(repeats), datasets, range(E13_FOLDS))]
    jobs += [Job('baseline', dict(model='gbdt', ds=ds, frac=f, r=r, k=k, out_dir=out, save_val=True, norm='both'),
                 cv_path('gbdt', ds, f, r, k, out, 'both'), BASELINE_S[('gbdt', ds)], f'gbdt-both {ds} f{f} cv{r}-{k}', (ds,))
             for f, r, ds, k in itertools.product(E20_DEV_FRACS, range(repeats), datasets, range(E13_FOLDS))]
    return jobs


# Đánh giá cuối (chốt sau giai đoạn phát triển, xem tai-lieu/de-cuong-E20.md): 3 lần lặp × 5 fold × 4 mức nhãn.
#   P  PINN-semi E13, thành viên 0–2, lưu dự đoán validation (thành viên 0 trùng từng bit với E13)
#   M  MLP E13, thành viên 1–2 (thành viên 0 lấy từ E13) — nhánh đối chứng không vật lý của mô hình lai
#   gbdt  GBDT E16, lưu dự đoán validation (dự đoán test trùng E16)
E20_REPEATS = 3
E20_MEMBERS = (0, 1, 2)


def E20(repeats=E20_REPEATS, datasets=DATASETS, seeds=None, out='runs/E20') -> List[Job]:
    from .baselines import cv_path
    rr = range(repeats)
    jobs = [_member_job('P', 'pinn_semi', f, E13_PHYS, r, ds, k, out, mem)
            for f, r, ds, k, mem in itertools.product(FRACS, rr, datasets, range(E13_FOLDS), E20_MEMBERS)]
    jobs += [_member_job('M', 'mlp', f, {}, r, ds, k, out, mem)
             for f, r, ds, k, mem in itertools.product(FRACS, rr, datasets, range(E13_FOLDS), E20_MEMBERS[1:])]
    jobs += [Job('baseline', dict(model='gbdt', ds=ds, frac=f, r=r, k=k, out_dir=out, save_val=True),
                 cv_path('gbdt', ds, f, r, k, out), BASELINE_S[('gbdt', ds)], f'gbdt {ds} f{f} cv{r}-{k}', (ds,))
             for f, r, ds, k in itertools.product(FRACS, rr, datasets, range(E13_FOLDS))]
    return jobs


# E21: huấn luyện trên một bộ, kiểm tra trên các bộ khác (pinnsoh/crossds.py)
#   base   2 giao thức (70/30 và 70/15/15) × 4 bộ nguồn × 3 hạt giống × {MLP, GBDT, 3 PINN}
#   chain  giao thức 70/30, 12 cặp nguồn–đích × 2 hạt giống × 2 nửa bộ đích; vòng 0 thích nghi không nhãn,
#          vòng 1–2 tự huấn luyện bằng nhãn giả của mô hình kết hợp
E21_SEEDS = (0, 1, 2)
E21_CHAIN_SEEDS = (0, 1)
E21_ROUNDS = 2


def E21(repeats=None, datasets=DATASETS, seeds=None, out='runs/E21') -> List[Job]:
    from .crossds import base_path, chain_path, PROTOCOLS
    sd = E21_SEEDS if seeds is None else tuple(seeds)
    cs = E21_CHAIN_SEEDS if seeds is None else tuple(seeds)
    jobs = []
    for proto, src, s in itertools.product(PROTOCOLS, datasets, sd):
        for model, mem, cost in [('mlp', 0, 40), ('gbdt', 0, 30)] + [('pinn_semi', m, 90) for m in range(3)]:
            jobs.append(Job('cross_base', dict(proto=proto, src=src, seed=s, model=model, member=mem),
                            base_path(proto, src, s, model, mem), cost, f'{proto} {src} s{s} {model} m{mem}',
                            tuple(DATASETS)))
    for src, tgt, s, h in itertools.product(datasets, DATASETS, cs, (0, 1)):
        if src == tgt:
            continue
        jobs.append(Job('cross_chain', dict(src=src, tgt=tgt, seed=s, half=h, rounds=E21_ROUNDS),
                        chain_path(src, tgt, s, h, E21_ROUNDS, 'gbdt'), (E21_ROUNDS + 1) * 300,
                        f'chain {src}->{tgt} s{s} h{h}', (src, tgt)))
    return jobs


# E22: dự báo SOH tương lai, dung lượng còn lại và RUL (pinnsoh/forecast.py) — cùng fold với E13/E20, lần lặp 0–2,
#      lịch sử ước lượng lấy từ PINN-lai của E20 (mức nhãn 70 %)
E22_COST = {'XJTU': 60, 'TJU': 90, 'MIT': 120, 'HUST': 200}


def E22(repeats=E20_REPEATS, datasets=DATASETS, seeds=None, out='runs/E22') -> List[Job]:
    from .forecast import out_path
    return [Job('forecast', dict(ds=ds, r=r, k=k, root=out), out_path(ds, r, k, out), E22_COST[ds],
                f'forecast {ds} cv{r}-{k}', (ds,))
            for r, ds, k in itertools.product(range(repeats), datasets, range(E13_FOLDS))]


# E23: dự báo có xét hồi phục dung lượng (pinnsoh/recovery.py) — cùng fold E22, học lại cả mô hình E22 để so ghép cặp
E23_COST = {'XJTU': 150, 'TJU': 200, 'MIT': 250, 'HUST': 300, 'NASA': 60}


def E23(repeats=E20_REPEATS, datasets=DATASETS, seeds=None, out='runs/E23') -> List[Job]:
    from .recovery import out_path
    return [Job('recovery', dict(ds=ds, r=r, k=k, root=out), out_path(ds, r, k, out), E23_COST[ds],
                f'recovery {ds} cv{r}-{k}', (ds,))
            for r, ds, k in itertools.product(range(repeats), datasets, range(E13_FOLDS))]


# E24: bộ NASA PCoE (18 cell, pinnsoh/nasa.py), có và không có thời gian nghỉ (NASAdt)
#   cv     kiểm định chéo 5 fold × 3 lần lặp, mức nhãn 30 và 70 %: 3 PINN-semi, MLP, GBDT -> PINN-lai
#   samba  cách chia NASA-L của SambaMixer (đánh giá trên #06, #07, #47), 3 hạt giống
#   dự báo E22 và E23 trên NASA (cần lượt cv ở 70 % nhãn làm lịch sử ước lượng)
E24_PHYS = dict(E13_PHYS, range_first=(0.40, 1.10))
NASA_SETS = ('NASA', 'NASAdt')


def E24(repeats=E20_REPEATS, datasets=NASA_SETS, seeds=None, out='runs/E24') -> List[Job]:
    from .baselines import cv_path, samba_path
    from .train import Config, run_name
    rr = range(repeats)
    datasets = [d for d in datasets if d in NASA_SETS] or list(NASA_SETS)    # run.py truyền bốn bộ chính: bỏ qua
    jobs = []
    for ds, f, r, k in itertools.product(datasets, (0.3, 0.7), rr, range(E13_FOLDS)):
        jobs += [_member_job('P', 'pinn_semi', f, E24_PHYS, r, ds, k, out, m) for m in E20_MEMBERS]
        jobs.append(_member_job('M', 'mlp', f, dict(range_first=(0.40, 1.10)), r, ds, k, out, 0))
        jobs.append(Job('baseline', dict(model='gbdt', ds=ds, frac=f, r=r, k=k, out_dir=out, save_val=True),
                        cv_path('gbdt', ds, f, r, k, out), 3, f'gbdt {ds} f{f} cv{r}-{k}', (ds,)))
    for ds, s in itertools.product(datasets, rr):
        for model, mem, extra in [('pinn_semi', m, E24_PHYS) for m in E20_MEMBERS] + [('mlp', 0, dict(range_first=(0.40, 1.10)))]:
            kw = dict(dataset=ds, label_frac=0.7, model=model, seed=s, split='samba', tag='S', out_dir=out,
                      member=mem, save_val_predictions=True, **E13_NET, **extra)
            jobs.append(Job('train', kw, os.path.join(out, run_name(Config(**kw)) + '.json'), _train_cost(model),
                            f'S {ds} {model} s{s} m{mem}', (ds,)))
        jobs.append(Job('baseline_samba', dict(model='gbdt', ds=ds, seed=s, out_dir=out), samba_path('gbdt', ds, s, out),
                        3, f'samba gbdt {ds} s{s}', (ds,)))
    return jobs


def E24f(repeats=E20_REPEATS, datasets=None, seeds=None, out=None) -> List[Job]:
    """Dự báo E22 và E23 trên NASA; chạy sau E24."""
    from .forecast import out_path as fo
    from .recovery import out_path as ro
    rr = range(repeats)
    return ([Job('forecast', dict(ds='NASA', r=r, k=k, root='runs/E22'), fo('NASA', r, k, 'runs/E22'), 40,
                 f'forecast NASA cv{r}-{k}', ('NASA',)) for r, k in itertools.product(rr, range(E13_FOLDS))]
            + [Job('recovery', dict(ds='NASA', r=r, k=k, root='runs/E23'), ro('NASA', r, k, 'runs/E23'), 60,
                   f'recovery NASA cv{r}-{k}', ('NASA',)) for r, k in itertools.product(rr, range(E13_FOLDS))])


# E26: tham số vật lý có xác định được không — hai bộ nhiều mức nhiệt độ (NASA 4/24/43 °C, TJU 25/35/45 °C), PINN-semi
#      ở 70 % nhãn, tăng trọng số phần dư động học alpha_ode, thêm phần dư Euler, và khởi tạo lam, Ea ở giá trị khác
#      (nếu giá trị học được đi theo giá trị khởi tạo thì dữ liệu không xác định được tham số đó). Mức gốc (alpha_ode = 0,02,
#      khởi tạo lam = 1, Ea = 30 kJ/mol) lấy lại từ E24 (NASA) và E13 nhánh G (TJU).
E26_ARMS = [('A0i', dict(lam_init=3.0, ea_init_kj=60.0)),
            ('A1', dict(alpha_ode=0.2)),
            ('A2', dict(alpha_ode=1.0)),
            ('A2i', dict(alpha_ode=1.0, lam_init=3.0, ea_init_kj=60.0)),
            ('A3', dict(alpha_ode=1.0, alpha_euler=2.0))]
E26_SETS = {'NASA': (0, 1, 2), 'TJU': (0,)}


def E26(repeats=None, datasets=None, seeds=None, out='runs/E26') -> List[Job]:
    jobs = []
    for ds, reps in E26_SETS.items():
        base = dict(E13_PHYS, range_first=(0.40, 1.10)) if ds == 'NASA' else dict(E13_PHYS)
        for (tag, extra), r, k in itertools.product(E26_ARMS, reps, range(E13_FOLDS)):
            jobs.append(_train_job('E26' + tag, 'pinn_semi', 0.7, dict(base, **extra), r, ds, k, out))
    return jobs


# E27: so với PINN4SOH trên đúng cách chia của bài (pinnsoh/paper.py) — 11 nhóm đầy đủ và 28 nhóm ít mẫu, 3 hạt giống,
#      hai cách tiền xử lý: 'strict' (của đồ án, không rò rỉ) và 'paper' (bản sao tiền xử lý của bài: lọc 3σ cả vòng đời
#      kể cả cột dung lượng, min–max theo từng cell cho đặc trưng và chỉ số chu kỳ) để so cùng điều kiện với số công bố.
#      Mỗi nhóm: 3 PINN-semi (-> PINN-lai cùng GBDT) và, ở các nhóm đầy đủ, một PINN động học hộp đen kiểu PINN4SOH.
E27_SEEDS = (0, 1, 2)
E27_PREPS = {'strict': dict(clean='causal', norm='global', cyc='global'),
             'paper': dict(clean='paper3s', norm='per_cell_minmax', cyc='per_cell_minmax')}


def E27(repeats=None, datasets=None, seeds=E27_SEEDS, out='runs/E27') -> List[Job]:
    from .paper import keys, gbdt_path
    from .train import Config, run_name
    seeds = E27_SEEDS if seeds is None else tuple(seeds)
    jobs = []
    for key, prep, s in itertools.product(keys(), E27_PREPS, seeds):
        ds, mode, _ = key.split('|')
        if datasets and ds not in datasets:
            continue
        models = [('pinn_semi', m) for m in E20_MEMBERS] + ([('pinn_bb', 0)] if mode == 'full' else [])
        for model, mem in models:
            kw = dict(dataset=ds, label_frac=1.0, model=model, seed=s, split='paper', paper_key=key, member=mem,
                      tag=f'E27{prep}', out_dir=out, norm_pool='train', pair_by='cycle', **E13_NET, **E13_PHYS,
                      **E27_PREPS[prep])
            cost = _train_cost(model) * (2.0 if ds in ('HUST', 'MIT') else 1.0)
            jobs.append(Job('train_cfg', kw, os.path.join(out, run_name(Config(**kw)) + '.json'), cost,
                            f'E27{prep} {key} {model} s{s} m{mem}', (ds,)))
        jobs.append(Job('baseline_paper', dict(key=key, seed=s, out_dir=out, prep=prep), gbdt_path(key, s, out, prep),
                        3 if ds in ('XJTU', 'TJU') else 20, f'E27{prep} gbdt {key} s{s}', (ds,)))
    return jobs


# E28: ngân sách đo dung lượng (pinnsoh/budget.py) — MỌI cell train được đo ở chu kỳ đầu và sau mỗi q·L chu kỳ, cùng
#      fold và hạt giống với E20 (lần lặp 0–2), để vẽ sai số theo số lần đo cạnh các mức nhãn theo cell của E20.
#      Mỗi (bộ, q, lần lặp, fold): 3 PINN-semi chỉ có nhãn ở lần đo (vật lý trên mọi hàng) -> PINN-lai cùng GBDT;
#      MLP chỉ có nhãn ở lần đo; GBDT trên lần đo và trên nhãn nội suy; một PINN-semi học nhãn nội suy (q = 0,1; 0,05).
E28_QS = (0.5, 0.2, 0.1, 0.05)


def E28(repeats=E20_REPEATS, datasets=DATASETS, seeds=None, out='runs/E28') -> List[Job]:
    from .budget import gbdt_path
    rr = range(repeats)
    jobs = []
    for q, r, ds, k in itertools.product(E28_QS, rr, datasets, range(E13_FOLDS)):
        meas = dict(E13_PHYS, meas_every=q)
        jobs += [_member_job(f'Q', 'pinn_semi', 0.7, meas, r, ds, k, out, m) for m in E20_MEMBERS]
        jobs.append(_member_job('Q', 'mlp', 0.7, dict(meas_every=q), r, ds, k, out, 0))
        if q in (0.1, 0.05):
            jobs.append(_member_job('Q', 'pinn_semi', 0.7, dict(meas, meas_interp=True), r, ds, k, out, 0))
        for mode in ('meas', 'interp'):
            jobs.append(Job('baseline_budget', dict(ds=ds, q=q, mode=mode, r=r, k=k, out_dir=out),
                            gbdt_path(ds, q, mode, r, k, out), BASELINE_S[('gbdt', ds)], f'gbdt-{mode} {ds} q{q} cv{r}-{k}',
                            (ds,)))
    return jobs


# E29dev: mô hình dùng lịch sử của cell — giai đoạn phát triển, lần lặp 0, nhãn 30 và 70 %, chỉ xem MAE validation (so
#         với thành viên 0 của E20, cùng hạt giống). Đặc trưng ở chu kỳ n chỉ dùng chu kỳ n và các chu kỳ trước của cùng cell
#         (data.history_features):
#           V   'ema'                    chỉ trung bình trượt mũ a = 0,3 thay cho đặc trưng gốc (lượt thử đầu)
#           H   'h:x+e0.1'               đặc trưng gốc + trung bình trượt mũ a = 0,1
#           H   'h:x+e0.3+e0.1+e0.03'    đặc trưng gốc + ba thang thời gian (khoảng 3, 10, 30 chu kỳ)
#           H   'h:x+e0.1+e0.03+s20'     đặc trưng gốc + hai thang + độ dốc trên 20 chu kỳ gần nhất
E29_DEV_SPECS = ('h:x+e0.1', 'h:x+e0.3+e0.1+e0.03', 'h:x+e0.1+e0.03+s20')


def E29dev(repeats=1, datasets=DATASETS, seeds=None, out='runs/E29dev') -> List[Job]:
    jobs = [_member_job('V', 'pinn_semi', f, dict(E13_PHYS, norm='ema'), r, ds, k, out, 0)
            for f, r, ds, k in itertools.product((0.3, 0.7), range(repeats), datasets, range(E13_FOLDS))]
    jobs += [_member_job('H', 'pinn_semi', f, dict(E13_PHYS, norm=spec), r, ds, k, out, 0)
             for spec, f, r, ds, k in itertools.product(E29_DEV_SPECS, (0.3, 0.7), range(repeats), datasets,
                                                       range(E13_FOLDS))]
    # NASA (có / không giờ nghỉ): cùng bốn dạng, kiểm định chéo như E24 và cách chia NASA-L của SambaMixer (hạt giống 0)
    from .train import Config, run_name
    for ds, spec in itertools.product(NASA_SETS, ('ema',) + E29_DEV_SPECS):
        jobs += [_member_job('H', 'pinn_semi', f, dict(E24_PHYS, norm=spec), r, ds, k, out, 0)
                 for f, r, k in itertools.product((0.3, 0.7), range(repeats), range(E13_FOLDS))]
        kw = dict(dataset=ds, label_frac=0.7, model='pinn_semi', seed=0, split='samba', tag='S', out_dir=out, member=0,
                  save_val_predictions=True, norm=spec, **E13_NET, **E24_PHYS)
        jobs.append(Job('train', kw, os.path.join(out, run_name(Config(**kw)) + '.json'), _train_cost('pinn_semi'),
                        f'S {ds} pinn_semi {spec}', (ds,)))
    return jobs


# E29: đánh giá cuối mô hình dùng lịch sử (xem tai-lieu/de-cuong-E29.md). Dạng lịch sử chọn trên validation ở E29dev:
#      chỉ giữ khi giảm MAE validation trung bình ít nhất 2 %; bộ không có trong bảng giữ đặc trưng gốc (không có lượt mới).
E29_SPEC_PINN = {'XJTU': 'h:x+e0.1', 'TJU': 'h:x+e0.1', 'HUST': 'h:x+e0.1+e0.03+s20'}
E29_SPEC_GBDT = {'XJTU': 'h:x+e0.1', 'HUST': 'h:x+e0.1+e0.03+s20'}


def E29(repeats=E20_REPEATS, datasets=DATASETS, seeds=None, out='runs/E29') -> List[Job]:
    from .baselines import cv_path
    jobs = []
    for f, r, ds, k in itertools.product(FRACS, range(repeats), datasets, range(E13_FOLDS)):
        if ds in E29_SPEC_PINN:
            spec = E29_SPEC_PINN[ds]
            jobs += [_member_job('H', 'pinn_semi', f, dict(E13_PHYS, norm=spec), r, ds, k, out, m) for m in E20_MEMBERS]
            jobs.append(_member_job('HM', 'mlp', f, dict(norm=spec), r, ds, k, out, 0))
        if ds in E29_SPEC_GBDT:
            spec = E29_SPEC_GBDT[ds]
            jobs.append(Job('baseline', dict(model='gbdt', ds=ds, frac=f, r=r, k=k, out_dir=out, save_val=True, norm=spec),
                            cv_path('gbdt', ds, f, r, k, out, spec), BASELINE_S[('gbdt', ds)],
                            f'gbdt-h {ds} f{f} cv{r}-{k}', (ds,)))
    return jobs


EXPERIMENTS: Dict[str, Callable[..., List[Job]]] = dict(E13=E13, E14=E14, E15=E15, E16=E16, E17=E17, E18=E18, E19=E19,
                                                        E20dev=E20dev, E20=E20, E21=E21, E22=E22, E23=E23, E24=E24,
                                                        E24f=E24f, E26=E26, E27=E27, E28=E28, E29dev=E29dev, E29=E29)
# thứ tự chạy cho `run.py experiments`: E14 nằm trong E18 nên không lặp lại
ORDER = ['E13', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20dev', 'E20', 'E21', 'E22', 'E23', 'E24', 'E24f', 'E26', 'E27',
         'E28', 'E29dev', 'E29']
