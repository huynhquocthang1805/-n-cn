# Giai đoạn đầu (tháng 9/2026) — pipeline v1

Các script ở đây sinh kết quả của các báo cáo tiến độ trước E13 (thí nghiệm E1–E12, báo cáo HTML/LaTeX,
bộ slide 14–29/09). Chúng **không** thuộc báo cáo đồ án cuối — báo cáo đó dùng E13–E19 ở
`pinnsoh/experiments.py`, chạy bằng `python run.py`.

Chạy được ở commit `2d66f75`, khi thư mục này còn là `legacy/` ở gốc dự án (xem `archive/README.md`):

| Script | Việc |
|---|---|
| `experiments_v1.py E1 … E12` | các thí nghiệm pipeline v1 (`python legacy/experiments_v1.py E1 --seeds 0 1 2`) |
| `analyze.py`, `figures.py` | `runs/E1…E12` -> `results/E?_table.*`, `results/fig_*` |
| `make_report.py`, `make_markdown.py`, `make_latex.py`, `section_proofs.py` | báo cáo HTML / Markdown / LaTeX cũ |
| `verify_losses.py` | 24 phép kiểm tính chất của các hàm mất mát |
| `seed_sweep.py`, `seed_sweep_tuned.py`, `stats_test.py`, `power_test.py` | kiểm định nhiều seed của cấu hình v1 |
| `corr_audit.py`, `leak_demo.py`, `scale_sweep.py`, `decode.py` | kiểm toán tương quan, minh hoạ rò rỉ, quét CYCLE_SCALE, giải mã đơn điệu (E8) |
| `scripts/*.sh` | script chạy nối tiếp dùng trong giai đoạn đó |

Chế độ v1 vẫn là mặc định của `Config` trong `pinnsoh/train.py` (`clean='life3sigma'`, `norm_pool='model'`,
`pair_by='row'`) để các log cũ tái lập được; `Config(...).v2()` bật pipeline v2.
