# Báo cáo đồ án chuyên ngành

`main.tex` là báo cáo tiếng Việt (khuôn đồ án Bách Khoa TP.HCM), bỏ toàn bộ phần bộ NASA, chỉ gồm: cơ sở lý thuyết;
dữ liệu và đặc trưng (dùng gì, bỏ gì, số liệu chứng minh); những việc đã thực hiện; kết quả chạy lại (mã gốc PINN4SOH
và các thí nghiệm của đồ án); kết quả đạt được; đối chiếu với bài báo đã công bố (chỉ số tốt hơn, chỉ số chưa tốt);
kết luận. `bao-cao-do-an.pdf` là bản đã dựng.

```bash
python run.py report          # = python report/build.py: bảng, đoạn văn có số, hình -> report/bao-cao-do-an.pdf
python report/build.py --no-pdf
```

Biên dịch bằng XeLaTeX (cần phông TeX Gyre Termes, TeX Gyre Heros, TeX Gyre Termes Math và DejaVu Sans Mono; trên
Ubuntu: `apt-get install texlive-xetex fonts-texgyre tex-gyre`). `generated/` và `hinh/` được đưa lên git, nên có thể
tải nguyên thư mục `report/` lên Overleaf (trình biên dịch XeLaTeX) mà không chạy gì thêm.

## Số liệu lấy từ đâu

Không số nào gõ tay vào bảng.

| Phần | Nguồn |
|---|---|
| Chương 2: phân tích đặc trưng | `results/features_*.csv` từ `analysis/features.py` (bản sao ở `generated/data/`) |
| Chương 2, 5: kiểm toán rò rỉ | `results/thesis_leakage.csv` (chạy lại E15 và nhánh F của E13; bản sao ở `generated/data/`) |
| Chương 3: số lượt chạy | `pinnsoh/experiments.py`, `benchmark/results/summary.csv`, `results/features_prune.csv` |
| Chương 4, 6: so với PINN4SOH | `benchmark/results/` từ `benchmark/compare.py` |
| Chương 4: kiểm tra tái lập | `tools/kiem_tai_lap.py` (nhật ký lưu ở `generated/data/kiem_tai_lap.txt`), `runs/E15`, `runs/E27` so với `generated/data/e27_lan_dau.json` (kết quả của lần chạy E27 đầu tiên, lấy từ `benchmark/results` ở commit `e174340`); tóm tắt ở `generated/data/tai_lap.json` |
| Chương 5: các bảng còn lại | bảng của báo cáo đã nộp (`archive/thesis/do-an/main.tex`), đọc bằng `paper/thesis_tables.py` |

Các con số trong lời văn của Chương 5 chép từ lời văn của báo cáo đã nộp; báo cáo đó sinh từ `results/` của lần chạy
ở commit `2d66f75`, thư mục `runs/` của lần chạy đó không còn. Mọi phần nói tới bộ NASA đã bỏ; các khoảng số trong
báo cáo đã nộp có gộp NASA (ví dụ đo cell mới một lần lúc nhận pin) không được dùng.

## Việc còn lại trước khi nộp

* Điền tên giảng viên hướng dẫn ở trang bìa.
* Đọc lại lời văn của Chương 5 (chép từ báo cáo đã nộp) nếu đã sửa gì ở báo cáo đó.
