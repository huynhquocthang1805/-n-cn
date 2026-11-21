# Báo cáo đồ án chuyên ngành

`main.tex` là báo cáo tiếng Việt theo mẫu Specialized-Project-Template-VI của Khoa Khoa học và Kỹ thuật Máy tính
(lớp `hcmut-thesis.cls`, dựa trên hoang-himself/hcmut-thesis, giấy phép LPPL 1.3c ở `LICENSE-hcmut-thesis.txt`), bỏ
toàn bộ phần bộ NASA. `bao-cao-do-an.pdf` là bản đã dựng.

```bash
python run.py report          # = python report/build.py: bảng, đoạn văn có số, hình -> report/bao-cao-do-an.pdf
python report/build.py --no-pdf
```

Biên dịch bằng pdfLaTeX và BibTeX như mẫu (`latexmk -pdf main.tex`), cần gói tiếng Việt vntex (Ubuntu:
`apt-get install latexmk texlive-latex-extra texlive-lang-other`). `generated/` và `graphics/` được đưa lên git, nên có
thể tải nguyên thư mục `report/` lên Overleaf (trình biên dịch pdfLaTeX) mà không chạy gì thêm.

## Cấu trúc

| Tệp | Nội dung |
|---|---|
| `main.tex` | thông tin bìa, gói, thứ tự các chương |
| `chapters/front/` | lời cam đoan, lời cảm ơn, tóm tắt, phân công, danh mục từ viết tắt |
| `chapters/main/01-introduction.tex` | lý do chọn đề tài, bài toán, mục tiêu, phạm vi, đóng góp |
| `chapters/main/02-background.tex` | cơ sở lý thuyết, các nghiên cứu liên quan |
| `chapters/main/03-methodology.tex` | dữ liệu, đặc trưng đầu vào (dùng gì, bỏ gì, số liệu chứng minh), giao thức, các mô hình, dự báo có đặc trưng lịch sử, mô hình HiPINN (E33) |
| `chapters/main/04-implementation.tex` | yêu cầu, kiến trúc và hiện thực hệ thống |
| `chapters/main/05-evaluation.tex` | thiết lập thực nghiệm, kết quả (chạy lại PINN4SOH, kết quả của đồ án, dự báo), thảo luận: so với bài báo đã công bố, chỉ số tốt hơn và chưa tốt |
| `chapters/main/06-conclusion.tex` | kết luận, hạn chế, hướng phát triển |
| `chapters/back/appendix-a.tex`, `appendix-b.tex` | kết quả bổ sung; mã nguồn và các kiểm tra tái lập |
| `refs/references.bib` | tài liệu tham khảo (BibTeX, kiểu `plain`) |
| `generated/` | bảng và đoạn văn có số, sinh bởi `build.py`; `generated/data/` giữ bản sao các bảng kết quả nhỏ |
| `graphics/` | hình, sinh bởi `build.py` hoặc chép từ báo cáo đã nộp |

## Số liệu lấy từ đâu

Không số nào gõ tay vào bảng.

| Phần | Nguồn |
|---|---|
| Chương 3: phân tích đặc trưng | `results/features_*.csv` từ `analysis/features.py` (bản sao ở `generated/data/`) |
| Chương 3, 5: kiểm toán rò rỉ | `results/thesis_leakage.csv` (chạy lại E15 và nhánh F của E13; bản sao ở `generated/data/`) |
| Chương 4: số lượt chạy | `pinnsoh/experiments.py`, `benchmark/results/summary.csv`, `results/features_prune.csv` |
| Chương 5: so với PINN4SOH | `benchmark/results/` từ `benchmark/compare.py` |
| Phụ lục B: kiểm tra tái lập | `tools/kiem_tai_lap.py` (nhật ký lưu ở `generated/data/kiem_tai_lap.txt`), `runs/E15`, `runs/E27` so với `generated/data/e27_lan_dau.json` (kết quả của lần chạy E27 đầu tiên, lấy từ `benchmark/results` ở commit `e174340`); tóm tắt ở `generated/data/tai_lap.json` |
| Chương 3, 5, Phụ lục A: dự báo SOH, dung lượng, RUL với HiGBDT và HiPINN (E33) | `runs/E33` (lịch sử ước lượng lấy từ `runs/E20`, 70 % nhãn): `analysis/analyze_e33.py` (bảng `results/e33_*.csv`, bản sao ở `generated/data/`) và `analysis/figures_report.py` (hình `graphics/fig_e33_*.pdf`); sơ đồ `graphics/fig_hipinn_arch.pdf` từ `analysis/diagrams.py` |
| Phụ lục A: HiGBDT cho bài toán cập nhật (E34) | `runs/E34`: `analysis/analyze_e34.py` (bảng `results/e34_*.csv`, bản sao ở `generated/data/`, hình `graphics/fig_e34_cost.pdf`) |
| Chương 5: các bảng còn lại | bảng của báo cáo đã nộp (`archive/thesis/do-an/main.tex`), đọc bằng `paper/thesis_tables.py` |

Khi `runs/E33` đủ 120 lượt (hoặc `runs/E34` đủ 60 lượt), `build.py` tính lại bảng và vẽ lại các hình tương ứng; không có thì dùng bản đã lưu ở `generated/data/` và `graphics/`. Chỉ chạy lại phần dự báo: `python run.py E20` (E33 chỉ cần các lượt PINN-semi và GBDT ở 70 % nhãn, cho kịch bản lịch sử ước lượng), `python run.py E33`, `python run.py E34`.

Mục dự báo E22 của báo cáo đã nộp không còn trong báo cáo: E33 thay nó trên cùng fold, với PINN-GBDT của E22 học lại làm mốc so sánh. Mã của E22 (`pinnsoh/forecast.py`, `analysis/analyze_e22.py`) vẫn giữ vì E33 dùng các khối của nó.

Trừ mục dự báo (sinh từ E33), các con số trong lời văn của Chương 5 chép từ lời văn của
báo cáo đã nộp; báo cáo đó sinh từ `results/` của lần chạy ở commit `2d66f75`, thư mục `runs/` của lần chạy đó không
còn. Mọi phần nói tới bộ NASA đã bỏ; các khoảng số trong báo cáo đã nộp có gộp NASA (ví dụ đo cell mới một lần lúc nhận
pin) không được dùng.

## Việc còn lại trước khi nộp

* Điền tên giảng viên hướng dẫn ở trang bìa (`\supervisor{...}` trong `main.tex`).
* Đọc lại lời văn của Chương 5 (chép từ báo cáo đã nộp) nếu đã sửa gì ở báo cáo đó.
