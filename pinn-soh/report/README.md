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

Quy ước bảng số liệu: mỗi bảng có một hàng nêu độ đo (MAE, RMSE, MAPE, tỉ số MAE, tỉ lệ cell tốt hơn, hệ số Spearman); mũi tên $\downarrow$ là thấp hơn tốt hơn, $\uparrow$ là cao hơn tốt hơn, và số tốt nhất được in đậm (phạm vi so ghi ở chú thích bảng). Bảng kiểm toán rò rỉ đo độ lệch của đánh giá chứ không so mô hình nên không có mũi tên hay chữ đậm. Các bảng sinh trong `build.py` dùng chung hàm `metric_row` và `bold_best`.

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
| `graphics/` | hình, sinh bởi `build.py` hoặc chép từ báo cáo đã nộp; bốn sơ đồ do tác giả vẽ: kiến trúc PINN-semi (`fig_pinn_semi.png`), PINN-lai và HiPINN-lai (`fig_pinn_lai.png`), tổng quan HiPINN (`fig_hipinn_tongquan.png`), quy trình thực nghiệm (`fig_quy_trinh.png`) |

## Số liệu lấy từ đâu

Không số nào gõ tay vào bảng.

| Phần | Nguồn |
|---|---|
| Chương 3: phân tích đặc trưng | `results/features_*.csv` từ `analysis/features.py` (bản sao ở `generated/data/`) |
| Chương 3, 5: kiểm toán rò rỉ | `results/thesis_leakage.csv` (chạy lại E15 và nhánh F của E13; bản sao ở `generated/data/`) |
| Chương 4: số lượt chạy | `pinnsoh/experiments.py`, `benchmark/results/summary.csv`, `results/features_prune.csv` |
| Chương 5: so với PINN4SOH | `benchmark/results/` từ `benchmark/compare.py` |
| Phụ lục A: kiểm tra tái lập | `tools/kiem_tai_lap.py` (nhật ký lưu ở `generated/data/kiem_tai_lap.txt`), `runs/E15`, `runs/E27` so với `generated/data/e27_lan_dau.json` (kết quả của lần chạy E27 đầu tiên, lấy từ `benchmark/results` ở commit `e174340`); tóm tắt ở `generated/data/tai_lap.json` |
| Chương 3, 5: dự báo SOH, dung lượng, RUL với HiGBDT và HiPINN (E33; báo cáo chỉ dùng lịch sử SOH đo) | `runs/E33`: `analysis/analyze_e33.py` (bảng `results/e33_*.csv`, bản sao ở `generated/data/`) và `analysis/figures_report.py` (hình `graphics/fig_e33_*.pdf`); sơ đồ `graphics/fig_hipinn_arch.pdf` từ `analysis/diagrams.py` |
| Chương 5: HiGBDT cho bài toán cập nhật (E34) | `runs/E34`: `analysis/analyze_e34.py` (bảng `results/e34_*.csv`, bản sao ở `generated/data/`, hình `graphics/fig_e34_cost.pdf`) |
| Chương 5: PINN-lai cố định, PINN-lai tinh chỉnh theo từng fold, HiPINN-lai (bảng MAE chính, RMSE và MAPE, bảng lịch sử, các đoạn văn về PINN-lai; bảng mô hình cuối ở Chương 6) | `runs/E20`, `runs/E29`: `analysis/analyze_e20_folds.py` (`results/e20f_*.csv`, bản sao ở `generated/data/`); cột GBDT và PINN-semi của bảng MAE chính được kiểm là trùng bảng của báo cáo đã nộp |
| Chương 5: cập nhật bằng bộ lọc Kalman và GBDT-cập nhật, cột cuối của bảng ngân sách đo | `runs/E32`: `analysis/analyze_e32.py` (`results/e32_tiers.csv`, thang đời $L$ lấy trên cell huấn luyện của fold) |
| Chương 3: ngưỡng cuối đời $\tau$; Chương 5: các dải song song ở hình dự báo XJTU; mục vai trò của phần vật lý | dữ liệu (`generated/data/tau.csv`, `xjtu_bands.json`), $\lambda$ và $E_a$ của mọi PINN-semi trong `runs/E20` (`generated/data/physics_params.csv`) |
| Chương 4: chi phí tính toán | `analysis/compute_cost.py` (`results/compute_cost.csv`) |
| Chương 5: các bảng còn lại | bảng của báo cáo đã nộp (`archive/thesis/do-an/main.tex`), đọc bằng `paper/thesis_tables.py` |

Khi `runs/E33` đủ 120 lượt (hoặc `runs/E34` đủ 60 lượt), `build.py` tính lại bảng và vẽ lại các hình tương ứng; không có thì dùng bản đã lưu ở `generated/data/` và `graphics/`. Chỉ chạy lại phần dự báo: `python run.py E20` (E33 chỉ cần các lượt PINN-semi và GBDT ở 70 % nhãn, cho kịch bản lịch sử ước lượng), `python run.py E33`, `python run.py E34`.

Mục dự báo E22 của báo cáo đã nộp không còn trong báo cáo: E33 thay nó trên cùng fold, với PINN-GBDT của E22 học lại làm mốc so sánh. Mã của E22 (`pinnsoh/forecast.py`, `analysis/analyze_e22.py`) vẫn giữ vì E33 dùng các khối của nó.

Trừ mục dự báo (sinh từ E33) và các phần ở bảng trên, các con số trong lời văn của Chương 5 chép từ lời văn của
báo cáo đã nộp; báo cáo đó sinh từ `results/` của lần chạy ở commit `2d66f75`, thư mục `runs/` của lần chạy đó không
còn. Mọi phần nói tới bộ NASA đã bỏ; các khoảng số trong báo cáo đã nộp có gộp NASA (ví dụ đo cell mới một lần lúc nhận
pin) không được dùng.

## Việc còn lại trước khi nộp

* Điền tên giảng viên hướng dẫn ở trang bìa (`\supervisor{...}` trong `main.tex`).
* Đọc lại lời văn của Chương 5 (chép từ báo cáo đã nộp) nếu đã sửa gì ở báo cáo đó.
