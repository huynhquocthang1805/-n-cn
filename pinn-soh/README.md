# Ước lượng SOH pin lithium-ion đơn với ràng buộc vật lý không cần nhãn

Mã nguồn và dữ liệu của Đồ án chuyên ngành HK261 — nhóm 15.
Huỳnh Quốc Thắng, Khoa KH&KT Máy tính, Trường ĐH Bách khoa — ĐHQG-HCM.

Bài toán: ước lượng SOH (dung lượng hiện tại chia dung lượng danh định) của **một cell
lithium-ion đơn**, từ 16 đặc trưng thống kê của đoạn sạc CC–CV, tại từng chu kỳ. Điểm khác
biệt so với PINN4SOH: động học suy giảm có **cấu trúc vật lý** thay vì mạng đen; mọi hàm
mất mát vật lý đều **không dùng nhãn** nên áp được lên cả cell chưa đo dung lượng; và
pipeline **không rò rỉ thông tin**.

Toàn bộ kết quả trong báo cáo dựa trên **1 363 lượt huấn luyện thật** trên **387 cell** của
bốn bộ dữ liệu công khai (XJTU, TJU, MIT, HUST).

---

## 1. Cài đặt

Cần Python ≥ 3.10. Phiên bản đã dùng để sinh mọi kết quả: **Python 3.11.15**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Chỉ cần Node.js nếu muốn dựng lại slide (`slide/`):

```bash
npm install -g pptxgenjs@4.0.1
```

Chỉ cần XeLaTeX (TeX Live / MiKTeX) nếu muốn biên dịch báo cáo ra PDF.

## 2. Giải nén dữ liệu

Repo này đi kèm bốn zip riêng. Giải nén **vào đúng thư mục** dưới đây:

| Zip | Giải nén vào | Nội dung |
|---|---|---|
| `du-lieu-XJTU.zip`, `du-lieu-TJU.zip`, `du-lieu-MIT.zip`, `du-lieu-HUST.zip` | `PINN4SOH/data/` | 387 file CSV, mỗi file là một cell — dữ liệu gốc bốn bộ |
| `ket-qua-runs-E1-{XJTU,TJU,MIT,HUST}.zip`, `ket-qua-runs-E9-TUNED.zip`, `ket-qua-runs-con-lai.zip` | thư mục gốc repo | 1 387 file JSON — kết quả từng lượt huấn luyện (1 363 lượt dùng trong báo cáo + 24 lượt quét `CYCLE_SCALE` ở `runs/SCALE`). Các zip này đã chứa sẵn đường dẫn `runs/...` nên giải nén thẳng vào thư mục gốc. |
| `ket-qua-bang-hinh.zip` | thư mục gốc repo | 36 bảng CSV + 13 bảng MD + 13 hình PNG + 11 hình PDF (zip đã chứa sẵn `results/`) |

Sau khi giải nén, cây thư mục phải như sau:

```
pinn-soh/
├── PINN4SOH/data/
│   ├── XJTU data/*.csv                              (55 file, phẳng)
│   ├── TJU data/Dataset_{1_NCA,2_NCM,3_NCM_NCA}_battery/*.csv   (130 file)
│   ├── MIT data/{2017-05-12, 2017-06-30, 2018-04-12}/*.csv      (125 file)
│   └── HUST data/*.csv                              (77 file, phẳng)
├── runs/{E1, E2, ..., E12, TUNED, SCALE}/*.json
└── results/*.{csv, md, png, pdf}
```

Kiểm nhanh:

```bash
python -c "from pinnsoh.data import load_dataset; \
print({d: len(load_dataset('PINN4SOH/data', d)) for d in ['XJTU','TJU','MIT','HUST']})"
# mong đợi: {'XJTU': 55, 'TJU': 130, 'MIT': 125, 'HUST': 77}   (tổng 387 cell)
```

> **Mọi lệnh Python bên dưới phải chạy từ thư mục gốc của repo** — các script dùng đường
> dẫn tương đối `PINN4SOH/data`, `runs/`, `results/`.

---

## 3. Cấu trúc

### Thư viện lõi — `pinnsoh/`

| File | Vai trò |
|---|---|
| `data.py` | Đọc 387 file CSV thành `Cell`; chuẩn hoá **nhân quả** (`global` / `first_cycle`) và bản rò rỉ để đối chứng (`per_cell_minmax`); chia cell 70/15/15; sinh cặp điểm cho loss vật lý. `CYCLE_SCALE = 1000` là **đơn vị thời gian**, không phải siêu tham số. |
| `models.py` | Mạng nghiệm $F_\varphi(\mathbf{x},t)$ (4 kiến trúc: `mlp`, `res`, `fourier`, `mono`) và động học xám $r = \mathrm{softplus}(\mathrm{MLP}_\theta(\mathbf{x}))\,e^{\lambda(1-u)}\,e^{-\frac{E_a}{R}(\frac{1}{T}-\frac{1}{T_\mathrm{ref}})}$. |
| `train.py` | `Config` (mọi siêu tham số) + `run()`; bốn hàm mất mát; lịch hâm nóng $w(s)$; early stopping theo validation. |
| `transfer.py` | Huấn luyện nguồn → thích nghi sang bộ đích với $k$ cell có nhãn. |
| `metrics.py` | MAE gộp hàng, MAE theo cell, MAE giai đoạn cuối, sai số EOL, chỉ số vi phạm đơn điệu, nhiễu quỹ đạo. |
| `plotstyle.py` | Khuôn vẽ hình cho báo cáo. |

### Thí nghiệm

| Lệnh | Thí nghiệm | Số lượt |
|---|---|---|
| `python experiments.py E1` | Hiệu quả theo lượng nhãn — 4 mức nhãn × 3 mô hình | 200 |
| `python experiments.py E2` | Chuyển miền zero-shot, cặp cùng hoá học | 48 |
| `python experiments.py E3` | Ablation — bỏ từng loss / từng số hạng vật lý | 57 |
| `python experiments.py E4` | **Kiểm toán rò rỉ** — chuẩn hoá per-cell so với nhân quả | 30 |
| `python experiments.py E5` | Thích nghi miền với $k \in \{0,3\}$ cell nhãn ở bộ đích | 48 |
| `python experiments.py E6` | Chọn trọng số vật lý theo validation, ở 30 % nhãn | 48 |
| `python experiments.py E7` | Như E6 nhưng quét ở **mọi** mức nhãn | 240 |
| `python experiments.py E9` | Kiến trúc × hàm kích hoạt, cho **cả** baseline lẫn PINN | 180 |
| `python experiments.py E10` | Ma trận chuyển miền đầy đủ 4×4 | 48 |
| `python experiments.py E12` | **Ngân sách tinh chỉnh cân bằng** — cùng một lưới cho cả hai mô hình | 384 |
| `python decode.py` | E8 — giải mã đơn điệu hậu kỳ (PAVA), không huấn luyện lại | — |

Tham số chung: `--seeds 0 1 2`, `--datasets XJTU TJU MIT HUST`, `--threads N`.
Kết quả ghi vào `runs/<E?>/` dạng JSON, đặt tên theo cấu hình nên **chạy lại được từ giữa
chừng** (lượt đã có file thì bỏ qua).

E11 (bộ chỉ số mở rộng) đọc lại dự đoán đã lưu của E1, không cần huấn luyện — nó chạy bên
trong `analyze.py`.

### Tổng hợp và vẽ hình

```bash
python analyze.py     # runs/**/*.json  ->  results/E?_table.csv + .md
python figures.py     # results/*.csv   ->  results/fig_*.png và .pdf
```

`figures.py` dùng phông **TeX Gyre Heros** (gói `tex-gyre` của TeX Live, hoặc
`fonts-texgyre` trên Debian/Ubuntu). Thiếu phông thì matplotlib tự lùi về DejaVu Sans —
hình vẫn vẽ ra, chỉ khác kiểu chữ.

### Kiểm định và chứng minh

```bash
python verify_losses.py --csv     # 24 phép kiểm tính chất của các loss -> results/verify_table.csv
python seed_sweep.py              # nâng phép so chính ở 30 % nhãn lên 10 seed   (~14 phút)
python seed_sweep_tuned.py        # như trên, cấu hình đã tinh chỉnh            (~26 phút)
python stats_test.py                                              # kiểm định cấu hình E1
python stats_test.py --runs runs/TUNED --tag wide --suffix _tuned  # kiểm định cấu hình tinh chỉnh
python power_test.py              # ước lượng số seed cần thiết
```

Ba script kiểm toán độc lập:

```bash
python corr_audit.py    # tính lại bảng tương quan Spearman  -> results/corr_table.csv
python leak_demo.py     # chứng minh rò rỉ chuẩn hoá theo ba bước
python scale_sweep.py   # quét CYCLE_SCALE in {1, 100, 1000, 10000}
```

Chứng minh toán học đầy đủ (8 mệnh đề + 1 nhận xét) nằm ở `tai-lieu/chung-minh.md`, và
được sinh vào báo cáo LaTeX bởi `section_proofs.py`.

### Dựng tài liệu

```bash
python make_report.py     # results/  -> report.html   (nguồn duy nhất của nội dung)
python make_markdown.py   # report.html -> bao-cao-md/bao-cao.md + hinh/
python make_latex.py      # results/  -> bao-cao-latex/  (xelatex + bibtex)
cd bao-cao-latex && xelatex main && bibtex main && xelatex main && xelatex main
cd slide && node make_slides_bk.js    # slide theo khuôn BK
```

`section_proofs.py` được `make_latex.py` import, sinh mục 15 "Tính đúng đắn: chứng minh và
kiểm định" thẳng từ `results/stats_*.csv` và `results/verify_table.csv`.

---

## 4. Nguyên tắc của repo

1. **Không có số nào gõ tay.** Mọi bảng và mọi con số trong báo cáo đều sinh từ
   `results/*.csv`, và các CSV đó sinh từ `runs/**/*.json`. Muốn kiểm chứng một con số thì
   lần ngược theo đúng chuỗi đó.
2. **Chuẩn hoá nhân quả.** Thống kê chuẩn hoá chỉ tính trên tập train. Chuẩn hoá min–max
   theo từng cell bằng thống kê cả vòng đời là **rò rỉ** — E4 và `leak_demo.py` đo đúng mức
   rò rỉ đó thay vì chỉ nói suông.
3. **Chia theo cell, không theo hàng.** Cell test không bao giờ xuất hiện trong train ở bất
   kỳ chu kỳ nào.
4. **So sánh công bằng.** E12 áp đúng một lưới siêu tham số tổng quát cho cả baseline lẫn
   PINN. Trước E12 thì PINN được quét còn MLP chạy cấu hình cố định — tức là thiên vị.
5. **Phát biểu đúng mức.** Với 10 seed, ở cấu hình đã tinh chỉnh có bằng chứng nhất quán
   rằng PINN giảm MAE 10–28 % trên ba trong bốn bộ, **nhưng chưa đạt mức có ý nghĩa thống kê
   sau hiệu chỉnh đa so sánh**. Chi tiết ở `tai-lieu/chung-minh.md` mục 5.

## 5. Nguồn dữ liệu

Bốn bộ dữ liệu là dữ liệu công khai, lấy qua repo `wang-fujin/PINN4SOH` (bản clone
05/09/2026). Bản quyền thuộc về nhóm tác giả gốc của từng bộ:

| Bộ | Cell | Hoá học | Dung lượng danh định | Nhiệt độ |
|---|---|---|---|---|
| XJTU | 55 | NCM | 2,0 Ah | 25 °C |
| TJU | 130 | NCA/NCM | — | — |
| MIT | 125 | LFP | 1,1 Ah | 30 °C |
| HUST | 77 | LFP | 1,1 Ah | 30 °C |
| **Tổng** | **387** | | | |

TJU là bộ **duy nhất** có nhiệt độ thay đổi giữa các cell (đọc từ tên file theo từng batch).
Ba bộ còn lại chỉ có một nhiệt độ duy nhất, nên số hạng Arrhenius không có gradient ở đó —
$E_a$ đứng yên ở giá trị khởi tạo. Vì vậy ablation `no_arrh` của E3 chỉ chạy trên TJU: đó là
chỗ duy nhất việc bỏ Arrhenius có thể đổi kết quả.

Mã nguồn trong repo này là của tác giả đồ án; phần `PINN4SOH/data` giữ nguyên dữ liệu gốc
để kết quả tái lập được.
