# Ước lượng SOH pin lithium-ion bằng PINN bán giám sát

Mã nguồn của Đồ án chuyên ngành HK261 — nhóm 15.
Huỳnh Quốc Thắng, Khoa KH&KT Máy tính, Trường ĐH Bách khoa — ĐHQG-HCM.

Bài toán: ước lượng SOH (dung lượng hiện tại / dung lượng danh định) của từng cell lithium-ion, tại từng
chu kỳ, từ 16 đặc trưng của đoạn sạc CC–CV. Mô hình là một mạng nơ-ron có thông tin vật lý (PINN): mạng
nghiệm dự đoán SOH, gắn với động học suy giảm dạng xám luôn không âm; ba hàm mất mát vật lý không dùng nhãn
nên áp được lên cả cell chưa đo dung lượng. Mô hình cải tiến PINN-lai (E20) lấy trung bình ba mạng PINN với
GBDT rồi qua một bộ lọc quỹ đạo nhân quả không cho SOH tăng, kèm khoảng dự đoán 90 %. E21 huấn luyện trên một bộ
dữ liệu (70/30 hoặc 70/15/15), kiểm tra trên ba bộ còn lại, rồi thích nghi không nhãn và tự huấn luyện hai vòng.
E22 dự báo SOH ở các chu kỳ tương lai, dung lượng còn lại (Ah) và số chu kỳ còn lại tới ngưỡng cuối đời (RUL):
PINN-dự báo (mạng trả tham số của nghiệm đóng của động học xám) kết hợp với GBDT, kèm khoảng dự báo 90 %.
E23 tách lịch sử SOH thành xu thế, phần hồi phục tắt dần và dao động đều để dự báo đúng hơn khi dung lượng hồi phục.
E24 thêm bộ NASA PCoE (18 cell), thử thời gian nghỉ làm đặc trưng như SambaMixer và so trên đúng cách chia của bài đó.
Báo cáo: `do-an/bao-cao-do-an.pdf`.

Kết quả của báo cáo dựa trên **5 100 lượt huấn luyện** (E13–E20) trên **387 cell** của bốn bộ dữ liệu công
khai XJTU, TJU, MIT, HUST.

---

## 1. Cài đặt

Cần Python ≥ 3.10 (đã dùng 3.11.15). XeLaTeX (TeX Live / MiKTeX) chỉ cần cho bước dựng báo cáo PDF.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Chạy lại trên máy cá nhân

Mọi bước đều qua một lệnh `python run.py <lệnh>`, chạy từ thư mục nào cũng được.

```bash
python run.py check          # kiểm tra Python, thư viện, dữ liệu, XeLaTeX, số lõi CPU
python run.py data           # tải 387 file CSV (~94 MB) từ repo gốc và dựng bộ nhớ đệm
python run.py reproduce      # ~2–5 phút: chạy lại 20 lượt mẫu, so từng bit với kết quả công bố
python run.py bench          # đo tốc độ máy để ước lượng thời gian chính xác hơn
python run.py all --dry-run  # xem còn bao nhiêu lượt và mất bao lâu, không chạy gì
python run.py all            # toàn bộ: thí nghiệm -> phân tích -> báo cáo PDF
```

`all` = `experiments` (E13, E15, E16, E17, E18, E19, E20dev, E20, E21, E22, E23, E24, E24f) → `analyze` → `thesis`. Có thể chạy từng bước:

```bash
python run.py E13 --workers 8          # một thí nghiệm, 8 tiến trình song song
python run.py E16 --datasets XJTU      # chỉ một bộ dữ liệu
python run.py experiments --quick      # chạy thử: 1 lần lặp thay vì 5, 1 hạt giống thay vì 5
python run.py analyze                  # runs/ -> results/ (bảng CSV/MD, hình PNG/PDF)
python run.py thesis                   # results/ -> do-an/bao-cao-do-an.pdf, tai-lieu/phan-tich-vat-ly-PINN.pdf
```

* **Dừng lúc nào cũng được** (Ctrl+C, tắt máy). Mỗi lượt ghi một file JSON trong `runs/`, ghi nguyên tử;
  chạy lại đúng lệnh cũ sẽ bỏ qua lượt đã xong và chạy tiếp.
* `--workers N`: mặc định bằng số lõi CPU. Mỗi tiến trình dùng 1 luồng — với mạng nhỏ thế này, N tiến
  trình × 1 luồng nhanh hơn nhiều so với 1 tiến trình × N luồng.
* `--quick` cho tập con của bản đầy đủ (lần lặp 0, hạt giống 0); chạy bản đầy đủ sau đó sẽ không phải chạy
  lại phần đã có. Kết quả `--quick` đi trọn được tới PDF nhưng chỉ để kiểm tra, không phải số liệu của báo cáo.
* `--device cuda`: dùng GPU. Kết quả lệch ở mức làm tròn số thực nên không tái lập từng bit.
* Dữ liệu để ở chỗ khác: đặt biến môi trường `PINNSOH_DATA=<thư mục chứa "XJTU data", "TJU data", ...>`.
  Không tải tự động được (mạng chặn GitHub): tải zip của repo `wang-fujin/PINN4SOH` rồi
  `python tools/tai_du_lieu.py --zip <file.zip>`.

### Thời gian

Ước lượng trên CPU của máy chạy thử, 1 luồng mỗi tiến trình. `python run.py bench` đo hệ số cho máy của bạn
và `--dry-run` sẽ dùng hệ số đó.

| Thí nghiệm | Nội dung | Lượt | Giờ CPU | Giờ, 4 lõi | Giờ, 8 lõi |
|---|---|---:|---:|---:|---:|
| E13 | MLP, PINN-semi ở 4 mức nhãn; PINN-sup, 2 biến thể cắt bỏ ở 30 % (CV 5 fold × 5 lần lặp) | 1 100 | 7,0 | 1,8 | 0,9 |
| E15 | Kiểm toán rò rỉ: 4 cách chia/chuẩn hoá sai | 160 | 0,5 | 0,1 | 0,1 |
| E16 | Ridge, GPR, GBDT, cùng cách chia và tập nhãn | 1 200 | 2,7 | 0,7 | 0,3 |
| E17 | PINN-sup ở 10, 50 %; PINN hộp đen ở 4 mức nhãn | 600 | 6,5 | 1,6 | 0,8 |
| E18 | Chuyển miền 12 cặp, MLP và PINN-semi, k ∈ {0, 3} | 240 | 1,0 | 0,3 | 0,1 |
| E19 | GBDT trên 12 cặp chuyển miền | 120 | 2,3 | 0,6 | 0,3 |
| E20dev | 9 hướng cải tiến, chọn trên validation (lần lặp 0, nhãn 10 và 50 %) | 400 | 2,2 | 0,6 | 0,3 |
| E20 | PINN-lai: 3 PINN-semi, 2 MLP thêm, GBDT lưu dự đoán validation (3 lần lặp) | 1 280 | 7,2 | 1,8 | 0,9 |
| E21 | Huấn luyện một bộ, kiểm tra bộ khác (2 giao thức, 3 hạt giống) + 48 chuỗi thích nghi, 2 vòng học lặp | 168 | 12 | 3 | 1,5 |
| E22 | Dự báo SOH tương lai, dung lượng, RUL: GBDT + 3 PINN-dự báo mỗi fold (CV 5 fold × 3 lần lặp, cần E20) | 60 | 0,6 | 0,2 | 0,1 |
| E23 | Dự báo có xét hồi phục, cùng fold E22, học lại cả mô hình E22 để so ghép cặp | 60 | 3 | 0,8 | 0,4 |
| E24 | Bộ NASA: 3 PINN-semi, MLP, GBDT ở 30/70 % nhãn, có/không giờ nghỉ; cách chia SambaMixer | 330 | 1,9 | 0,5 | 0,3 |
| E24f | Dự báo E22 và E23 trên NASA (cần E24) | 30 | 0,4 | 0,1 | 0,1 |
| **Tổng** | | **5 748** | **≈ 47** | **≈ 12** | **≈ 6** |

E20 gồm 1 440 lượt, trong đó 160 lượt trùng cấu hình với E20dev: chép `runs/E20dev/P_*` và
`runs/E20dev/gbdt_*` sang `runs/E20/` trước thì chỉ còn 1 280 lượt mới (bảng tính theo con số này). Cấu hình E20 chốt trong `tai-lieu/de-cuong-E20.md`.

`--quick` cần khoảng 1/5 thời gian trên. Phân tích (`analyze`) mất ~5 phút lần đầu cộng ~40 phút CPU để
`analyze_physics.py` huấn luyện lại 120 lượt (đo đạo hàm và tốc độ suy giảm cần chính mô hình, mà trọng số không
được lưu; huấn luyện tất định nên mô hình giống từng bit lượt đã công bố); các lần sau ~2 phút nhờ bộ nhớ đệm.
Dựng PDF ~10 giây.

### Đã có sẵn `runs/`?

Nếu đã có thư mục `runs/` của lần chạy công bố (ví dụ nhận kèm dạng zip, giải nén vào thư mục gốc) thì
không cần huấn luyện: `python run.py analyze && python run.py thesis` dựng lại đúng từng byte các bảng,
hình và file PDF của báo cáo.

## 3. Cấu trúc

```
pinn-soh/
├── run.py                  lệnh duy nhất cho mọi bước
├── pinnsoh/                thư viện
│   ├── data.py             đọc 387 CSV -> Cell; lọc ngoại lai nhân quả; chia theo cell (cv_split);
│   │                       chuẩn hoá; bộ nhớ đệm .cache/
│   ├── models.py           mạng nghiệm F_φ; động học xám r = softplus(G_θ(x))·e^{λ(1-u)}·A(T); hộp đen
│   ├── train.py            Config, bốn hàm mất mát, huấn luyện + dừng sớm
│   ├── transfer.py         huấn luyện chuyển miền nguồn -> đích
│   ├── baselines.py        Ridge, GBDT, GPR (E16, E19)
│   ├── leakage.py          các cách chia/chuẩn hoá sai cho kiểm toán rò rỉ (E15)
│   ├── metrics.py          MAE theo cell, R², EOL, vi phạm đơn điệu, ...
│   ├── experiments.py      định nghĩa E13–E20 thành danh sách lượt chạy
│   ├── postproc.py         bộ lọc quỹ đạo nhân quả (E20)
│   ├── crossds.py          E21: huấn luyện một bộ, kiểm tra bộ khác; thích nghi + tự huấn luyện
│   ├── forecast.py         E22: dự báo SOH tương lai, dung lượng còn lại, RUL (nghiệm đóng + GBDT)
│   ├── recovery.py         E23: tách xu thế / hồi phục tắt dần / dao động đều cho dự báo
│   ├── nasa.py             đọc bộ NASA PCoE (.mat) -> 16 đặc trưng sạc + giờ nghỉ, cùng định dạng PINN4SOH
│   ├── runner.py           chạy song song, bỏ qua lượt đã xong, ước lượng thời gian
│   ├── runcache.py         nhớ kết quả đọc runs/*.json giữa các lần phân tích
│   └── plotstyle.py        khuôn vẽ hình
├── analysis/               runs/ -> results/
│   ├── analyze_e13.py      bảng và kiểm định của E13, E14
│   ├── analyze_predictions.py  R², MAPE, hình dự đoán
│   ├── analyze_thesis.py   chọn số chia, mô phỏng, so sánh theo lượng nhãn, rò rỉ, chuyển miền
│   ├── analyze_metrics.py  RMSE, R² cho mọi mô hình; tỉ lệ dự đoán tăng ngược (tính đơn điệu)
│   ├── analyze_physics.py  phần vật lý tác động ở đâu: ∂F/∂t, phần dư ODE, tốc độ suy giảm học được
│   ├── error_decomp.py     phân rã sai số theo cell: lệch mức, trôi chậm, dao động nhanh
│   ├── dev_e20.py          E20dev: so 9 hướng cải tiến trên validation
│   ├── analyze_e20.py      E20: PINN-lai trên test, phân rã lợi ích, khoảng dự đoán phù hợp
│   ├── analyze_e21.py      E21: ma trận nguồn → đích, sai số qua các vòng học lặp
│   ├── analyze_e22.py      E22: sai số theo tầm dự báo, RUL, độ phủ khoảng 90 %
│   ├── analyze_e23.py      E23: so với E22 trên gốc thường và gốc ngay sau hồi phục
│   ├── analyze_e24.py      E24: NASA, có/không giờ nghỉ, so với SambaMixer
│   ├── verify_pipeline.py  21 phép kiểm cho pipeline (chia, lọc, chuẩn hoá)
│   └── section_e13.py      mục kết quả E13 (Markdown, slide)
├── report/                 results/ -> báo cáo
│   ├── make_thesis.py      do-an/main.tex + PDF; mọi con số đọc từ results/
│   ├── thesis_bk.py        khuôn đồ án Bách Khoa: bìa, lời cam đoan, sắp xếp 7 chương
│   ├── make_physics_doc.py tai-lieu/phan-tich-vat-ly-PINN (vật lý nằm ở đâu, R², RMSE)
│   └── diagrams.py         sơ đồ kiến trúc
├── tools/
│   ├── tai_du_lieu.py      tải dữ liệu
│   ├── kiem_tai_lap.py     kiểm tra tái lập từng bit (python run.py reproduce)
│   └── demo.py             ước lượng và dự báo SOH cho một cell (python run.py demo ...)
├── legacy/                 script giai đoạn đầu (E1–E12, pipeline v1, báo cáo HTML/LaTeX cũ)
├── slide/  tai-lieu/  do-an/  demo/
├── PINN4SOH/data/          dữ liệu (tải bằng python run.py data, không đưa lên git)
├── runs/                   kết quả từng lượt (JSON, không đưa lên git)
└── results/                bảng + hình tổng hợp (không đưa lên git)
```

## 4. Tối ưu và tái lập

Code được tối ưu mà **không đổi một bit kết quả nào** — `python run.py reproduce` kiểm đúng điều đó:

| Thay đổi | Tác dụng |
|---|---|
| Bộ nhớ đệm dữ liệu đã lọc (`.cache/`, tự dựng lại khi file nguồn hoặc tham số lọc đổi) | nạp 4 bộ dữ liệu 10,8 s → 0,02 s ở mỗi tiến trình và mỗi script |
| Tensor dựng một lần cho cả lượt; batch lấy bằng `index_select` | bỏ chép numpy → torch ở mỗi bước |
| Dừng sớm chỉ tính MAE validation (chỉ số đầy đủ tính một lần ở cuối) | bỏ phần tính theo cell/EOL 40 lần mỗi lượt |
| AdamW và cắt gradient dạng `foreach` | gộp phép tính trên các tensor tham số |
| Lượt dài chạy trước, mỗi tiến trình 1 luồng | các tiến trình kết thúc gần cùng lúc |
| Ghi kết quả nguyên tử | dừng giữa chừng không để lại file dở |
| Phân tích chỉ đọc file JSON mới hoặc đã đổi | `analyze` lần hai ~1 phút thay vì ~2 phút |
| Hình và PDF không ghi thời điểm tạo | dựng lại cho file giống từng byte, git không báo đổi |

Đo trên cùng máy, 1 luồng: một lượt MLP 13,6 s → 10,7 s, PINN-semi 34,6 s → 32,1 s, PINN hộp đen
28,5 s → 26,3 s. Bước huấn luyện của mạng nhỏ này đã bị giới hạn bởi chính phép tính (lan truyền xuôi/ngược
và đạo hàm bậc hai của ∂u/∂t), nên cách tăng tốc chính trên máy cá nhân là chạy song song nhiều tiến trình.
Hai hướng đã thử nhưng không dùng: gộp ba lần lan truyền xuôi thành một (chậm hơn gấp đôi vì đạo hàm bậc hai
phải đi qua cả batch gộp) và tính ∂u/∂t bằng đạo hàm xuôi (nhanh hơn ~10 % nhưng lệch ở mức làm tròn, mất
tính tái lập từng bit).

**Tái lập từng bit** được bảo đảm khi cùng phiên bản thư viện (`requirements.txt`) và cùng loại CPU. Khác
CPU (ví dụ AVX-512 so với AVX2, Apple Silicon) hoặc khác phiên bản torch có thể lệch ở mức làm tròn số thực;
khi đó `reproduce` báo độ lệch, các con số trong bảng có thể khác ở chữ số thứ ba–thứ tư nhưng kết luận
không đổi.

### Bộ NASA (E24)

```bash
mkdir -p NASA/raw && curl -o NASA/raw/nasa.zip "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
cd NASA/raw && unzip nasa.zip && cd "5. Battery Data Set" && for z in *.zip; do unzip -o "$z" -d "x_${z%.zip}"; done; cd ../../..
python -m pinnsoh.nasa NASA/raw        # -> PINN4SOH/data/NASA data/<nhóm>/B00xx.csv (18 cell)
python run.py E24 && python run.py E24f
```

## 5. Demo — ước lượng và dự báo SOH cho một cell

```bash
python run.py demo train --dataset XJTU          # PINN-semi, 30 % nhãn -> demo/XJTU.pt (~1 phút)
python run.py demo predict --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-8.csv"
python run.py demo predict --model demo/XJTU.pt --csv "..." --an-nhan   # coi như cell chưa đo dung lượng

# dự báo từ chu kỳ 150: SOH và dung lượng (Ah) sau 0,1L / 0,25L / 0,5L chu kỳ, số chu kỳ còn lại tới ngưỡng
# cuối đời, khoảng 90 %; bỏ --den-chu-ky để dự báo từ chu kỳ cuối cùng của file
python run.py demo forecast --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-8.csv" --den-chu-ky 150
python run.py demo forecast --model demo/XJTU.pt --csv "..." --den-chu-ky 150 --an-nhan   # lịch sử SOH ước lượng
```

Ước lượng ở chu kỳ N chỉ dùng dữ liệu tới chu kỳ N nên chạy trực tuyến được. `forecast` dùng bản có xét hồi phục
(E23) và in thêm phần hồi phục hiện tại, thời gian tắt τ và dao động đều nếu có. Lần đầu chạy `forecast`, bộ dự báo
được huấn luyện trên các cell có nhãn của cùng fold với mô hình ước lượng (vài phút) và lưu ở `demo/<bộ>_dubao.pkl`.
Đầu ra: `demo/<cell>_soh.csv`, `demo/<cell>_dubao.csv` và các hình `.png`.

## 6. Giai đoạn đầu

Các thí nghiệm E1–E12 (pipeline v1), báo cáo HTML/LaTeX cũ và các script kiểm toán của tháng 9/2026 nằm ở
`legacy/` — xem `legacy/README.md`. Chúng vẫn chạy được (`python legacy/<tên>.py`) nhưng không thuộc báo cáo
đồ án.

## 7. Nguyên tắc của repo

1. **Không có số nào gõ tay.** Mọi bảng và mọi con số trong báo cáo đều sinh từ `results/*.csv`, và các CSV
   đó sinh từ `runs/**/*.json`.
2. **Chia theo cell, không theo chu kỳ.** Cell test không bao giờ xuất hiện trong train ở bất kỳ chu kỳ
   nào; chuẩn hoá chỉ dùng đặc trưng của cell train; bộ lọc ngoại lai chỉ dùng quá khứ của chính cell.
   E15 đo định lượng mức sai lệch khi vi phạm từng điều.
3. **So sánh ghép cặp.** Mọi mô hình dùng đúng cùng cách chia, cùng tập cell có nhãn, cùng normaliser và
   cùng hạt giống; khác biệt duy nhất là mô hình và hàm mất mát.
4. **Chốt tiêu chí trước khi xem test.** Đề cương E13 (`tai-lieu/de-cuong-E13.md`) được commit trước lượt
   huấn luyện đầu tiên; cấu hình E13 trong `pinnsoh/experiments.py` không sửa sau đó.
5. **Phát biểu đúng mức.** Báo cáo nêu cả kết quả bất lợi: GBDT tinh chỉnh tốt có sai số thấp hơn PINN-semi
   trong cùng một bộ dữ liệu ở phần lớn trường hợp; lợi ích của vật lý lớn ở mức nhãn thấp và nhỏ dần khi
   nhiều nhãn.

## 8. Nguồn dữ liệu

Bốn bộ dữ liệu công khai, lấy qua repo [`wang-fujin/PINN4SOH`](https://github.com/wang-fujin/PINN4SOH)
(commit `cc3cc50`). Bản quyền thuộc nhóm tác giả gốc của từng bộ.

| Bộ | Cell | Hoá học | Dung lượng danh định | Nhiệt độ |
|---|---:|---|---|---|
| XJTU | 55 | NCM | 2,0 Ah | 25 °C |
| TJU | 130 | NCA, NCM, NCM+NCA | 3,5 / 2,5 Ah | 25, 35, 45 °C |
| MIT | 125 | LFP | 1,1 Ah | 30 °C |
| HUST | 77 | LFP | 1,1 Ah | 30 °C |
| **Tổng** | **387** | | | |
