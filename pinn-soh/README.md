# Ước lượng SOH pin lithium-ion bằng PINN bán giám sát

Mã nguồn của Đồ án chuyên ngành HK261 — nhóm 15, và bản thảo bài báo phát triển từ đồ án.
Huỳnh Quốc Thắng, Khoa KH&KT Máy tính, Trường ĐH Bách khoa — ĐHQG-HCM.

Bài toán: ước lượng SOH (dung lượng hiện tại / dung lượng danh định) của từng cell lithium-ion, tại từng chu kỳ, từ 16
đặc trưng của đoạn sạc CC–CV, khi chỉ một phần cell được đo dung lượng. Mô hình là mạng nơ-ron có thông tin vật lý
(PINN): mạng nghiệm dự đoán SOH, gắn với động học suy giảm dạng xám luôn không âm; ba hàm mất mát vật lý không dùng nhãn
nên áp được lên cả cell chưa đo dung lượng (PINN-semi). Mô hình cải tiến PINN-lai lấy trung bình ba PINN-semi với GBDT
rồi làm trơn bằng bộ lọc xu thế nhân quả.

Mọi so sánh dùng một giao thức không rò rỉ: chia theo cell (kiểm định chéo 5 fold), bộ lọc ngoại lai chỉ dùng quá khứ
của chính cell, chuẩn hoá chỉ học trên cell train. Các thí nghiệm, trên 387 cell của bốn bộ dữ liệu công khai XJTU,
TJU, MIT, HUST:

| Mã | Câu hỏi |
|---|---|
| E13, E16, E17 | PINN-semi so với MLP, PINN có giám sát, PINN hộp đen, Ridge, GPR, GBDT ở 10–70 % cell có nhãn; đề cương chốt trước |
| E15 | kiểm toán rò rỉ: bốn cách chia và chuẩn hoá sai làm sai số thấp đi bao nhiêu |
| E14, E18, E19, E21 | chuyển miền giữa các bộ, thích nghi không nhãn, tự huấn luyện |
| E20 | PINN-lai: 3 PINN-semi + GBDT + bộ lọc xu thế, khoảng dự đoán 90 % |
| E22, E23 | dự báo SOH tương lai, dung lượng còn lại, RUL; dự báo có xét hồi phục dung lượng |
| E25, E32 | cập nhật ước lượng khi đo được dung lượng vài lần trong đời cell: bộ lọc Kalman (E25) và mô hình học cập nhật (E32) |
| E26 | λ và Ea của động học có được dữ liệu xác định không (TJU, 25/35/45 °C) |
| E27 | so với PINN4SOH (Wang và cộng sự, Nature Communications 2024) trên đúng cách chia của bài |
| E28 | ngân sách đo: đo thưa mọi cell train thay vì đo đầy đủ một phần cell |
| E29 | thêm lịch sử của cell vào đầu vào |
| E30 | CNN, LSTM, Transformer trên cửa sổ nhân quả các chu kỳ gần nhất |
| E33 | đưa lịch sử của cell vào dự báo: HiGBDT (GBDT có đặc trưng lịch sử), PINN-dự báo có lịch sử, nhánh theo đời và HiPINN lấy trung vị của ba nhánh, so với PINN-GBDT của E22, trong giao thức đã sửa các điểm rò rỉ còn lại của E22 |
| E34 | HiGBDT cho bài toán cập nhật từ lần đo dung lượng gần nhất (mã gộp từ kho G-doan): GBDT-cập nhật, HiGBDT-Δ, HiGBDT-R, HiGBDT-V trên bốn lịch đo |
| benchmark | chạy lại mã gốc của PINN4SOH trên cách chia của bài, so với số công bố và với E27 |
| features | đặc trưng đầu vào: dùng gì, bỏ gì (tương quan hạng, bỏ từng nhóm, hoán vị, chọn trên validation) |

Tổng cộng 10 695 lượt huấn luyện (đếm theo tệp kết quả của `pinnsoh/experiments.py`), cộng các lần chạy lại mã gốc
PINN4SOH ở `benchmark/` và phân tích đặc trưng. Ba sản phẩm viết:

* `report/bao-cao-do-an.pdf`: báo cáo đồ án chuyên ngành bản hoàn thiện, tiếng Việt, không còn phần bộ NASA
  (`python run.py report`; xem `report/README.md`);
* `paper/manuscript.pdf`: bản thảo bài báo tiếng Anh (`python run.py paper`; xem `paper/README.md`);
* `archive/thesis/do-an/bao-cao-do-an.pdf`: báo cáo đồ án đã nộp (có phần bộ NASA PCoE), giữ nguyên.

---

## 1. Cài đặt

Cần Python ≥ 3.10 (đã dùng 3.11). TeX Live chỉ cần cho hai bước dựng PDF (bản thảo bài báo và báo cáo đồ án), cả hai
dùng pdfLaTeX; báo cáo đồ án theo mẫu hcmut-thesis cần thêm `latexmk` và gói tiếng Việt vntex (Ubuntu:
`apt-get install latexmk texlive-latex-extra texlive-lang-other`).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Chạy lại

Mọi bước đều qua một lệnh `python run.py <lệnh>`, chạy từ thư mục nào cũng được.

```bash
python run.py check          # kiểm tra Python, thư viện, dữ liệu, pdfLaTeX, số lõi CPU
python run.py data           # tải 387 file CSV (~94 MB) từ repo gốc và dựng bộ nhớ đệm
python run.py reproduce      # ~2–5 phút: chạy lại 20 lượt mẫu, so từng bit với kết quả công bố
python -m unittest discover -s tests   # vài giây: đặc trưng chỉ dùng quá khứ của cell, lịch đo, HiPINN
python run.py bench          # đo tốc độ máy để ước lượng thời gian chính xác hơn
python run.py all --dry-run  # xem còn bao nhiêu lượt và mất bao lâu, không chạy gì
python run.py all            # toàn bộ: thí nghiệm -> phân tích -> benchmark -> đặc trưng -> bản thảo -> báo cáo
```

`all` = `experiments` (E13, E15–E23, E26–E30, E32, E33 theo thứ tự trong `pinnsoh/experiments.py`) → `analyze` →
`benchmark` → `features` → `paper` → `report`. Có thể chạy từng bước:

```bash
python run.py E13 --workers 8          # một thí nghiệm, 8 tiến trình song song
python run.py E16 --datasets XJTU      # chỉ một bộ dữ liệu
python run.py experiments --quick      # chạy thử: 1 lần lặp thay vì 5, 1 hạt giống thay vì 5
python run.py analyze                  # runs/ -> results/ (bảng CSV, hình PNG/PDF)
python run.py benchmark                # mã gốc PINN4SOH trên cách chia của bài -> benchmark/results/
python run.py features --workers 4     # phân tích đặc trưng (GBDT, 120 fold) -> results/features_*.csv
python run.py paper                    # -> paper/manuscript.pdf
python run.py report                   # -> report/bao-cao-do-an.pdf
```

* **Dừng lúc nào cũng được** (Ctrl+C, tắt máy). Mỗi lượt ghi một file JSON trong `runs/`, ghi nguyên tử; chạy lại
  đúng lệnh cũ sẽ bỏ qua lượt đã xong và chạy tiếp.
* `--workers N`: mặc định bằng số lõi CPU. Mỗi tiến trình dùng 1 luồng — với mạng nhỏ thế này, N tiến trình × 1 luồng
  nhanh hơn nhiều so với 1 tiến trình × N luồng.
* `--quick` cho tập con của bản đầy đủ (lần lặp 0, hạt giống 0); chạy bản đầy đủ sau đó sẽ không phải chạy lại phần
  đã có. Kết quả `--quick` chỉ để kiểm tra, không phải số liệu công bố.
* `--device cuda`: dùng GPU. Kết quả lệch ở mức làm tròn số thực nên không tái lập từng bit.
* Dữ liệu để ở chỗ khác: đặt biến môi trường `PINNSOH_DATA=<thư mục chứa "XJTU data", "TJU data", ...>`. Không tải
  tự động được (mạng chặn GitHub): tải zip của repo `wang-fujin/PINN4SOH` rồi `python tools/tai_du_lieu.py --zip <file.zip>`.

### Thời gian

Ước lượng trên CPU của máy chạy thử, 1 luồng mỗi tiến trình. `python run.py bench` đo hệ số cho máy của bạn và
`--dry-run` sẽ dùng hệ số đó.

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
| E22 | Dự báo SOH tương lai, dung lượng, RUL: GBDT + 3 PINN-dự báo mỗi fold (cần E20) | 60 | 0,6 | 0,2 | 0,1 |
| E23 | Dự báo có xét hồi phục, cùng fold E22, học lại cả mô hình E22 để so ghép cặp | 60 | 3 | 0,8 | 0,4 |
| E26 | λ, Ea trên TJU: trọng số ODE 0,2/1, ràng buộc Euler, khởi tạo khác (PINN-semi, 70 % nhãn) | 25 | 0,2 | 0,1 | 0,1 |
| E27 | Cách chia của PINN4SOH (39 nhóm, đầy đủ + ít mẫu), 2 cách tiền xử lý × 3 hạt giống: 3 PINN-semi, GBDT, PINN hộp đen | 1 002 | 7,9 | 2,0 | 1,0 |
| E28 | Ngân sách đo: mọi cell train đo mỗi q·L, q ∈ {0,5; 0,2; 0,1; 0,05} (3 lần lặp × 5 fold) | 1 560 | 9 | 2,2 | 1,1 |
| E29dev | Chọn dạng lịch sử trên validation (lần lặp 0, nhãn 30 và 70 %): PINN-semi 4 dạng, GBDT 6 dạng | 400 | 2,2 | 0,6 | 0,3 |
| E29 | Đầu vào có lịch sử, cùng fold với E20: 3 PINN-semi, MLP, GBDT trên XJTU, TJU, HUST | 840 | 5,4 | 1,4 | 0,7 |
| E30dev | Chọn độ dài cửa sổ của CNN, LSTM, Transformer trên validation (L ∈ {16, 48}) | 240 | 1,8 | 0,5 | 0,2 |
| E30 | CNN, LSTM, Transformer cùng fold với E20 (3 lần lặp × 5 fold × 4 mức nhãn × 4 bộ) | 720 | 3,7 | 0,9 | 0,5 |
| E32 | GBDT-cập nhật: học cập nhật từ lần đo gần nhất, cùng fold với E20 (4 bộ × 4 mức nhãn) | 240 | 6,7 | 1,7 | 0,8 |
| E33 | Dự báo có đặc trưng lịch sử (HiGBDT, HiPINN và các nhánh), cùng fold với E22, 2 ngân sách nhãn (cần E20) | 120 | 6,8 | 1,7 | 0,8 |
| E34 | HiGBDT cho bài toán cập nhật: GBDT-cập nhật, HiGBDT-Δ, -R, -V trên 4 lịch đo, 30 % nhãn (3 lần lặp × 5 fold) | 60 | 1,1 | 0,3 | 0,2 |
| benchmark | Mã gốc PINN4SOH, 39 nhóm: bản công bố 10 lần, bản đã sửa 3 lần mỗi nhóm | 507 | ≈ 6 | ≈ 1,5 | ≈ 0,8 |
| **Tổng** | | **11 102** | **≈ 96** | **≈ 24** | **≈ 12** |

E20 gồm 1 440 lượt, trong đó 160 lượt trùng cấu hình với E20dev: chép `runs/E20dev/P_*` và `runs/E20dev/gbdt_*` sang
`runs/E20/` trước thì chỉ còn 1 280 lượt mới (bảng tính theo con số này). Cấu hình E20 chốt trong `docs/de-cuong-E20.md`.

`--quick` cần khoảng 1/5 thời gian trên. Phân tích (`analyze`) mất ~5 phút lần đầu cộng ~40 phút CPU để
`analyze_physics.py` huấn luyện lại 120 lượt (đo đạo hàm và tốc độ suy giảm cần chính mô hình, mà trọng số không được
lưu; huấn luyện tất định nên mô hình giống từng bit lượt đã công bố); các lần sau ~2 phút nhờ bộ nhớ đệm.

## 3. Cấu trúc

```
pinn-soh/
├── run.py                  lệnh duy nhất cho mọi bước
├── pinnsoh/                thư viện
│   ├── data.py             đọc 387 CSV -> Cell; lọc ngoại lai nhân quả; chia theo cell (cv_split); chuẩn hoá;
│   │                       bộ nhớ đệm .cache/
│   ├── models.py           mạng nghiệm F_φ; động học xám r = softplus(G_θ(x))·e^{λ(1-u)}·A(T); hộp đen
│   ├── train.py            Config, bốn hàm mất mát, huấn luyện + dừng sớm
│   ├── baselines.py        Ridge, GBDT, GPR
│   ├── seqmodels.py        CNN, LSTM, Transformer trên cửa sổ nhân quả các chu kỳ gần nhất của cell (E30)
│   ├── postproc.py         bộ lọc quỹ đạo nhân quả của PINN-lai (E20)
│   ├── metrics.py          MAE theo cell, R², EOL, vi phạm đơn điệu, ...
│   ├── leakage.py          các cách chia/chuẩn hoá sai cho kiểm toán rò rỉ (E15)
│   ├── transfer.py         huấn luyện chuyển miền nguồn -> đích (E14, E18)
│   ├── crossds.py          huấn luyện một bộ, kiểm tra bộ khác; thích nghi + tự huấn luyện (E21)
│   ├── forecast.py         dự báo SOH tương lai, dung lượng còn lại, RUL (E22)
│   ├── recovery.py         dự báo có xét hồi phục dung lượng (E23)
│   ├── paper.py            cách chia của PINN4SOH, GBDT trên cách chia đó (E27)
│   ├── budget.py           ngân sách đo dung lượng (E28)
│   ├── update.py           mô hình học cập nhật từ lần đo dung lượng gần nhất (E32)
│   ├── forecast_hist.py    GBDT có đặc trưng lịch sử SOH (E33; bản phát triển ở kho G-doan của đồ án)
│   ├── forecast_life.py    nhánh dự báo theo đời: đời còn lại dự đoán + khuôn quỹ đạo của cell train (E33)
│   ├── forecast_combo.py   mô hình HiPINN và các nhánh, giao thức E33
│   ├── hipinn.py           HiPINN học một lần rồi dự báo cho một cell bất kỳ (dùng trong tools/demo.py)
│   ├── update_hist.py      HiGBDT cho bài toán cập nhật từ lần đo gần nhất (E34, gộp từ kho G-doan)
│   ├── pinn4soh_splits.json, pinn4soh_reported.csv   cách chia và số công bố của PINN4SOH
│   ├── experiments.py      định nghĩa E13–E33 thành danh sách lượt chạy
│   ├── runner.py           chạy song song, bỏ qua lượt đã xong, ước lượng thời gian
│   ├── runcache.py         nhớ kết quả đọc runs/*.json giữa các lần phân tích
│   └── plotstyle.py        khuôn vẽ hình
├── analysis/               runs/ -> results/: một script cho mỗi thí nghiệm (analyze_eXX.py, dev_eXX.py cho giai
│                           đoạn chọn trên validation), cùng các phân tích chung (chỉ số, phần vật lý, phân rã sai số,
│                           tính đơn điệu, hình tổng quan, figures_report.py vẽ lại hình cho bản in, diagrams.py sơ đồ,
│                           features.py phân tích đặc trưng đầu vào)
├── benchmark/
│   ├── rerun_pinn4soh.py   chạy lại mã gốc PINN4SOH (nguyên trạng) trên 39 nhóm chia của bài, hai bản mã
│   ├── compare.py          bảng so sánh: số công bố, chạy lại, mô hình của đồ án (E27)
│   └── results/            kết quả so sánh theo từng cell và kiểm định (nhỏ, đưa lên git)
├── report/                 báo cáo đồ án chuyên ngành (LaTeX tiếng Việt), bảng và hình sinh tự động, script dựng
├── paper/                  bản thảo bài báo (LaTeX), hình và script dựng
├── docs/                   đề cương chốt trước (de-cuong-E13, E20, E27-E28, E29, E30, E32), phương pháp, chứng minh,
│                           phân tích phần vật lý, kết quả E13, các phương trình (equations.tex)
├── tools/
│   ├── tai_du_lieu.py      tải dữ liệu
│   ├── kiem_tai_lap.py     kiểm tra tái lập từng bit (python run.py reproduce)
│   ├── make_physics_doc.py docs/phan-tich-vat-ly-PINN (vật lý nằm ở đâu, R², RMSE)
│   └── demo.py             ước lượng và dự báo SOH cho một cell (python run.py demo ...)
├── demo/                   mô hình demo đã huấn luyện và ví dụ đầu ra
├── archive/                báo cáo đồ án đã nộp, slide, pipeline v1 — giữ nguyên như lúc nộp (archive/README.md)
├── PINN4SOH/data/          dữ liệu (tải bằng python run.py data, không đưa lên git)
├── runs/                   kết quả từng lượt (JSON, không đưa lên git)
└── results/                bảng + hình tổng hợp (không đưa lên git)
```

## 4. So với PINN4SOH (benchmark)

E27 huấn luyện mô hình của đồ án trên đúng 39 nhóm chia của PINN4SOH (dựng lại từ dòng `data path:` trong nhật ký
huấn luyện công bố kèm mã), với hai cách tiền xử lý: `strict` (không rò rỉ, của đồ án) và `paper` (bản sao tiền xử lý
của bài: lọc 3σ trên cả vòng đời kể cả cột dung lượng, min–max theo từng cell kể cả cell test).

`benchmark/rerun_pinn4soh.py` chạy lại chính mã của bài (lớp PINN và lớp đọc dữ liệu import nguyên trạng) trên cùng
39 nhóm, 10 lần chạy mỗi nhóm như bài, để có dự đoán theo từng cell và so ghép cặp được. Hai bản mã:

* `published` — bản tải lên đầu tiên (commit `5d57a9d`), siêu tham số lấy từ nhật ký công bố. Ở bản này loss đơn điệu
  là `relu(y2 − y1).sum()`, chỉ tính từ nhãn nên không có gradient. Ở cả 39 nhóm × 10 lần chạy, nhật ký công bố ghi
  "physics loss" là một hằng số qua mọi epoch, và ở cả 39 nhóm hằng số đó bằng đúng biểu thức này tính trên các hàng
  train mà dataloader gốc dựng từ dữ liệu ở đây (`runs/pinn4soh/logged_constant.json`; ví dụ 0,043214 ở XJTU 2C). Số
  liệu công bố của bài vì vậy đến từ mô hình không có ràng buộc đơn điệu, và cách chia dựng lại cùng bản dữ liệu ở đây
  cho đúng các hàng train của mọi nhóm công bố.
* `current` — bản hiện tại của repo (commit `cc3cc50`), loss đơn điệu đã sửa thành `relu((u2 − u1)(y1 − y2)).sum()`,
  α và β theo giá trị mặc định mới trong `main_*.py` (`fix.md` của repo: sau khi sửa lỗi, α và β khác Supplementary).

Mã gốc viết cho pandas 1.3.5 và lỗi kiểu dữ liệu với pandas 3; `python run.py benchmark` tự dựng một môi trường ảo
có pandas 2.3.3 (dùng chung các gói khác) khi cần. Kết quả, nhỏ nên được đưa lên git: `benchmark/results/summary.csv`
(theo bộ × chế độ), `cells.csv` (theo cell và lần chạy), `tests.csv` (Wilcoxon ghép cặp theo cell), `table.md` (bảng
đọc nhanh), `logged_constant.json` (đối chiếu nhật ký công bố).

Đọc kết quả (số cụ thể ở `benchmark/results/table.md` và mục 5.4 của `paper/manuscript.pdf`, đều sinh từ dữ liệu):

* Chạy lại bản công bố cho sai số hơi cao hơn số công bố, cùng cỡ với độ lệch giữa các lần chạy; dữ liệu train trùng
  từng hàng, phần chênh còn lại do phần cứng, phiên bản thư viện, hạt giống.
* Cùng cách chia và cùng tiền xử lý của bài, PINN-lai của đồ án có sai số thấp hơn mã gốc chạy lại ở cả 12 chế độ (có ý
  nghĩa ở 7). GBDT cùng tiền xử lý tốt ngang PINN-lai (thấp hơn có ý nghĩa ở 2 chế độ ít mẫu của HUST, không chế độ
  nào cao hơn): với min–max theo cell, chỉ số chu kỳ đã chuẩn hoá là phần đời đã qua, mô hình cây khai thác thẳng điều
  đó, nên lợi thế ở điều kiện này không quy được cho phần vật lý.
* Với tiền xử lý không rò rỉ, so với số công bố: thấp hơn nhiều trên TJU và MIT, ngang trên XJTU, cao hơn trên HUST —
  đúng chiều mà kiểm toán rò rỉ dự đoán.

### Đặc trưng đầu vào (thí nghiệm F)

`analysis/features.py` kiểm tra vì sao mọi mô hình dùng 16 đặc trưng đoạn sạc cùng chỉ số chu kỳ `t = n/1000`: GBDT
(cùng cấu hình E16, trùng từng bit với E16 khi dùng đủ đầu vào) được huấn luyện lại với từng tập đầu vào trên đúng các
fold không rò rỉ (4 bộ × 2 mức nhãn 10 và 70 % × 3 lần lặp × 5 fold), cộng tương quan hạng Spearman, trùng lặp, độ quan
trọng hoán vị và chọn đặc trưng trên validation. Kết quả (`results/features_*.csv`, Chương 2 của báo cáo):

* Bỏ `t`, bỏ một trong ba nhóm (điện áp pha CC, dòng pha CV, điện lượng và thời gian), hay chỉ giữ một phần: cả 6 tập
  rút gọn đều kém bộ đầy đủ có ý nghĩa ở ít nhất một trường hợp; chỉ dùng `t` kém 1,2–12 lần.
* Chọn đặc trưng tự động trên validation giữ 7–17 đầu vào nhưng sai số bằng 0,99–1,02 lần bộ đầy đủ (không khác có ý
  nghĩa ở trường hợp nào), nên giữ đủ 17 đầu vào.
* Có trường hợp ngược chiều và báo cáo nêu rõ: bỏ `t` có lợi trên MIT ở 10 % nhãn; 16 đặc trưng tương đối có lợi trên
  TJU ở 70 % nhãn nhưng có hại trên MIT; thêm nhiệt độ trên TJU thay đổi sai số dưới 1 %.
* Chuẩn hoá theo cả vòng đời cell, lọc ngoại lai dùng cả vòng đời và fit normaliser trên cell test bị loại vì là rò rỉ
  (kiểm toán E15).

### Dự báo so với SOH đo được (E22, không còn trong báo cáo)

`pinnsoh/forecast.py`: tại gốc k, GBDT theo tầm và PINN-dự báo (mạng trả hai tham số của nghiệm đóng của động học xám)
dự báo SOH ở mọi chu kỳ sau k; PINN-GBDT lấy trung bình hai nhánh, rồi suy ra dung lượng, RUL và khoảng 90 %. Cell
test không được đo dung lượng: lịch sử tới gốc là ước lượng PINN-lai (E20, 70 % nhãn), SOH đo được chỉ dùng để chấm.
Báo cáo đã nộp có mục dự báo này; báo cáo hiện tại thay nó bằng E33 (mục dưới), chạy trên cùng fold, với PINN-GBDT
học lại làm mốc so sánh. Bảng và hình của E22 vẫn tính lại được từ `runs/E22`.

```bash
python run.py E20       # E22 chỉ cần các lượt PINN-semi và GBDT ở 70 % nhãn
python run.py E22
python analysis/analyze_e22.py                 # bảng results/e22_*.csv
python analysis/figures_report.py e22_traj     # và e22_cells, e22_origins, e22_parity, e22_horizon, e22_rul
```

* Chạy lại toàn bộ (240 lượt E20 ở 70 % nhãn, 60 lượt E22): 188/188 ô của ba bảng dự báo trùng với báo cáo đã nộp.
* Trên phần đời còn lại, sai số của PINN-GBDT bằng 0,21–0,46 lần giữ nguyên, 0,39–0,60 lần ngoại suy tuyến tính
  và 0,27–0,58 lần khớp vật lý theo cell; so với GBDT là 0,91–1,06. Ở tầm 0,5L, sai số quy ra dung lượng là 20–57 mAh.
* Chưa tốt: RUL nằm trong ±20 % RUL thực ở 38–65 % số gốc; khoảng RUL 90 % chỉ phủ 65–84 %. Từ gốc ở 20 % đời,
  nghiệm đóng có thể đi qua điểm kỳ dị và kéo dự báo xuống giới hạn dưới 0,40 (2–5 % số quỹ đạo).

### Dự báo có đặc trưng lịch sử: HiGBDT và HiPINN (E33)

Hai mô hình dự báo có đặc trưng lịch sử của đồ án: HiGBDT, và HiPINN lấy trung vị của ba nhánh tại từng (gốc, tầm):

* HiGBDT (`pinnsoh/forecast_hist.py`, phát triển trong kho G-doan của đồ án): GBDT dự báo độ giảm SOH với đặc
  trưng gốc và đặc trưng lịch sử SOH (trung bình, độ lệch chuẩn, độ dốc trên nhiều cửa sổ; thay đổi theo độ trễ);
* PINN-dự báo của E22 (23 đặc trưng gốc, nghiệm đóng của đường suy giảm);
* nhánh theo đời (`pinnsoh/forecast_life.py`): GBDT dự đoán đời còn lại, rồi đặt khuôn quỹ đạo của K cell train gần nhất
  lên đời dự đoán.

Trung vị không có tham số cần chọn; khi một nhánh sai xa, nó đi theo hai nhánh còn lại. Phương án đầu HiPINN-w (tổng
có trọng số, trọng số chọn trên nửa validation dùng để chọn, chỉ 3–7 cell mỗi lượt) không giữ được ưu thế trên nửa còn
lại; dạng trung vị được chọn sau khi đã xem kết quả test của HiPINN-w, nên là kết quả sau thí nghiệm. E33 vẫn lưu
HiPINN-w (`hipinn_w`), HiPINN-tb là trung bình đều ba nhánh (`tb3`), PINN-dự báo có lịch sử (`pinn_ls`) và sai số theo
cell trên hai nửa validation (mảng `vcell`) để đối chiếu. Mô hình dự báo của E22 là PINN-GBDT (½ GBDT theo tầm +
½ PINN-dự báo, cột `ket_hop`).

Giao thức (`pinnsoh/forecast_combo.py`): cùng fold với E22; cell validation chia đôi theo tên, một nửa để chọn cấu hình
từng nhánh, nửa còn lại để hiệu chỉnh khoảng 90 % và so các mô hình trên dữ liệu không thuộc test. Thang đời L và chuẩn
hoá chỉ tính trên cell train, gốc dự báo ở 0,2L, 0,4L, 0,6L (E22 tính L trên mọi cell và đặt gốc theo số hàng của cell
test). Hai ngân sách nhãn: 30 % số cell và mọi cell train; lịch sử SOH tới gốc là dung lượng đo, hoặc ước lượng PINN-lai
như E22.

```bash
python run.py data                      # tải dữ liệu nếu chưa có
python run.py E20 --nhan 0.7            # E33 chỉ cần E20 ở 70 % nhãn (360 lượt, khoảng 2 giờ CPU)
python run.py E33                       # 120 lượt, khoảng 8 giờ CPU (2 giờ với 4 lõi)
python run.py demo tat-ca               # HiPINN cho một cell test của mỗi bộ -> demo/hipinn_bon_bo.*
GDOAN=<thư mục kho G-doan> python run.py report   # runs/E33 đủ 120 lượt thì tính lại bảng và hình
```

Kết quả của mỗi lượt E33 nặng 4–7 MB (cả thí nghiệm khoảng 600 MB) nên không đưa lên git; `report/generated/data/`
giữ các bảng đã tổng hợp, đủ để dựng báo cáo mà không cần `runs/`.

Kết quả (Mục 5.2.10 và Phụ lục A của báo cáo; MAE theo cell, ×10⁻³):

* HiGBDT chạy lại ở 30 % nhãn với lịch sử đo: XJTU 17,34, TJU 7,07, MIT 10,24, HUST 12,78; lần chạy trước ở kho G-doan
  là 18,23, 7,21, 10,11, 12,56 (tương quan theo cell 0,931–0,982; chênh do phiên bản scikit-learn và NumPy). Đặc trưng
  lịch sử giúp GBDT ở mọi trường hợp: HiGBDT bằng 0,85–0,94 lần GBDT theo tầm khi lịch sử đo.
* HiPINN có MAE thấp nhất trong tám mô hình của bảng ở 11/12 trường hợp (bộ × thiết lập). So với HiGBDT, mô hình
  đứng riêng tốt nhất: 0,86–1,01 lần, tốt hơn có ý nghĩa ở 10/12 trường hợp, 62–82 % số cell có sai số thấp hơn. So với
  PINN-GBDT của E22 khi lịch sử đo: 0,62–0,91 lần, tốt hơn có ý nghĩa ở 8/8.
* HiPINN-w (trọng số chọn trên 3–7 cell validation) có MAE cao hơn HiGBDT ở 7/8 trường hợp lịch sử đo; HiPINN thấp hơn
  HiPINN-w và HiPINN-tb (trung bình ba nhánh) ở 12/12. Dạng trung vị được chọn sau khi xem kết quả test của HiPINN-w.
* Kiểm tra trên nửa hiệu chỉnh (không thuộc test, chưa xem khi chọn trung vị): HiPINN thấp hơn HiPINN-w ở 7/8 trường hợp;
  thấp hơn HiGBDT ở 4/4 bộ khi học trên mọi cell train (3 có ý nghĩa) nhưng chỉ 2/4 bộ với 30 % nhãn.
* Phần cuối đời (10 % đời cuối, MIT, mọi cell train, lịch sử đo): PINN-GBDT 26,2, HiPINN 20,6, HiGBDT 20,8, GBDT theo
  tầm 19,2 (thấp nhất); dự báo vẫn cao hơn thực tế ở đoạn này (độ lệch +12,7 với HiPINN, +15,3 với HiGBDT).
* PINN-dự báo có lịch sử bằng 0,87–1,11 lần bản không lịch sử, chỉ tốt hơn có ý nghĩa trên HUST, nên HiPINN giữ
  PINN-dự báo của E22.
* RUL (mọi cell train): HiPINN bằng 0,90–1,70 lần PINN-GBDT khi lịch sử đo (thấp hơn có ý nghĩa trên TJU và MIT; trên
  HUST cao hơn, không có ý nghĩa) và 0,34–0,80 lần HiGBDT, vốn dự báo RUL dài hơn thực tế trên MIT và HUST.
* Khoảng SOH 90 % của HiPINN phủ 89–93 % khi lịch sử đo, 69–90 % khi lịch sử ước lượng.

### HiGBDT cho bài toán cập nhật (E34)

`pinnsoh/update_hist.py` (gộp từ kho G-doan): ở chu kỳ không đo, ước lượng SOH từ lần đo dung lượng gần nhất trước nó.
GBDT-cập nhật dùng đặc trưng của E32; HiGBDT-Δ và HiGBDT-R thêm hai trung bình trượt mũ nhân quả của đặc trưng sạc,
học độ giảm SOH hoặc phần dư so với ngoại suy tuyến tính; HiGBDT-V chọn trên validation, theo từng lịch đo, một trong ba
mô hình và trọng số trộn với giữ lần đo hoặc ngoại suy. Lịch đo: chu kỳ đầu rồi mỗi 0,2L / 0,1L / 0,05L (q20, q10,
q05), hoặc biết mọi lần đo trước (M2). Cùng fold, 30 % nhãn, L và chuẩn hoá từ cell train.

```bash
python run.py E34                       # 60 lượt, khoảng 17 phút với 4 lõi
python analysis/analyze_e34.py          # results/e34_*.csv, results/fig_e34_cost.pdf
```

Kết quả (Phụ lục A của báo cáo; MAE theo cell, ×10⁻³):

* Lịch q10, XJTU / TJU / MIT / HUST: giữ lần đo 11,79 / 12,66 / 6,46 / 13,35; GBDT-cập nhật 3,82 / 1,87 / 0,91 / 1,09;
  HiGBDT-V 3,91 / 2,21 / 0,92 / 1,09.
* Khi đo thưa, lịch sử không giúp: HiGBDT-V thấp hơn GBDT-cập nhật ở 1/4 bộ với q20 và q10, kém hơn có ý nghĩa trên
  TJU ở cả ba lịch đo thưa; chỉ thấp hơn trên cả bốn bộ khi biết mọi lần đo trước (M2). Vì vậy giữ GBDT-cập nhật.
* GBDT-cập nhật ở lịch q20 (5,6–6,6 lần đo mỗi cell) có MAE thấp hơn giữ lần đo ở lịch q05 (21–25 lần đo) trên cả bốn
  bộ, tức đo ít hơn 3,7–3,8 lần.
* So với lần chạy trước ở kho G-doan trên cùng cell và lịch đo: tương quan sai số theo cell 0,992–0,999
  (GBDT-cập nhật) và 0,945–0,993 (HiGBDT-V); phần chênh do phiên bản thư viện.

## 5. Tối ưu và tái lập

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

**Tái lập từng bit** được bảo đảm khi cùng phiên bản thư viện (`requirements.txt`) và cùng loại CPU. Khác CPU (ví dụ
AVX-512 so với AVX2, Apple Silicon) hoặc khác phiên bản torch có thể lệch ở mức làm tròn số thực; khi đó `reproduce`
báo độ lệch, các con số trong bảng có thể khác ở chữ số thứ ba–thứ tư nhưng kết luận không đổi.

Đã kiểm trên toàn bộ E27: chạy lại cả 1 002 lượt cho `benchmark/results/summary.csv` và `tests.csv` trùng từng byte với bản sinh từ lần chạy đầu, và `cells.csv` (6 chữ số có nghĩa) trùng từng dòng.

## 6. Demo — ước lượng và dự báo SOH cho một cell

```bash
python run.py demo train --dataset XJTU          # PINN-semi, 30 % nhãn -> demo/XJTU.pt (~1 phút)
python run.py demo predict --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-8.csv"
python run.py demo predict --model demo/XJTU.pt --csv "..." --an-nhan   # coi như cell chưa đo dung lượng

# dự báo từ chu kỳ 150: SOH và dung lượng (Ah) sau 0,1L / 0,25L / 0,5L chu kỳ, số chu kỳ còn lại tới ngưỡng
# cuối đời, khoảng 90 %; bỏ --den-chu-ky để dự báo từ chu kỳ cuối cùng của file
python run.py demo forecast --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-8.csv" --den-chu-ky 150
python run.py demo forecast --model demo/XJTU.pt --csv "..." --den-chu-ky 150 --an-nhan   # lịch sử SOH ước lượng

# cập nhật: cell được đo dung lượng ở vài chu kỳ (lịch đo mỗi 0,1L, hoặc --do-chu-ky "1,60,120"); HiGBDT-V (E34)
# ước lượng SOH ở mọi chu kỳ còn lại, mỗi chu kỳ chỉ dùng các lần đo trước nó
python run.py demo cap-nhat --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-8.csv" --lich q10
python run.py demo tat-ca                        # HiPINN cho một cell test của mỗi bộ, hình 2 × 2
```

Ước lượng ở chu kỳ N chỉ dùng dữ liệu tới chu kỳ N nên chạy trực tuyến được. `forecast` dùng HiPINN (trung vị của GBDT
lịch sử, PINN-dự báo và nhánh theo đời); lần đầu cho một bộ, HiPINN được học trên các cell có nhãn của cùng fold với mô
hình ước lượng (vài phút) và lưu ở `demo/<bộ>_hipinn.pkl`. `--mo-hinh e23` dùng bản có xét hồi phục của E23 (lưu ở
`demo/<bộ>_dubao.pkl`), in thêm phần hồi phục hiện tại, tỉ lệ được giữ lại, thời gian tắt của phần còn lại và dao động
đều nếu có. `tat-ca` chạy HiPINN cho một cell test của mỗi bộ (cell có số chu kỳ ở trung vị, dự báo từ 40 % số chu kỳ
của nó) và vẽ hình 2 × 2. `cap-nhat` dùng HiGBDT-V của E34, học trên cùng fold với PINN-semi của công cụ (lưu ở
`demo/<bộ>_capnhat.pkl`). Đầu ra: `demo/<cell>_soh.csv`, `demo/<cell>_dubao.csv`, `demo/<cell>_capnhat.csv`,
`demo/hipinn_bon_bo.csv` và các hình.

## 7. Nguyên tắc của repo

1. **Không có số nào gõ tay.** Mọi bảng và mọi con số đều sinh từ `results/*.csv`, và các CSV đó sinh từ
   `runs/**/*.json`.
2. **Chia theo cell, không theo chu kỳ.** Cell test không bao giờ xuất hiện trong train ở bất kỳ chu kỳ nào; chuẩn hoá
   chỉ dùng đặc trưng của cell train; bộ lọc ngoại lai chỉ dùng quá khứ của chính cell. E15 đo định lượng mức sai lệch
   khi vi phạm từng điều.
3. **So sánh ghép cặp.** Mọi mô hình dùng đúng cùng cách chia, cùng tập cell có nhãn, cùng normaliser và cùng hạt
   giống; khác biệt duy nhất là mô hình và hàm mất mát.
4. **Chốt tiêu chí trước khi xem test.** Đề cương E13 (`docs/de-cuong-E13.md`) được commit trước lượt huấn luyện đầu
   tiên; cấu hình E13 trong `pinnsoh/experiments.py` không sửa sau đó. Các phân tích thêm sau khi đã xem test được ghi
   rõ là phân tích phụ trong đề cương tương ứng.
5. **Phát biểu đúng mức.** Nêu cả kết quả bất lợi: GBDT tinh chỉnh tốt có sai số thấp hơn PINN-semi trong cùng một bộ
   dữ liệu ở phần lớn trường hợp; lợi ích của vật lý lớn ở mức nhãn thấp và nhỏ dần khi nhiều nhãn.

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
