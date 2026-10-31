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
| benchmark | chạy lại mã gốc của PINN4SOH trên cách chia của bài, so với số công bố và với E27 |

Tổng cộng 10 575 lượt huấn luyện (đếm theo tệp kết quả của `pinnsoh/experiments.py`), cộng các lần chạy lại mã gốc
PINN4SOH ở `benchmark/`. Bản thảo bài báo: `paper/main.pdf` (dựng bằng `python run.py paper`). Báo cáo đồ án đã nộp
(có thêm phần bộ NASA PCoE): `archive/thesis/do-an/bao-cao-do-an.pdf`.

---

## 1. Cài đặt

Cần Python ≥ 3.10 (đã dùng 3.11). XeLaTeX (TeX Live / MiKTeX) chỉ cần cho bước dựng bản thảo PDF.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Chạy lại

Mọi bước đều qua một lệnh `python run.py <lệnh>`, chạy từ thư mục nào cũng được.

```bash
python run.py check          # kiểm tra Python, thư viện, dữ liệu, XeLaTeX, số lõi CPU
python run.py data           # tải 387 file CSV (~94 MB) từ repo gốc và dựng bộ nhớ đệm
python run.py reproduce      # ~2–5 phút: chạy lại 20 lượt mẫu, so từng bit với kết quả công bố
python run.py bench          # đo tốc độ máy để ước lượng thời gian chính xác hơn
python run.py all --dry-run  # xem còn bao nhiêu lượt và mất bao lâu, không chạy gì
python run.py all            # toàn bộ: thí nghiệm -> phân tích -> benchmark -> bản thảo PDF
```

`all` = `experiments` (E13, E15–E23, E26–E30, E32 theo thứ tự trong `pinnsoh/experiments.py`) → `analyze` →
`benchmark` → `paper`. Có thể chạy từng bước:

```bash
python run.py E13 --workers 8          # một thí nghiệm, 8 tiến trình song song
python run.py E16 --datasets XJTU      # chỉ một bộ dữ liệu
python run.py experiments --quick      # chạy thử: 1 lần lặp thay vì 5, 1 hạt giống thay vì 5
python run.py analyze                  # runs/ -> results/ (bảng CSV, hình PNG/PDF)
python run.py benchmark                # mã gốc PINN4SOH trên cách chia của bài -> benchmark/results/
python run.py paper                    # results/ -> paper/main.pdf
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
| E32 | Mô hình học cập nhật từ lần đo gần nhất (GBDT), cùng fold với E20 (4 bộ × 4 mức nhãn) | 240 | 6,7 | 1,7 | 0,8 |
| benchmark | Mã gốc PINN4SOH, 39 nhóm: bản công bố 10 lần, bản đã sửa 3 lần mỗi nhóm | 507 | ≈ 6 | ≈ 1,5 | ≈ 0,8 |
| **Tổng** | | **10 922** | **≈ 88** | **≈ 22** | **≈ 11** |

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
│   ├── pinn4soh_splits.json, pinn4soh_reported.csv   cách chia và số công bố của PINN4SOH
│   ├── experiments.py      định nghĩa E13–E32 thành danh sách lượt chạy
│   ├── runner.py           chạy song song, bỏ qua lượt đã xong, ước lượng thời gian
│   ├── runcache.py         nhớ kết quả đọc runs/*.json giữa các lần phân tích
│   └── plotstyle.py        khuôn vẽ hình
├── analysis/               runs/ -> results/: một script cho mỗi thí nghiệm (analyze_eXX.py, dev_eXX.py cho giai
│                           đoạn chọn trên validation), cùng các phân tích chung (chỉ số, phần vật lý, phân rã sai số,
│                           tính đơn điệu, hình tổng quan, figures_report.py vẽ lại hình cho bản in, diagrams.py sơ đồ)
├── benchmark/
│   ├── rerun_pinn4soh.py   chạy lại mã gốc PINN4SOH (nguyên trạng) trên 39 nhóm chia của bài, hai bản mã
│   └── compare.py          bảng so sánh: số công bố, chạy lại, mô hình của đồ án (E27)
├── paper/                  bản thảo bài báo (LaTeX), hình và script dựng
├── docs/                   đề cương chốt trước (de-cuong-E13, E20, E27-E28, E29, E30, E32), phương pháp, chứng minh,
│                           phân tích phần vật lý, kết quả E13, các phương trình (equations.tex)
├── tools/
│   ├── tai_du_lieu.py      tải dữ liệu
│   ├── kiem_tai_lap.py     kiểm tra tái lập từng bit (python run.py reproduce)
│   ├── make_physics_doc.py docs/phan-tich-vat-ly-PINN (vật lý nằm ở đâu, R², RMSE)
│   └── demo.py             ước lượng và dự báo SOH cho một cell (python run.py demo ...)
├── demo/                   mô hình demo đã huấn luyện và ví dụ đầu ra
├── archive/                báo cáo đồ án, slide, pipeline v1 — giữ nguyên như lúc nộp (archive/README.md)
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

## 6. Demo — ước lượng và dự báo SOH cho một cell

```bash
python run.py demo train --dataset XJTU          # PINN-semi, 30 % nhãn -> demo/XJTU.pt (~1 phút)
python run.py demo predict --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-8.csv"
python run.py demo predict --model demo/XJTU.pt --csv "..." --an-nhan   # coi như cell chưa đo dung lượng

# dự báo từ chu kỳ 150: SOH và dung lượng (Ah) sau 0,1L / 0,25L / 0,5L chu kỳ, số chu kỳ còn lại tới ngưỡng
# cuối đời, khoảng 90 %; bỏ --den-chu-ky để dự báo từ chu kỳ cuối cùng của file
python run.py demo forecast --model demo/XJTU.pt --csv "PINN4SOH/data/XJTU data/2C_battery-8.csv" --den-chu-ky 150
python run.py demo forecast --model demo/XJTU.pt --csv "..." --den-chu-ky 150 --an-nhan   # lịch sử SOH ước lượng
```

Ước lượng ở chu kỳ N chỉ dùng dữ liệu tới chu kỳ N nên chạy trực tuyến được. `forecast` dùng bản có xét hồi phục (E23)
và in thêm phần hồi phục hiện tại, tỉ lệ được giữ lại, thời gian tắt của phần còn lại và dao động đều nếu có. Lần đầu
chạy `forecast`, bộ dự báo được huấn luyện trên các cell có nhãn của cùng fold với mô hình ước lượng (vài phút) và lưu ở
`demo/<bộ>_dubao.pkl`. Đầu ra: `demo/<cell>_soh.csv`, `demo/<cell>_dubao.csv` và các hình `.png`.

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
