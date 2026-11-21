# Đề cương thí nghiệm xác nhận E13 (và E14) — ấn định trước khi chạy

**Ngày chốt:** 29/09/2026, trước lượt huấn luyện E13 đầu tiên. Tài liệu này được commit
**trước** khi có bất kỳ kết quả E13 nào, nên lịch sử git là bằng chứng rằng tiêu chí không bị
chọn lại sau khi nhìn thấy dữ liệu test.

Mục đích: trả lời hai câu hỏi ở slide 3 của báo cáo 29/09 bằng một phép so **công bằng** (slide 18):
cùng cách chia, cùng normaliser, cùng lượng nhãn, cùng ngân sách chọn mô hình. Chỉ khác nhau ở hàm
mất mát.

---

## 1. Pipeline — v2 (sửa năm điểm ở slide 16)

| # | Bản rà soát 29/09 | v2 |
|---|---|---|
| 1 | Lọc 3σ dùng mean/std cả vòng đời cell | `clean='causal'`: cửa sổ 50 chu kỳ **quá khứ**, z bền vững (median, IQR/1,349), ngưỡng 5, cần ≥ 10 chu kỳ lịch sử |
| 2 | `dropna()` làm mất hàng thiếu dung lượng | Chỉ loại hàng có **đặc trưng** không hữu hạn; nhãn thiếu → `soh = NaN`, hàng vẫn vào pool vật lý |
| 3 | MLP và PINN fit normaliser trên hai pool khác nhau | `norm_pool='train'`: **mọi** mô hình fit trên đặc trưng của mọi cell train (có nhãn + không nhãn) |
| 4 | `h` trong `sample_pairs` đếm theo hàng sau lọc | `pair_by='cycle'`: j là hàng đầu tiên có chu kỳ gốc ≥ chu kỳ(i) + h |
| 5 | EOL theo chỉ số hàng, trộn quan sát bị kiểm duyệt | EOL theo **chu kỳ gốc**; tách `miss` (nhãn cắt ngưỡng, dự đoán không cắt) và `false` |

Tham số bộ lọc (50 / 10 / 5 / 1 %) chọn trước khi huấn luyện, chỉ dựa trên đặc trưng; tỉ lệ hàng
bị loại 1,58 % so với 2,15 % của bộ lọc v1. Mọi sửa đổi được kiểm bằng `verify_pipeline.py`
(21 phép kiểm). Chế độ v1 vẫn là mặc định và tái lập **từng bit** kết quả cũ.

## 2. Giao thức chia — kiểm định chéo 5 fold theo cell, lặp 5 lần

Cách chia 70/15/15 theo seed ở E1–E12 có một hạn chế mà slide 11 đã nêu: các seed chia chồng lấp
nhau, và mỗi cell xuất hiện trong test một số lần ngẫu nhiên (có cell không lần nào). E13 dùng
`data.cv_split`:

* 5 fold theo **cell**, phân tầng theo batch giao thức; lặp **R = 5** lần với cách chia khác nhau.
* Trong một lần lặp, mỗi cell nằm trong test **đúng một lần** → mỗi cell có đúng R dự đoán test cho
  mỗi mô hình, ghép cặp theo cell được.
* Mỗi fold: test 20 %, validation 10 % (chọn vòng tròn theo batch từ phần còn lại), train 70 %.
* Tỉ lệ nhãn tính trên **tổng số cell** như E1: 30 % nhãn nghĩa là round(0,3 · n) cell train có
  nhãn, phần còn lại của train không nhãn. Tập có nhãn lồng nhau: 10 % ⊂ 30 % ⊂ 50 % ⊂ 70 %.
* Validation (10 % cell) cần nhãn **ngoài** ngân sách 30 %. Điều này giống nhau cho mọi mô hình.
  Báo cáo phải ghi rõ, không được gộp vào "30 % nhãn".
* Seed khởi tạo và seed chọn nhãn = 100 · lần_lặp + fold, **giống nhau cho mọi nhánh** → ghép cặp.

## 3. Các nhánh (cùng cấu hình mạng, chỉ khác hàm mất mát)

Cấu hình chung (cố định, lấy từ TUNED — không quét lại): mạng nghiệm `wide` (128, 128, 64), SiLU,
MLP; AdamW lr 2·10⁻³, wd 10⁻⁵, batch 512, tối đa 4 000 bước, đánh giá validation mỗi 100 bước,
dừng sớm sau 12 lần không cải thiện, lấy checkpoint tốt nhất theo MAE validation.
Cấu hình vật lý (chọn theo validation ở E7, không chọn lại): β = 0,5, phần dư `autograd`, α = 0,02, γ = 1.

| Nhánh | Mô hình | Nhãn | Vai trò |
|---|---|---|---|
| **A** | `mlp` | 30 % | mốc so sánh |
| **B** | `pinn_sup` | 30 % | vật lý chỉ trên cell có nhãn — tách tác dụng regularization |
| **C** | `pinn_semi` | 30 % | **mô hình đề xuất** — vật lý trên mọi cell train |
| **D** | `pinn_semi`, α = 0, γ = 0 | 30 % | chỉ loss đơn điệu — tách ích lợi của ODE |
| **E** | `pinn_semi`, phần dư `euler` dọc quỹ đạo, α = 2 | 30 % | slide 17: ∂u/∂t giữ x cố định ≠ du/dt dọc quỹ đạo |
| **F** | `mlp` | 70 % | mốc cho phép kiểm tiết kiệm nhãn |
| **G** | `pinn_semi` | 70 % | mô tả |
| L | `mlp`, `pinn_semi` | 10 %, 50 % | đường cong theo lượng nhãn — mô tả |

Tổng: 11 nhánh × 4 bộ × 5 fold × 5 lần lặp = **1 100 lượt**. Số lần lặp R = 5 cố định, không chạy
thêm hay dừng sớm theo p-value.

## 4. Chỉ số và đơn vị phân tích

* **Chỉ số chính:** MAE theo cell. Với cell c và nhánh a:
  e_a(c) = trung bình qua R lần lặp của MAE trên mọi chu kỳ có nhãn của c.
  Tổng hợp một bộ dữ liệu: trung bình e_a(c) qua các cell (mỗi cell một phiếu).
* **Chỉ số phụ:** MAE gộp hàng; MAE vùng SOH ≤ 0,90; sai số EOL theo chu kỳ tại ngưỡng 0,85
  (chỉ trên cell mà nhãn và dự đoán cùng cắt), kèm số `miss` và `false`; vi phạm đơn điệu trên
  100 chu kỳ; jitter.
* Mọi chỉ số tính lại từ `test_predictions` đã lưu (y, p, chu kỳ, id cell), không đọc số tóm tắt.

## 5. Giả thuyết và phép kiểm (ấn định trước)

**H1 — ưu thế ở cùng lượng nhãn (C so với A, 30 %).**
d(c) = e_C(c) − e_A(c). Wilcoxon signed-rank **hai phía** trên các cell, từng bộ dữ liệu; hiệu chỉnh
Holm cho 4 bộ; α = 0,05. Kết luận "PINN tốt hơn trên bộ X" chỉ khi p_Holm < 0,05 **và** trung vị d < 0.
Cỡ hiệu ứng: tỉ số mean(e_C) / mean(e_A), KTC 95 % bootstrap theo cell (20 000 lần).

**H2 — tiết kiệm nhãn (C 30 % so với F 70 %), biên không kém hơn δ = 10 %.**
H0: PINN 30 % kém hơn MLP 70 % quá 10 %. Wilcoxon signed-rank **một phía** trên
e_C(c) − 1,10 · e_F(c) (đối thuyết: nhỏ hơn 0); Holm cho 4 bộ; α = 0,05.
Kèm cận trên của KTC 90 % hai phía (tức KTC 95 % một phía) của tỉ số mean(e_C)/mean(e_F); kết luận
"không kém hơn" chỉ khi p_Holm < 0,05 **và** cận trên < 1,10.
Biên 10 % chọn theo thực tế BMS: sai số SOH ~0,1 điểm phần trăm tuyệt đối ở mức MAE hiện tại,
nhỏ hơn nhiều so với biến thiên giữa các cell.

**Phân tích độ nhạy (không thay thế phép kiểm chính).** Phép t hiệu chỉnh cho kiểm định chéo lặp
lại (Nadeau & Bengio 2003; Bouckaert & Frank 2004) ở mức fold: d_{r,k} = MAE_cell(C) − MAE_cell(A)
trên fold k lần lặp r; t = mean(d) / sqrt((1/(KR) + n_test/n_train) · s²), bậc tự do KR − 1.
Phép này bảo thủ hơn vì tính cả biến thiên do huấn luyện lại. Nếu hai phép cho kết luận khác nhau,
báo cáo **cả hai** và ghi "kết luận phụ thuộc đơn vị phân tích".

**So sánh phụ (thăm dò, Holm trong từng họ, không dùng để khẳng định):**
B − A (tác dụng regularization), C − B (ích lợi của cell không nhãn), D − C (ích lợi của ODE ngoài
đơn điệu), E − C (phần dư dọc quỹ đạo so với đạo hàm riêng), G − F (ở 70 %).

**Điều sẽ không làm:** không đổi chỉ số chính, biên δ, số lần lặp hay cấu hình sau khi thấy kết quả;
không bỏ bộ dữ liệu nào khỏi hiệu chỉnh Holm; p lớn **không** được diễn giải là "tương đương".

## 6. E14 — chuyển miền với cùng normaliser (slide 15)

E5 so MLP fit normaliser trên nguồn với PINN fit trên nguồn + đích, nên phần lớn chênh lệch có thể
đến từ normaliser. E14 lặp lại đúng E5 (4 cặp HUST↔MIT, XJTU↔TJU; k ∈ {0, 3} cell đích có nhãn;
cấu hình vật lý E5; 2 500 bước cố định, không chọn mô hình theo nhãn đích) nhưng **cả hai mô hình**
dùng normaliser fit trên cùng pool đặc trưng nguồn + đích, pipeline v2, 5 seed.
Chỉ số: MAE theo cell trên tập test đích. Phân tích mô tả (5 seed, không kiểm định khẳng định).

## 7. Cách chạy

```bash
python experiments.py E13 --workers 4 --repeats 5     # 1 100 lượt, ~2–3 giờ trên 4 lõi CPU
python experiments.py E14 --workers 4 --seeds 0 1 2 3 4
python analyze_e13.py                                  # -> results/E13_*.csv, E14_*.csv, hình
```
