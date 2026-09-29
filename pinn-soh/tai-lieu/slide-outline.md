# Báo cáo tiến độ 2 tuần — ĐACN HK261, nhóm 15

**Đề tài:** Ước lượng SOH pin Li-ion đơn bằng ràng buộc vật lý không cần nhãn
**Người trình bày:** Huỳnh Quốc Thắng · 14/09/2026 · ~6 phút, 8 slide

Mọi con số dưới đây đọc trực tiếp từ `results/*.csv`, không có số nào gõ tay.

---

## Slide 1 — Trang bìa

**Tiêu đề:** Ước lượng SOH pin Li-ion đơn bằng ràng buộc vật lý không cần nhãn
**Phụ đề:** ĐACN HK261 · Nhóm 15 · Báo cáo tiến độ 2 tuần

**Dải số ở chân slide:** 4 bộ dữ liệu công khai · 387 cell · 1 227 lượt huấn luyện · 10 protocol thí nghiệm

**Ghi chú nói:** Mở đầu 20 giây. Nêu tên đề tài, và chốt ngay rằng giai đoạn 2 tuần này đã đi hết 3 mục tiêu bắt buộc và đã chạy được phần optional.

---

## Slide 2 — Bài toán và vị trí hiện tại trong 4 mục tiêu ĐACN

**Bài toán**

- Đầu vào: 16 đặc trưng thống kê của **một** chu kỳ sạc CC–CV. Đầu ra: SOH của đúng chu kỳ đó.
- Nút thắt thực tế: muốn có nhãn SOH phải xả đầy cell — tốn hàng giờ và phải dừng vận hành. Trong khi đó đường sạc thì BMS ghi được của **mọi** cell, miễn phí.
- Câu hỏi nghiên cứu: **ràng buộc vật lý có thay được nhãn không?** Cụ thể: mô hình huấn luyện với 30 % cell có nhãn có đuổi kịp mô hình thường dùng đủ 70 % không?
- Phạm vi: cell đơn (chưa xét pack), 4 bộ công khai, 387 cell, 2 họ hoá học (NCM/NCA và LFP).

**Trạng thái 4 mục tiêu ĐACN**

| # | Mục tiêu | Trạng thái |
|---|---|---|
| 1 | Xác định bài toán | Xong |
| 2 | Khảo sát research / công nghệ | Xong |
| 3 | Baseline và thiết kế lời giải | Xong |
| 4 | Thí nghiệm baseline / prototype *(optional)* | Đã chạy, đang mở rộng |

**Ghi chú nói:** Nhấn vào chữ "miễn phí" — đó là toàn bộ lý do bài toán này đáng làm. Nếu nhãn rẻ thì không cần vật lý.

---

## Slide 3 — Đã làm (1/4): Khảo sát — bốn khoảng trống của PINN4SOH

Xuất phát từ survey **Su, Xu & Dong 2024** (*Energy Conversion and Economics*) và bài **PINN4SOH** (Wang et al., *Nature Communications* 2024). Đọc thẳng mã nguồn công bố `wang-fujin/PINN4SOH`, tìm được bốn điểm bài báo không nói rõ — mỗi điểm là một chỗ để đóng góp.

1. **Phần dư PDE có thể tự triệt tiêu** (`Model.py:232–234`) — mạng động học nhận chính `u_t` làm đầu vào, nên có thể học `F ≈ u_t` và đưa phần dư về 0 mà không cần chút vật lý nào.
2. **"Physics loss" phụ thuộc nhãn** (`Model.py:255`) — phạt theo hướng thay đổi của *nhãn*, nên **không áp được lên cell chưa đo dung lượng**. Về cấu trúc, nó không thể là nguồn thông tin bù cho nhãn.
3. **Chuẩn hoá theo từng cell trên cả vòng đời** (`dataloader.py:50, 58–64`) — chỉ số chu kỳ sau chuẩn hoá chính là "phần trăm tuổi thọ đã đi qua", tức rò rỉ tương lai vào đầu vào.
4. **Validation chia theo hàng, không theo cell** (`dataloader.py:157–158`) — early-stopping chấm điểm trên đúng những cell đã học.

> **Số đo được cho điểm 3:** trên XJTU, chỉ riêng cái rò rỉ chỉ số chu kỳ đã kéo MAE từ 0.0092 xuống 0.0078 — **giả tạo −15 %**.

**Ghi chú nói:** Đây là phần "khảo sát" của mục tiêu 2, nhưng làm theo kiểu đọc code chứ không chỉ đọc abstract. Nếu thầy hỏi vì sao không dùng thẳng PINN4SOH — câu trả lời nằm ở điểm 2.

---

## Slide 4 — Đã làm (2/4): Thiết kế lời giải

**Hình:** `fig_pipeline.png` (bên phải)

**Mạng nghiệm** — `u(N) = F_φ(x_N, Ñ)`, MLP 17→64→64→32→1, SiLU. Ñ = N/1000, hằng số toàn cục, **không bao giờ** chuẩn hoá theo cell.

**Động học xám** — `r = softplus(MLP_θ(x)) · exp(λ(1−u)) · exp(−Eₐ/R·(1/T − 1/T_ref))`
Ba thừa số mang ba mẩu vật lý tách rời nhau, ablation được từng cái:
- `softplus(·) ≥ 0` → tốc độ mất dung lượng không âm ⇒ nghiệm đơn điệu **theo cấu trúc**, không cần nhãn để dạy.
- `exp(λ(1−u))` → tốc độ tăng dần khi SOH giảm, mô tả pha "đầu gối".
- Arrhenius → chỉ có tác dụng ở bộ nhiều nhiệt độ (TJU 25/35/45 °C).

**Ba loss KHÔNG cần nhãn** — ODE dạng Euler, đơn điệu có dung sai ε = 0.002, miền hợp lý. Chỉ `L_data` cần nhãn. Nhờ vậy loss vật lý chạy được trên cả cell chưa đo dung lượng — đây chính là điều PINN4SOH không làm được.

**Baseline:** MLP cùng kích thước, chỉ `L_data`. **Pipeline không rò rỉ:** chia theo **cell** 70/15/15 có phân tầng, chuẩn hoá nhân quả, assert không cell nào nằm ở hai tập.

Tổng ≈ 8 000 tham số — đủ nhỏ để chạy trên vi điều khiển sau lượng tử hoá.

**Ghi chú nói:** Điểm cần bảo vệ: khác PINN4SOH, mạng động học ở đây **không** nhận đạo hàm của u làm đầu vào, nên phần dư không thể tự triệt tiêu.

---

## Slide 5 — Đã làm (3/4): Ít nhãn thì vật lý thắng

**Hình:** `fig_E1_label_efficiency.png` (chiếm phần lớn slide)

**Ba con số:**

- **Ở 10 % nhãn, vật lý thắng cả bốn bộ.** Tỉ số MAE(MLP)/MAE(PINN-semi): XJTU 1.7× · TJU 1.1× · MIT 1.3× · HUST 1.9×.
- **15/16** ô thắng khi trọng số vật lý β được chọn theo validation (thay vì cố định β = 5 như ban đầu).
- **Giả thuyết "30 % ≈ 70 %":** xác nhận trên MIT (0.91×) và HUST (0.96×); chưa xác nhận trên XJTU (1.25×) và TJU (1.17×).

**Đọc cho đúng:** vật lý là một *prior* — nó bù thiếu hụt nhãn, và bù nhiều nhất ở nơi mô hình chỉ-dữ-liệu đói nhãn nhất. Ở nơi 30 % nhãn đã đủ cho MLP (TJU), ràng buộc trở thành thiên kiến và làm MAE tăng nhẹ. Câu chuyện trung thực là **"ít nhãn thì vật lý thắng, nhãn đủ thì hoà"**, không phải "vật lý luôn thắng".

**Ghi chú nói:** Trục dọc là thang log dùng chung cho cả bốn bộ — cùng khoảng cách dọc là cùng một tỉ số sai số. Chỉ vào panel MIT và HUST trước, đó là chỗ câu chuyện rõ nhất.

---

## Slide 6 — Đã làm (4/4): Hai phép kiểm chứng khắt khe

**Hình:** `fig_E12_fair.png`

**(a) Cân bằng ngân sách tinh chỉnh (E12, 384 lượt).** Lần đầu so sánh, baseline MLP chưa được tinh chỉnh ngang với PINN — không công bằng. Khi cho MLP hưởng **đúng** lưới siêu tham số mà PINN được hưởng:

- Tỉ số PINN/MLP trung bình đi từ **0.72× lên 0.90×** — tức một phần lợi thế của PINN biến mất thật.
- PINN vẫn thắng **7/8** ô, tỉ số 0.75–1.01×.
- Riêng MIT @ 30 %: baseline tự nó cải thiện 0.0103 → 0.0048 chỉ nhờ dropout 0.1.

**(b) Đem sang bộ dữ liệu khác (E5).** Zero-shot thất bại với mọi mô hình — đặc trưng đường sạc phụ thuộc protocol nên đầu vào bộ đích nằm ngoài phân bố. Nhưng vì loss vật lý không cần nhãn, chúng chạy được ngay trên cell bộ đích và trở thành cơ chế **thích nghi miền không nhãn**:

| Cặp chuyển | MLP | PINN-semi | Cải thiện |
|---|---|---|---|
| XJTU → TJU | 2.944 | 0.041 | 72× |
| HUST → MIT | 0.369 | 0.027 | 13× |
| TJU → XJTU | 0.329 | 0.103 | 3.2× |
| MIT → HUST | 0.062 | 0.059 | 1.05× |

**Ghi chú nói:** Con số 0.72 → 0.90 là chỗ nên chủ động nói ra trước khi bị hỏi. Nó cho thấy mình tự kiểm chứng chứ không chọn số đẹp.

---

## Slide 7 — Khó khăn gặp phải

| Khó khăn | Cách xử lý | Trạng thái |
|---|---|---|
| **Phần dư dạng đạo hàm khuếch đại nhiễu 1000×.** Chia cho Δt = h/1000, với h = 1 thì dao động 10⁻³ của mạng thành phần dư cỡ 1 — lớn hơn cả tín hiệu thật. MAE xấu gấp đôi MLP thuần. | Viết lại phần dư ở **dạng Euler** (nhân hai vế với h/1000) — cặp horizon ngắn đóng góp nhỏ, nhiễu không bị khuếch đại. | Đã xử lý |
| **Dấu tương quan đặc trưng–SOH không bất biến theo hoá học.** CV Q có tương quan −0.94 ở XJTU nhưng +0.41 ở MIT; ngay cả hai bộ cùng NCM cũng lệch (CC Q: −0.18 vs +0.95). | Loại **toàn bộ** ràng buộc dấu trên đặc trưng khỏi thiết kế; vật lý chỉ đặt lên **quỹ đạo SOH** (đơn điệu, tốc độ không âm, Arrhenius). | Đã xử lý |
| **So sánh ban đầu không công bằng** — baseline chưa được tinh chỉnh ngang. | Chạy thêm E12 (384 lượt) cho hai mô hình cùng lưới siêu tham số. Kết quả tự hạ: 0.72× → 0.90×. | Đã xử lý |
| **Zero-shot chuyển miền thất bại hoàn toàn** (sai 2–3 bậc độ lớn). | Hiện chỉ khai thác được dạng thích nghi miền không nhãn. Zero-shot thật giữa hai họ hoá học có thể không khả thi với bộ đặc trưng này. | Chưa xong |
| **Chỉ số đánh giá mâu thuẫn nhau.** Trên TJU @ 30 %: MLP thắng theo MAE (0.0088 vs 0.0099) nhưng PINN thắng theo cell tệ nhất (0.0259 vs 0.0283), vi phạm đơn điệu (0.20 vs 0.68) và sai số dự báo EOL (24.5 vs 29.8 chu kỳ). | Chưa chốt được chỉ số chính để báo cáo. | Cần quyết |
| **Chạy trên CPU** → chỉ đủ 3 seed, chưa đủ để kết luận thống kê. | Chuyển sang GPU. | Kế hoạch |

**Ghi chú nói:** Nếu hết giờ thì chỉ nói 3 dòng đầu. Dòng "chỉ số mâu thuẫn" là chỗ nên xin ý kiến thầy.

---

## Slide 8 — Công việc 2 tuần tiếp theo

**Tuần 1 — làm cho kết quả hiện có đủ tin cậy**

1. **Chuyển toàn bộ sang GPU** — tăng 3 → 10 seed; kiểm định thống kê ghép cặp theo cell để có khoảng tin cậy thay vì chỉ trung bình.
2. **Mở rộng lưới ngân sách công bằng** — E12 hiện chỉ cân bằng ở 30 % và 70 %; bổ sung mức 10 % và 50 %, thêm learning rate và số bước vào lưới.
3. **Chốt bộ chỉ số chính để báo cáo** — chọn giữa MAE trung bình và sai số dự báo EOL, rồi viết lại mục kết quả theo chỉ số đã chốt.

**Tuần 2 — mở rộng đóng góp và dựng prototype**

4. **Quét số cell đích cần đo nhãn** — k = 0 / 1 / 3 / 5 / 10 cho bài toán thích nghi miền, để trả lời "dây chuyền mới cần đo bao nhiêu cell".
5. **Sửa đầu ra đơn điệu** — đơn điệu dọc quỹ đạo thật, không chỉ đơn điệu theo biến t như hiện nay.
6. **Dựng demo prototype** — suy luận online trên một cell, không cần biết tuổi thọ tương lai, đúng điều kiện BMS chạy thật.

**Cần hỗ trợ từ thầy:** chọn chỉ số chính để báo cáo — MAE trung bình hay sai số dự báo EOL? Hai chỉ số cho hai kết luận ngược nhau trên TJU.

**Ghi chú nói:** Kết bằng câu hỏi xin ý kiến — để buổi họp có đầu ra chứ không chỉ là báo cáo một chiều.

---

## Hình dùng trong slide

| Slide | File | Nội dung |
|---|---|---|
| 4 | `fig_pipeline.png` | Pipeline đúng như mã nguồn chạy |
| 5 | `fig_E1_label_efficiency.png` | MAE theo phần trăm cell mang nhãn, 4 bộ |
| 6 | `fig_E12_fair.png` | So sánh sau khi cân bằng ngân sách tinh chỉnh |

**Hình dự phòng nếu thầy hỏi sâu:** `fig_ratio_matrix.png` (tỉ số PINN/MLP), `fig_E7_tuned.png` (quét trọng số vật lý), `fig_traj_XJTU.png` (quỹ đạo dự đoán), `fig_E5_adapt.png` (thích nghi miền), `fig_E10_matrix.png` (ma trận chuyển miền 4×4), `fig_E9_arch.png` (kiến trúc & hàm kích hoạt).
