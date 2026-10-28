# Đề cương E32 — mô hình học cập nhật từ lần đo dung lượng gần nhất

Chốt trước khi chạy E32. Thiết kế dựa trên khám phá chỉ trên cell không thuộc tập đánh giá: NASA bỏ một cell ra trên
10 cell train và validation của NASA-L (như E31); bốn bộ chính ở lần lặp 0, 5 fold, 70 % nhãn, đánh giá trên cell
validation (cell test không được nhìn tới).

## Vì sao cần E32

E31 cho thấy khi cell cần ước lượng có vài lần đo dung lượng (mức M1, M2), bộ lọc Kalman trên dự đoán của PINN-lai giảm
sai số nhiều so với không đo, nhưng:

* ở M2 (biết mọi lần xả trước), cách "giữ nguyên" lần đo trước thắng bộ lọc ở 23/25 trường hợp, vì thay đổi mà PINN-lai
  dự đoán giữa hai chu kỳ liền nhau nhiễu hơn thay đổi thật;
* trên NASA-L, bộ lọc chốt trước chỉ xuống dưới 0,96 % của SambaMixer ở M2, và trên kiểm định chéo của NASA giữ nguyên tốt
  hơn bộ lọc ở lịch 0,05L và M2.

Bộ lọc chỉ dùng mô hình M0 (học không có lần đo nào của cell) rồi sửa độ lệch. E32 học thẳng điều cần biết ở M1, M2:
dung lượng đã thay đổi bao nhiêu kể từ lần đo gần nhất.

## Mô hình (chốt)

Ở chu kỳ (hàng) n của cell, gọi m là hàng được đo gần nhất trước n và m' là hàng được đo trước m (nếu có). Ước lượng

    ŷ_n = y_m + f(u_n),

với f là GBDT và u_n gồm:

* z_n: đặc trưng đã chuẩn hoá ở chu kỳ n (16 cột ở bốn bộ chính và NASA, 24 cột ở NASAfc như E31);
* z_n − z_m: thay đổi đặc trưng kể từ lần đo;
* y_m: SOH đo được ở lần đo gần nhất;
* (c_n − c_m)/L: số chu kỳ kể từ lần đo, theo đơn vị L (trung vị số chu kỳ của bộ);
* độ dốc của hai lần đo gần nhất: y_m − y_m', (c_m − c_m')/L, s = (y_m − y_m')/((c_m − c_m')/L), s·(c_n − c_m)/L, và một
  cờ cho biết đã có m' chưa (khi chưa có, bốn đại lượng trước bằng 0).

Huấn luyện: cặp (n, m) lấy trên cell train có nhãn của chính fold, theo đúng các lịch đo của E31 — đo ở chu kỳ đầu và sau
mỗi qL chu kỳ (q = 0,2; 0,1; 0,05) với pha 0 và hai pha ngẫu nhiên trong [0, qL), cộng lịch M2 (m = n − 1); một mô hình
chung cho mọi lịch đo; tối đa 300 000 cặp (lấy mẫu ngẫu nhiên). Mục tiêu y_n − y_m. GBDT như các bộ khác của đồ án
(HistGradientBoosting, tốc độ học 0,05, 31 lá, ít nhất 40 mẫu mỗi lá, L2 = 1); số cây chọn theo MAE trên cặp của cell
validation của fold (bước 50, dừng sau 3 lần không cải thiện, tối đa 1 000). Chuẩn hoá đặc trưng như các mô hình khác
(z-score trên cell train). Hạt giống của lấy mẫu = 100·r + k (NASA-L: hạt giống s).

## Giao thức đánh giá

* Cùng cách chia, cùng cell có nhãn với E20 (bốn bộ chính, 3 lần lặp × 5 fold × 4 mức nhãn), E24/E31 (NASA và NASAfc,
  3 × 5 × 2 mức nhãn) và cách chia NASA-L (NASA, NASAfc, 3 hạt giống).
* Lịch đo trên cell test như E31 (pha 0): M1 tính sai số trên các chu kỳ không được đo; M2 trên mọi chu kỳ trừ chu kỳ
  đầu. Hàng đánh giá trùng đúng hàng của E31, nên so ghép cặp được với bộ lọc Kalman (chốt trước và κ), gần nhất và giữ
  nguyên.
* So ghép cặp theo cell (trung bình qua lần lặp), Wilcoxon; Holm qua bốn bộ chính cho mỗi (lịch đo, mức nhãn, đối chứng).

## Tiêu chí

* Chính: mô hình cập nhật so với bộ lọc Kalman chốt trước của E31 và với giữ nguyên, ở mọi lịch đo, trên bốn bộ chính và
  trên kiểm định chéo của NASA (24 cột).
* NASA-L: so MAE trung bình ba cell, trung bình 3 hạt giống, với 0,96 % của SambaMixer ở từng mức; chỉ nói "thấp hơn
  SambaMixer" ở mức nào thoả, luôn kèm mức thông tin; nói kèm kết quả trên kiểm định chéo 18 cell ở cùng mức (mô hình cập
  nhật có tốt hơn giữ nguyên ở đó không), vì NASA-L chỉ có ba cell.

## Khám phá (trước khi chốt)

MAE theo cell (% SOH); bốn bộ chính: cell validation, lần lặp 0, 70 % nhãn; NASA: bỏ một cell ra trên 10 cell không đánh
giá. "Gần nhất" ở đây dịch dự đoán của một GBDT M0 theo sai số của lần đo gần nhất.

| Bộ | Lịch | Cập nhật | Giữ nguyên | Gần nhất |
|---|---|---|---|---|
| XJTU | 0,2L / 0,05L / M2 | 0,43 / 0,25 / 0,14 | 1,84 / 0,62 / 0,19 | 1,13 / 0,73 / 0,52 |
| TJU | 0,2L / 0,05L / M2 | 0,26 / 0,14 / 0,04 | 2,03 / 0,71 / 0,11 | 0,84 / 0,61 / 0,42 |
| MIT | 0,2L / 0,05L / M2 | 0,12 / 0,07 / 0,05 | 1,01 / 0,36 / 0,05 | 0,21 / 0,16 / 0,14 |
| HUST | 0,2L / 0,05L / M2 | 0,21 / 0,05 / 0,01 | 2,43 / 0,67 / 0,02 | 1,61 / 0,87 / 0,49 |
| NASAfc | 0,2L / 0,05L / M2 | 1,14 / 0,85 / 0,77 | 2,41 / 1,14 / 0,77 | 3,08 / 2,65 / 2,29 |

Bản không có độ dốc của hai lần đo (chỉ y_m, khoảng chu kỳ và đặc trưng) cho NASAfc 1,12 / 0,90 / 0,78: độ dốc giúp ở lịch
dày, không giúp ở lịch 0,2L; giữ lại vì trên bốn bộ chính, nơi SOH giảm đều, nó là thông tin chính giữa hai lần đo thưa.

## Kết quả (runs/E32, analysis/analyze_e31.py, results/e31_tiers*.csv)

306 lượt, không lượt nào lỗi, 6,8 giờ CPU. Ước lượng của mô hình cập nhật được ghép theo (cell, chu kỳ) vào đúng các
hàng đánh giá của E31 (kiểm tra trùng hàng ở mọi đơn vị), nên mọi so sánh dưới đây là ghép cặp.

NASA-L, 24 cột (MAE trung bình ba cell × 3 hạt giống, %; SambaMixer 0,96):

| Cách | mỗi 0,2L (8 lần đo) | mỗi 0,1L (16) | mỗi 0,05L (31) | M2 |
|---|---|---|---|---|
| mô hình cập nhật | 0,63 | 0,40 | 0,35 | 0,33 |
| Kalman (E31, chốt trước) | 1,89 | 1,58 | 1,08 | 0,80 |
| giữ nguyên | 2,54 | 1,48 | 0,99 | 0,60 |

Theo cell (B0006, B0007, B0047; SambaMixer 1,17; 1,20; 0,51): mỗi 0,2L 0,85; 0,33; 0,70; mỗi 0,1L 0,54; 0,23; 0,43;
mỗi 0,05L 0,41; 0,22; 0,43; M2 0,39; 0,22; 0,39. RMSE ở mỗi 0,1L: 0,86; 0,31; 0,53 (SambaMixer 2,07; 1,29; 0,65). Với 16
đặc trưng (NASA): 1,11 / 0,75 / 0,62 / 0,59, tức xuống dưới 0,96 từ lịch 0,1L.

Kiểm định chéo NASA, 24 cột, 18 cell (mô hình cập nhật so với giữ nguyên, Wilcoxon): 70 % nhãn — mỗi 0,2L 0,79 so với
2,12 (18/18 cell tốt hơn, p < 10⁻⁴), mỗi 0,1L 0,75 so với 1,36 (17/18), mỗi 0,05L 0,67 so với 1,02 (15/18, p = 0,0007),
M2 0,60 so với 0,68 (11/18, p = 0,12); 30 % nhãn — 1,02 / 0,91 / 0,81 / 0,73 so với 2,12 / 1,36 / 1,02 / 0,68 (có ý nghĩa ở
0,2L và 0,1L, không ở 0,05L và M2).

Bốn bộ chính (3 lần lặp × 5 fold × 4 mức nhãn, Holm qua bốn bộ):

* Đo thưa: mô hình cập nhật thấp hơn bộ lọc Kalman ở 47/48 trường hợp bộ × mức nhãn × lịch đo (tỉ số 0,15–1,06; ngoại lệ
  MIT 10 % nhãn ở lịch 0,2L, nơi phần lớn cell tốt hơn nhưng MAE trung bình cao hơn một chút) và thấp hơn giữ nguyên ở 48/48
  (tỉ số 0,07–0,53).
* M2: thấp hơn bộ lọc ở 16/16; thấp hơn giữ nguyên có ý nghĩa ở 12/16, kém hơn ở 1 (XJTU 10 % nhãn), không khác ở 3 (XJTU
  30–70 % nhãn), tỉ số 0,37–1,13.

## Đối chiếu với tiêu chí

* Trên NASA-L với 24 cột, mô hình cập nhật thấp hơn SambaMixer ở mọi lịch đo, từ 8 lần đo mỗi cell; từ lịch 0,1L thấp
  hơn trên cả ba cell. Trên kiểm định chéo 18 cell, nó thấp hơn giữ nguyên có ý nghĩa ở 5/6 trường hợp mức nhãn × lịch đo
  thưa và không khác ở M2, nên lợi thế không chỉ đến từ ba cell của NASA-L.
* Cách nói được phép: "khi đo thêm dung lượng của cell 8 lần trong đời (mỗi 0,2L), mô hình học cập nhật từ lần đo gần
  nhất có MAE trung bình ba cell 0,63 % trên NASA-L, thấp hơn 0,96 % của SambaMixer, mô hình nhận chính lần xả cần ước
  lượng ở mọi chu kỳ".
