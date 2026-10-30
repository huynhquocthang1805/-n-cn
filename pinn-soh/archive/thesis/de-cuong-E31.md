# Đề cương E31 — so với SambaMixer trên NASA theo lượng thông tin đo được, rồi kiểm tra lại trên mọi bộ

Chốt trước khi chạy E31. Mọi lựa chọn dưới đây dựa trên khám phá chỉ trên 10 cell không thuộc tập đánh giá của cách chia
NASA-L (8 cell train và 2 cell validation: B0005, B0018, B0031, B0034, B0036, B0046, B0048, B0054, B0055, B0056); ba cell
đánh giá B0006, B0007, B0047 chưa được nhìn tới.

## Vì sao cần E31

Trên cách chia NASA-L, PINN-lai có MAE trung bình ba cell 4,12 % SOH, SambaMixer công bố 0,96 %. SambaMixer nhận chuỗi
điện áp, dòng điện, nhiệt độ và thời gian của chính lần xả cần ước lượng; với dòng xả không đổi, dung lượng gần như là
thời gian xả nhân dòng điện. Đồ án chỉ dùng 16 đặc trưng của đoạn cuối lần sạc trước đó (4,0–4,2 V và pha CV).

Khám phá (bỏ một cell ra, GBDT, 10 cell nói trên):

| Đầu vào | MAE trung bình (% SOH) |
|---|---|
| 16 đặc trưng | 10,47 |
| 16 đặc trưng + giờ nghỉ | 8,96 |
| 16 đặc trưng + 7 đặc trưng của cả lần sạc | 5,71 |
| 16 + 7 + giờ nghỉ | 5,57 |
| giữ nguyên dung lượng của lần xả trước (cần đo mọi chu kỳ) | 0,83 |

Ở NASA mỗi lần sạc bắt đầu từ gần cạn, nên cả lần sạc mang nhiều thông tin về dung lượng hơn đoạn cuối; và vì mỗi chu kỳ
đều xả đầy, dung lượng của lần xả trước luôn đo được. Tỉ số điện lượng nạp / dung lượng xả gần như cố định trong một cell
nhưng khác nhau giữa các cell (trung vị 1,00–1,36), nên một cell chưa đo lần nào vẫn bị lệch mức. Hiệu chỉnh bằng đúng một
lần đo ở chu kỳ đầu làm kém đi (7,9–9,2 %): chu kỳ đầu không đại diện cho về sau, như E25 đã thấy.

## Ba mức thông tin, đều ít hơn SambaMixer

* M0 — không đo cell đánh giá: đầu vào là lần sạc ngay trước lần xả (và giờ nghỉ).
* M1 — đo thưa: đo dung lượng cell đánh giá ở chu kỳ đầu và mỗi qL chu kỳ, q ∈ {0,2; 0,1; 0,05} (L = trung vị số chu kỳ
  của bộ). Ước lượng ở chu kỳ n chỉ dùng các lần đo ở chu kỳ trước n; sai số tính trên các chu kỳ không được đo.
* M2 — đo mọi chu kỳ trước: dung lượng của mọi lần xả trước n đã biết (máy thử xả đầy mỗi chu kỳ như ở NASA). Ước lượng ở
  chu kỳ n là dự báo một bước; sai số tính trên mọi chu kỳ trừ chu kỳ đầu của cell.
* SambaMixer dùng cả chính lần xả n, tức nhiều thông tin hơn M2.

## Đặc trưng (chốt)

NASAfc = 16 đặc trưng như cũ + 7 đặc trưng của cả lần sạc (pinnsoh/nasa.py: FULL_FEATURES — điện lượng nạp, thời gian và
điện lượng pha CC tới 4,195 V, thời gian pha CV, điện áp đầu pha CC, độ tăng nhiệt, tổng thời gian sạc) + log(1 + giờ
nghỉ). Lọc ngoại lai nhân quả, chuẩn hoá theo cell train như mọi bộ khác.

## Mô hình và giao thức

* Như E24 trên NASAfc: 3 PINN-semi, MLP, GBDT; kiểm định chéo 5 fold × 3 lần lặp ở 30 và 70 % nhãn; cách chia NASA-L với
  3 hạt giống. Như E30: CNN, LSTM, Transformer với độ dài cửa sổ đã chốt ở E30dev.
* PINN-lai: (ω, w) chọn trên validation như PINN-lai tinh chỉnh (ω ∈ {0,3; 0,4; 0,5; 0,6; 0,7}, w ∈ {5, 10, 20, 40});
  trên NASA-L, validation là B0018 và B0055.
* M1, M2: bộ lọc Kalman của E25 (độ lệch theo cell là quá trình Ornstein–Uhlenbeck) trên dự đoán của PINN-lai ở M0; μ0, R,
  P0 và độ dài tương quan ℓ ước lượng trên cell validation của chính fold (NASA-L: B0018, B0055), ℓ chọn riêng cho từng
  lịch đo. Đối chứng: "gần nhất" (dịch theo sai số của lần đo gần nhất) và "giữ nguyên" (lấy đúng dung lượng của lần đo gần
  nhất, không dùng mô hình).

## Kiểm tra lại trên mọi bộ

* M0 ở bốn bộ chính không đổi: dữ liệu công bố kèm PINN4SOH chỉ có 16 đặc trưng, không có chuỗi đo để tính đặc trưng của
  cả lần sạc.
* M1, M2 với cùng bộ lọc và cùng lịch đo trên dự đoán của PINN-lai tinh chỉnh ở XJTU, TJU, MIT, HUST (3 lần lặp × 5 fold ×
  4 mức nhãn) và ở NASA, NASAfc (kiểm định chéo).
* So ghép cặp theo cell (Wilcoxon), như mọi so sánh khác.

## Tiêu chí

Trên NASA-L, mỗi mức thông tin được so với 0,96 % (trung bình ba cell của SambaMixer) bằng MAE trung bình ba cell, trung
bình 3 hạt giống; báo cáo cả từng cell. Chỉ nói "thấp hơn SambaMixer" ở mức nào có MAE trung bình ba cell thấp hơn 0,96 %,
và luôn nói kèm mức thông tin đó.

## Bổ sung trước khi có kết quả NASAfc (hậu kiểm, ghi rõ trong báo cáo)

Chạy thử phần phân tích trên các lượt đã có (bốn bộ chính, NASA và NASAdt; các lượt NASAfc chưa xong) cho thấy ở M2 cách
"giữ nguyên" tốt hơn bộ lọc ở mọi bộ, kể cả trên NASA-L (0,60 % so với 0,70–0,84 %). Lý do: với dự báo một bước, bộ lọc
gần như cộng thay đổi của mô hình giữa hai chu kỳ liền nhau, p_n − p_{n−1}, vào lần đo trước, mà thay đổi này nhiễu hơn
thay đổi thật. Vì vậy thêm một biến thể, quyết định trước khi có bất kỳ kết quả M1, M2 nào trên NASAfc (nhật ký chạy chỉ in
MAE ở M0 của từng lượt):

* kalman_k: cùng bộ lọc, nhưng độ lệch nhận thêm đầu vào −(1 − κ)(p_n − p_{n−1}) ở mỗi bước, tức ước lượng chỉ tin κ phần
  thay đổi của mô hình. κ = 1 trùng bộ lọc chốt ở trên; κ = 0 với nhiễu đo nhỏ trùng "giữ nguyên". (ℓ, κ) chọn cùng lúc
  trên cell validation, κ ∈ {0; 0,25; 0,5; 0,75; 1}, ℓ trên cùng lưới, riêng cho từng lịch đo; μ0, R, P0 như trên.
* Bộ lọc chốt ở trên vẫn là kết quả chính; kalman_k được báo cáo là phân tích hậu kiểm. Kết quả M1, M2 của NASAfc là phần
  duy nhất mà thiết kế của kalman_k chưa nhìn thấy.

## Kết quả (analysis/analyze_e31.py, results/e31_*.csv)

264 lượt, không lượt nào lỗi (lần chạy đầu bị ngắt giữa chừng vì thư mục dữ liệu bị xoá nhầm; dữ liệu tải lại và dựng
lại trùng từng bit với bộ nhớ đệm dựng trước đó ở cả bảy biến thể, các lượt đã xong được giữ nguyên, các lượt còn lại
chạy tiếp).

M0 — đặc trưng của cả lần sạc (tỉ số MAE 24 cột / 16 đặc trưng, kiểm định chéo, Wilcoxon theo 18 cell):

| Mô hình | 30 % nhãn | 70 % nhãn | NASA-L, 16 → 24 cột (MAE %, trung bình 3 cell × 3 hạt giống) |
|---|---|---|---|
| MLP | 0,95 | 0,78* | 4,64 → 3,16 |
| CNN | 0,66* | 0,58* | 5,04 → 2,25 |
| LSTM | 0,63* | 0,61* | 6,49 → 2,16 |
| Transformer | 0,65* | 0,61* | 7,08 → 2,31 |
| GBDT | 0,73* | 0,64* | 5,54 → 2,74 |
| PINN-semi | 0,97 | 0,86 | 5,23 → 3,32 |
| PINN-lai (ω = ½, w = 10) | 0,81 | 0,80 | 4,12 → 2,52 |
| PINN-lai tinh chỉnh | 0,73* | 0,71* | 4,64 → 2,54 |

\* p < 0,05. Ở M0, mô hình tốt nhất trên NASA-L (LSTM, 2,16 %) vẫn gấp 2,2 lần SambaMixer.

M1, M2 trên NASA-L với 24 cột (MAE trung bình ba cell, %; trên dự đoán của PINN-lai tinh chỉnh):

| Cách | M0 | mỗi 0,2L | mỗi 0,1L | mỗi 0,05L | M2 |
|---|---|---|---|---|---|
| Kalman (chốt trước) | 2,54 | 1,89 | 1,58 | 1,08 | 0,80 |
| Kalman κ (hậu kiểm) | – | 1,64 | 1,32 | 0,82 | 0,64 |
| gần nhất | – | 2,18 | 1,57 | 1,19 | 0,75 |
| giữ nguyên | – | 2,54 | 1,48 | 0,99 | 0,60 |
| số lần đo mỗi cell | 0 | 8 | 16 | 31 | mọi chu kỳ |

Theo cell ở mỗi 0,05L, Kalman κ: B0006 1,09, B0007 0,47, B0047 0,89 % (SambaMixer 1,17; 1,20; 0,51). Ở M2, Kalman
chốt trước: 1,03; 0,41; 0,96 %. B0047 cao hơn SambaMixer ở mọi mức và mọi cách.

Kiểm tra lại trên mọi bộ (kiểm định chéo, Wilcoxon theo cell, Holm qua bốn bộ chính):

* Đo thưa (M1): bộ lọc thấp hơn không đo ở 46/48 trường hợp bộ × mức nhãn × lịch đo, thấp hơn giữ nguyên ở 48/48
  (tỉ số 0,11–0,94); so với gần nhất, tốt hơn có ý nghĩa ở 12, kém ở 11, còn lại không khác. Ở mức đo thưa, validation
  chọn κ = 1 ở 71–91 % số fold của bốn bộ chính, nên Kalman κ gần như trùng bộ lọc gốc; ở M2 chỉ 6 %.
* Đo mọi chu kỳ trước (M2): giữ nguyên thấp nhất ở 23/25 trường hợp (bốn bộ × 4 mức nhãn, NASA ba bộ đặc trưng × 2 mức
  nhãn, NASA-L ba bộ đặc trưng). Kalman κ đưa tỉ số so với giữ nguyên từ 1,17–1,52 về 0,98–1,30 trên bốn bộ chính.
* NASA, kiểm định chéo 18 cell: giữ nguyên tốt hơn cả hai bộ lọc ở mỗi 0,05L và ở M2 (p < 0,05 ở từng phép so, chưa
  hiệu chỉnh), với cả ba bộ đặc trưng; lợi thế của Kalman κ trên ba cell của NASA-L không lặp lại.

## Đối chiếu với tiêu chí

* M0: không thấp hơn SambaMixer (thấp nhất 2,16 %).
* M1: bộ lọc chốt trước không thấp hơn ở lịch nào (thấp nhất 1,08 % ở mỗi 0,05L). Biến thể κ (hậu kiểm) thấp hơn ở mỗi
  0,05L (0,82 %), nhưng không lặp lại trên kiểm định chéo của NASA.
* M2: thấp hơn SambaMixer (bộ lọc chốt trước 0,80 %, κ 0,64 %), nhưng giữ nguyên, không dùng mô hình, cũng thấp hơn
  (0,60 %): ở mức thông tin này NASA-L dễ, và phần thắng đến từ thông tin chứ không từ mô hình.
* Cách nói được phép: "khi biết dung lượng của mọi lần xả trước, PINN-lai với bộ lọc Kalman có MAE trung bình ba cell
  0,80 %, thấp hơn 0,96 % của SambaMixer (mô hình nhận chính lần xả cần ước lượng); cách giữ nguyên lần đo trước cũng
  đạt 0,60 %".
