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
