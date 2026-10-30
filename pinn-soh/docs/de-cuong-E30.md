# Đề cương E30 — CNN, LSTM, Transformer theo đề cương đồ án nhóm 15

Đề cương đồ án (giai đoạn 1) yêu cầu so với ít nhất bốn baseline trong nhóm MLP, CNN, LSTM, Transformer và báo cáo
RMSE, MAE, MAPE. Đồ án đã có MLP (E13); E30 thêm ba mô hình chuỗi thời gian theo chu kỳ. Chốt trước khi chạy.

## Mô hình (pinnsoh/seqmodels.py)

* Đầu vào ở chu kỳ n: cửa sổ nhân quả L chu kỳ gần nhất của chính cell (16 đặc trưng sạc đã chuẩn hoá + chỉ số chu kỳ);
  đầu cửa sổ lặp chu kỳ đầu khi cell chưa đủ L chu kỳ. Đầu ra: SOH ở chu kỳ n.
* CNN: 2 lớp tích chập nhân quả 32 kênh (kernel 5, dilation 1 và 2), bước cuối -> MLP 32 -> 1.
* LSTM: 1 lớp, 32 nút ẩn, trạng thái ở bước cuối -> tuyến tính.
* Transformer: chiếu lên 32 chiều + mã hoá vị trí sin, 1 lớp encoder (4 đầu, FFN 64), token cuối -> tuyến tính.
* Chỉ học có nhãn, cùng cách chia, cùng cell có nhãn (hạt giống 100·r + k), cùng normaliser với MLP và GBDT.
  AdamW (lr 1e-3, weight decay 1e-4), batch 128, cắt gradient ở chuẩn 1, tối đa 3 000 bước, đánh giá validation mỗi
  150 bước, dừng khi 6 lần liên tiếp không cải thiện, giữ trọng số tốt nhất.

## E30dev: chọn độ dài cửa sổ

* L ∈ {16, 48}; lần lặp 0, 30 và 70 % nhãn, 5 fold, bốn bộ (240 lượt).
* Với mỗi mô hình: tỉ số MAE theo cell trên validation của L = 48 so với L = 16, trung bình theo fold rồi qua bộ và mức
  nhãn. Chọn L = 48 nếu tỉ số trung bình < 1, ngược lại L = 16. Một giá trị L cho mỗi mô hình, dùng cho mọi bộ.

## E30: đánh giá cuối

* Đúng giao thức E20: 3 lần lặp × 5 fold × 4 mức nhãn × 4 bộ (720 lượt).
* NASA: kiểm định chéo 30 và 70 % nhãn, có / không giờ nghỉ, 3 lần lặp × 5 fold; cách chia NASA-L của SambaMixer với
  3 hạt giống.
* So ghép cặp theo cell test (trung bình 3 lần lặp), Wilcoxon: mỗi mô hình mới so với MLP, GBDT, PINN-semi, PINN-lai;
  p hiệu chỉnh Holm qua bốn bộ như mọi so sánh khác của báo cáo (ghi thêm khi viết phần phân tích, theo phương pháp thống kê chung của báo cáo).
* Chỉ số: MAE theo cell (chính), RMSE theo cell, MAPE, R² (gộp năm fold của một lần lặp, từ chu kỳ thứ 10, trung bình
  qua lần lặp).

## Kết quả E30dev (chốt trước khi chạy E30)

| Mô hình | Tỉ số validation L48 / L16 | L chọn |
|---|---|---|
| CNN | 1,000 | 16 |
| LSTM | 0,989 | 48 |
| Transformer | 1,005 | 16 |

CNN cho đúng cùng kết quả ở hai độ dài: vùng nhìn của hai lớp tích chập (kernel 5, dilation 1 và 2) là 1 + 4 + 8 = 13
chu kỳ, ngắn hơn cả L = 16, nên phần cửa sổ dài hơn không ảnh hưởng tới đầu ra ở bước cuối. Theo quy tắc, CNN giữ
L = 16. Chi tiết theo bộ và mức nhãn: results/e30_dev_detail.csv.

## Kết quả E30 (results/e30_metrics.csv, e30_tests.csv)

* So với MLP: tỉ số MAE theo cell 0,36–0,93 (CNN 0,37–0,90, LSTM 0,36–0,90, Transformer 0,43–0,93); thấp hơn có ý
  nghĩa ở 42/48 phép so. Lợi ích lớn nhất ở 10 % nhãn.
* So với GBDT: mô hình chuỗi kém ở 40/48 phép so; mô hình chuỗi tốt nhất chỉ thấp hơn GBDT ở cả bốn mức nhãn trên HUST.
* So với PINN-semi (một mạng, không lịch sử): mô hình chuỗi tốt nhất thấp hơn ở 14/16 trường hợp.
* PINN-lai thấp hơn cả ba mô hình chuỗi ở 47/48 phép so; mô hình cuối có MAE bằng 0,77–0,96 lần mô hình chuỗi tốt nhất,
  thấp hơn có ý nghĩa ở 14/16 trường hợp.
* NASA: kiểm định chéo cho MAE 7,86–9,18 % SOH (30 % nhãn) và 6,34–7,16 % (70 % nhãn), bằng 0,86–1,02 lần MLP; cách
  chia NASA-L cho 5,04–7,08 % so với 4,12 % của PINN-lai.
