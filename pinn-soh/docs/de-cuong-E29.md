# Đề cương E29 — mô hình dùng lịch sử của cell, và bộ lọc quỹ đạo

Chốt trước khi đọc kết quả test của E29. Mọi lựa chọn dựa trên tập validation (lần lặp 0, 5 fold, 30 và 70 % nhãn).

## Hai điểm cần sửa

1. Mạng nghiệm của PINN-semi dự đoán SOH ở mỗi chu kỳ chỉ từ 16 đặc trưng của chính chu kỳ đó (và chỉ số chu kỳ).
2. Bộ lọc quỹ đạo của PINN-lai (khớp đường thẳng có độ dốc <= 0 trên 10 chu kỳ gần nhất) không bảo đảm chuỗi đầu ra không
   tăng: mỗi chu kỳ dùng một cửa sổ khác. Trên test của E20, đầu ra tăng ở 28–44 % số bước giữa hai chu kỳ liên tiếp.

## Bộ lọc

* Bản có bảo đảm `causal_mono`: m_n = m_{n-1} + a·min(0, z_n − m_{n-1}), z = bộ lọc xu thế. (w, a) chọn trên validation
  riêng cho từng bộ (w ∈ {5, 10, 20, 40}, a ∈ {1; 0,5; 0,3; 0,1}); so với bộ lọc xu thế trên test (analysis/analyze_mono.py).
* Kết quả validation đã biết trước khi chốt: bản đơn điệu tốt hơn rất ít trên TJU và HUST, kém hơn trên MIT và kém rất
  nhiều trên XJTU, vì dung lượng đo được của XJTU (và NASA) có tăng thật (đầu đời, sau khi nghỉ). PINN-lai giữ bộ lọc xu
  thế; mô tả trong báo cáo sửa thành "làm trơn theo xu thế, không bảo đảm đơn điệu".

## Đặc trưng lịch sử (data.history_features)

Ở chu kỳ n chỉ dùng chu kỳ n và các chu kỳ trước của cùng cell. Bốn dạng đã thử với PINN-semi và GBDT: 'ema' (chỉ trung
bình trượt mũ a = 0,3), 'h:x+e0.1', 'h:x+e0.3+e0.1+e0.03', 'h:x+e0.1+e0.03+s20' (GBDT thêm 'h:x+e0.1+r5', 'h:x+s20').

Quy tắc chọn: với mỗi bộ và mỗi loại mô hình, lấy dạng có MAE validation trung bình nhỏ nhất qua hai mức nhãn, và chỉ giữ
nếu nó thấp hơn bản không lịch sử ít nhất 2 %; nếu không, mô hình giữ đặc trưng gốc.

| Bộ | PINN-semi | GBDT |
|---|---|---|
| XJTU | `h:x+e0.1` (tỉ số validation 0,93) | `h:x+e0.1` (0,97) |
| TJU | `h:x+e0.1` (0,87) | đặc trưng gốc (tốt nhất 0,997) |
| MIT | đặc trưng gốc (tốt nhất `ema` 0,99) | đặc trưng gốc (mọi dạng > 1) |
| HUST | `h:x+e0.1+e0.03+s20` (0,94) | `h:x+e0.1+e0.03+s20` (0,95) |
| NASA, NASAdt | đặc trưng gốc (mọi dạng kém hơn ở ít nhất một cách chia) | đặc trưng gốc (1,07–1,97) |

## Đánh giá cuối

* Cùng fold, hạt giống và mức nhãn với E20 (3 lần lặp × 5 fold × 4 mức nhãn), trên XJTU, TJU, HUST: PINN-semi có lịch sử
  (3 thành viên), MLP có lịch sử (thành viên 0, để tách tác động của vật lý khi đã có lịch sử), GBDT có lịch sử (XJTU, HUST).
* PINN-lai có lịch sử: ½ trung bình 3 PINN-semi có lịch sử + ½ GBDT (bản đã chọn cho bộ đó), bộ lọc xu thế w = 10 — đúng
  công thức PINN-lai của E20. Bản tinh chỉnh: (ω, w) chọn trên validation như phần tinh chỉnh E20.
* So sánh ghép cặp theo cell test (trung bình 3 lần lặp), Wilcoxon: PINN-semi có/không lịch sử, PINN-lai có/không lịch
  sử, PINN-lai có lịch sử với GBDT tốt nhất của bộ, MLP có lịch sử với PINN-semi có lịch sử.
* MIT và NASA không có lượt mới: dạng lịch sử không được chọn trên validation, kết quả giữ như E20 và E24.

## Kết quả và phân tích hậu kiểm (ghi sau khi đã xem test)

* Theo đề cương: lịch sử giúp trên TJU và HUST khi có từ 30 % nhãn (PINN-lai tinh chỉnh 0,89–0,97 lần bản không lịch
  sử, có ý nghĩa ở 6/6 trường hợp), nhưng làm PINN-semi kém đi trên XJTU ở mọi mức nhãn (1,05–1,42 lần) và không giảm
  sai số có ý nghĩa ở 10 % nhãn trên bộ nào.
* Lựa chọn trên XJTU sai vì chỉ dựa trên lần lặp 0: tỉ số MAE validation của PINN-semi có / không lịch sử ở 30 và 70 %
  nhãn là 0,92–0,94 với lần lặp 0, 1,30–1,50 với lần lặp 2, trung bình ba lần lặp 1,06–1,17 (test 1,06–1,15). Mỗi
  fold của XJTU chỉ có 6 cell validation.
* Phân tích hậu kiểm, đặt ra sau khi đã xem kết quả test ở trên nhưng chỉ dùng validation của đủ 3 lần lặp × 5 fold:
  với mỗi (bộ, mức nhãn), PINN-lai dùng hay không dùng lịch sử cùng (ω, w) được chọn theo MAE validation; GBDT có hay
  không lịch sử chọn theo cùng cách (analysis/analyze_e29.py: pinn_lai_sel_v2, gbdt_best). Validation chọn lịch sử ở
  TJU và HUST với 30, 50, 70 % nhãn, trùng các trường hợp lịch sử có lợi trên test. Mô hình này thấp hơn GBDT ở 16/16
  trường hợp (tỉ số 0,80–0,99, 13 có ý nghĩa); so với GBDT có hay không lịch sử, có hay không bộ lọc xu thế
  (w ∈ {5, 10, 20, 40}) chọn trên validation: 16/16 thấp hơn (0,82–0,99), 11 có ý nghĩa, không trường hợp nào cao hơn
  có ý nghĩa.
* Khi cả hai cùng có lịch sử, PINN-semi so với MLP: 0,31–0,89 ở 10 % nhãn; 0,68–0,92 trên XJTU và HUST từ 30 % nhãn;
  1,03–1,06 trên TJU (MLP tốt hơn, có ý nghĩa ở 70 %), giống như khi không có lịch sử.
* Tỉ số GBDT trong bảng chọn ở trên lấy từ lượt khám phá; khi chạy lại đầy đủ trong E29dev (analysis/dev_e29.py),
  TJU là 1,00 thay vì 0,997, các lựa chọn không đổi.
