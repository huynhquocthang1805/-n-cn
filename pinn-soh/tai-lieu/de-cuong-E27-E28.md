# Đề cương E27 và E28

Chốt trước khi đọc kết quả test của hai thí nghiệm. Mọi lựa chọn dưới đây dựa trên tập validation (lần lặp 0) hoặc
trên lập luận, không dựa trên tập test.

## E27 — so với PINN4SOH trên đúng cách chia của bài

* Cách chia: dựng lại từ nhật ký huấn luyện công bố kèm mã nguồn PINN4SOH (dòng `data path:` của mỗi lần chạy).
  39 nhóm: XJTU 6 batch, TJU 3 batch, MIT và HUST cả bộ; thí nghiệm ít mẫu K = 1..4 cell có nhãn cho XJTU (theo
  batch) và HUST. Danh sách lưu ở `pinnsoh/pinn4soh_splits.json`; danh sách giống nhau ở cả 10 lần chạy của bài.
* Số liệu so sánh: MAE theo cell trên cell test, trung bình qua 10 lần chạy rồi qua nhóm, lấy từ các tệp kết quả
  công bố (`results analysis/processed results*`). Giá trị từng cell (chế độ đầy đủ) ở `pinnsoh/pinn4soh_reported.csv`.
* Mô hình của đồ án, không chỉnh siêu tham số cho cách chia này: 3 PINN-semi (cấu hình E13) → PINN-lai đúng công thức
  E20 (½ trung bình 3 mạng + ½ GBDT, bộ lọc quỹ đạo w = 10); GBDT cấu hình E16; ở các nhóm đầy đủ thêm một PINN động
  học hộp đen kiểu PINN4SOH. 3 hạt giống.
* Validation như bài: 20 % hàng ngẫu nhiên của các cell có nhãn. Ở chế độ ít mẫu, đặc trưng sạc của các cell train
  còn lại của nhóm vào các loss vật lý (không dùng nhãn của chúng). Cell test không được dùng dưới dạng nào.
* Hai cách tiền xử lý: `strict` (của đồ án) và `paper` (bản sao đúng hàm `delete_3_sigma` và min–max theo từng cell của
  bài, cho cả đặc trưng lẫn chỉ số chu kỳ). Kết luận chính dựa trên `strict`; `paper` chỉ để so cùng điều kiện.
* Kiểm định: Wilcoxon ghép cặp theo cell test giữa PINN-lai (trung bình 3 hạt giống) và giá trị từng cell của PINN4SOH.

## E28 — ngân sách đo dung lượng

* Cùng fold và hạt giống với E20 (lần lặp 0–2, 5 fold), mọi cell train (70 % số cell) được đo ở chu kỳ đầu, sau mỗi
  q·L chu kỳ (L = trung vị số chu kỳ của các cell train), và ở chu kỳ cuối của thử nghiệm; q ∈ {0,5; 0,2; 0,1; 0,05}.
* Lần đo cuối được thêm sau khi khám phá trên validation (lần lặp 0, chỉ GBDT): không có nó, các cell ngắn hơn một
  khoảng đo (nhóm 1C của TJU, 23–35 chu kỳ) chỉ có một lần đo ở chu kỳ đầu và nhãn nội suy sai tới 0,18 SOH.
* Mô hình: 3 PINN-semi chỉ có nhãn ở lần đo, loss vật lý trên mọi hàng → PINN-lai (GBDT học nhãn nội suy tuyến tính);
  đối chứng: MLP chỉ trên lần đo, GBDT trên lần đo, GBDT trên nhãn nội suy, một PINN-semi học nhãn nội suy (q = 0,1;
  0,05). Cell validation có nhãn đầy đủ như E20; cell test không có nhãn.
* So sánh chính: PINN-lai đo thưa so với PINN-lai và GBDT theo cell (E20) ở 30 % và 70 % nhãn, Wilcoxon ghép cặp
  theo cell test (trung bình 3 lần lặp). Trục chung: phần trăm số chu kỳ của cell train có đo dung lượng.
* Đo thưa cả khi dùng: cell test được đo theo cùng lịch q (không có lần đo cuối), bộ lọc Kalman của E25 (tham số trên
  validation); sai số trên các chu kỳ không được đo. So với cùng bộ lọc trên PINN-lai học với 70 % cell có nhãn.

## Đã thử trên validation và không giữ lại

* Đặc trưng tương đối Δx = x − x(3 chu kỳ đầu) theo đơn vị z: GBDT kém đi trên XJTU, MIT, HUST.
* Đặc trưng làm trơn mũ nhân quả (α = 0,3): GBDT lẫn lộn theo fold; PINN-semi ở 30 % nhãn tốt hơn trên XJTU, kém hơn
  trên HUST.
* GBDT tinh chỉnh (tốc độ học, số lá, số mẫu tối thiểu mỗi lá): lẫn lộn theo fold.
* Làm trơn nhãn của cell train (trung vị/trung bình trượt): XJTU kém đi, TJU tốt hơn, MIT không đổi.
