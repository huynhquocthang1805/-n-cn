# Đề cương E20: cải tiến mô hình, chốt trước khi đánh giá trên test

Ngày chốt: 20/10/2026. Tài liệu này được commit trước khi đọc bất kỳ dự đoán test nào của lượt chạy cuối
E20, nên lịch sử git cho thấy cấu hình không bị chọn lại sau khi nhìn kết quả test.

## 1. Giai đoạn phát triển (E20dev, đã xong)

* Dữ liệu: lần lặp 0, 5 fold, 4 bộ, mức nhãn 10 % và 50 %. Mọi quyết định dựa trên MAE theo cell của
  tập validation (khoảng 10 % số cell của mỗi fold, tách khỏi train và test). Dự đoán test của các lượt
  này không được đọc.
* Thước đo chọn: tỉ số MAE theo cell so với PINN-semi đơn lẻ (cấu hình E13), lấy trung bình nhân trên 8
  tổ hợp bộ × mức nhãn (`analysis/dev_e20.py`, `results/e20_dev.csv`).

| Hướng thử | Tỉ số validation |
|---|---|
| 3 PINN-semi + GBDT (trung bình đều) + bộ lọc quỹ đạo w = 10 | 0,823 |
| 3 PINN-semi + GBDT (trung bình đều) | 0,828 |
| GBDT đơn lẻ | 0,906 |
| chưng cất từ GBDT, trọng số 0,3 | 0,961 |
| 3 PINN-semi (ensemble) | 0,963 |
| chưng cất từ GBDT, trọng số 1 | 0,976 |
| ràng buộc đơn điệu mạnh hơn (β = 2) | 1,065 |
| thêm phần dư Euler (α_E = 2) | 1,289 |
| đặc trưng tuyệt đối + tương đối so với 3 chu kỳ đầu | 1,692 |

Trọng số GBDT 0,5 và 0,6 cho kết quả gần như nhau (0,828 và 0,824); chọn 0,5 vì không cần ước lượng.
Cửa sổ lọc 5, 10, 20 gần như nhau (0,822, 0,823, 0,827); chọn 10 vì gần tốt nhất với mọi mô hình nền
đã thử. Chưng cất không còn lợi ích khi đã kết hợp với GBDT (0,849 so với 0,851) nên bị bỏ.

## 2. Mô hình cải tiến (PINN-lai)

1. Ba mạng PINN-semi cấu hình E13, chỉ khác hạt giống khởi tạo và thứ tự lấy batch
   (`Config.member` 0, 1, 2); cùng cách chia, cùng tập cell có nhãn. Thành viên 0 trùng từng bit với E13.
2. GBDT cấu hình E16 học trên đúng các cell có nhãn đó, cùng normaliser.
3. Dự đoán thô: ŷ = ½·(trung bình 3 PINN-semi) + ½·GBDT.
4. Bộ lọc quỹ đạo nhân quả: tại chu kỳ n của một cell, hồi quy tuyến tính trên 10 điểm gần nhất (kể cả n)
   của chính ŷ, độ dốc bị chặn trên bởi 0, lấy giá trị đường thẳng tại n (`pinnsoh/postproc.py`).
   Chỉ dùng dự đoán của các chu kỳ đã qua, không dùng nhãn.

## 3. Đánh giá cuối (E20)

* Lần lặp 0, 1, 2 (giới hạn tính toán), 5 fold, 4 bộ, 4 mức nhãn 10/30/50/70 %. Mọi so sánh trong chương
  cải tiến tính trên đúng 3 lần lặp này, kể cả cho các mô hình cũ.
* Chỉ số chính: MAE theo cell. Phụ: RMSE theo cell, R² gộp từ chu kỳ 10, tỉ lệ cặp tăng ngược.
* So sánh chính: PINN-lai với PINN-semi, với GBDT và với MLP. Kiểm định Wilcoxon ghép cặp theo cell
  (MAE trung bình qua 3 lần lặp), hiệu chỉnh Holm trên 4 bộ ở mỗi mức nhãn; tỉ số kèm KTC 95 % bootstrap.
* Phân rã: PINN-semi → 3 PINN-semi → + GBDT → + bộ lọc. Nhánh đối chứng không vật lý: thay 3 PINN-semi bằng
  3 MLP (thành viên 0 từ E13, 1–2 chạy mới), giữ nguyên GBDT và bộ lọc.
* Khoảng dự đoán: dự đoán phù hợp tách tập (split conformal) mức 90 %, hiệu chỉnh trên dự đoán validation
  của chính fold, hai dạng: độ rộng cố định, và độ rộng tỉ lệ với độ lệch giữa trung bình PINN và GBDT.
  Báo cáo độ phủ thực tế và độ rộng trung bình trên test.

## 4. Diễn giải

Kết quả trên test được báo cáo nguyên trạng, kể cả khi PINN-lai không hơn GBDT ở mức nhãn nào đó.
Giai đoạn phát triển dùng các cell validation của lần lặp 0; các cell này nằm trong test ở fold khác, nên
có một thiên lệch chọn nhỏ. Với 9 hướng thử và chênh lệch lớn giữa hướng được chọn và các hướng còn lại,
thiên lệch này nhỏ so với hiệu ứng cần đo; chương kết quả so sánh mức cải thiện trên validation và trên test.
