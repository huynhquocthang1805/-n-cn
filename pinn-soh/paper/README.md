# Bản thảo bài báo

`main.tex` là bản thảo tiếng Anh, viết từ kết quả của đồ án (bỏ phần bộ NASA) cộng phần chạy lại PINN4SOH mới làm.
`manuscript.pdf` là bản đã dựng.

```bash
python run.py paper          # = python paper/build.py: bảng, con số, hình -> paper/manuscript.pdf (cần TeX Live)
```

Thư mục này tự đủ: `generated/` (bảng, đoạn văn có số) và `figures/` được đưa lên git, nên có thể tải nguyên thư mục
`paper/` lên Overleaf và biên dịch `main.tex` (pdfLaTeX) mà không chạy gì thêm.

## Cấu trúc bài

1. Introduction — bài toán, hai vấn đề của tài liệu (rò rỉ khi đánh giá, thiếu đối chứng mạnh), năm đóng góp.
2. Related work.
3. Data and evaluation protocol — bốn bộ dữ liệu, lọc ngoại lai nhân quả, kiểm định chéo theo cell phân tầng, chỉ số và
   thống kê.
4. Methods — PINN bán giám sát với động học xám, PINN-hybrid (tên tiếng Anh của PINN-lai), các mô hình đối chứng,
   ngân sách đo và cập nhật khi dùng (Kalman, mô hình học cập nhật).
5. Results — kiểm toán rò rỉ; hiệu quả nhãn; nguồn gốc lợi ích; mô hình lai và mô hình chuỗi; so với PINN4SOH; ngân
   sách đo và cập nhật cell đang dùng.
6. Discussion, Limitations. 7. Conclusion.

## Số liệu lấy từ đâu

Không số nào gõ tay vào bảng:

| Phần | Nguồn |
|---|---|
| Bảng dữ liệu, sai số theo mức nhãn, tách nguồn lợi ích, mô hình chính, ngân sách đo | bảng của báo cáo đồ án (`archive/thesis/do-an/main.tex`), đọc bằng `thesis_tables.py` |
| So với PINN4SOH (bảng, hình, cả mục 5.4) | `results/benchmark_summary.csv`, `benchmark_tests.csv` từ `benchmark/compare.py`; đối chiếu nhật ký công bố ở `runs/pinn4soh/logged_constant.json` |
| Kiểm toán rò rỉ | `results/thesis_leakage.csv` (chạy lại E15 và nhánh F của E13); khi chưa có thì dùng các tỉ số báo cáo đồ án nêu |

Các con số trong lời văn của mục 5.1–5.3, 5.5 lấy từ lời văn và bảng của báo cáo đồ án (sinh từ `results/` của lần
chạy ở commit `2d66f75`; thư mục `runs/` của lần chạy đó không còn). Bảng của các mục đó sinh lại đúng từ báo cáo; khi
chạy lại các thí nghiệm, `results/` mới là nguồn chính.

## Việc còn lại trước khi nộp

* Tác giả: thêm giảng viên hướng dẫn / đồng tác giả, email liên hệ (đã để chú thích `TODO` trong `main.tex`).
* Chọn tạp chí và đổi sang khuôn của tạp chí (ví dụ Journal of Energy Storage, Applied Energy, eTransportation, Energy
  and AI); hiện dùng lớp `article` để dễ chuyển.
* Phạm vi: bản thảo gom cả năm đóng góp. Nếu tạp chí giới hạn độ dài, có thể tách thành hai bài: (a) giao thức không
  rò rỉ, PINN bán giám sát, mô hình lai, so với PINN4SOH; (b) ngân sách đo và cập nhật cell đang dùng.
* Mục cập nhật cell đang dùng (5.5, đoạn cuối) mới có số trường hợp thắng và khoảng tỉ số; muốn có bảng MAE theo từng
  bộ × mức đo cần chạy lại E20 và E32 rồi `analysis/analyze_e32.py` (khoảng 15 giờ CPU).
* Kiểm tra lại từng tài liệu tham khảo (`refs.bib`) và thêm các công trình mới nhất của tạp chí định nộp.
* Có thể thêm một hình tổng quan phương pháp (giao thức, mô hình, cập nhật khi dùng).

## Các điểm cần nói rõ khi viết tiếp

* Các biến thể "tuned" và lựa chọn có/không lịch sử là phân tích hậu kiểm (chọn trên validation nhưng đặt ra sau khi đã
  xem kết quả test của bản cố định); bài đã ghi rõ ở bảng và ở mục Limitations.
* Chạy lại PINN4SOH dùng CPU và thư viện mới (pandas 2.3, torch 2.14); bài gốc dùng GPU, torch 1.7.1. Độ lệch giữa các
  lần chạy được báo cáo cạnh mỗi số.
* Số công bố của PINN4SOH đến từ bản mã có loss đơn điệu không có gradient (đối chiếu ở cả 39 nhóm). Bài nêu đây là một
  khác biệt giữa mã công bố và phương pháp mô tả, không suy đoán thêm về ý định.
