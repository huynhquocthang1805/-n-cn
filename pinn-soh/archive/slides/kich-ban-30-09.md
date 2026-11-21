# Kịch bản thuyết trình — Báo cáo kết quả 30/09/2026

Nhóm 15 · Huỳnh Quốc Thắng · 15 slide · khoảng 10 phút

---

## Slide 1 — Bìa (20 giây)

Em chào thầy/cô. Em là Huỳnh Quốc Thắng, nhóm 15. Đề tài của em là dự đoán thông số sức khoẻ
của pin bằng mô hình học sâu có hỗ trợ từ thông tin hoá lý của hệ pin. Hôm nay em báo cáo kết
quả thí nghiệm chính của đề tài.

## Slide 2 — Bài toán (40 giây)

Bài toán là ước lượng SOH, tức dung lượng hiện tại chia cho dung lượng danh định. Đầu vào là 16
đặc trưng lấy từ đoạn sạc CC–CV của một chu kỳ, cộng với chỉ số chu kỳ. Đầu ra là SOH của chính
chu kỳ đó.

Khó khăn thực tế là muốn có nhãn SOH thì phải xả đầy cell để đo dung lượng, rất tốn thời gian.
Còn dữ liệu sạc thì BMS ghi được cho mọi cell. Vì vậy em đặt hai câu hỏi:

- Nếu chỉ 30 % số cell có nhãn, PINN có cho sai số thấp hơn mạng MLP thông thường không?
- PINN dùng 30 % nhãn có đạt được mức của MLP dùng tới 70 % nhãn không?

## Slide 3 — Dữ liệu và đặc trưng (40 giây)

Em dùng bốn bộ dữ liệu công khai: XJTU, TJU, MIT và HUST, tổng cộng 387 cell và khoảng 306
nghìn chu kỳ. Các bộ này khác nhau về hoá học: XJTU là NCM, TJU là NCA và NCM, còn MIT và HUST
là LFP. TJU là bộ duy nhất có nhiều mức nhiệt độ: 25, 35 và 45 độ.

Mỗi chu kỳ được mô tả bằng 16 đặc trưng thống kê của điện áp và dòng điện trong lúc sạc. Bước
lọc ngoại lai chỉ dùng các chu kỳ đã qua của cell, nên mô hình dùng được trực tuyến.

## Slide 4 — Mô hình PINN (1 phút)

Mô hình gồm hai mạng.

Mạng thứ nhất là mạng nghiệm: nhận 16 đặc trưng và chỉ số chu kỳ, cho ra SOH dự đoán. Đây là
mạng duy nhất dùng khi suy luận.

Mạng thứ hai mô tả tốc độ suy giảm. Tốc độ này là tích của ba thành phần: một mạng nhỏ theo đặc
trưng, qua softplus nên luôn không âm; một hệ số tăng dần khi SOH giảm, để mô tả hiện tượng
suy giảm nhanh ở cuối đời; và hệ số Arrhenius theo nhiệt độ. Nhờ cấu trúc này, tốc độ suy giảm
luôn lớn hơn hoặc bằng không, tức là dung lượng không thể tự tăng.

Hai mạng được nối với nhau bằng các hàm mất mát vật lý.

## Slide 5 — Huấn luyện bán giám sát (1 phút)

Điểm quan trọng nhất ở slide này là cách dùng cell không có nhãn.

Nhánh trên: với cell có nhãn, mô hình học như bình thường bằng sai số bình phương giữa dự đoán
và nhãn.

Nhánh dưới: với mọi cell trong tập huấn luyện, kể cả cell chưa đo dung lượng, em lấy hai chu
kỳ của cùng một cell và áp ba ràng buộc. Thứ nhất, độ giảm SOH giữa hai chu kỳ phải khớp với
tốc độ suy giảm. Thứ hai, SOH không được tăng theo thời gian. Thứ ba, SOH phải nằm trong khoảng
hợp lý. Cả ba ràng buộc này đều không cần nhãn.

Tổng mất mát là phần dữ liệu cộng phần vật lý, trong đó trọng số phần vật lý tăng dần trong
giai đoạn đầu để mô hình khớp dữ liệu trước.

## Slide 6 — Thiết lập thí nghiệm (50 giây)

Để so sánh công bằng, em chia dữ liệu theo cell chứ không theo chu kỳ, dùng kiểm định chéo 5
fold và lặp lại 5 lần. Như vậy mỗi cell đều được dùng làm test, và cell test không bao giờ xuất
hiện trong lúc huấn luyện.

Em so sánh ba mô hình: MLP chỉ học từ nhãn; PINN-sup có thêm ràng buộc vật lý nhưng chỉ trên
cell có nhãn; và PINN-semi áp ràng buộc vật lý lên cả cell không nhãn. Ba mô hình dùng cùng kiến
trúc mạng, cùng cách chuẩn hoá, cùng cách chia, nên khác biệt chỉ nằm ở hàm mất mát.

Lượng nhãn thay đổi từ 10 đến 70 % số cell. Chỉ số đánh giá là MAE tính riêng cho từng cell rồi
lấy trung bình, để cell có vòng đời dài không lấn át kết quả.

## Slide 7 — Kết quả 1: cùng 30 % nhãn (1 phút)

Đây là kết quả chính. Khi cả hai mô hình cùng có 30 % cell mang nhãn, PINN-semi có sai số thấp
hơn MLP trên cả bốn bộ dữ liệu.

Mức giảm lớn nhất là trên MIT, khoảng 24 %, tiếp theo là XJTU 16 % và HUST 13 %. Trên TJU mức
giảm nhỏ hơn, khoảng 4 %.

Cột cuối cho thấy cải thiện không chỉ đến từ vài cell: từ 65 đến 82 % số cell có sai số thấp hơn
khi dùng PINN-semi. Mức giảm này có ý nghĩa thống kê trên cả bốn bộ khi kiểm định theo từng cell.

## Slide 8 — Tỉ số sai số giữa các mô hình (1 phút)

Hình này gom các phép so sánh lại. Mỗi chấm là tỉ số sai số giữa hai mô hình, vạch ngang là
khoảng tin cậy. Nhỏ hơn 1 nghĩa là mô hình đứng trước có sai số thấp hơn.

Có ba ý em muốn nhấn mạnh:

- Hàng đầu tiên: PINN-semi luôn nằm bên trái vạch 1 so với MLP.
- So PINN-semi với PINN-sup: dùng thêm cell không nhãn giúp giảm sai số trên ba bộ, XJTU, MIT
  và HUST. Tức là dữ liệu không nhãn thật sự có ích.
- Hàng "chỉ đơn điệu": nếu bỏ ràng buộc phương trình suy giảm, chỉ giữ ràng buộc đơn điệu, thì
  trên MIT sai số tăng lại. Nghĩa là phần phương trình động học có đóng góp riêng.

## Slide 9 — Sai số theo lượng nhãn (1 phút)

Hình này trả lời câu hỏi thứ hai. Đường xanh dương là MLP, đường xanh lục là PINN-semi, vạch đứt
là mức của MLP khi dùng 70 % nhãn.

Nhận xét đầu tiên: càng ít nhãn, lợi thế của PINN càng rõ. Ở 10 % nhãn, sai số của PINN-semi chỉ
bằng khoảng 40 % sai số MLP trên MIT, và khoảng 60 % trên XJTU và HUST.

Nhận xét thứ hai: trên MIT và HUST, PINN-semi với 30 % nhãn đã chạm tới mức của MLP dùng 70 %
nhãn, tức là tiết kiệm được hơn một nửa lượng nhãn. Trên XJTU và TJU thì chưa đạt được mức đó.

## Slide 10 — Sai số từng cell (40 giây)

Mỗi chấm ở đây là một cell. Trục ngang là sai số của MLP, trục dọc là sai số của PINN-semi.
Chấm nằm dưới đường chéo, màu xanh lục, là cell mà PINN tốt hơn.

Có thể thấy phần lớn các chấm nằm dưới đường chéo ở cả bốn bộ, rõ nhất trên MIT. Những cell khó,
tức sai số lớn ở phía bên phải, cũng thường được PINN cải thiện.

## Slide 11 — Quỹ đạo SOH trên cell test (50 giây)

Đây là ví dụ quỹ đạo SOH trên một cell test của mỗi bộ. Em chọn cell có sai số MLP ở mức trung
bình của bộ đó, để ví dụ không bị chọn theo hướng có lợi cho PINN.

Đường đen là giá trị đo thật. Có thể thấy đường của PINN-semi trơn hơn, ít dao động hơn MLP, vì
ràng buộc vật lý không cho SOH tăng ngược. Ở HUST, cả hai mô hình đều đánh giá SOH cao hơn thực
tế ở giai đoạn cuối đời. Đây là điểm em cần cải thiện tiếp.

## Slide 12 — Cuối đời pin và độ trơn của dự đoán (50 giây)

Bảng này xem xét phần quan trọng nhất trong thực tế là giai đoạn cuối đời.

Cột thứ ba là sai số khi SOH đã xuống dưới 0,9: PINN-semi thấp hơn MLP trên cả bốn bộ. Cột thứ
tư là sai số khi xác định thời điểm SOH chạm 0,85, tính bằng số chu kỳ: PINN-semi cũng tốt hơn,
ví dụ trên XJTU giảm từ khoảng 27 xuống 17 chu kỳ.

Cột cuối đo mức dự đoán tăng ngược, tức phi vật lý. PINN-semi giảm chỉ số này khoảng một nửa
trên mọi bộ.

## Slide 13 — Chuyển sang bộ dữ liệu khác (50 giây)

Em thử huấn luyện trên một bộ rồi dùng cho bộ khác, cả hai mô hình dùng cùng cách chuẩn hoá.

Khi bộ đích không có cell nào có nhãn, MLP gần như không dùng được, sai số rất lớn. Còn
PINN-semi, nhờ áp ràng buộc vật lý lên dữ liệu sạc của bộ đích, vẫn giữ sai số ở mức vài phần
trăm. Khi có thêm 3 cell đích có nhãn thì khoảng cách giữa hai mô hình thu hẹp lại, nhưng PINN
vẫn tốt hơn hoặc ngang bằng.

## Slide 14 — Demo (40 giây)

Đây là demo ước lượng SOH cho một cell mà mô hình chưa từng thấy. Đầu vào chỉ là dữ liệu sạc từng
chu kỳ, không cần cột dung lượng. Mô hình trả về SOH theo từng chu kỳ và cảnh báo khi SOH xuống
dưới 0,85.

Với cell này, mô hình cảnh báo ở chu kỳ 367, trong khi giá trị đo thật chạm ngưỡng ở chu kỳ 374,
tức là báo sớm khoảng 7 chu kỳ.

## Slide 15 — Kết luận (50 giây)

Tóm lại:

- Cùng lượng nhãn, PINN-semi có sai số thấp hơn MLP trên cả bốn bộ dữ liệu, rõ nhất trên MIT.
- Về tiết kiệm nhãn, PINN dùng 30 % nhãn đạt mức của MLP dùng 70 % trên MIT và HUST; trên XJTU
  và TJU thì chưa đạt.
- Hướng tiếp theo của em là bổ sung thêm các mô hình đối chứng khác, ước lượng độ bất định của dự
  đoán, và cải thiện dự đoán ở giai đoạn cuối đời pin, đặc biệt trên bộ HUST.

Em xin cảm ơn thầy/cô đã lắng nghe, em rất mong nhận được góp ý.

---

## Câu hỏi có thể gặp

**Vì sao chia theo cell mà không chia theo chu kỳ?**
Nếu chia theo chu kỳ, các chu kỳ gần nhau của cùng một cell rơi vào cả train lẫn test, mô hình
gần như "nhìn thấy trước" đáp án, nên sai số thấp giả tạo.

**Vì sao cải thiện trên TJU nhỏ?**
TJU có nhiều cell và nhiều điều kiện vận hành hơn, nên MLP với 30 % nhãn đã học khá tốt; phần
ràng buộc vật lý bổ sung ít thông tin mới hơn. Ngoài ra TJU pha trộn hai hoá học NCA và NCM.

**Kết quả có chắc chắn không?**
Mỗi cấu hình được chạy 25 lần (5 fold × 5 lần lặp). Kiểm định theo từng cell cho kết quả có ý
nghĩa trên cả bốn bộ. Khi dùng phép kiểm định chặt hơn tính cả dao động do huấn luyện lại, MIT
vẫn có ý nghĩa rõ, ba bộ còn lại có xu hướng giống nhau nhưng cần thêm dữ liệu để khẳng định.

**Mô hình có dự báo được tuổi thọ còn lại (RUL) không?**
Hiện tại mô hình ước lượng SOH tại chu kỳ hiện tại từ dữ liệu sạc của chính chu kỳ đó. Dự báo
tương lai cần một thiết kế đánh giá riêng, em để ở hướng phát triển.
