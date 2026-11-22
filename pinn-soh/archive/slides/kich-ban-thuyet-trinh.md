# Kịch bản thuyết trình — ĐACN HK261, nhóm 15

**Bộ slide:** `DACN_16_09_Group 15.pdf`, 16 slide · **Thời lượng đo được:** ~14 phút + hỏi đáp
**Người trình bày:** Huỳnh Quốc Thắng · 16/09/2026

Phần in thường là lời nói. Phần **in đậm** là chỗ nhấn giọng. Phần *in nghiêng trong ngoặc* là ghi chú thao tác, không đọc.

---

## Slide 1 — Trang bìa · 20 s

Em chào Thầy và các bạn. Em là Huỳnh Quốc Thắng, nhóm 15.

Đề tài của em là **dự đoán thông số sức khoẻ của pin bằng mô hình học sâu được hỗ trợ từ thông tin hoá lý của hệ pin**.

Hôm nay em báo cáo tiến độ: những việc đã làm, các khó khăn gặp phải, và kế hoạch hai tuần tới.

*(Chuyển slide ngay, đừng dừng lâu ở bìa.)*

---

## Slide 2 — Mục tiêu và tiến độ · 45 s

Trước hết em xin đối chiếu với bốn yêu cầu của đồ án chuyên ngành.

**Xác định bài toán** — đã xong: ước lượng SOH của từng cell trong điều kiện ít nhãn dung lượng.

**Khảo sát research và công nghệ** — em đã khảo sát trọng tâm vào PINN4SOH, gồm cả dữ liệu và cách họ đánh giá.

**Baseline và thiết kế lời giải** — đã có: một MLP thuần dữ liệu làm đối chứng, và một PINN bán giám sát làm lời giải đề xuất.

**Phần thí nghiệm, vốn là optional** — em đã chạy được kết quả sơ bộ trên CPU với ba seed. Phần demo thì **chưa hoàn tất**.

Ba mục bắt buộc đã đi hết, mục thứ tư mới ở mức sơ bộ.

*(Chuyển tiếp)* Em xin đi vào chi tiết, bắt đầu từ bài toán.

---

## Slide 3 — Bài toán và câu hỏi nghiên cứu · 55 s

Đầu vào của mô hình là **16 đặc trưng thống kê của một đoạn sạc CC–CV**, cộng với chỉ số chu kỳ. Đầu ra là SOH, tức dung lượng hiện tại chia cho dung lượng danh định.

Điểm mấu chốt nằm ở bối cảnh. Trong thực tế, hệ quản lý pin **ghi được đường sạc của mọi cell** — cái đó gần như miễn phí. Nhưng muốn có **nhãn** SOH thì phải xả đầy cell để đo dung lượng, tốn hàng giờ và phải dừng vận hành. Nên nhãn chỉ có cho một phần nhỏ số cell.

Từ đó em đặt câu hỏi nghiên cứu: **liệu các ràng buộc suy giảm có khai thác được những cell không nhãn hay không?** Cụ thể hoá thành một giả thuyết đo được:

> PINN dùng **30 %** cell có nhãn có đạt được MAE tương đương MLP dùng đủ **70 %** hay không?

Để so sánh công bằng, em **giữ nguyên tập validation và test** khi giảm lượng nhãn. Chỉ số cell mang nhãn trong tập train thay đổi, còn lại giữ nguyên.

*(Chuyển tiếp)* Để trả lời câu hỏi đó, em đã khảo sát các hướng hiện có.

---

## Slide 4 — Khảo sát và hướng thiết kế được chọn · 55 s

Em khảo sát bốn nhóm.

**MLP thuần dữ liệu** — em lấy làm **baseline**, vì muốn chứng minh vật lý có ích thì phải so với một mô hình không có vật lý, cùng kích thước.

**PINN4SOH của Wang và cộng sự năm 2024** — công trình gần nhất với đề tài. Em kế thừa dữ liệu và bộ đặc trưng của họ để kết quả đối chiếu được, đồng thời **kiểm tra lại cách họ viết loss và chia dữ liệu**. Phần kiểm tra này dẫn tới các phát hiện ở slide 12.

**PINN bán giám sát** — hướng em chọn: viết loss vật lý sao cho **không dùng tới nhãn**, nhờ vậy áp được lên cả cell chỉ có đường sạc.

Nhóm thứ tư gồm GPR, XGBoost, CNN, GRU và TCN. Em xin nói thẳng là nhóm này **chưa có kết quả** trong báo cáo hiện tại. Đây là thiếu sót, nằm trong kế hoạch mở rộng.

*(Chuyển tiếp)* Tiếp theo là dữ liệu em dùng.

---

## Slide 5 — Dữ liệu và đặc trưng · 40 s

Em dùng bốn bộ dữ liệu đơn pin công khai: **XJTU, TJU, MIT và HUST**. Tổng cộng **387 cell** và hơn **304 nghìn chu kỳ** sau khi lọc.

Điểm đáng chú ý là bốn bộ này **không đồng nhất**: XJTU là NCM ở 25 độ, TJU là NCA và NCM ở **ba mức nhiệt độ**, MIT và HUST đều là LFP ở 30 độ. Khác biệt về hoá học và nhiệt độ này sẽ quay lại ở phần kết quả — nó là lý do mô hình hành xử khác nhau giữa các bộ.

Mỗi chu kỳ có 16 đặc trưng thống kê ở đoạn cuối pha CC và pha CV. Em dùng đúng bản tiền xử lý từ repo PINN4SOH.

*(Chuyển tiếp)* Với dữ liệu đó, em thiết kế protocol so sánh như sau.

---

## Slide 6 — Protocol và các baseline đối chứng · 50 s

Em so ba mô hình, khác nhau **đúng một chỗ**: loss vật lý được áp lên đâu.

**MLP** không dùng loss vật lý.
**PINN-sup** dùng loss vật lý, nhưng **chỉ trên cell có nhãn**.
**PINN-semi** dùng loss vật lý trên **mọi cell train, kể cả cell không nhãn**.

Tách ba mô hình như vậy là để phân biệt hai cơ chế. PINN-sup cho biết vật lý có tác dụng như một **regularizer** hay không. PINN-semi cho biết thêm phần **khai thác cell không nhãn** đóng góp bao nhiêu.

Về chia dữ liệu: em chia **theo cell**, 70 – 15 – 15, và có assert kiểm tra không cell nào nằm ở hai tập. Các mức nhãn 10, 30, 50, 70 phần trăm tính trên tổng số cell, và **validation với test giữ nguyên** trong cùng một seed.

Chuẩn hoá thì z-score fit trên tập train, chỉ số chu kỳ chia cho hằng số 1000. Chọn checkpoint theo MAE validation.

*(Chuyển tiếp)* Đây là kiến trúc cụ thể.

---

## Slide 7 — Pipeline model · 1 ph 25 s

*(Slide này dày, đi theo thứ tự từ trái sang phải, chỉ tay theo từng khối.)*

Bắt đầu từ trái. Dữ liệu sạc của bốn họ hoá học đi qua bước trích xuất đặc trưng, cho ra **vector 16 chiều** cộng với chỉ số chu kỳ đã chia 1000. Tổng cộng **17 chiều** vào mạng.

*(Chỉ khối "Mạng nghiệm F")* Phần dưới bên trái là **mạng nghiệm**. Đây là một MLP 17 vào, ba lớp ẩn 64, 64, 32, kích hoạt SiLU, ra một số là SOH. **Khi triển khai thật thì chỉ dùng mạng này** — phần còn lại chỉ tồn tại lúc huấn luyện.

*(Chỉ khối "Động học xám")* Bên phải là **động học xám**. Nó không dự đoán SOH mà dự đoán **tốc độ suy giảm r**. Cấu trúc gồm ba thừa số:

Thừa số đầu là **softplus của một mạng nhỏ**, nên **r luôn không âm** — đây là ràng buộc theo cấu trúc, không phụ thuộc tham số học được. Nghĩa là dung lượng không thể tự tăng lên.

Thừa số thứ hai là **e mũ lambda nhân một trừ SOH**, mô tả giai đoạn suy giảm tăng tốc khi pin đã yếu.

Thừa số thứ ba là **Arrhenius theo nhiệt độ**, phản ánh việc các phản ứng phụ gây suy giảm là phản ứng được kích hoạt nhiệt.

*(Chỉ hàng loss ở dưới)* Cuối cùng là bốn hàm mất mát. **Chỉ Ldata cần nhãn.** Ba hàm còn lại — Lode, Lmono, Lrange — **không dùng tới nhãn**. Tổng loss cập nhật đồng thời tham số của cả hai mạng, cộng với hai tham số vật lý lambda và Ea.

*(Chuyển tiếp)* Cơ chế bán giám sát hoạt động như thế nào, em xin làm rõ ở slide sau.

---

## Slide 8 — Huấn luyện bán giám sát · 55 s

Trong một bước huấn luyện có **hai luồng dữ liệu chạy song song**.

*(Chỉ luồng trên)* Luồng trên là các cell **có nhãn**. Mạng nghiệm dự đoán SOH, so với nhãn thật, ra **Ldata**.

*(Chỉ luồng dưới)* Luồng dưới mới là điểm chính. Nó lấy các **cặp chu kỳ N và N cộng h** từ **mọi cell train**, kể cả cell không có nhãn. Cùng một mạng nghiệm dự đoán cho cả hai chu kỳ, rồi ba loss vật lý so sánh hai dự đoán đó với nhau. **Không có y trong biểu thức nào cả.**

Vì không cần nhãn, nên những cell chỉ có đường sạc vẫn tham gia huấn luyện được. Chúng không dạy mạng SOH bằng bao nhiêu, nhưng **buộc mạng phải gán cho chúng một quỹ đạo hợp lý**: đơn điệu giảm, tốc độ không âm, nằm trong miền hợp lý.

Hai luồng cộng lại thành tổng loss. Có một trọng số w tăng dần theo bước, để mạng bám dữ liệu trước rồi vật lý mới siết vào sau.

*(Chuyển tiếp)* Bây giờ là kết quả.

---

## Slide 9 — Kết quả ít nhãn · 1 ph 15 s

Đây là kết quả chính, trả lời trực tiếp giả thuyết ban đầu.

Trục dọc là **tỉ số MAE so với MLP dùng đủ 70 % nhãn**. Mốc tham chiếu là **1,00** — dưới mốc là tốt hơn, trên mốc là kém hơn. Cả hai cột đều chỉ dùng **30 % nhãn**.

*(Chỉ MIT và HUST)* Trên **MIT**, PINN-semi đạt **0,91 lần**, tức dùng 30 % nhãn mà **tốt hơn** MLP dùng 70 %. Trên **HUST** là **0,96 lần**, cũng đạt. Trong khi MLP cùng 30 % nhãn thì kém hơn hẳn, 1,35 và 1,17 lần.

*(Chỉ XJTU và TJU)* Nhưng trên **XJTU** và **TJU** thì **chưa đạt**. XJTU là 1,25 lần. TJU thì đáng chú ý hơn: PINN 1,17 lần trong khi MLP chỉ 1,04 — tức ở TJU, **thêm vật lý lại làm kém đi**.

Em xin phát biểu trung thực: **giả thuyết đạt ở 2 trên 4 bộ**, ở cấu hình E1.

Cách em hiểu hiện tượng này: vật lý là một **tiên nghiệm**. Nó bù được chỗ thiếu nhãn, và bù nhiều nhất ở nơi mô hình chỉ-dữ-liệu đói nhãn nhất. Còn ở TJU, 30 % nhãn vốn đã gần đủ cho MLP — MLP chỉ kém 1,04 lần thôi — nên ràng buộc thêm vào trở thành **thiên kiến** chứ không còn là thông tin.

*(Chuyển tiếp)* Kết quả trên vẫn có một điểm yếu, và em đã tự kiểm chứng ở thí nghiệm sau.

---

## Slide 10 — E12: đối chứng sau tinh chỉnh siêu tham số · 1 ph 05 s

**Điểm yếu là thế này.** Ở kết quả vừa rồi, PINN có siêu tham số được chọn theo validation, còn MLP baseline thì dùng cấu hình mặc định. So sánh như vậy **chưa công bằng** — phần thắng có thể chỉ đến từ việc một bên được tinh chỉnh còn một bên thì không.

Nên em chạy lại thí nghiệm E12: cho **cả hai mô hình cùng một lưới 8 biến thể siêu tham số**, cùng cách chọn theo validation, rồi mới đọc kết quả test.

*(Chỉ cột cuối)* Sau khi công bằng hoá, **PINN vẫn tốt hơn ở 7 trên 8 cấu hình**, tỉ số trung bình khoảng **0,90 lần**.

*(Chỉ dòng TJU / 30 %)* Trường hợp duy nhất PINN không thắng là **TJU ở 30 % nhãn, 1,01 lần** — tức hoà.

Em cũng xin nói rõ một điều: **khoảng cách bị thu hẹp lại đáng kể** sau khi công bằng hoá. Riêng trên MIT ở 30 % nhãn, baseline MLP tự nó cải thiện từ 0,0103 xuống **0,0048** chỉ nhờ được tinh chỉnh. Nên **con số 0,90 lần này mới là con số nên dùng khi báo cáo**, không phải con số ở slide trước.

*(Chuyển tiếp)* Thí nghiệm tiếp theo là chỗ em thấy ràng buộc không nhãn có giá trị rõ nhất.

---

## Slide 11 — E5: thích nghi miền · 55 s

Câu hỏi ở đây là: huấn luyện trên một bộ rồi đem sang bộ khác thì sao?

Cột giữa là **MLP zero-shot** — đem thẳng sang bộ đích, không tinh chỉnh. Kết quả **thất bại hoàn toàn**: XJTU sang TJU sai tới **2,94**, lớn hơn cả giá trị cần đoán. Lý do là đặc trưng đường sạc phụ thuộc protocol, nên đầu vào bộ đích nằm ngoài phân bố đã học.

Cột phải là PINN với **k bằng 0** — **không một nhãn nào của bộ đích**. Nhưng vì loss vật lý không cần nhãn, chúng chạy được ngay trên cell train của bộ đích. Kết quả: **2,94 xuống 0,041**; HUST sang MIT từ 0,369 xuống **0,027**.

Em xin nói rõ: đây **không phải zero-shot** — mô hình có nhìn thấy đặc trưng của bộ đích. Nhưng nó **không cần đo dung lượng cell nào cả**. Với dây chuyền mới, đó là khác biệt lớn về chi phí.

*(Chuyển tiếp)* Trong quá trình làm, em có ba phát hiện ảnh hưởng ngược lại tới thiết kế.

---

## Slide 12 — Các phát hiện ảnh hưởng đến lời giải · 1 ph 25 s

**Thứ nhất, chuẩn hoá có thể tạo ra kết quả giả.**

Trong thí nghiệm E4 trên XJTU, MAE đi từ 0,0092 xuống **0,0078** chỉ vì đổi cách chuẩn hoá chỉ số chu kỳ — chuẩn hoá theo min-max trên **cả vòng đời của cell**. Lý do là phép chia đó biến chỉ số chu kỳ thành **phần trăm tuổi thọ đã đi qua**. Mà lúc dự đoán thật thì cell chưa hỏng, **chưa biết tuổi thọ tổng** là bao nhiêu, nên đại lượng đó không tồn tại. Từ đó em rút ra nguyên tắc: mọi phép biến đổi phải **tính được tại thời điểm dự đoán**.

**Thứ hai, đơn điệu theo t chưa đảm bảo đơn điệu thật.**

Thí nghiệm E9 cho thấy kiến trúc đơn điệu có cải thiện MAE. Nhưng nó chỉ đảm bảo SOH giảm theo **biến t** khi giữ nguyên đặc trưng. Trong thực tế **đặc trưng cũng thay đổi theo chu kỳ**, nên SOH dự đoán dọc theo một cell **vẫn có thể tăng ngược**. Đây là việc em còn phải xử lý.

**Thứ ba, thứ hạng phụ thuộc vào chỉ số đánh giá.**

Trên TJU, MLP có MAE tốt hơn — 0,0088 so với 0,0099. Nhưng PINN lại có **sai số dự báo thời điểm hết đời thấp hơn** — 24,5 so với 29,8 chu kỳ. Hai chỉ số cho **hai kết luận ngược nhau trên cùng một bộ dữ liệu**.

Nên em đề xuất chốt **MAE theo cell** làm chỉ số chính, và báo cáo thêm MAE ở vùng cuối vòng đời.

*(Chuyển tiếp)* Đó cũng dẫn sang phần khó khăn.

---

## Slide 13 — Khó khăn và trạng thái xử lý · 55 s

Bốn khó khăn chính.

**Sai phân khuếch đại nhiễu.** Ban đầu em viết phần dư ở dạng đạo hàm, tức chia cho delta t. Với hai chu kỳ liền nhau thì mẫu số rất nhỏ, **nhiễu bị khuếch đại hàng nghìn lần**, lấn át tín hiệu thật. Em đã xử lý bằng **dạng Euler với horizon ngẫu nhiên** — nhân hai vế lên thay vì chia xuống.

**Ràng buộc vật lý quá mạnh.** Trọng số beta cố định thì có bộ bị siết quá tay. Em đã thử chọn beta theo validation và một biến thể trọng số tự thích ứng.

**Chuyển miền và quỹ đạo SOH** — đang tiếp tục. Thích nghi miền đã có kết quả, nhưng kiểm tra đơn điệu dọc theo cell thì chưa xong.

**Và điều em muốn nói thật:** em thấy mình **cần tìm hiểu thêm về phần vật lý** và cách áp dụng nó cho đúng. Hiện em đang đọc thêm bài báo và tìm thêm bộ dữ liệu.

*(Chuyển tiếp)* Từ các khó khăn đó, kế hoạch hai tuần tới của em như sau.

---

## Slide 14 — Công việc cho hai tuần tiếp theo · 50 s

**Tuần thứ nhất** em tập trung vào **độ tin cậy của những gì đã có**:

Rà soát lại toàn bộ khâu lọc dữ liệu để đảm bảo mọi phép biến đổi đều nhân quả, kiểm tra lại cách chia cell và cách chuẩn hoá.

Bổ sung thêm các chỉ số đánh giá để xem chúng tương quan với nhau ra sao — vì phát hiện thứ ba ở slide trước cho thấy chọn chỉ số nào là quyết định chứ không phải chi tiết nhỏ.

**Tuần thứ hai** em mở rộng:

Chạy E12 thêm ở mức **10 % và 50 % nhãn** — hiện mới chỉ cân bằng ở 30 và 70. Đồng thời bổ sung các nhóm đối chứng ưu tiên. Sản phẩm là một **bảng so sánh hoàn chỉnh**.

Cuối cùng là **dựng demo**: nhận đặc trưng của một cell và hiển thị SOH theo chu kỳ.

*(Chuyển tiếp)* Em xin tóm lại.

---

## Slide 15 — Kết luận và nội dung xin góp ý · 55 s

**Về thiết kế:** em đã xác định bài toán ít nhãn, khảo sát PINN4SOH, và xây dựng cả baseline MLP lẫn lời giải PINN bán giám sát. Ba mục tiêu bắt buộc đã hoàn thành.

**Về thí nghiệm:** PINN tốt hơn ở phần lớn cấu hình sau khi công bằng hoá, và có giá trị rõ nhất khi chuyển sang bộ dữ liệu mới. Nhưng kết quả **phụ thuộc vào bộ dữ liệu, lượng nhãn và cách tinh chỉnh**. Với ba seed chạy trên CPU, **chưa đủ để khẳng định PINN luôn tốt hơn, càng chưa thể nói đạt SOTA**.

**Em xin Thầy cho ý kiến hai điểm.** Thứ nhất, chốt **MAE theo cell** làm chỉ số chính có hợp lý không, hay nên ưu tiên sai số dự báo thời điểm hết đời. Thứ hai, em dự định **hoàn thiện protocol và tăng độ tin cậy kết quả trước**, rồi mới dựng demo — thứ tự như vậy có đúng không.

Em xin hết. Em cảm ơn Thầy và các bạn đã lắng nghe.

---

## Slide 16 — Phụ lục · chỉ mở khi được hỏi

*(Đừng chủ động trình bày. Chỉ mở khi Thầy hỏi chi tiết công thức. Khi mở thì nói một câu:)*

Đây là dạng đầy đủ của bốn hàm mất mát. Điểm cần chú ý là **chỉ Ldata có y trong công thức**. Ba hàm còn lại chỉ so sánh dự đoán ở hai chu kỳ với nhau, nên chạy được trên cell không nhãn.

---

# Bảng canh giờ

| Slide | Nội dung | Thời gian | Cộng dồn |
|---|---|---|---|
| 1 | Bìa | 0:21 | 0:21 |
| 2 | Mục tiêu và tiến độ | 0:44 | 1:05 |
| 3 | Bài toán | 0:55 | 2:00 |
| 4 | Khảo sát | 0:53 | 2:53 |
| 5 | Dữ liệu | 0:42 | 3:35 |
| 6 | Protocol | 0:52 | 4:27 |
| 7 | Pipeline | 1:23 | 5:50 |
| 8 | Huấn luyện bán giám sát | 0:55 | 6:45 |
| 9 | Kết quả ít nhãn | 1:13 | 7:58 |
| 10 | E12 | 1:06 | 9:04 |
| 11 | E5 | 0:55 | 9:59 |
| 12 | Các phát hiện | 1:24 | 11:23 |
| 13 | Khó khăn | 0:57 | 12:20 |
| 14 | Kế hoạch | 0:49 | 13:09 |
| 15 | Kết luận | 0:56 | **14:05** |

Đo ở nhịp **190 âm tiết/phút** — nhịp trình bày chậm rãi, rõ ràng. Nếu nói nhanh hơn thì rút xuống khoảng 12 phút; nếu hay dừng lại thì có thể lên 16. **Nên bấm giờ tập thử ít nhất một lần** để biết nhịp thật của mình.

**Nếu bị nhắc còn 5 phút:** rút gọn slide 4, 5, 6 xuống mỗi slide hai câu, và bỏ hẳn phần diễn giải hiện tượng ở slide 9. Giữ nguyên slide 9, 10, 12, 15 — đó là bốn slide có nội dung thật.

**Nếu bị nhắc còn 3 phút:** chỉ nói slide 2, 9, 12, 15.

---

# Chuẩn bị hỏi đáp

### 1. Vì sao không dùng thẳng PINN4SOH?

Loss vật lý của họ được viết là `relu((u2 − u1)·(y1 − y2))`, trong đó **y là nhãn**. Hướng phạt do nhãn quyết định, nên **không có nhãn thì không tính được biểu thức**. Về cấu trúc nó không thể là nguồn thông tin bù cho nhãn thiếu. Bài toán của em là bài toán ít nhãn, nên em cần loss không phụ thuộc nhãn.

### 2. Định luật vật lý cụ thể ở đây là định luật gì?

Em xin trả lời chính xác: **chỉ có một thứ là định luật đúng nghĩa, đó là Arrhenius** — định luật thực nghiệm của động hoá học, mô tả việc phản ứng phụ gây suy giảm được kích hoạt nhiệt.

Phần `r ≥ 0` và loss đơn điệu là một **nguyên lý**, không phải phương trình — nó dựa trên tính bất thuận nghịch của suy giảm dung lượng, gốc rễ từ nguyên lý hai nhiệt động lực học. Nhưng nó không đúng tuyệt đối vì có hiện tượng phục hồi dung lượng sau khi nghỉ, nên em phải để dung sai epsilon.

Thừa số `exp(λ(1−u))` thì em xin nói thẳng là **một giả định hiện tượng luận**, không phải định luật.

Nên cách phát biểu đúng là: mô hình **không mã hoá định luật bảo toàn nào**. Giá trị của nó không nằm ở chỗ đúng vật lý hơn, mà ở chỗ ba ràng buộc đó **tính được mà không cần nhãn**.

### 3. Vì sao chia chỉ số chu kỳ cho 1000?

Đó **không phải siêu tham số được tinh chỉnh mà là đơn vị đo thời gian**. Điều kiện bắt buộc duy nhất là hằng số phải **toàn cục**, giống nhau cho mọi cell — đó mới là thứ chặn rò rỉ. Giá trị cụ thể chỉ ảnh hưởng tới điều kiện số học: chọn sao cho cả đầu vào t lẫn tốc độ r đều nằm quanh 1.

Em có chạy quét thử. Trong vùng từ 100 đến 10 000 thì kết quả nằm trong sai số giữa các seed. Chỉ khi không chia gì cả thì HUST mới hỏng, vì lúc đó t chạy tới hơn 2 600 còn r phải nhỏ cỡ 10 mũ trừ 4.

### 4. Sao biết chuẩn hoá theo cell là rò rỉ chứ không phải mô hình tốt lên?

Em kiểm bằng ba bước.

Bước một là đại số: rút gọn công thức min-max ra thì nó **đúng bằng hai lần phần trăm tuổi thọ đã đi qua, trừ một**. Em kiểm trên toàn bộ 257 cell, sai lệch bằng không tuyệt đối.

Bước hai: em **bỏ hết 16 đặc trưng, chỉ giữ đúng một con số đó**, rồi khớp bằng hồi quy đẳng hướng — không dùng mạng nơ-ron. Trên XJTU được 0,0097 trong khi cả mạng 16 đặc trưng được 0,0092. Nghĩa là đáp án đã nằm sẵn trong đầu vào.

Bước ba: thay tuổi thọ tổng bằng ước lượng — đúng như lúc chạy thật — thì sai số **bung ra ba đến bốn lần**.

### 5. PINN có thật sự tốt hơn MLP không?

Sau khi công bằng hoá ngân sách tinh chỉnh, PINN thắng ở **7 trên 8 cấu hình**, tỉ số trung bình **0,90 lần**. Nhưng khoảng cách **thu hẹp đáng kể** so với trước khi công bằng hoá, và có một cấu hình hoà — TJU ở 30 % nhãn. Với ba seed thì em chưa dám nói mạnh hơn thế.

### 6. Ba seed có ít quá không?

Dạ đúng, đó là hạn chế. Hiện em chạy trên CPU nên bị giới hạn. Em mới chỉ có trung bình và độ lệch chuẩn, **chưa có kiểm định thống kê**. Kế hoạch là chuyển sang GPU và tăng số seed.

### 7. Thừa số knee đóng góp gì?

Em xin trả lời thẳng: **ablation cho thấy bỏ nó đi thì MAE gần như không đổi** — trên XJTU là giống hệt. Tham số lambda cũng đi từ giá trị khởi tạo 1,0 xuống khoảng 0,84, tức dữ liệu đẩy nó **yếu đi** so với tiên nghiệm. Nên hiện tại em chưa đo được đóng góp của thành phần này.

### 8. Loss đơn điệu có làm MAE xấu đi không?

Dạ có. Bỏ nó ra thì MAE tốt lên. Nhưng nó **đánh đổi lấy chất lượng quỹ đạo**: vi phạm đơn điệu giảm từ 1,28 xuống 0,27 trên XJTU, và sai số dự báo thời điểm hết đời giảm từ 29 xuống 17 chu kỳ. Với ứng dụng BMS thật thì đó mới là thứ đáng giá.

### 9. Vì sao chưa so với XGBoost, GPR hay CNN?

Dạ em chưa làm. Đây là thiếu sót thật và nằm trong kế hoạch tuần thứ hai. Lý do em ưu tiên MLP trước là vì muốn **cô lập đúng một biến** — cùng kiến trúc, cùng kích thước, chỉ khác có loss vật lý hay không. Thêm mô hình khác kiến trúc vào sẽ trộn hai yếu tố.

### 10. Vì sao TJU lại kém đi khi thêm vật lý?

Ở TJU, MLP dùng 30 % nhãn vốn đã gần bằng MLP dùng 70 % — chỉ kém 1,04 lần. Nghĩa là **dữ liệu đã đủ**, không có chỗ trống cho tiên nghiệm lấp vào. Lúc đó ràng buộc thêm vào trở thành thiên kiến chứ không còn là thông tin. Đây là hành vi em cho là đúng với bản chất của một tiên nghiệm, chứ không phải lỗi cài đặt.

### 11. Tại sao dùng tương quan Spearman?

Câu hỏi em cần trả lời là về **hướng đơn điệu**, không phải độ dốc tuyến tính. Suy giảm dung lượng cong mạnh, nên quan hệ có thể đơn điệu hoàn hảo mà vẫn phi tuyến — Pearson sẽ báo yếu, tức âm tính giả. Em cũng tính Kendall tau, dấu và kết luận giống hệt, nên kết luận không phải sản phẩm của việc chọn Spearman.

### 12. Demo bao giờ có?

Dạ tuần thứ hai. Phạm vi demo em đặt vừa phải: nhận đặc trưng của một cell, chạy suy luận **chỉ bằng mạng nghiệm**, hiển thị SOH theo chu kỳ. Điều kiện là không được dùng thông tin tương lai của cell đó.

---

# Hai chỗ nên sửa trên slide trước khi trình

1. **Slide 2** có thanh tiêu đề màu **xanh lá**, trong khi 14 slide còn lại đều màu **xanh navy**. Nhiều khả năng là sót lại từ template. Nên đổi cho đồng bộ.

2. **Slide 14**, cột **"Sản phẩm cần hoàn thành"** đang **trống ở 3 trên 4 dòng**. Chỉ dòng Tuần 2 có ghi "Bảng so sánh". Nên điền nốt, ví dụ: dòng 1 — "Báo cáo rà soát pipeline"; dòng 2 — "Bảng tương quan giữa các chỉ số"; dòng 4 — "Bản demo chạy được trên một cell". Cột trống trên slide kế hoạch dễ bị hỏi.

---

# Ba nguyên tắc khi nói

**Đừng đọc bảng.** Slide 10 có 8 dòng số — chỉ nói cột cuối cùng và dòng ngoại lệ. Thầy tự đọc được phần còn lại.

**Chủ động nói ra điểm yếu trước khi bị hỏi.** Cụ thể là ba chỗ: giả thuyết chỉ đạt 2 trên 4 bộ, khoảng cách thu hẹp sau khi công bằng hoá, và nhóm đối chứng chưa có. Nói trước thì đó là sự nghiêm cẩn; bị hỏi ra mới nói thì thành lỗ hổng.

**Khi không biết thì nói không biết.** Có ít nhất ba câu em chưa trả lời được: đóng góp thật của thừa số knee, vì sao TJU khác ba bộ còn lại ở mức cơ chế, và ngưỡng nhãn tối thiểu. Trả lời "em chưa đo được, em sẽ kiểm" mạnh hơn nhiều so với đoán bừa.
