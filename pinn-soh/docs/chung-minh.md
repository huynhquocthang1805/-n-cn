# Tính đúng đắn: chứng minh và kiểm chứng

Tài liệu này làm ba việc tách bạch nhau:

1. **Chứng minh** các tính chất mà thiết kế dựa vào — bằng toán, không bằng thực nghiệm.
2. **Kiểm chứng số** rằng cài đặt thật sự thoả các tính chất đó (`verify_losses.py`, 24 phép kiểm).
3. **Kiểm định thống kê** các kết quả thực nghiệm (`stats_test.py`) — và một đính chính quan trọng.

Cần phân biệt rõ: phép kiểm số **không chứng minh** mệnh đề. Nó bắt lỗi cài đặt — sai dấu, quên softplus, nhãn lọt vào loss — chứ không thay được chứng minh.

---

## 1. Ký hiệu

| Ký hiệu | Nghĩa |
|---|---|
| $\mathbf{x} \in \mathbb{R}^{16}$ | vector đặc trưng đường sạc đã chuẩn hoá |
| $t = N/1000$ | chỉ số chu kỳ đã chia hằng số toàn cục |
| $u(\mathbf{x},t) = F_\varphi(\mathbf{x},t)$ | SOH dự đoán — nghiệm của phương trình |
| $y$ | nhãn SOH thật |
| $r(\mathbf{x},u,T)$ | tốc độ suy giảm, đơn vị SOH trên 1000 chu kỳ |
| $\varphi,\ \theta$ | tham số mạng nghiệm / mạng tốc độ nền |
| $\lambda,\ E_a$ | hệ số đầu gối / năng lượng hoạt hoá — hai tham số vật lý học được |
| $\alpha,\beta,\gamma$ | trọng số ba loss vật lý (siêu tham số) |
| $\varepsilon = 0{,}002$ | dung sai đơn điệu |
| $h \sim \mathcal{U}\{5,\dots,50\}$ | horizon giữa hai chu kỳ trong một cặp |
| $\Delta = h/1000$ | bước thời gian tương ứng |
| $R = 8{,}314$, $T_\text{ref} = 298{,}15$ | hằng số khí, nhiệt độ tham chiếu |

Động học xám:

$$r(\mathbf{x},u,T) \;=\; \underbrace{\text{softplus}\big(\text{MLP}_\theta(\mathbf{x})\big)}_{k(\mathbf{x})} \cdot \underbrace{e^{\lambda(1-u)}}_{g(u)} \cdot \underbrace{e^{-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_\text{ref}}\right)}}_{A(T)}$$

Bốn hàm mất mát:

$$
\begin{aligned}
L_\text{data} &= \mathbb{E}\big[(u-y)^2\big] &&\text{(chỉ trên cell có nhãn)}\\
L_\text{ode} &= \mathbb{E}\big[(u_2-u_1+r_1\Delta)^2\big] \\
L_\text{mono} &= \mathbb{E}\big[\text{ReLU}(u_2-u_1-\varepsilon)\big] \\
L_\text{range} &= \mathbb{E}\big[\text{ReLU}(u-1{,}15)+\text{ReLU}(0{,}4-u)\big] \\
L &= L_\text{data} + w(s)\big(\alpha L_\text{ode}+\beta L_\text{mono}+\gamma L_\text{range}\big)
\end{aligned}
$$

trong đó $u_i = u(\mathbf{x}_i,t_i)$, $r_1 = r(\mathbf{x}_1,u_1,T)$, $\Delta = t_2-t_1$, và $w(s)=\min(1,\ s/0{,}2S)$.

---

## 2. Các mệnh đề

### Mệnh đề 1 — tốc độ suy giảm không âm

> Với **mọi** $\theta,\lambda,E_a \in \mathbb{R}$ và **mọi** $(\mathbf{x},u,T)$ với $T>0$, ta có $r(\mathbf{x},u,T) > 0$.

**Chứng minh.** $\text{softplus}(z)=\ln(1+e^z)$. Vì $e^z>0\ \forall z\in\mathbb{R}$ nên $1+e^z>1$, do đó $\ln(1+e^z)>0$. Hai thừa số còn lại là $\exp$ của một số thực nên dương ngặt. Tích ba số dương ngặt là dương ngặt. $\blacksquare$

**Hệ quả.** $\dfrac{du}{dt} = -r \le 0$: dung lượng không thể tự tăng, **theo cấu trúc mô hình**, không phụ thuộc giá trị tham số học được.

**Chú ý về số học dấu phẩy động.** Trong float32, $\text{softplus}(z)$ tràn dưới về đúng $0$ khi $z < -104$. Nên trong cài đặt, phát biểu đúng là $r \ge 0$ — **không bao giờ âm**, đó mới là tính chất mà hệ quả cần. Đo trên 50 mạng khởi tạo Xavier với đặc trưng đã chuẩn hoá, tiền kích hoạt nhỏ nhất quan sát được là $-2{,}93$, cách ngưỡng tràn 101 đơn vị.

### Mệnh đề 2 — đơn điệu theo cấu trúc của đầu ra `mono`

> Cho $\displaystyle u(\mathbf{x},t) = u_0(\mathbf{x}) - \sum_{k=1}^{K} s_k(\mathbf{x})\,\text{softplus}\big(\omega_k t + \phi_k(\mathbf{x})\big)$ với $s_k = \text{softplus}(\cdot) \ge 0$ và $\omega_k = \text{softplus}(\cdot) \ge 0$. Khi đó $\dfrac{\partial u}{\partial t} \le 0$ với **mọi** giá trị tham số.

**Chứng minh.** $\dfrac{d}{dz}\text{softplus}(z) = \sigma(z) = \dfrac{1}{1+e^{-z}} \in (0,1)$. Theo quy tắc dây chuyền:

$$\frac{\partial u}{\partial t} = -\sum_{k=1}^{K} s_k(\mathbf{x})\,\omega_k\,\sigma\big(\omega_k t+\phi_k(\mathbf{x})\big)$$

Mỗi số hạng là tích của ba đại lượng không âm ($s_k \ge 0$, $\omega_k \ge 0$, $\sigma > 0$), nên tổng $\ge 0$, do đó $\partial u/\partial t \le 0$. Dấu bằng xảy ra khi và chỉ khi $s_k\omega_k = 0$ với mọi $k$. $\blacksquare$

**Ý nghĩa.** Kết luận không phụ thuộc giá trị tham số, nên đúng ngay từ bước khởi tạo và giữ nguyên suốt quá trình huấn luyện. Khác hẳn $L_\text{mono}$ vốn chỉ **phạt** vi phạm chứ không **ngăn** được, và còn phải cân trọng số $\beta$.

### Mệnh đề 3 — ba loss vật lý không phụ thuộc nhãn

> $L_\text{ode}$, $L_\text{mono}$, $L_\text{range}$ là hàm của $(\mathbf{x}_1,\mathbf{x}_2,t_1,t_2,T,\varphi,\theta,\lambda,E_a)$ và không chứa $y$. Do đó $\partial L/\partial y \equiv 0$.

**Chứng minh.** Trực tiếp từ định nghĩa ở mục 1: không biểu thức nào chứa $y$. $\blacksquare$

**Hệ quả — đây là điều kiện cần để cơ chế bán giám sát tồn tại.** Ba hàm này tính được trên mọi cell chỉ có đường sạc mà chưa đo dung lượng.

**Đối chiếu.** Loss vật lý của PINN4SOH là $\text{ReLU}\big((u_2-u_1)(y_1-y_2)\big)$ — hướng phạt do **nhãn** quyết định, nên **không thoả** mệnh đề này. Về cấu trúc, nó không thể là nguồn thông tin bù cho nhãn thiếu.

### Mệnh đề 4 — $L_\text{mono}$ là hàm phạt chính xác theo tập không

> Với $\varepsilon \ge 0$ và tập cặp mẫu $\mathcal{P}$:
> $L_\text{mono}(\mathcal{P}) = 0 \iff u_2 - u_1 \le \varepsilon$ với **mọi** cặp trong $\mathcal{P}$.
> Hơn nữa, nếu mọi cặp vi phạm đúng một lượng $d>0$ thì $L_\text{mono} = d$.

**Chứng minh.** $\text{ReLU}(z)=\max(0,z) \ge 0$, bằng $0 \iff z \le 0$. Trung bình của các số không âm bằng $0$ khi và chỉ khi mọi số hạng bằng $0$. Áp cho $z = u_2-u_1-\varepsilon$ được vế thứ nhất. Vế thứ hai: nếu $u_2-u_1-\varepsilon = d$ với mọi cặp thì $\text{ReLU}(d)=d$ và trung bình bằng $d$. $\blacksquare$

**Ý nghĩa.** Tập nghiệm của $L_\text{mono}=0$ **trùng đúng** tập khả thi của ràng buộc — không nới rộng, không thu hẹp. Và giá trị của nó bằng đúng mức vi phạm trung bình, nên đọc được như một đại lượng có đơn vị SOH.

### Mệnh đề 5 — bậc chặt cụt của phần dư Euler

> Giả sử $u \in C^2([t,t+\Delta])$ và $u' = -r$. Đặt
> $R_\text{Euler}(\Delta) = u(t+\Delta)-u(t)+r(t)\Delta$ và $R_\text{đh}(\Delta) = \dfrac{u(t+\Delta)-u(t)}{\Delta}+r(t)$.
> Khi đó $R_\text{Euler}(\Delta) = O(\Delta^2)$, $R_\text{đh}(\Delta)=O(\Delta)$, và $R_\text{Euler} = \Delta \cdot R_\text{đh}$.

**Chứng minh.** Khai triển Taylor bậc hai với phần dư Lagrange: tồn tại $\xi \in (t,t+\Delta)$ sao cho

$$u(t+\Delta) = u(t) + \Delta u'(t) + \tfrac{\Delta^2}{2}u''(\xi)$$

Thay $u'(t) = -r(t)$:

$$R_\text{Euler} = \Big[u(t)-\Delta r(t)+\tfrac{\Delta^2}{2}u''(\xi)\Big] - u(t) + r(t)\Delta = \tfrac{\Delta^2}{2}u''(\xi)$$

Nếu $|u''| \le M$ trên đoạn thì $|R_\text{Euler}| \le M\Delta^2/2$. Chia hai vế cho $\Delta$: $R_\text{đh} = \tfrac{\Delta}{2}u''(\xi) = O(\Delta)$. $\blacksquare$

**Kiểm chứng số.** Với nghiệm giải tích $u(t)=e^{-at}$, $a=1{,}7$, $r=au$: hồi quy $\log|R|$ theo $\log\Delta$ trên 10 giá trị $\Delta$ cho độ dốc **1,9957** (Euler) và **0,9957** (đạo hàm) — khớp bậc 2 và bậc 1.

### Mệnh đề 6 — hệ số khuếch đại nhiễu

> Giả sử đầu ra mạng có nhiễu cộng độc lập $\eta_1,\eta_2$ với $|\eta_i| \le \delta$. Khi đó
> $|\Delta R_\text{Euler}| \le 2\delta$ — **chặn, độc lập với $\Delta$**;
> $|\Delta R_\text{đh}| \le 2\delta/\Delta$.
> Tỉ số khuếch đại đúng bằng $1/\Delta$.

**Chứng minh.** $R_\text{Euler}$ phụ thuộc $u$ tuyến tính với hệ số $\pm 1$, nên $\Delta R_\text{Euler} = \eta_2-\eta_1$ và $|\Delta R_\text{Euler}| \le 2\delta$. Dạng đạo hàm chia thêm cho $\Delta$: $\Delta R_\text{đh} = (\eta_2-\eta_1)/\Delta$, nên $|\Delta R_\text{đh}| \le 2\delta/\Delta$. Tỉ số hai chặn là $1/\Delta$. $\blacksquare$

**Hệ quả định lượng.** Với $\Delta = h/1000$, ở $h=1$ tỉ số là **1000** — đúng con số nêu ở mục hạn chế. Kiểm chứng số với $\delta = 10^{-3}$ cho $|\Delta R_\text{đh}| \approx 1{,}13$ trong khi $|du/d\tilde{N}|$ thực tế chỉ khoảng $0{,}2$–$0{,}5$: **nhiễu lấn át tín hiệu 2–5 lần**.

**Hệ quả thiết kế.** Dạng Euler nhân hai vế với $\Delta$ thay vì chia, nên cặp horizon ngắn đóng góp ít và cặp horizon dài chi phối một cách tự nhiên — không cần lọc hay cân trọng số theo $h$.

### Mệnh đề 7 — $L_\text{range}$ là khoảng cách tới hộp

> $L_\text{range}=0 \iff u \in [0{,}4;\ 1{,}15]$ tại mọi điểm mẫu. Khi $u$ ra ngoài, $L_\text{range}$ bằng đúng khoảng cách trung bình tới hộp.

**Chứng minh.** Tương tự Mệnh đề 4. Với $u>1{,}15$: $\text{ReLU}(u-1{,}15)=u-1{,}15=\text{dist}(u,[0{,}4;1{,}15])$ và $\text{ReLU}(0{,}4-u)=0$; đối xứng cho $u<0{,}4$; trong hộp cả hai bằng 0. $\blacksquare$

### Mệnh đề 8 — lịch trình trọng số vật lý

> $w(s)=\min(1,\ s/(0{,}2S))$ liên tục, không giảm, $w(0)=0$, và $w(s)=1$ với mọi $s \ge 0{,}2S$.

**Chứng minh.** $\min$ của hai hàm liên tục không giảm là liên tục không giảm; các giá trị biên thay trực tiếp. $\blacksquare$

**Ý nghĩa.** Ở bước đầu $L = L_\text{data}$ thuần — mạng bám dữ liệu trước; vật lý siết vào dần và đạt trọng số đầy đủ sau 20 % số bước. Nhờ vậy ràng buộc không đẩy mạng vào nghiệm suy biến ngay từ lúc khởi tạo, khi $u$ còn ngẫu nhiên.

---

## 3. Nhận xét 1 — hạn chế của Mệnh đề 2

> Mệnh đề 2 cho $\partial u/\partial t \le 0$ khi **giữ nguyên $\mathbf{x}$**. Dọc theo quỹ đạo thật của một cell, $\mathbf{x}$ cũng thay đổi theo chu kỳ. Đạo hàm toàn phần là
> $$\frac{du}{dN} = \frac{\partial u}{\partial t}\cdot\frac{dt}{dN} + \Big\langle \nabla_\mathbf{x} u,\ \frac{d\mathbf{x}}{dN}\Big\rangle = \frac{1}{1000}\frac{\partial u}{\partial t} + \Big\langle \nabla_\mathbf{x} u,\ \frac{d\mathbf{x}}{dN}\Big\rangle$$
> Số hạng thứ nhất $\le 0$ theo Mệnh đề 2. Số hạng thứ hai **không có ràng buộc dấu nào**. Do đó **đơn điệu theo $t$ không kéo theo đơn điệu dọc quỹ đạo**.

Đây là phát biểu chính xác của hạn chế mà báo cáo nêu ở mục "các phát hiện".

**Kiểm chứng số.** Dựng quỹ đạo giả trong đó $\mathbf{x}$ đi theo hướng làm $u_0(\mathbf{x})$ tăng. Đạo hàm riêng theo $t$ vẫn $\le 0$ tại mọi điểm (cực đại $= -2{,}82\times 10^{-1}$), nhưng **4/299 bước dọc quỹ đạo tăng ngược**, mức tăng lớn nhất $+0{,}0013$.

**Bằng chứng thực nghiệm khớp với lý thuyết.** Ở E9 trên HUST, kiến trúc đơn điệu `silu-mono` có chỉ số vi phạm đơn điệu **2,61** so với **1,18** của đầu ra MLP thường *trong cùng mô hình PINN-semi* — tức kiến trúc *đảm bảo* đơn điệu theo $t$ lại vi phạm đơn điệu *dọc quỹ đạo* **nhiều hơn**. Lý thuyết dự đoán đúng hiện tượng này.

---

## 4. Kiểm chứng số — 24 phép kiểm

Chạy bằng `python verify_losses.py`. Toàn bộ **24/24 đạt**.

| Mệnh đề | Phép kiểm | Kết quả đo |
|---|---|---|
| MĐ1 | $r \ge 0$ trên 200 mạng phá hoại (trọng số tới $10^2$) | min $r$ trên 51 200 điểm $= 0$, không có giá trị âm |
| MĐ1 | ngưỡng tràn dưới softplus float32 | $r>0$ ngặt khi tiền kích hoạt $>-104$ |
| MĐ1 | tiền kích hoạt thực tế | min $=-2{,}93$, cách ngưỡng 101 đơn vị |
| MĐ2 | $\partial u/\partial t \le 0$ trên 200 đầu ngẫu nhiên | max $= 0$ trên 25 600 điểm |
| MĐ3 | $\partial L/\partial y = $ None cho cả ba loss | autograd không nối được tới $y$ |
| MĐ3 | đối chứng: $\partial L_\text{data}/\partial y \ne 0$ | $5{,}07\times10^{-3}$ |
| MĐ4 | quỹ đạo giảm $\to L_\text{mono}=0$ | $0{,}000$ |
| MĐ4 | tăng ngược đúng $\varepsilon \to L_\text{mono}=0$ | đúng bằng biên |
| MĐ4 | vượt $\varepsilon$ một lượng $d \to L_\text{mono}=d$ | $0{,}010000 \approx 0{,}010000$ |
| MĐ5 | bậc của $R_\text{Euler}$ theo $\Delta$ | **1,9957** (lý thuyết 2) |
| MĐ5 | bậc của $R_\text{đh}$ theo $\Delta$ | **0,9957** (lý thuyết 1) |
| MĐ6 | tỉ số khuếch đại $=1/\Delta$ | $h{=}1$: 1000 · $h{=}5$: 200 · $h{=}20$: 50 · $h{=}50$: 20 |
| MĐ6 | ở $h{=}1$ nhiễu lấn át tín hiệu | $1{,}13$ so với $0{,}2$–$0{,}5$ |
| MĐ7 | $L_\text{range}$ = khoảng cách tới hộp | $0{,}200000 \approx 0{,}200000$ |
| MĐ8 | $w$ không giảm, $w(0)=0$, $w(0{,}2S)=1$ | đạt |
| cài đặt | `gradcheck` $L_\text{ode}$, $L_\text{mono}$, $L_\text{range}$ | autograd khớp sai phân hữu hạn (float64) |
| Nhận xét 1 | $\partial u/\partial t \le 0$ nhưng $du/dN > 0$ có xảy ra | 4/299 bước tăng ngược |

Hai phép kiểm **thất bại ở lần chạy đầu** và đó chính là giá trị của bộ test:

1. **MĐ1 phát biểu quá mạnh.** Bản đầu viết $r>0$ ngặt; phép kiểm bắt được $r=0$ chính xác khi trọng số bị đẩy tới $10^2$. Nguyên nhân là tràn dưới float32, không phải lỗi lập luận — nhưng phát biểu phải sửa thành $r \ge 0$, và đó mới đúng là tính chất cần dùng.
2. **`gradcheck` thất bại do lỗi của chính phép kiểm.** Bản đầu gán `p.data = ...` để thay tham số, việc này **cắt đồ thị autograd** nên gradcheck luôn báo sai. Sửa bằng `torch.func.functional_call` thì cả ba loss đều đạt.

---

## 5. Kiểm định thống kê kết quả — và một đính chính

### 5.1 Thiết kế

Đơn vị quan sát là **seed**. Với cùng một seed, hai mô hình dùng **đúng một** cách chia cell (`split_cells` gieo theo seed) và được đánh giá trên **đúng một** tập cell test, theo đúng thứ tự hàng — đã kiểm bằng assert. Nên mỗi seed cho một **cặp ghép đôi hợp lệ** $(\text{MAE}_\text{MLP}, \text{MAE}_\text{PINN})$, và biến thiên do chia dữ liệu lẫn do khởi tạo đều nằm trong biến thiên giữa các seed.

Ba lớp phân tích, tất cả **ghép cặp**:

1. **Wilcoxon signed-rank** trên $n$ seed — không giả định phân phối chuẩn.
2. **Bootstrap ghép cặp trên seed** — khoảng tin cậy 95 % cho hiệu và cho tỉ số, 20 000 lần lấy lại mẫu.
3. **Bootstrap cụm theo cell** trong từng seed — các hàng trong một cell tương quan mạnh nên phải lấy lại mẫu theo **cell**, không theo hàng; bỏ qua điều này cho khoảng tin cậy hẹp giả tạo.

Hiệu chỉnh **Holm–Bonferroni** trên bốn bộ dữ liệu. Chỉ số chính là **MAE theo cell** — chỉ số mà mục "các phát hiện" đề xuất.

Số seed nâng từ **3 lên 10**: 56 lượt huấn luyện bổ sung cho cấu hình E1 và 80 lượt cho cấu hình tinh chỉnh.

### 5.2 Kết quả — cấu hình E1 ($\beta = 5$ cố định)

| bộ | MLP | PINN | tỉ số | KTC 95 % của tỉ số | $p$ thô | $p$ Holm | seed PINN thắng |
|---|---|---|---|---|---|---|---|
| XJTU | 0,0217 | 0,0147 | 0,676 | [0,487; 1,034] | 0,322 | 0,697 | 6/10 |
| TJU | 0,0108 | 0,0126 | **1,167** | [1,116; 1,223] | 0,0020 | **0,0078** | **0/10** |
| MIT | 0,0048 | 0,0046 | 0,957 | [0,853; 1,081] | 0,846 | 0,846 | 5/10 |
| HUST | 0,0180 | 0,0166 | 0,920 | [0,816; 1,026] | 0,232 | 0,697 | 7/10 |

**Đính chính.** Với 10 seed và kiểm định ghép cặp, **không bộ nào cho thấy PINN tốt hơn một cách có ý nghĩa thống kê**, và trên **TJU thì PINN kém hơn có ý nghĩa** ($p_\text{Holm} = 0{,}0078$; thua cả 10/10 seed). Phát biểu "giả thuyết đạt ở 2/4 bộ" ở cấu hình E1 **không đứng vững** khi tăng số seed.

**Vì sao ba seed đầu cho bức tranh khác:**

| bộ | tỉ số với 3 seed | tỉ số với 10 seed | biên độ MAE của MLP qua 10 seed | số cell test |
|---|---|---|---|---|
| XJTU | 0,935 | 0,676 | 0,0104–0,0569 (**5,5×**) | **7** |
| TJU | 1,152 | 1,167 | 0,0095–0,0118 (1,2×) | 18 |
| MIT | 0,828 | 0,957 | 0,0032–0,0068 (**2,1×**) | 18 |
| HUST | 0,813 | 0,919 | 0,0151–0,0224 (1,5×) | 12 |

Nguyên nhân gốc là **tập test quá nhỏ**, nên MAE của baseline là một xổ số theo seed — và ba seed đầu rơi về **hai phía ngược nhau** ở hai bộ:

- **XJTU** (**7 cell test**, tức 55 cell × 15 %) — một cell xấu chi phối toàn bộ MAE, nên MLP nhảy từ 0,0104 lên 0,0569. Hai lượt tệ nhất là seed 9 và seed 4, tức **ngoài** ba seed đầu. Ba seed đầu vì thế là mẫu **thuận lợi** cho MLP (trung bình 0,0122 so với 0,0258 của bảy seed còn lại), nên tỉ số 3 seed 0,935 đã **đánh giá thấp** PINN.
- **MIT** — ngược lại. Hai lượt MLP tệ nhất trong mười nằm **trong** ba seed đầu (seed 1 và seed 2): trung bình ba seed đầu 0,0060 so với 0,0043 của bảy seed còn lại, tức tệ hơn 40 %. Tỉ số 3 seed 0,828 vì thế đã **đánh giá cao** PINN.

Hai bộ lệch **ngược chiều nhau** là bằng chứng rõ nhất rằng cái đo được ở ba seed là nhiễu chia dữ liệu, không phải hiệu ứng của mô hình.

### 5.3 Kết quả — cấu hình đã tinh chỉnh

Cấu hình **ấn định trước**, không chọn lại theo 10 seed nên không có thiên lệch do chọn mô hình: mạng `wide` $(128,128,64)$ cho **cả hai** mô hình — biến thể mà E12 chọn theo validation ở 6/8 ô — cộng cấu hình vật lý tốt nhất theo E7 cho PINN ($\beta = 0{,}5$, phần dư autograd, $\alpha = 0{,}02$).

| bộ | MLP | PINN | tỉ số | KTC 95 % của tỉ số | $p$ thô | $p$ Holm | seed PINN thắng |
|---|---|---|---|---|---|---|---|
| XJTU | 0,0178 | 0,0138 | **0,774** | **[0,680; 0,893]** | 0,0137 | 0,055 | 8/10 |
| TJU | 0,0105 | 0,0103 | 0,979 | [0,956; 1,005] | 0,232 | 0,232 | 6/10 |
| MIT | 0,0042 | 0,0030 | **0,721** | **[0,596; 0,935]** | 0,0273 | 0,082 | 9/10 |
| HUST | 0,0183 | 0,0165 | **0,901** | **[0,828; 0,973]** | 0,0273 | 0,082 | 8/10 |

Bức tranh đổi hẳn và theo hướng có lợi:

- **Ba trên bốn bộ có khoảng tin cậy của tỉ số nằm hoàn toàn dưới 1** — XJTU, MIT, HUST. Mức cải thiện 10 %–28 %.
- **Ba trên bốn bộ có $p$ thô $< 0{,}05$.** Sau hiệu chỉnh Holm cho bốn so sánh thì **không bộ nào đạt mức 0,05**, nhưng cả ba đều nằm trong khoảng 0,055–0,082 — sát ngưỡng.
- **TJU không còn kém hơn có ý nghĩa** (1,167 → 0,979). Tức phần lớn "thất bại trên TJU" là do cấu hình $\beta = 5$ siết quá tay, chứ không phải do bản chất của bộ dữ liệu.

**Phát biểu trung thực nhất với dữ liệu hiện có:** *ở cấu hình đã tinh chỉnh, có bằng chứng nhất quán rằng PINN giảm được MAE khoảng 10 %–28 % trên ba trong bốn bộ; bằng chứng chưa đạt mức có ý nghĩa thống kê sau hiệu chỉnh đa so sánh với 10 seed.* Không được nói "PINN tốt hơn" mà không kèm mệnh đề sau.

### 5.4 Cần bao nhiêu seed

Ước lượng lực thống kê bằng bootstrap từ chính phân bố hiệu đã quan sát (600 lần mô phỏng mỗi ô, Holm ×4):

| bộ | hiệu TB | SD | $n{=}10$ | $n{=}15$ | $n{=}20$ | $n{=}30$ | $n{=}50$ |
|---|---|---|---|---|---|---|---|
| XJTU | −0,00401 | 0,00454 | 0,51 | 0,86 | **0,96** | 1,00 | 1,00 |
| TJU | −0,00022 | 0,00047 | 0,05 | 0,11 | 0,20 | 0,39 | 0,72 |
| MIT | −0,00118 | 0,00156 | 0,39 | 0,66 | **0,85** | 0,95 | 1,00 |
| HUST | −0,00181 | 0,00242 | 0,34 | 0,68 | **0,83** | 0,98 | 1,00 |

**Kết luận thực hành: 20 seed là đủ** cho XJTU, MIT và HUST (lực 0,83–0,96). Nên đổi kế hoạch từ 10 seed lên **20 seed**.

TJU không bao giờ đạt được, và đó **chính là kết quả đúng**: hiệu thật ở đó chỉ −0,00022, tức khoảng 2 %. Không phải thiếu lực thống kê mà là **thật sự không có khác biệt**. Với TJU, phát biểu đúng là "hai mô hình tương đương", không phải "chưa đủ bằng chứng".

### 5.5 Những gì phép kiểm này CHƯA phủ

- Chỉ kiểm ở **mức 30 % nhãn**. Các mức 10, 50, 70 % vẫn đang là ba seed.
- Chỉ kiểm **hai cấu hình**. Kết luận 7/8 của E12 vẫn dựa trên ba seed và **chịu đúng điểm yếu này**.
- Không kiểm các thí nghiệm chuyển miền (E5, E10) và ablation (E3, E9).
- Chỉ số EOL và vi phạm đơn điệu chưa được kiểm định, mới chỉ có MAE.

---

## 6. Cách chạy lại

```bash
python verify_losses.py --csv     # 24 phép kiểm tính chất, ghi results/verify_table.csv
python seed_sweep.py              # nâng E1 @30 % lên 10 seed  (~14 phút)
python seed_sweep_tuned.py        # cấu hình tinh chỉnh, 10 seed (~26 phút)
python stats_test.py                                              # kiểm định cấu hình E1
python stats_test.py --runs runs/TUNED --tag wide --suffix _tuned  # kiểm định cấu hình tinh chỉnh
python power_test.py              # ước lượng số seed cần thiết
```

Mọi bảng trong tài liệu này sinh từ `results/stats_*.csv` và `results/verify_table.csv`.
