---
title: "Phương pháp: xử lý dữ liệu, mô hình và huấn luyện"
subtitle: "Ước lượng SOH pin lithium-ion bằng PINN bán giám sát — Nhóm 15"
lang: vi
---

# 1. Xử lý dữ liệu

## 1.1. Dữ liệu đầu vào

Mỗi file CSV là một cell, mỗi hàng là một chu kỳ sạc–xả. Mỗi chu kỳ có 16 đặc trưng thống kê của
đoạn sạc CC–CV và dung lượng đo được:

| Nhóm | Đặc trưng |
|---|---|
| Điện áp (pha CC) | trung bình, độ lệch chuẩn, độ nhọn, độ lệch, độ dốc, entropy |
| Dòng điện (pha CV) | trung bình, độ lệch chuẩn, độ nhọn, độ lệch, độ dốc, entropy |
| Điện lượng và thời gian | CC Q, thời gian sạc CC, CV Q, thời gian sạc CV |

Ký hiệu: $\mathbf{x}_n \in \mathbb{R}^{16}$ là vector đặc trưng ở chu kỳ $n$ của một cell,
$Q_n$ là dung lượng đo được.

## 1.2. Nhãn SOH

$$
\mathrm{SOH}_n = \frac{Q_n}{Q_\text{danh định}}
$$

| Bộ | Hoá học | $Q_\text{danh định}$ | Nhiệt độ |
|---|---|---|---|
| XJTU | NCM | 2,0 Ah | 25 °C |
| TJU | NCA / NCM / NCM+NCA | 3,5 / 3,5 / 2,5 Ah | 25, 35, 45 °C |
| MIT | LFP | 1,1 Ah | 30 °C |
| HUST | LFP | 1,1 Ah | 30 °C |

Cell chưa đo dung lượng vẫn được giữ lại, nhãn để trống. Các cell này chỉ dùng cho phần loss vật lý.

## 1.3. Lọc ngoại lai nhân quả

Một chu kỳ bị loại nếu đặc trưng không hữu hạn (NaN, ±∞), hoặc nếu có đặc trưng lệch quá xa so
với các chu kỳ trước đó của chính cell. Với đặc trưng thứ $j$ ở chu kỳ $n$, lấy cửa sổ 50 chu
kỳ liền trước:

$$
m_{n,j} = \operatorname{median}\big(x_{n-50..n-1,\,j}\big), \qquad
\mathrm{IQR}_{n,j} = q_{75} - q_{25}
$$

$$
s_{n,j} = \max\!\left(\frac{\mathrm{IQR}_{n,j}}{1{,}349},\; 0{,}01\,|m_{n,j}|\right), \qquad
z_{n,j} = \frac{|x_{n,j} - m_{n,j}|}{s_{n,j}}
$$

Chu kỳ $n$ bị loại nếu $\max_j z_{n,j} > 5$. Cần ít nhất 10 chu kỳ lịch sử mới xét; 10 chu kỳ đầu
chỉ bị loại khi không hữu hạn. Hệ số 1,349 đổi IQR sang độ lệch chuẩn của phân phối chuẩn.

Vì chỉ dùng quá khứ, bộ lọc chạy được trực tuyến trong BMS. Tỉ lệ chu kỳ bị loại là 1,6 %.

## 1.4. Biến thời gian

$$
t_n = \frac{n}{1000}
$$

$n$ là chỉ số chu kỳ gốc (trước khi lọc). Hằng số 1000 dùng chung cho mọi cell, không chuẩn hoá
theo tuổi thọ riêng của từng cell.

## 1.5. Chuẩn hoá đặc trưng

Chuẩn hoá z-score, với trung bình và độ lệch chuẩn tính trên mọi cell train (có nhãn và
không nhãn), dùng chung cho mọi mô hình:

$$
\tilde{\mathbf{x}}_n = \frac{\mathbf{x}_n - \boldsymbol{\mu}}{\boldsymbol{\sigma} + 10^{-6}}
$$

Cell validation và test chỉ được biến đổi bằng $\boldsymbol{\mu}, \boldsymbol{\sigma}$ này, không
tham gia tính thống kê.

## 1.6. Chia dữ liệu

- Chia theo cell: toàn bộ chu kỳ của một cell nằm trọn trong train, validation hoặc test.
- Kiểm định chéo 5 fold, lặp 5 lần: cell được xáo trong từng nhóm giao thức thí nghiệm rồi chia
  vòng tròn vào 5 fold, nên fold nào cũng có đủ các giao thức. Mỗi lần lặp dùng một cách xáo khác.
- Mỗi fold: 20 % cell làm test, 10 % làm validation, 70 % làm train.
- Lượng nhãn tính trên tổng số cell $N$: chọn $\operatorname{round}(f \cdot N)$ cell train có
  nhãn, $f \in \{0{,}1;\,0{,}3;\,0{,}5;\,0{,}7\}$, lấy luân phiên theo giao thức. Các cell train
  còn lại là cell không nhãn.

| Bộ | Train | Validation | Test | Có nhãn khi $f = 0{,}3$ |
|---|---|---|---|---|
| XJTU | 38 | 6 | 11 | 16 |
| TJU | 91 | 13 | 26 | 39 |
| MIT | 88 | 12 | 25 | 38 |
| HUST | 53 | 8 | 16 | 23 |

# 2. Mô hình

## 2.1. Mạng nghiệm

$$
u = F_\varphi(\tilde{\mathbf{x}}, t)
$$

MLP 17 → 128 → 128 → 64 → 1, hàm kích hoạt SiLU, khởi tạo Xavier. Đầu ra $u$ là SOH dự đoán.

## 2.2. Động học suy giảm

Tốc độ suy giảm $r$ (đơn vị: SOH trên 1000 chu kỳ):

$$
r(\tilde{\mathbf{x}}, u, T) = \underbrace{\operatorname{softplus}\!\big(G_\theta(\tilde{\mathbf{x}})\big)}_{k(\mathbf{x})}
\cdot \underbrace{\exp\!\big(\lambda \cdot \operatorname{clip}(1-u,\,-0{,}5,\,1)\big)}_{g(u)}
\cdot \underbrace{\exp\!\left(-\frac{E_a}{R}\Big(\frac{1}{T} - \frac{1}{T_\text{ref}}\Big)\right)}_{A(T)}
$$

- $G_\theta$: MLP 16 → 32 → 32 → 1, SiLU. $\operatorname{softplus}(z) = \ln(1 + e^z)$.
- $g(u)$: SOH càng thấp thì suy giảm càng nhanh (hiện tượng "đầu gối" cuối đời).
- $A(T)$: hệ số Arrhenius, $R = 8{,}314$ J/(mol·K), $T_\text{ref} = 298{,}15$ K.
- $\lambda = \operatorname{softplus}(\lambda_\text{raw})$, khởi tạo $\lambda = 1$.
- $E_a = 10^4 \cdot \operatorname{softplus}(e_\text{raw})$ J/mol, khởi tạo 30 kJ/mol.

Cả ba thừa số đều dương nên $r \ge 0$ với mọi tham số: mô hình vật lý không cho phép dung lượng
tự tăng.

Phương trình vật lý mà mạng nghiệm phải thoả:

$$
\frac{\partial u}{\partial t} = -\,r(\tilde{\mathbf{x}}, u, T)
$$

# 3. Hàm mất mát

## 3.1. Loss dữ liệu (chỉ cell có nhãn)

$$
\mathcal{L}_\text{data} = \frac{1}{B}\sum_{i=1}^{B}\big(u_i - y_i\big)^2
$$

## 3.2. Cặp chu kỳ cho loss vật lý

Lấy ngẫu nhiên chu kỳ $n_1$ của một cell, chọn bước $h \sim \mathcal{U}\{5, \dots, 50\}$ chu kỳ,
và $n_2$ là chu kỳ đầu tiên của cùng cell có chỉ số $\ge n_1 + h$. Ký hiệu
$u_1 = F_\varphi(\tilde{\mathbf{x}}_{n_1}, t_{n_1})$, $u_2 = F_\varphi(\tilde{\mathbf{x}}_{n_2}, t_{n_2})$,
$\Delta t = t_{n_2} - t_{n_1}$.

Ba loss dưới đây không dùng nhãn, nên áp được lên cả cell chưa đo dung lượng.

## 3.3. Loss phương trình động học

$$
\mathcal{L}_\text{ode} = \frac{1}{B}\sum\left(\frac{\partial u_1}{\partial t} + r(\tilde{\mathbf{x}}_{n_1}, u_1, T)\right)^2
$$

Đạo hàm $\partial u / \partial t$ tính bằng autograd. Biến thể dùng sai phân dọc quỹ đạo (dạng
Euler), dùng để so sánh:

$$
\mathcal{L}_\text{ode}^\text{Euler} = \frac{1}{B}\sum\big(u_2 - u_1 + r_1\,\Delta t\big)^2
$$

## 3.4. Loss đơn điệu

$$
\mathcal{L}_\text{mono} = \frac{1}{B}\sum \operatorname{ReLU}\big(u_2 - u_1 - \varepsilon\big), \qquad \varepsilon = 0{,}002
$$

SOH không được tăng giữa hai chu kỳ; dung sai $\varepsilon$ cho phép hiện tượng phục hồi dung lượng nhỏ.

## 3.5. Loss miền giá trị

$$
\mathcal{L}_\text{range} = \frac{1}{B}\sum\Big[\operatorname{ReLU}(u_1 - 1{,}15) + \operatorname{ReLU}(0{,}4 - u_1)
+ \mathbb{1}_\text{đầu}\big(\operatorname{ReLU}(u_1 - 1{,}10) + \operatorname{ReLU}(0{,}85 - u_1)\big)\Big]
$$

$\mathbb{1}_\text{đầu} = 1$ nếu $n_1$ là chu kỳ đầu tiên của cell (cell mới có SOH trong
$[0{,}85;\,1{,}10]$).

## 3.6. Tổng loss

$$
\mathcal{L} = \mathcal{L}_\text{data} + w(s)\Big(\alpha\,\mathcal{L}_\text{ode} + \beta\,\mathcal{L}_\text{mono}
+ \gamma\,\mathcal{L}_\text{range}\Big)
$$

$$
w(s) = \min\!\left(1,\; \frac{s}{0{,}2\,S}\right)
$$

$s$ là bước hiện tại, $S = 4000$. Trọng số vật lý tăng tuyến tính trong 800 bước đầu để mạng khớp
dữ liệu trước. Giá trị dùng: $\alpha = 0{,}02$, $\beta = 0{,}5$, $\gamma = 1$.

## 3.7. Ba mô hình so sánh

| Mô hình | $\mathcal{L}_\text{data}$ | Loss vật lý | Cặp chu kỳ lấy từ |
|---|---|---|---|
| MLP | cell có nhãn | không | — |
| PINN-sup | cell có nhãn | có | cell có nhãn (512 cặp) |
| PINN-semi | cell có nhãn | có | 256 cặp từ cell có nhãn + 256 cặp từ cell không nhãn |

Ba mô hình dùng cùng mạng nghiệm, cùng chuẩn hoá, cùng cách chia và cùng quá trình tối ưu.

# 4. Suy luận

Với một cell mới, ở mỗi chu kỳ $n$:

1. Lọc ngoại lai bằng các chu kỳ trước đó (mục 1.3).
2. Chuẩn hoá bằng $\boldsymbol{\mu}, \boldsymbol{\sigma}$ đã lưu khi huấn luyện.
3. $\widehat{\mathrm{SOH}}_n = F_\varphi(\tilde{\mathbf{x}}_n, n/1000)$.
4. Cảnh báo cuối đời khi $\widehat{\mathrm{SOH}}_n \le 0{,}85$ lần đầu.

Không cần dung lượng đo, không cần dữ liệu các chu kỳ tương lai.

# 5. Đánh giá

## 5.1. MAE theo cell

Chỉ số chính, với $\mathcal{C}$ là tập cell test và $n_c$ là số chu kỳ của cell $c$:

$$
\mathrm{MAE}_c = \frac{1}{n_c}\sum_{n}\big|\widehat{\mathrm{SOH}}_{c,n} - \mathrm{SOH}_{c,n}\big|, \qquad
\mathrm{MAE}_\text{cell} = \frac{1}{|\mathcal{C}|}\sum_{c \in \mathcal{C}} \mathrm{MAE}_c
$$

Mỗi cell có một phiếu, nên cell có vòng đời dài không lấn át kết quả. Vì mỗi cell được test một lần
trong mỗi lần lặp, $\mathrm{MAE}_c$ được lấy trung bình qua 5 lần lặp.

## 5.2. Tỉ số và mức giảm sai số

$$
\rho = \frac{\overline{\mathrm{MAE}}_\text{PINN}}{\overline{\mathrm{MAE}}_\text{MLP}}, \qquad
\text{giảm sai số} = 1 - \rho
$$

Khoảng tin cậy 95 % của $\rho$: bootstrap 20 000 lần, lấy lại mẫu theo cell.

## 5.3. Kiểm định

- So sánh cùng lượng nhãn: Wilcoxon dấu–hạng hai phía trên hiệu
  $d_c = \mathrm{MAE}_c^\text{PINN} - \mathrm{MAE}_c^\text{MLP}$ qua các cell; hiệu chỉnh Holm cho 4 bộ.
- PINN 30 % so với MLP 70 %, không kém hơn quá 10 %: Wilcoxon một phía trên
  $\mathrm{MAE}_c^\text{PINN,30\%} - 1{,}1 \cdot \mathrm{MAE}_c^\text{MLP,70\%} < 0$, kèm cận trên
  khoảng tin cậy 90 % của $\rho$ nhỏ hơn 1,10.
- Kiểm định chặt hơn ở mức fold (Nadeau–Bengio), với $J = 25$ fold và $d_j$ là hiệu MAE của fold $j$:

$$
t = \frac{\bar d}{\sqrt{\left(\dfrac{1}{J} + \dfrac{n_\text{test}}{n_\text{train}}\right) s_d^2}}, \qquad \text{bậc tự do } J - 1
$$

## 5.4. Chỉ số phụ

- MAE khi SOH ≤ 0,9 (giai đoạn cuối đời).
- Sai số EOL: $\hat n_\text{EOL} - n_\text{EOL}$, với $n_\text{EOL}$ là chu kỳ đầu tiên SOH ≤ 0,85;
  chỉ tính trên các cell mà cả giá trị đo lẫn dự đoán cùng chạm ngưỡng.
- Tăng ngược trên 100 chu kỳ:

$$
\frac{\sum_n \max(0,\, \hat u_{n+1} - \hat u_n)}{(n_\text{cuối} - n_\text{đầu})/100}
$$
