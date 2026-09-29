// Sinh slide báo cáo tiến độ 2 tuần — ĐACN HK261, nhóm 15
//   node make_slides.js
// Mọi con số trong đây đã đối chiếu với results/*.csv.

const pptxgen = require('pptxgenjs');
const path = require('path');

const IMG = (f) => path.join(__dirname, 'slide-hinh', f);

// ─────────────────────────────────────────────────────────────── bảng màu
// Lấy đúng hệ màu của báo cáo và của các hình: xanh lục đậm = ràng buộc vật lý
// giúp ích, cam cháy = tệ đi. Không dùng xanh dương mặc định.
const DARK = '0E3B33';      // nền slide mở/đóng
const DARK2 = '15544A';
const TEAL = '177A68';
const TEAL_LT = '7FD5C0';
const TEAL_SOFT = 'E8F2EE';
const MINT = 'A9C9BF';
const ORANGE = 'B8531F';
const ORANGE_SOFT = 'FAEDE4';
const INK = '1B2330';
const INK2 = '41505A';
const MUTED = '6B7A82';
const LINE = 'DCE3E0';
const WHITE = 'FFFFFF';

const HEAD = 'Cambria';     // phông tiêu đề, có chân, nằm trong danh sách an toàn
const BODY = 'Calibri';     // phông nội dung

const W = 13.333, H = 7.5, M = 0.62;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'Huỳnh Quốc Thắng';
pres.title = 'ĐACN HK261 — Báo cáo tiến độ 2 tuần';

// ─────────────────────────────────────────────────────────── khối dùng lại
function title(s, text, kicker) {
  if (kicker) {
    s.addText(kicker, {
      x: M, y: 0.42, w: 11.5, h: 0.28, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, bold: true, color: TEAL, charSpacing: 1.4,
    });
  }
  s.addText(text, {
    x: M, y: kicker ? 0.72 : 0.52, w: 12.1, h: 0.72, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 30, bold: true, color: INK, valign: 'top',
  });
}

// chip số tròn — mô-típ lặp lại xuyên suốt bộ slide
function chip(s, x, y, n, { fill = TEAL, txt = WHITE, d = 0.36 } = {}) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill } });
  s.addText(String(n), {
    x, y, w: d, h: d, isTextBox: true, margin: 0, align: 'center', valign: 'middle',
    fontFace: BODY, fontSize: 13, bold: true, color: txt,
  });
}

function card(s, x, y, w, h, { fill = WHITE, line = LINE, shadow = true } = {}) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: fill }, line: { color: line, width: 0.75 },
    ...(shadow ? { shadow: { type: 'outer', blur: 6, offset: 1, angle: 90, color: '9AA8A2', opacity: 0.22 } } : {}),
  });
}

function stat(s, x, y, w, big, label, { bigColor = TEAL, labColor = MUTED, size = 30 } = {}) {
  s.addText(big, {
    x, y, w, h: 0.56, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: size, bold: true, color: bigColor,
  });
  s.addText(label, {
    x, y: y + 0.54, w, h: 0.52, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, color: labColor,
  });
}

// ══════════════════════════════════════════════════════ 1 · trang bìa (nền tối)
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  // khối nhấn góc phải, thay cho thanh trang trí
  s.addShape(pres.ShapeType.roundRect, {
    x: 10.35, y: -1.1, w: 4.2, h: 4.2, rectRadius: 0.1, rotate: 18,
    fill: { color: DARK2 }, line: { color: DARK2 },
  });

  s.addText('ĐACN HK261  ·  NHÓM 15  ·  BÁO CÁO TIẾN ĐỘ 2 TUẦN', {
    x: M, y: 1.62, w: 10.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12.5, bold: true, color: TEAL_LT, charSpacing: 1.8,
  });
  s.addText('Ước lượng SOH pin Li-ion đơn\nbằng ràng buộc vật lý không cần nhãn', {
    x: M, y: 2.06, w: 11.4, h: 1.85, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, lineSpacingMultiple: 1.12,
  });
  s.addText('Huỳnh Quốc Thắng  ·  14 / 09 / 2026', {
    x: M, y: 4.08, w: 8, h: 0.36, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15.5, color: MINT,
  });

  s.addShape(pres.ShapeType.line, {
    x: M, y: 4.82, w: 12.1, h: 0, line: { color: '2A6459', width: 1 },
  });

  const stats = [
    ['4', 'bộ dữ liệu công khai'],
    ['387', 'cell đơn'],
    ['1 227', 'lượt huấn luyện đã chạy'],
    ['10', 'protocol thí nghiệm'],
  ];
  stats.forEach(([b, l], i) => {
    stat(s, M + i * 3.05, 5.12, 2.9, b, l, { bigColor: TEAL_LT, labColor: MINT, size: 32 });
  });

  s.addNotes('Mở đầu 20 giây. Nêu tên đề tài, rồi chốt ngay: giai đoạn 2 tuần này đã đi hết 3 mục tiêu bắt buộc của ĐACN và đã chạy được cả phần optional. Bốn con số ở dưới là để thầy thấy khối lượng đã làm.');
}

// ══════════════════════════════════════════ 2 · bài toán + trạng thái mục tiêu
{
  const s = pres.addSlide();
  title(s, 'Bài toán và vị trí trong bốn mục tiêu ĐACN', 'BỐI CẢNH');

  const items = [
    ['Đầu vào / đầu ra', 'Vào: 16 đặc trưng thống kê của MỘT chu kỳ sạc CC–CV. Ra: SOH của đúng chu kỳ đó.'],
    ['Nút thắt thực tế', 'Muốn có nhãn SOH phải xả đầy cell — tốn hàng giờ và phải dừng vận hành. Còn đường sạc thì BMS ghi được của MỌI cell, gần như miễn phí.'],
    ['Câu hỏi nghiên cứu', 'Ràng buộc vật lý có thay được nhãn không? Cụ thể: mô hình huấn luyện với 30 % cell có nhãn có đuổi kịp mô hình dùng đủ 70 % không?'],
    ['Phạm vi', 'Cell đơn (chưa xét pack) · 4 bộ công khai · 387 cell · 2 họ hoá học NCM/NCA và LFP.'],
  ];
  let y = 1.72;
  items.forEach(([h, d]) => {
    s.addText(h, {
      x: M, y, w: 6.5, h: 0.27, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13.5, bold: true, color: TEAL,
    });
    s.addText(d, {
      x: M, y: y + 0.28, w: 6.5, h: 0.76, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13.5, color: INK2, lineSpacingMultiple: 1.14,
    });
    y += 1.14;
  });

  // cột phải: 4 mục tiêu ĐACN kèm trạng thái
  const gx = 7.55, gw = 5.16;
  s.addText('BỐN MỤC TIÊU CỦA ĐỒ ÁN CHUYÊN NGÀNH', {
    x: gx, y: 1.72, w: gw, h: 0.26, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11, bold: true, color: MUTED, charSpacing: 1.1,
  });
  const goals = [
    [1, 'Xác định bài toán', 'Xong', TEAL],
    [2, 'Khảo sát research / công nghệ', 'Xong', TEAL],
    [3, 'Baseline và thiết kế lời giải', 'Xong', TEAL],
    [4, 'Thí nghiệm baseline / prototype  (optional)', 'Đang mở rộng', ORANGE],
  ];
  let gy = 2.1;
  goals.forEach(([n, t, st, c]) => {
    card(s, gx, gy, gw, 1.0, { fill: c === TEAL ? TEAL_SOFT : ORANGE_SOFT, line: c === TEAL ? 'C6E0D6' : 'EDD3C2', shadow: false });
    chip(s, gx + 0.24, gy + 0.32, n, { fill: c });
    s.addText(t, {
      x: gx + 0.74, y: gy + 0.16, w: gw - 1.0, h: 0.68, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13.5, bold: true, color: INK, valign: 'middle',
    });
    s.addText(st, {
      x: gx + 0.74, y: gy + 0.62, w: gw - 1.0, h: 0.26, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11, bold: true, color: c,
    });
    gy += 1.12;
  });

  s.addNotes('Nhấn vào chữ "gần như miễn phí" — đó là toàn bộ lý do bài toán này đáng làm. Nếu nhãn rẻ thì không cần vật lý. Bên phải cho thầy thấy ba mục tiêu bắt buộc đã xong, phần optional đang chạy tiếp.');
}

// ══════════════════════════════════ 3 · khảo sát — bốn khoảng trống PINN4SOH
{
  const s = pres.addSlide();
  title(s, 'Khảo sát: bốn khoảng trống của PINN4SOH', 'ĐÃ LÀM  ·  1 / 4');
  s.addText([
    { text: 'Xuất phát từ survey ', options: {} },
    { text: 'Su, Xu & Dong 2024', options: { italic: true } },
    { text: ' và bài ', options: {} },
    { text: 'PINN4SOH', options: { bold: true } },
    { text: ' (Wang et al., Nature Communications 2024). Đọc thẳng mã nguồn công bố — mỗi khoảng trống là một chỗ để đóng góp.', options: {} },
  ], {
    x: M, y: 1.52, w: 12.1, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12.5, color: MUTED,
  });

  const gaps = [
    ['Phần dư PDE có thể tự triệt tiêu', 'Mạng động học nhận chính u_t làm đầu vào, nên có thể học F ≈ u_t và đưa phần dư về 0 mà không cần chút vật lý nào.', 'Model.py:232–234'],
    ['"Physics loss" phụ thuộc nhãn', 'Phạt theo hướng thay đổi của NHÃN, nên không áp được lên cell chưa đo dung lượng — về cấu trúc, nó không thể bù cho nhãn.', 'Model.py:255'],
    ['Chuẩn hoá theo cell trên cả vòng đời', 'Chỉ số chu kỳ sau chuẩn hoá chính là "phần trăm tuổi thọ đã đi qua" — rò rỉ tương lai vào đầu vào.', 'dataloader.py:50, 58–64'],
    ['Validation chia theo hàng, không theo cell', 'Early-stopping chấm điểm trên đúng những cell mô hình đã học.', 'dataloader.py:157–158'],
  ];
  let y = 2.02;
  const cw = 8.55;
  gaps.forEach(([h, d, ref], i) => {
    card(s, M, y, cw, 1.06);
    chip(s, M + 0.26, y + 0.35, i + 1, { fill: i === 1 ? ORANGE : TEAL });
    s.addText(h, {
      x: M + 0.78, y: y + 0.12, w: cw - 1.05, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, bold: true, color: INK,
    });
    s.addText(d, {
      x: M + 0.78, y: y + 0.41, w: cw - 2.05, h: 0.56, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.1,
    });
    s.addText(ref, {
      x: M + cw - 2.28, y: y + 0.72, w: 2.05, h: 0.26, isTextBox: true, margin: 0, align: 'right',
      fontFace: 'Courier New', fontSize: 9.5, color: MUTED,
    });
    y += 1.16;
  });

  // callout: số đo được cho khoảng trống 3
  const bx = 9.5, bw = 3.22;
  card(s, bx, 2.02, bw, 2.46, { fill: ORANGE_SOFT, line: 'EDD3C2', shadow: false });
  s.addText('ĐO ĐƯỢC — KHOẢNG TRỐNG 3', {
    x: bx + 0.28, y: 2.24, w: bw - 0.56, h: 0.25, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10, bold: true, color: ORANGE, charSpacing: 0.8,
  });
  s.addText('−15 %', {
    x: bx + 0.28, y: 2.56, w: bw - 0.56, h: 0.72, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 42, bold: true, color: ORANGE,
  });
  s.addText('MAE trên XJTU giảm giả tạo từ 0.0092 xuống 0.0078 chỉ do rò rỉ chỉ số chu kỳ — không phải do mô hình tốt hơn.', {
    x: bx + 0.28, y: 3.32, w: bw - 0.56, h: 1.0, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.12,
  });

  card(s, bx, 4.62, bw, 2.0, { fill: TEAL_SOFT, line: 'C6E0D6', shadow: false });
  s.addText('HỆ QUẢ CHO THIẾT KẾ', {
    x: bx + 0.28, y: 4.84, w: bw - 0.56, h: 0.25, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10, bold: true, color: TEAL, charSpacing: 0.8,
  });
  s.addText('Loss vật lý phải KHÔNG cần nhãn, và pipeline phải chia theo cell với chuẩn hoá nhân quả. Hai điều này định hình toàn bộ thiết kế ở slide sau.', {
    x: bx + 0.28, y: 5.14, w: bw - 0.56, h: 1.2, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.12,
  });

  s.addNotes('Đây là phần khảo sát của mục tiêu 2, nhưng làm theo kiểu đọc mã nguồn chứ không chỉ đọc abstract. Nếu thầy hỏi vì sao không dùng thẳng PINN4SOH — câu trả lời nằm ở khoảng trống số 2: loss vật lý của họ phụ thuộc nhãn nên không thể là nguồn thông tin bù nhãn.');
}

// ══════════════════════════════════════════════ 4 · thiết kế lời giải
{
  const s = pres.addSlide();
  title(s, 'Thiết kế lời giải: động học xám + ba loss không cần nhãn', 'ĐÃ LÀM  ·  2 / 4');

  // ── sơ đồ vẽ trực tiếp bằng khối, đọc được khi chiếu (bản đầy đủ ở fig_pipeline.png)
  const dy = 1.62, bh = 0.92;
  const box = (x, w, t, sub, fill, lineC, txtC) => {
    s.addShape(pres.ShapeType.roundRect, {
      x, y: dy, w, h: bh, rectRadius: 0.07,
      fill: { color: fill }, line: { color: lineC, width: 0.9 },
    });
    s.addText(t, {
      x: x + 0.12, y: dy + 0.14, w: w - 0.24, h: 0.3, isTextBox: true, margin: 0, align: 'center',
      fontFace: BODY, fontSize: 13, bold: true, color: txtC,
    });
    s.addText(sub, {
      x: x + 0.12, y: dy + 0.46, w: w - 0.24, h: 0.36, isTextBox: true, margin: 0, align: 'center',
      fontFace: BODY, fontSize: 10.5, color: txtC === WHITE ? MINT : MUTED,
    });
  };
  const arrow = (x, w, y) => s.addShape(pres.ShapeType.line, {
    x, y, w, h: 0, line: { color: INK2, width: 1.25, endArrowType: 'triangle' },
  });

  box(M, 2.5, 'x  (16 đặc trưng)  +  N/1000', 'một chu kỳ sạc CC–CV', WHITE, LINE, INK);
  arrow(M + 2.5, 0.42, dy + bh / 2);
  box(M + 2.92, 2.72, 'Mạng nghiệm  u = Fᵩ(x, t)', 'MLP 17→64→64→32→1 · SiLU', TEAL_SOFT, TEAL, INK);
  arrow(M + 5.64, 0.42, dy + bh / 2);
  box(M + 6.06, 2.62, 'Động học xám  r(x, u, T)', 'r ≥ 0 theo cấu trúc', ORANGE_SOFT, ORANGE, INK);
  box(M + 9.02, 3.09, 'SOH  của chu kỳ đó', 'suy luận: CHỈ mạng nghiệm', DARK, DARK, WHITE);
  arrow(M + 8.68, 0.34, dy + bh / 2);

  s.addText('r  =  softplus(MLP_θ(x))  ·  exp( λ(1 − u) )  ·  exp( −Eₐ/R · (1/T − 1/T_ref) )', {
    x: M, y: dy + bh + 0.2, w: 12.1, h: 0.34, isTextBox: true, margin: 0, align: 'center',
    fontFace: 'Cambria', fontSize: 15, color: INK,
  });
  s.addText('không âm  →  đơn điệu theo cấu trúc          |          pha "đầu gối"          |          Arrhenius theo nhiệt độ', {
    x: M, y: dy + bh + 0.56, w: 12.1, h: 0.28, isTextBox: true, margin: 0, align: 'center',
    fontFace: BODY, fontSize: 10.5, color: MUTED,
  });

  // ── ba thẻ nội dung
  const cy = 3.68, cw = 3.87, gap = 0.25;
  const cards = [
    ['Bốn hàm mất mát', 'L_data cần nhãn. L_ode (dạng Euler) · L_mono (ε = 0.002) · L_range đều KHÔNG cần nhãn, nên chạy được cả trên cell chưa đo dung lượng. Đây chính là điều PINN4SOH không làm được.', TEAL],
    ['Khác PINN4SOH ở đâu', 'Mạng động học KHÔNG nhận đạo hàm của u làm đầu vào, nên phần dư không thể tự triệt tiêu. Chỉ số chu kỳ chia cho hằng số toàn cục 1000, không bao giờ chuẩn hoá theo cell.', ORANGE],
    ['Baseline và pipeline', 'Baseline: MLP cùng kích thước, chỉ L_data. Chia theo CELL 70/15/15 có phân tầng, chuẩn hoá nhân quả, assert không cell nào nằm ở hai tập. Tổng ≈ 8 000 tham số.', TEAL],
  ];
  cards.forEach(([h, d, c], i) => {
    const x = M + i * (cw + gap);
    card(s, x, cy, cw, 2.06);
    chip(s, x + 0.26, cy + 0.24, i + 1, { fill: c });
    s.addText(h, {
      x: x + 0.74, y: cy + 0.26, w: cw - 1.0, h: 0.32, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, bold: true, color: INK, valign: 'middle',
    });
    s.addText(d, {
      x: x + 0.26, y: cy + 0.72, w: cw - 0.52, h: 1.2, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12, color: INK2, lineSpacingMultiple: 1.16,
    });
  });

  s.addText('Sơ đồ đầy đủ đúng như mã nguồn chạy (dữ liệu → mạng → một bước huấn luyện → suy luận) nằm ở hình fig_pipeline.png, mở khi cần đi sâu.', {
    x: M, y: 6.0, w: 12.1, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10.5, color: MUTED,
  });

  s.addNotes('Điểm cần bảo vệ nếu bị hỏi: khác PINN4SOH, mạng động học ở đây không nhận đạo hàm của u làm đầu vào, nên phần dư không thể tự triệt tiêu. Sơ đồ đầy đủ đúng như mã nguồn chạy nằm ở file fig_pipeline.png, mở ra nếu thầy muốn xem chi tiết một bước huấn luyện.');
}

// ══════════════════════════════════════ 5 · kết quả — hiệu quả theo lượng nhãn
{
  const s = pres.addSlide();
  title(s, 'Ít nhãn thì vật lý thắng, nhãn đủ thì hoà', 'ĐÃ LÀM  ·  3 / 4');

  s.addImage({ path: IMG('fig_E1_label_efficiency.png'), x: 1.28, y: 1.5, w: 10.78, h: 3.93 });
  const st = [
    ['4 / 4', 'Ở mức 10 % nhãn, PINN thắng MLP trên cả bốn bộ. Tỉ số MAE(MLP)/MAE(PINN): XJTU 1.7× · TJU 1.1× · MIT 1.3× · HUST 1.9×', TEAL],
    ['15 / 16', 'Số ô PINN thắng MLP ở cùng mức nhãn, khi trọng số vật lý β được chọn theo validation thay vì cố định β = 5.', TEAL],
    ['2 / 4', 'Giả thuyết "30 % đuổi kịp 70 %": xác nhận trên MIT (0.91×) và HUST (0.96×); chưa trên XJTU (1.25×) và TJU (1.17×).', ORANGE],
  ];
  const sw = 3.87, sgap = 0.25;
  st.forEach(([b, l, c], i) => {
    const x = M + i * (sw + sgap);
    card(s, x, 5.62, sw, 1.36, { fill: c === TEAL ? TEAL_SOFT : ORANGE_SOFT, line: c === TEAL ? 'C6E0D6' : 'EDD3C2', shadow: false });
    s.addText(b, {
      x: x + 0.22, y: 5.62, w: 1.16, h: 1.36, isTextBox: true, margin: 0, valign: 'middle',
      fontFace: HEAD, fontSize: 26, bold: true, color: c,
    });
    s.addText(l, {
      x: x + 1.4, y: 5.62, w: sw - 1.62, h: 1.36, isTextBox: true, margin: 0, valign: 'middle',
      fontFace: BODY, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.12,
    });
  });

  s.addNotes('Trục dọc là thang log dùng chung cho cả bốn bộ, nên cùng khoảng cách dọc là cùng một tỉ số sai số. Chỉ vào panel MIT và HUST trước, đó là chỗ câu chuyện rõ nhất. Nói thêm: vật lý là một prior, nó bù nhiều nhất ở nơi mô hình chỉ-dữ-liệu đói nhãn nhất; ở TJU thì 30 % nhãn đã đủ cho MLP nên ràng buộc trở thành thiên kiến. Câu chuyện trung thực là "ít nhãn thì vật lý thắng, nhãn đủ thì hoà".');
}

// ══════════════════════════════════ 6 · hai phép kiểm chứng khắt khe
{
  const s = pres.addSlide();
  title(s, 'Hai phép kiểm chứng khắt khe với chính mình', 'ĐÃ LÀM  ·  4 / 4');

  // (a) ngân sách tinh chỉnh cân bằng
  s.addText('(a)   Cân bằng ngân sách tinh chỉnh — E12, 384 lượt', {
    x: M, y: 1.54, w: 7.35, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14, bold: true, color: TEAL,
  });
  s.addImage({ path: IMG('fig_E12_fair.png'), x: M, y: 1.92, w: 7.35, h: 3.03 });
  s.addText([
    { text: 'Lần đầu so sánh, baseline MLP chưa được tinh chỉnh ngang với PINN — không công bằng. Khi cho MLP hưởng đúng lưới siêu tham số mà PINN được hưởng: tỉ số PINN/MLP trung bình đi từ ', options: {} },
    { text: '0.72× lên 0.90×', options: { bold: true, color: ORANGE } },
    { text: ' — một phần lợi thế của PINN biến mất thật. PINN vẫn thắng ', options: {} },
    { text: '7/8 ô', options: { bold: true, color: TEAL } },
    { text: ', tỉ số 0.75–1.01×. Riêng MIT @ 30 %, baseline tự nó cải thiện 0.0103 → 0.0048 chỉ nhờ dropout 0.1.', options: {} },
  ], {
    x: M, y: 5.06, w: 7.35, h: 1.5, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.15,
  });

  // (b) chuyển miền
  const bx = 8.34, bw = 4.37;
  s.addText('(b)   Đem sang bộ dữ liệu khác — E5', {
    x: bx, y: 1.54, w: bw, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14, bold: true, color: TEAL,
  });
  card(s, bx, 1.92, bw, 4.64, { fill: TEAL_SOFT, line: 'C6E0D6', shadow: false });
  s.addText('Zero-shot thất bại với mọi mô hình — đặc trưng đường sạc phụ thuộc protocol nên đầu vào bộ đích nằm ngoài phân bố. Nhưng vì loss vật lý không cần nhãn, chúng chạy được ngay trên cell của bộ đích và trở thành cơ chế thích nghi miền không nhãn.', {
    x: bx + 0.26, y: 2.14, w: bw - 0.52, h: 1.32, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.15,
  });

  const hdr = [['Cặp chuyển', 'MLP', 'PINN', '']];
  const rows = [
    ['XJTU → TJU', '2.944', '0.041', '72×'],
    ['HUST → MIT', '0.369', '0.027', '13×'],
    ['TJU → XJTU', '0.329', '0.103', '3.2×'],
    ['MIT → HUST', '0.062', '0.059', '1.05×'],
  ];
  const colX = [bx + 0.26, bx + 1.66, bx + 2.52, bx + 3.42];
  const colW = [1.4, 0.86, 0.9, 0.72];
  let ry = 3.56;
  hdr.concat(rows).forEach((r, ri) => {
    r.forEach((c, ci) => {
      s.addText(c, {
        x: colX[ci], y: ry, w: colW[ci], h: 0.32, isTextBox: true, margin: 0,
        align: ci === 0 ? 'left' : 'right', valign: 'middle',
        fontFace: BODY, fontSize: ri === 0 ? 10 : 12,
        bold: ri === 0 || ci === 3, charSpacing: ri === 0 ? 0.6 : 0,
        color: ri === 0 ? MUTED : (ci === 3 ? TEAL : INK),
      });
    });
    if (ri === 0) {
      s.addShape(pres.ShapeType.line, { x: bx + 0.26, y: ry + 0.33, w: bw - 0.52, h: 0, line: { color: 'BFD8CE', width: 0.9 } });
    }
    ry += ri === 0 ? 0.42 : 0.5;
  });
  s.addText('MAE trên cell test của bộ đích, không có cell đích nào mang nhãn (k = 0).', {
    x: bx + 0.26, y: 5.9, w: bw - 0.52, h: 0.5, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10.5, color: MUTED, lineSpacingMultiple: 1.1,
  });

  s.addNotes('Con số 0.72 lên 0.90 là chỗ nên chủ động nói ra trước khi bị hỏi — nó cho thấy mình tự kiểm chứng chứ không chọn số đẹp. Bên phải: zero-shot thất bại hoàn toàn, nhưng chính vì loss vật lý không cần nhãn nên nó biến thành cơ chế thích nghi miền, đây là chỗ vật lý trả lại giá trị thật khi đổi bộ dữ liệu.');
}

// ══════════════════════════════════════════════════════ 7 · khó khăn
{
  const s = pres.addSlide();
  title(s, 'Khó khăn gặp phải', 'BÁO CÁO  ·  PHẦN 2');

  const rows = [
    ['Phần dư dạng đạo hàm khuếch đại nhiễu 1000×', 'Chia cho Δt = h/1000; với h = 1 thì dao động 10⁻³ của mạng thành phần dư cỡ 1 — lớn hơn cả tín hiệu thật. MAE xấu gấp đôi MLP thuần.', 'Viết lại phần dư ở dạng Euler', 'Đã xử lý', TEAL],
    ['Dấu tương quan đặc trưng–SOH đổi theo hoá học', 'CV Q tương quan −0.94 ở XJTU nhưng +0.41 ở MIT; ngay cả hai bộ cùng NCM cũng lệch (CC Q: −0.18 so với +0.95).', 'Bỏ mọi ràng buộc dấu trên đặc trưng; vật lý chỉ đặt lên quỹ đạo SOH', 'Đã xử lý', TEAL],
    ['So sánh ban đầu không công bằng', 'Baseline MLP chưa được tinh chỉnh ngang với PINN nên lợi thế bị thổi phồng.', 'Chạy thêm E12 cho hai mô hình cùng lưới; kết quả tự hạ 0.72× → 0.90×', 'Đã xử lý', TEAL],
    ['Zero-shot chuyển miền thất bại hoàn toàn', 'Sai 2–3 bậc độ lớn. Có thể không khả thi giữa hai họ hoá học với bộ đặc trưng này.', 'Hiện chỉ khai thác được dạng thích nghi miền không nhãn', 'Chưa xong', ORANGE],
    ['Các chỉ số đánh giá mâu thuẫn nhau', 'TJU @ 30 %: MLP thắng theo MAE (0.0088 / 0.0099) nhưng PINN thắng theo cell tệ nhất (0.0259 / 0.0283), vi phạm đơn điệu (0.20 / 0.68) và sai số EOL (24.5 / 29.8 chu kỳ).', 'Chưa chốt được chỉ số chính để báo cáo', 'Cần ý kiến thầy', ORANGE],
    ['Chạy trên CPU nên chỉ đủ 3 seed', 'Chưa đủ để kết luận thống kê, mới chỉ có trung bình và độ lệch chuẩn.', 'Chuyển sang GPU trong 2 tuần tới', 'Kế hoạch', MUTED],
  ];

  const cw = 6.18, gap = 0.34, ch = 1.62;
  rows.forEach((r, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = M + col * (cw + gap), y = 1.52 + row * (ch + 0.2);
    card(s, x, y, cw, ch);
    chip(s, x + 0.24, y + 0.22, i + 1, { fill: r[4] === MUTED ? '94A3A0' : r[4], d: 0.32 });
    s.addText(r[0], {
      x: x + 0.66, y: y + 0.16, w: cw - 2.05, h: 0.44, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12.5, bold: true, color: INK, lineSpacingMultiple: 1.0,
    });
    // nhãn trạng thái
    s.addShape(pres.ShapeType.roundRect, {
      x: x + cw - 1.34, y: y + 0.17, w: 1.12, h: 0.3, rectRadius: 0.14,
      fill: { color: r[4] === TEAL ? TEAL_SOFT : (r[4] === ORANGE ? ORANGE_SOFT : 'EEF1F0') },
      line: { color: r[4] === TEAL ? 'C6E0D6' : (r[4] === ORANGE ? 'EDD3C2' : 'DDE3E1'), width: 0.75 },
    });
    s.addText(r[3], {
      x: x + cw - 1.34, y: y + 0.17, w: 1.12, h: 0.3, isTextBox: true, margin: 0,
      align: 'center', valign: 'middle',
      fontFace: BODY, fontSize: 9.5, bold: true, color: r[4] === MUTED ? MUTED : r[4],
    });
    s.addText(r[1], {
      x: x + 0.66, y: y + 0.62, w: cw - 0.92, h: 0.58, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.1,
    });
    s.addText([
      { text: 'Hướng xử lý:  ', options: { bold: true, color: r[4] === MUTED ? MUTED : r[4] } },
      { text: r[2], options: { color: INK2 } },
    ], {
      x: x + 0.66, y: y + 1.22, w: cw - 0.92, h: 0.32, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 10.5,
    });
  });

  s.addNotes('Nếu hết giờ thì chỉ nói ba ô đầu — đó là ba khó khăn đã tự giải quyết được. Ô số 5 (chỉ số mâu thuẫn) là chỗ nên chủ động xin ý kiến thầy, vì nó quyết định cách viết mục kết quả.');
}

// ══════════════════════════════════════════════ 8 · kế hoạch (nền tối)
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  s.addShape(pres.ShapeType.roundRect, {
    x: 11.1, y: 5.4, w: 3.6, h: 3.6, rectRadius: 0.1, rotate: 18,
    fill: { color: DARK2 }, line: { color: DARK2 },
  });

  s.addText('BÁO CÁO  ·  PHẦN 3', {
    x: M, y: 0.5, w: 11.5, h: 0.28, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, bold: true, color: TEAL_LT, charSpacing: 1.4,
  });
  s.addText('Công việc hai tuần tiếp theo', {
    x: M, y: 0.8, w: 12.1, h: 0.7, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 30, bold: true, color: WHITE,
  });

  const weeks = [
    ['TUẦN 1', 'Làm cho kết quả hiện có đủ tin cậy', [
      ['Chuyển toàn bộ sang GPU', 'Tăng 3 → 10 seed; kiểm định thống kê ghép cặp theo cell để có khoảng tin cậy thay vì chỉ trung bình.'],
      ['Mở rộng lưới ngân sách công bằng', 'E12 hiện chỉ cân bằng ở 30 % và 70 % — bổ sung mức 10 % và 50 %, thêm learning rate và số bước vào lưới.'],
      ['Chốt bộ chỉ số chính để báo cáo', 'Chọn giữa MAE trung bình và sai số dự báo EOL, rồi viết lại mục kết quả theo chỉ số đã chốt.'],
    ]],
    ['TUẦN 2', 'Mở rộng đóng góp và dựng prototype', [
      ['Quét số cell đích cần đo nhãn', 'k = 0 / 1 / 3 / 5 / 10 cho bài toán thích nghi miền, để trả lời "dây chuyền mới cần đo bao nhiêu cell".'],
      ['Sửa đầu ra đơn điệu', 'Đơn điệu dọc quỹ đạo thật, không chỉ đơn điệu theo biến t như hiện nay.'],
      ['Dựng demo prototype', 'Suy luận online trên một cell, không cần biết tuổi thọ tương lai — đúng điều kiện BMS chạy thật.'],
    ]],
  ];

  let x = M;
  weeks.forEach(([wk, sub, items], wi) => {
    const cw = 5.96;
    s.addText(wk, {
      x, y: 1.78, w: cw, h: 0.28, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, bold: true, color: TEAL_LT, charSpacing: 1.4,
    });
    s.addText(sub, {
      x, y: 2.06, w: cw, h: 0.34, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 17, bold: true, color: WHITE,
    });
    let y = 2.62;
    items.forEach((it, i) => {
      const n = wi * 3 + i + 1;
      chip(s, x, y + 0.02, n, { fill: TEAL_LT, txt: DARK, d: 0.34 });
      s.addText(it[0], {
        x: x + 0.5, y, w: cw - 0.5, h: 0.3, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 13.5, bold: true, color: WHITE,
      });
      s.addText(it[1], {
        x: x + 0.5, y: y + 0.3, w: cw - 0.62, h: 0.72, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 11.5, color: MINT, lineSpacingMultiple: 1.14,
      });
      y += 1.12;
    });
    x += cw + 0.18;
  });

  // ô xin ý kiến
  card(s, M, 6.1, 12.1, 0.92, { fill: '17544A', line: '2E6E61', shadow: false });
  s.addText([
    { text: 'Cần hỗ trợ từ thầy:  ', options: { bold: true, color: TEAL_LT } },
    { text: 'chọn chỉ số chính để báo cáo — MAE trung bình hay sai số dự báo EOL? Hai chỉ số cho hai kết luận ngược nhau trên TJU.', options: { color: WHITE } },
  ], {
    x: M + 0.32, y: 6.1, w: 11.46, h: 0.92, isTextBox: true, margin: 0, valign: 'middle',
    fontFace: BODY, fontSize: 13.5,
  });

  s.addNotes('Kết bằng câu hỏi xin ý kiến, để buổi họp có đầu ra chứ không chỉ là báo cáo một chiều. Nếu thầy hỏi tiến độ tổng thể: ba mục tiêu bắt buộc đã xong, hai tuần tới là củng cố thống kê và dựng prototype.');
}

pres.writeFile({ fileName: path.join(__dirname, 'bao-cao-2-tuan.pptx') })
  .then((f) => console.log('đã ghi:', f));
