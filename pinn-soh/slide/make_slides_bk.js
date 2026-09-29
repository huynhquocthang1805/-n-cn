// Báo cáo tiến độ 2 tuần — ĐACN HK261, nhóm 15
// Dựng theo đúng khuôn mẫu file DACN_99_Group6.pdf (nền, logo, thanh tiêu đề navy,
// hộp có đầu đề, bullet tròn, chip dẫn dắt, dòng "Trích dẫn", số trang n/15).
//   node make_slides_bk.js
// Nội dung lấy từ bao-cao-md/bao-cao.md; số liệu đối chiếu results/*.csv.

const pptxgen = require('pptxgenjs');
const path = require('path');

const A = (f) => path.join(__dirname, 'mau', f);
const IMG = (f) => path.join(__dirname, 'slide-hinh', f);

// ───────────────────────────────────────────── hệ màu lấy mẫu từ file PDF mẫu
const NAVY = '1E4E79';            // (30,78,121) — thanh tiêu đề và đầu hộp
const NAVY_D = '163B5C';
const BLUE = '2E75B6';
const BLUE_LT = 'DEEBF7';
const ORANGE = 'C55A11';
const ORANGE_LT = 'FBE5D6';
const GREEN_LT = 'E2EFDA';
const GREEN = '548235';
const INK = '1A1A1A';
const INK2 = '3B3B3B';
const GREY = '7F7F7F';
const LINE = 'BFBFBF';
const WHITE = 'FFFFFF';

const F = 'Arial';                // phông an toàn, có đủ dấu tiếng Việt
const W = 13.333, H = 7.5;
const M = 0.52;                   // lề trái/phải
const BAR_H = 0.78;               // chiều cao thanh tiêu đề
const TOP = BAR_H + 0.30;         // mốc y nội dung bắt đầu
const TOTAL = 16;                 // tổng số slide nội dung, dùng cho "n / 16"

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'Huỳnh Quốc Thắng';
pres.title = 'ĐACN HK261 — Báo cáo tiến độ 2 tuần';

let page = 0;

// ───────────────────────────────────────────────────────────── khối dùng lại
function slide(titleText) {
  const s = pres.addSlide();
  s.background = { path: A('noi-dung-bg.png') };
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: W, h: BAR_H, fill: { color: NAVY }, line: { color: NAVY } });
  s.addText(titleText, {
    x: 0.34, y: 0, w: W - 1.0, h: BAR_H, isTextBox: true, margin: 0, valign: 'middle',
    fontFace: F, fontSize: 25, bold: true, color: WHITE,
  });
  page += 1;
  s.addShape(pres.ShapeType.rect, { x: 11.55, y: 7.16, w: W - 11.55, h: 0.34, fill: { color: NAVY }, line: { color: NAVY } });
  s.addText(`${page} / ${TOTAL}`, {
    x: 12.02, y: 7.16, w: 0.85, h: 0.34, isTextBox: true, margin: 0, align: 'center', valign: 'middle',
    fontFace: F, fontSize: 11.5, bold: true, color: WHITE,
  });
  return s;
}

// bullet tròn xanh + đoạn chữ (đầu dòng in đậm)
function bullet(s, x, y, w, lead, rest, { size = 13.5, gap = 0.30, color = INK2 } = {}) {
  s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.055, w: 0.135, h: 0.135, fill: { color: NAVY }, line: { color: BLUE, width: 1 } });
  const runs = [];
  if (lead) runs.push({ text: lead + ' ', options: { bold: true, color: INK } });
  if (rest) runs.push({ text: rest, options: { color } });
  return s.addText(runs, {
    x: x + gap, y, w: w - gap, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F, fontSize: size, lineSpacingMultiple: 1.12, valign: 'top',
  });
}

// hộp có đầu đề navy, đúng kiểu trong file mẫu
function boxed(s, x, y, w, h, head, body, { fill = WHITE, headFill = NAVY, headH = 0.42, size = 13 } = {}) {
  s.addShape(pres.ShapeType.rect, {
    x, y, w, h, fill: { color: fill }, line: { color: LINE, width: 0.75 },
    shadow: { type: 'outer', blur: 7, offset: 2, angle: 45, color: '808080', opacity: 0.35 },
  });
  s.addShape(pres.ShapeType.rect, { x, y, w, h: headH, fill: { color: headFill }, line: { color: headFill } });
  s.addText(head, {
    x: x + 0.16, y, w: w - 0.32, h: headH, isTextBox: true, margin: 0, valign: 'middle',
    fontFace: F, fontSize: 14, bold: true, color: WHITE,
  });
  const by = y + headH + 0.16;
  if (typeof body === 'string') {
    s.addText(body, {
      x: x + 0.18, y: by, w: w - 0.36, h: h - headH - 0.3, isTextBox: true, margin: 0, valign: 'top',
      fontFace: F, fontSize: size, color: INK2, lineSpacingMultiple: 1.16,
    });
  } else if (Array.isArray(body)) {
    // dùng bullet gốc của thư viện: tự xuống dòng và tự giãn cách, không ước lượng bằng tay
    s.addText(body.map((t, i) => ({
      text: t, options: { bullet: { code: '25CF' }, breakLine: i < body.length - 1, paraSpaceAfter: 7 },
    })), {
      x: x + 0.2, y: by, w: w - 0.4, h: h - headH - 0.3, isTextBox: true, margin: 0, valign: 'top',
      fontFace: F, fontSize: 12.5, color: INK2, lineSpacingMultiple: 1.14,
    });
  }
}

// chip dẫn dắt "Dẫn tới" / "Khoảng trống" như trong mẫu
function chipLine(s, x, y, w, label, text, { fill = BLUE_LT, edge = NAVY, ink = NAVY } = {}) {
  const lw = 0.26 + label.length * 0.085;
  s.addShape(pres.ShapeType.rect, { x, y, w: lw, h: 0.32, fill: { color: fill }, line: { color: edge, width: 1 } });
  s.addText(label, {
    x, y, w: lw, h: 0.32, isTextBox: true, margin: 0, align: 'center', valign: 'middle',
    fontFace: F, fontSize: 12, bold: true, color: ink,
  });
  s.addText(text, {
    x: x + lw + 0.18, y: y - 0.03, w: w - lw - 0.18, h: 0.6, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 13, color: INK2, lineSpacingMultiple: 1.12,
  });
}

// dòng "Trích dẫn:" ở chân slide
function cite(s, runs, y = 6.62) {
  s.addShape(pres.ShapeType.line, { x: M, y, w: 4.2, h: 0, line: { color: LINE, width: 0.75 } });
  s.addText([{ text: 'Trích dẫn: ', options: { bold: true, color: INK } }, ...runs], {
    x: M, y: y + 0.08, w: W - 2 * M - 1.3, h: 0.42, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 10.5, color: GREY, lineSpacingMultiple: 1.1,
  });
}

// bảng kiểu mẫu: đầu bảng in đậm, chỉ có đường kẻ ngang mảnh
function simpleTable(s, x, y, colW, rows, { size = 12, head = true, rowH = 0.42, headSize = 12.5, aligns = null } = {}) {
  let ry = y;
  rows.forEach((r, ri) => {
    let cx = x;
    const isHead = head && ri === 0;
    const lines = r.map((c, ci) => Math.ceil(String(c).length / Math.max(6, Math.round(colW[ci] * (size === 12 ? 10.5 : 12)))));
    const h = Math.max(rowH, 0.245 * Math.max(1, ...lines) + 0.18);
    r.forEach((c, ci) => {
      s.addText(String(c), {
        x: cx, y: ry, w: colW[ci], h, isTextBox: true, margin: 0, valign: 'middle',
        align: aligns ? aligns[ci] : 'left',
        fontFace: F, fontSize: isHead ? headSize : size, bold: isHead,
        color: isHead ? INK : INK2, lineSpacingMultiple: 1.1,
      });
      cx += colW[ci];
    });
    ry += h;
    s.addShape(pres.ShapeType.line, {
      x, y: ry - 0.04, w: colW.reduce((a, b) => a + b, 0), h: 0,
      line: { color: isHead ? GREY : 'D9D9D9', width: isHead ? 1 : 0.6 },
    });
  });
  return ry;
}

// ══════════════════════════════════════════════════════════ 0 · trang bìa
{
  const s = pres.addSlide();
  s.background = { path: A('bia-bg.png') };
  s.addImage({ path: A('vnu.png'), x: 5.20, y: 0.34, w: 1.42, h: 0.70 });
  s.addImage({ path: A('bk.png'), x: 6.92, y: 0.30, w: 0.68, h: 0.78 });
  s.addText('Đồ án chuyên ngành — HK261', {
    x: 1.0, y: 1.62, w: W - 2.0, h: 0.5, isTextBox: true, margin: 0, align: 'center',
    fontFace: F, fontSize: 25, bold: true, color: NAVY,
  });
  s.addText('Ước lượng SOH pin Li-ion đơn bằng ràng buộc vật lý không cần nhãn', {
    x: 0.9, y: 2.16, w: W - 1.8, h: 0.92, isTextBox: true, margin: 0, align: 'center',
    fontFace: F, fontSize: 21, bold: true, color: BLUE, lineSpacingMultiple: 1.14,
  });
  s.addText('Huỳnh Quốc Thắng     —     Nhóm 15', {
    x: 1.0, y: 3.42, w: W - 2.0, h: 0.4, isTextBox: true, margin: 0, align: 'center',
    fontFace: F, fontSize: 17, color: INK,
  });
  s.addText('Báo cáo tiến độ hai tuần  ·  14 / 09 / 2026', {
    x: 1.0, y: 3.90, w: W - 2.0, h: 0.34, isTextBox: true, margin: 0, align: 'center',
    fontFace: F, fontSize: 13.5, color: GREY,
  });
  s.addImage({ path: A('footer.png'), x: 4.17, y: 6.62, w: 5.0, h: 0.44 });
  s.addNotes('Chào Thầy. Em báo cáo tiến độ hai tuần của đề tài ước lượng SOH pin đơn. Bố cục theo yêu cầu: việc đã làm, khó khăn, và kế hoạch hai tuần tới; xen giữa là phần khảo sát và thiết kế để Thầy tiện theo dõi.');
}

// ══════════════════════════════════════════════════════ 1 · bài toán đặt ra
{
  const s = slide('Bài toán đặt ra');
  let y = TOP;
  bullet(s, M, y, 12.3, 'Ràng buộc thực tế.',
    'Nhãn SOH chỉ có được bằng cách xả đầy cell — tốn hàng giờ và phải dừng vận hành, nên chỉ đo được cho vài cell. ' +
    'Trong khi đó đường sạc CC–CV thì BMS ghi được của MỌI cell, gần như miễn phí.');
  y += 0.86;
  bullet(s, M, y, 12.3, 'Câu hỏi nghiên cứu.',
    'Ràng buộc vật lý có thay được nhãn không? Cụ thể: mô hình huấn luyện với 30 % cell mang nhãn có đuổi kịp mô hình dùng đủ 70 % không?');
  y += 0.78;

  s.addText('Bốn tiêu chí lời giải phải đạt', {
    x: M, y, w: 12.3, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 13.5, bold: true, color: NAVY,
  });
  y += 0.38;
  const crit = [
    ['1', 'Loss vật lý KHÔNG cần nhãn', 'để áp được lên cell chưa đo dung lượng'],
    ['2', 'Đơn điệu theo cấu trúc', 'SOH không tăng ngược, không phụ thuộc tham số học'],
    ['3', 'Pipeline không rò rỉ', 'chia theo cell, chuẩn hoá nhân quả, tính được online'],
    ['4', 'Đủ nhẹ cho thiết bị nhúng', 'khoảng 8 000 tham số, chạy được trên vi điều khiển'],
  ];
  crit.forEach(([n, t, d], i) => {
    const cx = M + (i % 2) * 6.25, cy = y + Math.floor(i / 2) * 0.88;
    s.addShape(pres.ShapeType.ellipse, { x: cx, y: cy + 0.02, w: 0.34, h: 0.34, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(n, {
      x: cx, y: cy + 0.02, w: 0.34, h: 0.34, isTextBox: true, margin: 0, align: 'center', valign: 'middle',
      fontFace: F, fontSize: 13, bold: true, color: WHITE,
    });
    s.addText(t, { x: cx + 0.48, y: cy, w: 5.6, h: 0.3, isTextBox: true, margin: 0, fontFace: F, fontSize: 13.5, bold: true, color: INK });
    s.addText(d, { x: cx + 0.48, y: cy + 0.3, w: 5.6, h: 0.5, isTextBox: true, margin: 0, fontFace: F, fontSize: 12, color: GREY, lineSpacingMultiple: 1.1 });
  });

  chipLine(s, M, 5.66, 12.3, 'Phạm vi',
    'Cell đơn, chưa xét pack · 4 bộ dữ liệu công khai · 387 cell · 2 họ hoá học NCM/NCA và LFP.');
  cite(s, [{ text: 'Su, Xu & Dong (2024), ' }, { text: 'Energy Conversion and Economics', options: { italic: true } },
           { text: ' 5(4) · Wang et al. (2024), ' }, { text: 'Nature Communications', options: { italic: true } }, { text: ' 15, 4332.' }]);
  s.addNotes('Nhấn vào chữ "gần như miễn phí" — đó là toàn bộ lý do bài toán này đáng làm. Nếu nhãn rẻ thì không cần vật lý. Bốn tiêu chí ở dưới là thước đo để chấm chính lời giải của mình ở các slide sau.');
}

// ══════════════════════════════════════════════════════════ 2 · dữ liệu
{
  const s = slide('Dữ liệu: 387 cell, 4 bộ, 2 họ hoá học');
  s.addText('Dùng đúng dữ liệu đã tiền xử lý trong repo PINN4SOH để kết quả đối chiếu được. Mỗi hàng là một chu kỳ; 16 đặc trưng thống kê của đoạn cuối CC và pha CV; nhãn là dung lượng xả.', {
    x: M, y: TOP, w: 12.3, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 13, color: INK2, lineSpacingMultiple: 1.14,
  });
  simpleTable(s, M, TOP + 0.58, [1.25, 1.0, 1.55, 3.15, 1.85, 3.5], [
    ['Bộ', 'Cell', 'Chu kỳ', 'Hoá học · danh định', 'Nhiệt độ', 'Điều kiện'],
    ['XJTU', '55', '22 212', 'NCM · 2.0 Ah', '25 °C', '6 nhóm: 2C, 3C, R2.5, R3, RW, vệ tinh'],
    ['TJU', '130', '56 779', 'NCA / NCM / NCM+NCA · 3.5 / 3.5 / 2.5 Ah', '25 / 35 / 45 °C', 'sạc 0.25–1C, xả 1–4C'],
    ['MIT', '125', '81 865', 'LFP (A123) · 1.1 Ah', '30 °C', '3 đợt, sạc nhanh đa dạng'],
    ['HUST', '77', '143 172', 'LFP (A123) · 1.1 Ah', '30 °C', '77 profile xả nhiều bậc'],
  ], { size: 12, rowH: 0.46 });

  boxed(s, M, 4.42, 6.0, 1.98, 'Bước lọc dữ liệu',
    'Bỏ hàng có giá trị vô hạn hoặc thiếu, và hàng có đặc trưng lệch quá 3σ so với chính cell đó — chỉ xét đặc trưng, không xét nhãn.');
  boxed(s, 6.83, 4.42, 6.0, 1.98, 'SOH không bắt đầu từ 1',
    'XJTU bắt đầu ở 0.92–0.99 còn HUST ở 1.06–1.12 do danh định thấp hơn thực tế. Vì thế ràng buộc "cell mới ≈ 1" phải nới thành khoảng [0.85, 1.10].');

  cite(s, [{ text: 'Wang et al. (2024) XJTU · Zhu et al. (2022) TJU · Severson et al. (2019) MIT · Ma et al. (2022) HUST.' }], 6.62);
  s.addNotes('Bốn bộ này là toàn bộ dữ liệu đơn pin công khai mà PINN4SOH dùng, nên kết quả đối chiếu được trực tiếp. Điểm cần nói: SOH không bắt đầu từ 1 nên mọi ràng buộc điều kiện đầu phải mềm, không cứng.');
}

// ═══════════════════════════════════ 3 · phát hiện định hình thiết kế
{
  const s = slide('Một phát hiện định hình thiết kế');
  s.addText('Trước khi đưa bất kỳ "tiên nghiệm dấu" nào vào loss, đo tương quan Spearman trong từng cell rồi lấy trung vị:', {
    x: M, y: TOP, w: 12.3, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 13, color: INK2,
  });
  simpleTable(s, M, TOP + 0.4, [4.4, 1.85, 1.85, 1.85, 1.85], [
    ['Đặc trưng', 'XJTU (NCM)', 'TJU (NCA/NCM)', 'MIT (LFP)', 'HUST (LFP)'],
    ['CC Q — điện lượng pha CC', '−0.18', '+0.95', '+0.98', '+0.97'],
    ['CV Q — điện lượng pha CV', '−0.94', '−1.00', '+0.41', '+0.55'],
    ['CV charge time', '−0.94', '−1.00', '+0.62', '+0.64'],
    ['voltage mean', '+0.60', '+0.04', '−0.99', '−0.94'],
    ['current kurtosis', '−0.65', '−0.97', '+0.77', '+0.87'],
  ], { size: 12.5, rowH: 0.4, aligns: ['left', 'center', 'center', 'center', 'center'] });

  let y = 4.34;
  y = TOP + 0.4 + 6 * 0.42 + 0.3;
  bullet(s, M, y, 12.3, 'Cùng một đặc trưng đổi dấu giữa hai họ hoá học.',
    'CV Q: −0.94 / −1.00 ở NCM-NCA nhưng +0.41 / +0.55 ở LFP — thậm chí lệch giữa hai bộ cùng NCM (CC Q: −0.18 ở XJTU, +0.95 ở TJU, do XJTU có protocol ngẫu nhiên).');
  chipLine(s, M, y + 0.86, 12.3, 'Dẫn tới',
    'Loại BỎ mọi ràng buộc dấu trên đặc trưng vì không phổ quát. Vật lý dùng được phải nằm ở QUỸ ĐẠO SOH — đơn điệu, tốc độ không âm, Arrhenius — chứ không ở đặc trưng.',
    { fill: ORANGE_LT, edge: ORANGE, ink: ORANGE });
  cite(s, [{ text: 'Tương quan Spearman tính trong từng cell rồi lấy trung vị trên toàn bộ cell của mỗi bộ — 387 cell, 304 028 chu kỳ sau lọc.' }]);
  s.addNotes(
    'Đây là phát hiện quyết định toàn bộ thiết kế. Nếu không đo bảng này mà cứ thế đặt ràng buộc dấu lên đặc trưng thì mô hình sẽ sai hệ thống trên một nửa số bộ dữ liệu.\n\n' +
    'CÁCH TÍNH: với MỖI cell, lấy tương quan Spearman giữa đặc trưng và SOH qua các chu kỳ của chính cell đó, ra một số; rồi lấy TRUNG VỊ trên các cell của bộ. Không gộp cell lại, vì gộp sẽ trộn biến thiên giữa-cell (cell A có CV Q nhỏ hơn cell B vì sai khác sản xuất) với biến thiên trong-cell, là thứ duy nhất liên quan tới lão hoá.\n\n' +
    'VÌ SAO SPEARMAN: câu hỏi ở đây là về HƯỚNG ĐƠN ĐIỆU chứ không phải độ dốc tuyến tính, vì ràng buộc dấu trong loss là một phát biểu đơn điệu. Suy giảm dung lượng cong mạnh ở pha đầu gối, nên quan hệ có thể đơn điệu hoàn hảo mà vẫn phi tuyến — Pearson sẽ báo yếu, tức âm tính giả. Bằng chứng: CV Q trên MIT, Pearson trung vị +0.04 còn Spearman +0.41, hai kết luận trái ngược. Spearman còn bất biến với mọi phép đổi thang đơn điệu, quan trọng vì 16 đặc trưng có đơn vị rất khác nhau và so chéo bốn bộ. Em cũng tính Kendall tau: dấu và kết luận giống hệt ở cả 24 ô, nên kết luận không phải sản phẩm của việc chọn Spearman.\n\n' +
    'ĐIỂM MẠNH HƠN BẢNG ĐANG CÓ, nói nếu còn giờ: trung vị che mất phân bố. CV Q trên XJTU rất chặt, khoảng tứ phân vị −0.96 tới −0.90, không một cell nào dương. Nhưng trên MIT thì trải từ −0.85 tới +0.99, chỉ 59 phần trăm cell dương; HUST 60 phần trăm. Tức trên LFP, quan hệ này không nhất quán ngay giữa các cell trong cùng một bộ — đặt ràng buộc dấu sẽ sai trên khoảng 40 phần trăm số cell của MIT.\n\n' +
    'DIỄN GIẢI VẬT LÝ, nói rõ đây là suy luận chứ không phải số đo: cell già thì nội trở tăng, chạm ngưỡng áp sớm hơn, pha CC nạp được ít hơn nên phần còn lại dồn sang CV — CV Q tăng khi SOH giảm, ra dấu âm, đúng với XJTU và TJU. Với LFP thì đường áp rất phẳng nên pha CC chạy gần hết dung lượng, đồng thời dung lượng tổng suy giảm cũng kéo CV Q xuống; hai hiệu ứng ngược nhau nên kết quả bất định.'
  );
}

// ═════════════════════════════════════════════ 4 · khoảng trống nghiên cứu
{
  const s = slide('Khoảng trống nghiên cứu — đọc từ mã nguồn PINN4SOH');
  let y = TOP;
  const gaps = [
    ['Phần dư PDE có thể tự triệt tiêu.', 'Mạng động học nhận chính u_t làm đầu vào, nên có thể học F ≈ u_t và đưa phần dư về 0 mà không cần chút vật lý nào.  (Model.py:232–234)'],
    ['"Physics loss" phụ thuộc nhãn.', 'Phạt theo hướng thay đổi của NHÃN, nên không áp được lên cell chưa đo dung lượng — về cấu trúc, nó không thể là nguồn thông tin bù nhãn.  (Model.py:255)'],
    ['Chuẩn hoá theo từng cell trên cả vòng đời.', 'Chỉ số chu kỳ sau chuẩn hoá chính là "phần trăm tuổi thọ đã đi qua" — rò rỉ tương lai vào đầu vào, và không tính được online.  (dataloader.py:50, 58–64)'],
    ['Validation chia theo hàng, không theo cell.', 'Early-stopping chấm điểm trên đúng những cell mô hình đã học.  (dataloader.py:157–158)'],
  ];
  gaps.forEach(([a, b]) => { bullet(s, M, y, 9.3, a, b); y += 0.92; });

  boxed(s, 10.1, TOP, 2.72, 2.15, 'Đo được', null);
  s.addText('−15 %', {
    x: 10.26, y: TOP + 0.52, w: 2.4, h: 0.72, isTextBox: true, margin: 0, align: 'center',
    fontFace: F, fontSize: 34, bold: true, color: ORANGE,
  });
  s.addText('MAE trên XJTU giảm giả tạo từ 0.0092 xuống 0.0078 chỉ do rò rỉ chỉ số chu kỳ.', {
    x: 10.26, y: TOP + 1.26, w: 2.4, h: 0.8, isTextBox: true, margin: 0, align: 'center',
    fontFace: F, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.12,
  });

  chipLine(s, M, 5.24, 12.3, 'Khoảng trống',
    'Không điểm nào phủ nhận kết quả của PINN4SOH. Chúng chỉ nói rằng CƠ CHẾ vì sao PINN4SOH cần ít dữ liệu chưa được chứng minh, và con số có thể đã được nâng bởi rò rỉ chuẩn hoá.',
    { fill: ORANGE_LT, edge: ORANGE, ink: ORANGE });
  cite(s, [{ text: 'Đọc bản clone repo wang-fujin/PINN4SOH ngày 05/09/2026 · Wang et al. (2024), ' },
           { text: 'Nature Communications', options: { italic: true } }, { text: ' 15, 4332.' }]);
  s.addNotes(
    'Phần khảo sát của mục tiêu 2, nhưng làm bằng cách đọc mã nguồn chứ không chỉ đọc abstract. Nếu Thầy hỏi vì sao không dùng thẳng PINN4SOH thì câu trả lời nằm ở khoảng trống số 2.\n\n' +
    'KHOẢNG TRỐNG 1: mạng động học nhận chính u_t làm một trong các đầu vào, rồi phần dư lấy u_t trừ đi đầu ra của nó. Nghiệm tầm thường là mạng học F bằng đúng u_t, cho phần dư bằng 0 với MỌI u. Tức ràng buộc tự thoả, loss về 0 mà không ép u phải thoả vật lý nào.\n\n' +
    'KHOẢNG TRỐNG 2: loss3 = relu((u2−u1)·(y1−y2)). Hướng phạt do NHÃN quyết định, nên không có nhãn thì không tính được biểu thức. Về cấu trúc nó không thể là nguồn thông tin bù cho nhãn thiếu. Chi tiết phụ nếu bị hỏi sâu: đây là .sum() trong khi loss1 và loss2 lấy trung bình, nên trọng số hiệu dụng của nó co giãn theo batch size.\n\n' +
    'KHOẢNG TRỐNG 3 — con số −15 phần trăm: cùng mạng MLP đầy đủ trên XJTU, chỉ đổi cách xử lý chỉ số chu kỳ, MAE đi từ 0.0092 với chuẩn hoá nhân quả xuống 0.0078 với chuẩn hoá theo cell. Slide sau chứng minh vì sao đó là rò rỉ chứ không phải mô hình tốt lên.\n\n' +
    'KHOẢNG TRỐNG 4: train_test_split cắt ngẫu nhiên theo HÀNG trên tensor đã trộn phẳng từ mọi cell huấn luyện, nên chu kỳ 100 của một cell vào train còn chu kỳ 101 của chính cell đó vào validation. Early-stopping vì thế chấm điểm trên cell đã học. Cần nói rõ là tập TEST của họ thì làm đúng, lấy từ danh sách cell riêng — vấn đề chỉ ở validation.'
  );
}

// ══════════════════════════ 4b · rò rỉ chuẩn hoá: chứng minh bằng ba bước
{
  const s = slide('Rò rỉ chuẩn hoá: chứng minh bằng ba bước');

  // ── bước A: đẳng thức đại số
  s.addText('BƯỚC A — chỉ số chu kỳ sau chuẩn hoá CHÍNH LÀ phần trăm tuổi thọ đã đi qua', {
    x: M, y: TOP - 0.1, w: 12.3, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 12.5, bold: true, color: NAVY,
  });
  s.addShape(pres.ShapeType.rect, { x: M, y: TOP + 0.2, w: 5.9, h: 0.92, fill: { color: 'F2F2F2' }, line: { color: LINE, width: 0.75 } });
  s.addText("df.insert(…, 'cycle index', np.arange(len(df)))   # dòng 50\nf_df = 2*(f_df-f_df.min())/(f_df.max()-f_df.min())-1  # dòng 60", {
    x: M + 0.14, y: TOP + 0.28, w: 5.62, h: 0.76, isTextBox: true, margin: 0,
    fontFace: 'Courier New', fontSize: 9.5, color: INK, lineSpacingMultiple: 1.2,
  });
  s.addText('min và max lấy trên CHÍNH FILE CELL ĐÓ, tức trên cả vòng đời. Thay vào công thức:', {
    x: 6.62, y: TOP + 0.18, w: 6.2, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 11.5, color: INK2,
  });
  s.addText('2(N − N_min) / (N_max − N_min) − 1  =  2 × (% tuổi thọ đã đi) − 1', {
    x: 6.62, y: TOP + 0.48, w: 6.2, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 12.5, bold: true, color: NAVY,
  });
  s.addText('Kiểm trên dữ liệu thật: sai lệch lớn nhất giữa hai vế trên cả 257 cell = 0.00e+00. Không phải xấp xỉ — là CÙNG MỘT ĐẠI LƯỢNG.', {
    x: 6.62, y: TOP + 0.78, w: 6.2, h: 0.44, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 11, color: GREY, lineSpacingMultiple: 1.12,
  });

  // ── bước B: bỏ hết đặc trưng, chỉ giữ con số đó
  s.addText('BƯỚC B — bỏ hết 16 đặc trưng, chỉ giữ MỘT con số đó', {
    x: M, y: 2.42, w: 6.6, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 12.5, bold: true, color: NAVY,
  });
  simpleTable(s, M, 2.76, [0.92, 1.52, 1.52, 2.2], [
    ['Bộ', 'chỉ số rò rỉ', 'chỉ số N/1000', 'MLP đủ 16 đặc trưng'],
    ['XJTU', '0.0097', '0.0253', '0.0092'],
    ['MIT', '0.0091', '0.0163', '0.0076'],
    ['HUST', '0.0089', '0.0236', '0.0167'],
  ], { size: 12, headSize: 10.5, rowH: 0.36, aligns: ['left', 'right', 'right', 'right'] });
  s.addText([
    { text: 'XJTU: ', options: { bold: true, color: INK } },
    { text: 'một con số duy nhất đạt 0.0097, cả mạng 16 đặc trưng đạt 0.0092 — gần như bằng nhau.  ', options: { color: INK2 } },
    { text: 'HUST: ', options: { bold: true, color: INK } },
    { text: 'con số rò rỉ một mình (0.0089) ', options: { color: INK2 } },
    { text: 'tốt hơn cả mạng đầy đủ không rò rỉ', options: { bold: true, color: ORANGE } },
    { text: ' (0.0167).', options: { color: INK2 } },
  ], { x: M, y: 4.34, w: 6.6, h: 0.64, isTextBox: true, margin: 0, fontFace: F, fontSize: 11, lineSpacingMultiple: 1.14 });

  // ── bước C: chạy thật thì mất thứ đó
  s.addText('BƯỚC C — chạy online thì KHÔNG biết N_max', {
    x: 7.28, y: 2.42, w: 5.54, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 12.5, bold: true, color: NAVY,
  });
  simpleTable(s, 7.28, 2.76, [0.92, 1.44, 1.44, 1.3], [
    ['Bộ', 'offline', 'online', 'xấu đi'],
    ['XJTU', '0.0097', '0.0385', '4.0×'],
    ['MIT', '0.0091', '0.0301', '3.3×'],
    ['HUST', '0.0089', '0.0268', '3.0×'],
  ], { size: 12, headSize: 10.5, rowH: 0.36, aligns: ['left', 'right', 'right', 'right'] });
  s.addText([
    { text: 'Vì sao mạnh đến vậy: ', options: { bold: true, color: INK } },
    { text: 'tuổi thọ lệch nhau rất xa — XJTU 116–912, MIT 34–2 116 chu kỳ. "Chu kỳ 300" nghĩa khác nhau ở mỗi cell, còn "đã đi 50 % vòng đời" thì ứng với cùng một mức sức khoẻ.', options: { color: INK2 } },
  ], { x: 7.28, y: 4.34, w: 5.54, h: 0.64, isTextBox: true, margin: 0, fontFace: F, fontSize: 11, lineSpacingMultiple: 1.14 });

  // ── cách sửa + vì sao là 1000
  boxed(s, M, 5.02, 12.3, 1.50, 'Cách đồ án sửa — và vì sao hằng số là 1000', null);
  s.addText([
    { text: 'Chia cho HẰNG SỐ TOÀN CỤC: t = N / 1000. Tại chu kỳ 300 luôn bằng 0.3, không cần biết gì về tương lai.  ', options: { color: INK2 } },
    { text: '1000 không phải siêu tham số được tinh chỉnh mà là ĐƠN VỊ ĐO thời gian: ', options: { bold: true, color: INK } },
    { text: 'chọn sao cho cả đầu vào t lẫn đầu ra r = −du/dt đều nằm quanh 1. Với bốn bộ (34–2 672 chu kỳ) thì t ∈ [0, 2.7] và r ≈ 0.1–0.3 SOH mỗi 1 000 chu kỳ — λ và Eₐ vì thế đọc được như tham số vật lý.\n', options: { color: INK2 } },
    { text: 'Quét S ∈ {1, 100, 1 000, 10 000}: ', options: { bold: true, color: INK } },
    { text: 'trong vùng 100–10 000 kết quả nằm trong sai số giữa các seed (XJTU 0.0113–0.0124, HUST 0.0150–0.0178). Chỉ khi S = 1 thì HUST hỏng: 0.0531, độ lệch chuẩn 0.0669 vì một seed phân kỳ.', options: { color: INK2 } },
  ], { x: M + 0.2, y: 5.52, w: 11.9, h: 0.94, isTextBox: true, margin: 0, fontFace: F, fontSize: 11, lineSpacingMultiple: 1.14 });

  cite(s, [{ text: 'Kiểm chứng chạy lại được bằng leak_demo.py (bước A–C) và scale_sweep.py (quét hằng số chia). PINN4SOH: dataloader/dataloader.py:50, 58–64.' }], 6.62);
  s.addNotes(
    'Đây là slide bằng chứng mạnh nhất của phần khảo sát — nếu Thầy hỏi "sao dám nói kết quả của họ bị nâng" thì trả lời bằng slide này.\n\n' +
    'BƯỚC A: mấu chốt nằm ở chỗ min và max được lấy trên chính file cell đó, tức trên cả vòng đời. Rút gọn công thức min-max ra thì nó đúng bằng hai lần phần trăm tuổi thọ đã đi qua, trừ một. Em kiểm trên toàn bộ 257 cell, sai lệch bằng 0 tuyệt đối chứ không phải xấp xỉ. Chỉ số chu kỳ thô thì vô hại, nhưng sau phép chia này nó biến thành câu "cell đã đi hết 63 % vòng đời" — mà biết được điều đó thì gần như đã biết SOH.\n\n' +
    'BƯỚC B: để chứng minh đáp án nằm sẵn trong đầu vào, em bỏ hết 16 đặc trưng sạc, chỉ giữ đúng một con số đó, rồi khớp bằng hồi quy đẳng hướng — không mạng nơ-ron. Trên XJTU được 0.0097 trong khi cả mạng 16 đặc trưng được 0.0092. Trên HUST còn mạnh hơn: một con số rò rỉ đạt 0.0089, tốt hơn mạng đầy đủ không rò rỉ vốn chỉ đạt 0.0167. Tức điểm số đang đo độ rò rỉ chứ không đo năng lực mô hình.\n\n' +
    'BƯỚC C: lúc triển khai thật, cell đang ở chu kỳ 300 và chưa hỏng, nên N_max chưa tồn tại — muốn biết thì phải chạy cell tới hỏng rồi quay ngược thời gian. Thay N_max bằng tuổi thọ trung bình của tập train thì sai số bung ra ba đến bốn lần. Đó là khoảng cách giữa con số báo cáo và con số chạy thật.\n\n' +
    'VÌ SAO 1000: nếu Thầy hỏi "sao không phải 500 hay 2000" thì nhấn rằng điều kiện bắt buộc duy nhất là hằng số phải TOÀN CỤC, giống nhau cho mọi cell — đó mới là thứ chặn rò rỉ. Giá trị cụ thể chỉ là chọn đơn vị đo, ảnh hưởng tới điều kiện số học chứ không tới tính đúng đắn. Em có chạy quét để kiểm: S = 100 cho XJTU 0.0124 / HUST 0.0150, S = 1 000 cho 0.0115 / 0.0156, S = 10 000 cho 0.0113 / 0.0178 — chênh lệch nằm trong độ lệch chuẩn giữa các seed, nên không có giá trị nào tối ưu rõ rệt. Chỉ khi S = 1, tức giữ nguyên chỉ số chu kỳ thô, thì HUST hỏng: 0.0531 với độ lệch chuẩn 0.0669 vì một seed phân kỳ — đúng như dự đoán từ lý do điều kiện số học, vì lúc đó t chạy tới 2 672 và r phải nhỏ cỡ 1e-4, nằm sâu trong đuôi trái của softplus. Em chọn 1 000 vì nó nằm giữa vùng an toàn và cho r một đơn vị đọc được: SOH mất đi trên mỗi 1 000 chu kỳ.'
  );
}

// ═══════════════════════════════════════════════════════ 5 · kiến trúc
{
  const s = slide('Kiến trúc: mạng nghiệm + động học xám');
  const dy = TOP, bh = 0.92;
  const box = (x, w, t, sub, fill, edge, txt) => {
    s.addShape(pres.ShapeType.rect, { x, y: dy, w, h: bh, fill: { color: fill }, line: { color: edge, width: 1 } });
    s.addText(t, { x: x + 0.1, y: dy + 0.14, w: w - 0.2, h: 0.3, isTextBox: true, margin: 0, align: 'center', fontFace: F, fontSize: 12.5, bold: true, color: txt });
    s.addText(sub, { x: x + 0.1, y: dy + 0.46, w: w - 0.2, h: 0.36, isTextBox: true, margin: 0, align: 'center', fontFace: F, fontSize: 10.5, color: txt === WHITE ? 'C9D9E8' : GREY });
  };
  const arw = (x, w) => s.addShape(pres.ShapeType.line, { x, y: dy + bh / 2, w, h: 0, line: { color: NAVY, width: 1.5, endArrowType: 'triangle' } });

  box(M, 2.62, 'x (16 đặc trưng) + N/1000', 'một chu kỳ sạc CC–CV', WHITE, LINE, INK);
  arw(M + 2.62, 0.4);
  box(M + 3.04, 2.86, 'Mạng nghiệm  u = Fᵩ(x, t)', 'MLP 17→64→64→32→1 · SiLU', BLUE_LT, BLUE, INK);
  arw(M + 5.90, 0.4);
  box(M + 6.32, 2.78, 'Động học xám  r(x, u, T)', 'r ≥ 0 theo cấu trúc', ORANGE_LT, ORANGE, INK);
  arw(M + 9.10, 0.4);
  box(M + 9.50, 2.78, 'SOH của chu kỳ đó', 'suy luận: CHỈ mạng nghiệm', NAVY, NAVY, WHITE);

  s.addText('r  =  softplus(MLP_θ(x))  ·  exp( λ(1 − u) )  ·  exp( −Eₐ/R · (1/T − 1/T_ref) )', {
    x: M, y: dy + bh + 0.26, w: 12.3, h: 0.36, isTextBox: true, margin: 0, align: 'center',
    fontFace: F, fontSize: 15.5, bold: true, color: NAVY,
  });

  const cw = 3.94, gap = 0.24;
  const cards = [
    ['softplus(·) ≥ 0', 'Tốc độ mất dung lượng không âm, nên nghiệm của phương trình đơn điệu không tăng THEO CẤU TRÚC — không cần nhãn để dạy.'],
    ['exp(λ(1 − u)),  λ ≥ 0 học được', 'Tốc độ tăng dần khi SOH giảm, mô tả pha "đầu gối" khi LAM lấn át LLI. λ dùng chung cả bộ nên đọc được như tham số vật lý.'],
    ['Arrhenius,  Eₐ ≥ 0 học được', 'Chỉ có tác dụng ở bộ nhiều nhiệt độ (TJU 25/35/45 °C). Bộ đơn nhiệt độ thì thừa số bằng 1 và Eₐ giữ nguyên khởi tạo.'],
  ];
  cards.forEach(([h, d], i) => boxed(s, M + i * (cw + gap), 3.24, cw, 2.0, h, d));

  chipLine(s, M, 5.5, 12.3, 'Khác PINN4SOH',
    'Mạng động học KHÔNG nhận đạo hàm của u làm đầu vào, nên phần dư không thể tự triệt tiêu. Chỉ số chu kỳ chia cho hằng số toàn cục 1000, không bao giờ chuẩn hoá theo cell.');
  cite(s, [{ text: 'Raissi, Perdikaris & Karniadakis (2019), ' }, { text: 'Journal of Computational Physics', options: { italic: true } },
           { text: ' 378 · Sơ đồ đầy đủ đúng như mã nguồn chạy: hình fig_pipeline.png.' }]);
  s.addNotes('Hai mạng nhỏ, tổng khoảng 8 000 tham số, đủ nhẹ để chạy trên vi điều khiển sau lượng tử hoá. Ba thừa số của r mang ba mẩu vật lý tách rời nhau nên ablation được từng cái.');
}

// ══════════════════════════════════════════ 6 · các hàm mất mát vật lý
{
  const s = slide('Các hàm mất mát vật lý — chỉ một hàm cần nhãn');
  const eqs = [
    ['L_data', 'mean ( u(N) − y_N )²', 'CẦN nhãn — chỉ tính trên cell có nhãn', ORANGE_LT, ORANGE],
    ['L_ode', 'mean [ u(N+h) − u(N) + r(x_N, u(N), T) · h/1000 ]²', 'không cần nhãn — dạng Euler', BLUE_LT, BLUE],
    ['L_mono', 'mean ReLU( u(N+h) − u(N) − ε ),  ε = 0.002', 'không cần nhãn — đơn điệu có dung sai', BLUE_LT, BLUE],
    ['L_range', 'mean [ ReLU(u − 1.15) + ReLU(0.4 − u) ] + điều kiện chu kỳ đầu ∈ [0.85, 1.10]', 'không cần nhãn — miền hợp lý', BLUE_LT, BLUE],
  ];
  let y = TOP;
  eqs.forEach(([n, f, note, fill, edge]) => {
    s.addShape(pres.ShapeType.rect, { x: M, y, w: 1.34, h: 0.62, fill: { color: fill }, line: { color: edge, width: 1 } });
    s.addText(n, { x: M, y, w: 1.34, h: 0.62, isTextBox: true, margin: 0, align: 'center', valign: 'middle', fontFace: F, fontSize: 13, bold: true, color: INK });
    s.addText(f, { x: M + 1.5, y: y + 0.02, w: 7.4, h: 0.36, isTextBox: true, margin: 0, valign: 'middle', fontFace: F, fontSize: 12.5, color: INK });
    s.addText(note, { x: M + 1.5, y: y + 0.33, w: 7.4, h: 0.28, isTextBox: true, margin: 0, fontFace: F, fontSize: 11, color: edge === ORANGE ? ORANGE : GREY });
    y += 0.72;
  });
  s.addText('L  =  L_data  +  w(s) · ( α L_ode + β L_mono + γ L_range ),      w(s) = min(1, s / 0.2 S)', {
    x: M, y: y + 0.06, w: 8.9, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 13, bold: true, color: NAVY,
  });
  s.addText('α = 2, β = 5, γ = 1 chọn trên validation của XJTU rồi đóng băng; w(s) tăng tuyến tính trong 20 % số bước đầu.', {
    x: M, y: y + 0.42, w: 8.9, h: 0.3, isTextBox: true, margin: 0, fontFace: F, fontSize: 11.5, color: GREY,
  });

  boxed(s, 9.28, TOP, 3.54, 3.4, 'Vì sao chọn dạng Euler',
    'Phần dư dạng đạo hàm chia cho Δt = h/1000. Với h = 1 chu kỳ, mẫu số 0.001 khuếch đại nhiễu 1000 lần: dao động 10⁻³ của mạng thành phần dư cỡ 1, lớn hơn hẳn tín hiệu thật |du/dÑ| ≈ 0.2–0.5. Mạng "học" cách giảm phần dư bằng cách trở nên vô cảm với đặc trưng — MAE xấu gấp đôi MLP thuần.');

  chipLine(s, M, 5.5, 12.3, 'Dẫn tới',
    'Ba loss không nhãn chạy được trên MỌI cell huấn luyện, kể cả cell chỉ có đường sạc mà chưa đo dung lượng. Đây chính là cơ chế "vật lý thay nhãn" mà PINN4SOH về cấu trúc không thể có.');
  cite(s, [{ text: 'Cặp chu kỳ (N, N+h) lấy trong cùng một cell, h ngẫu nhiên trong {5, …, 50}. Ba biến thể phần dư euler / fd / autograd đều giữ lại trong code để ablation.' }]);
  s.addNotes('Bài học quan trọng nhất ở slide này là về CÁCH VIẾT phần dư chứ không phải về vật lý. Hộp bên phải là chỗ mất nhiều thời gian nhất trong hai tuần.');
}

// ═══════════════════════════════════ 7 · đóng góp: kế thừa và đề xuất
{
  const s = slide('Đóng góp nghiên cứu: Kế thừa và Đề xuất');
  boxed(s, M, TOP, 5.9, 3.5, 'Kế thừa nền tảng', [
    'Bộ 16 đặc trưng đường sạc CC–CV và bốn bộ dữ liệu đơn pin công khai của PINN4SOH.',
    'Khung physics-informed neural network (Raissi et al., 2019).',
    'Ý tưởng ràng buộc đơn điệu của quỹ đạo suy giảm dung lượng.',
    'Mô hình tốc độ suy giảm phụ thuộc nhiệt độ theo Arrhenius.',
    'Baseline MLP cùng kích thước để so sánh ngang hàng.',
  ]);
  boxed(s, 6.93, TOP, 5.9, 3.5, 'Đóng góp khoa học cốt lõi', [
    'Động học xám có cấu trúc thay cho mạng động học đen: r ≥ 0 theo cấu trúc nên đơn điệu không phụ thuộc tham số học.',
    'Toàn bộ loss vật lý KHÔNG dùng nhãn, nên áp được lên cell chưa đo dung lượng — biến vật lý thành nhãn thay thế.',
    'Pipeline không rò rỉ: chia theo cell, chuẩn hoá nhân quả, và phần kiểm toán định lượng mức rò rỉ của cách làm cũ.',
    'Phép so sánh cân bằng ngân sách tinh chỉnh cho cả hai mô hình — hiếm gặp trong các bài PINN cho pin.',
  ]);
  chipLine(s, M, 4.94, 12.3, 'Phát biểu đúng mức',
    'Không tuyên bố "PINN luôn thắng". Kết luận trung thực là: ít nhãn thì vật lý thắng, nhãn đủ thì hoà — và phần thắng phải đo sau khi baseline được tinh chỉnh ngang.');
  cite(s, [{ text: 'Raissi et al. (2019) · Wang et al. (2024) · Ziyin, Hartwig & Ueda (2020), ' },
           { text: 'NeurIPS', options: { italic: true } }, { text: ' 33 (hàm kích hoạt tuần hoàn).' }]);
  s.addNotes('Slide này để Thầy thấy rõ ranh giới giữa cái mượn và cái mới. Bốn gạch đầu dòng bên phải là thứ sẽ bảo vệ khi phản biện.');
}

// ══════════════════════════════════════════════════════ 8 · phạm vi đồ án
{
  const s = slide('Phạm vi đồ án');
  boxed(s, M, TOP, 5.9, 2.9, 'Giai đoạn 1 — đang thực hiện',
    'Thiết kế kiến trúc và ba loss vật lý không nhãn; dựng pipeline thống nhất bốn bộ dữ liệu; chạy đủ bộ thí nghiệm kiểm chứng giả thuyết "ít nhãn"; kiểm toán rò rỉ của cách làm cũ; cân bằng ngân sách tinh chỉnh giữa baseline và PINN. Toàn bộ chạy trên CPU, 3 seed.');
  boxed(s, 6.93, TOP, 5.9, 2.9, 'Giai đoạn 2 — tiếp sau',
    'Chuyển sang GPU và tăng số seed để có kết luận thống kê; mở rộng bài toán thích nghi miền sang dây chuyền mới; sửa đầu ra để đơn điệu dọc quỹ đạo thật; dựng bản suy luận online trên một cell, không cần biết tuổi thọ tương lai — đúng điều kiện BMS chạy thật.');

  const st = [['4', 'bộ dữ liệu'], ['387', 'cell đơn'], ['1 227', 'lượt huấn luyện'], ['10', 'protocol thí nghiệm']];
  st.forEach(([b, l], i) => {
    const x = M + i * 3.14;
    s.addText(b, { x, y: 4.40, w: 2.9, h: 0.56, isTextBox: true, margin: 0, fontFace: F, fontSize: 28, bold: true, color: NAVY });
    s.addText(l, { x, y: 4.96, w: 2.9, h: 0.32, isTextBox: true, margin: 0, fontFace: F, fontSize: 12.5, color: GREY });
  });
  cite(s, [{ text: 'Ba mục tiêu bắt buộc của ĐACN đã hoàn tất; phần thí nghiệm / prototype (optional) đang được mở rộng.' }]);
  s.addNotes('Ba mục tiêu bắt buộc của ĐACN đã xong, phần optional đang chạy tiếp. Bốn con số ở dưới là khối lượng đã làm trong hai tuần.');
}

// ═══════════════════════════ 9 · đã làm (1/3) — protocol thí nghiệm
{
  const s = slide('Công việc đã làm hai tuần qua (1/3) — protocol thí nghiệm');
  const tEnd = simpleTable(s, M, TOP, [1.05, 8.0, 3.25], [
    ['Mã', 'Nội dung', 'Quy mô'],
    ['E1', 'Hiệu quả theo lượng nhãn: 4 mức nhãn × 3 mô hình, chỉ tiêu chính MAE(PINN-semi 30 %) / MAE(MLP 70 %)', '144 lượt'],
    ['E2 · E10', 'Chuyển miền zero-shot, và ma trận chuyển miền đầy đủ 4×4', '48 + 48 lượt'],
    ['E3', 'Ablation: bỏ từng loss, bỏ knee, bỏ Arrhenius, đổi dạng phần dư, thay bằng động học đen', '57 lượt'],
    ['E4', 'Kiểm toán rò rỉ chuẩn hoá: nhân quả so với cách chuẩn hoá theo cell của PINN4SOH', '30 lượt'],
    ['E5', 'Thích nghi miền bằng chính các loss không nhãn, k ∈ {0, 3} cell đích có nhãn', '48 lượt'],
    ['E7', 'Quét trọng số vật lý β ở MỌI mức nhãn, chọn theo validation rồi mới đọc test', '240 lượt'],
    ['E9', 'Thiết kế lại mạng: kiến trúc × hàm kích hoạt, chạy cho cả baseline lẫn PINN', '180 lượt'],
    ['E12', 'Ngân sách tinh chỉnh cân bằng: cùng 8 biến thể siêu tham số cho hai mô hình', '384 lượt'],
  ], { size: 12, rowH: 0.42, aligns: ['left', 'left', 'right'] });

  chipLine(s, M, tEnd + 0.26, 12.3, 'Nguyên tắc',
    'Chia theo CELL 70/15/15 có phân tầng, assert không cell nào nằm ở hai tập. Mọi mức nhãn giữ nguyên val/test của cùng seed. Trung bình ± độ lệch chuẩn trên 3 seed, seed đổi cả cách chia lẫn khởi tạo.');
  s.addText('Tổng cộng 1 227 lượt huấn luyện đã lưu lại, mọi số trong báo cáo sinh tự động từ results/*.csv — không có số nào gõ tay.', {
    x: M, y: tEnd + 0.94, w: 12.3, h: 0.3, isTextBox: true, margin: 0, fontFace: F, fontSize: 12.5, bold: true, color: NAVY,
  });
  cite(s, [{ text: 'AdamW, lr 2e-3 cosine, batch 512, tối đa 4 000 bước, early stopping theo MAE validation (kiên nhẫn 12 lần đánh giá).' }], 6.86);
  s.addNotes('Slide này trả lời câu "đã làm được gì" bằng khối lượng cụ thể. Nếu Thầy hỏi sâu vào protocol nào thì mở bảng chi tiết trong báo cáo.');
}

// ═══════════════════════════ 10 · đã làm (2/3) — hiệu quả theo lượng nhãn
{
  const s = slide('Công việc đã làm (2/3) — ít nhãn thì vật lý thắng');
  s.addImage({ path: IMG('fig_E1_label_efficiency.png'), x: 1.62, y: TOP - 0.06, w: 10.1, h: 3.69 });
  const st = [
    ['4 / 4', 'Ở mức 10 % nhãn, PINN thắng MLP trên cả bốn bộ. Tỉ số MAE(MLP)/MAE(PINN): XJTU 1.7× · TJU 1.1× · MIT 1.3× · HUST 1.9×', NAVY],
    ['15 / 16', 'Số ô PINN thắng MLP ở cùng mức nhãn, khi trọng số vật lý β được chọn theo validation thay vì cố định β = 5.', NAVY],
    ['2 / 4', 'Giả thuyết "30 % đuổi kịp 70 %": xác nhận trên MIT (0.91×) và HUST (0.96×); chưa trên XJTU (1.25×) và TJU (1.17×).', ORANGE],
  ];
  const sw = 3.94, gap = 0.24;
  st.forEach(([b, l, c], i) => {
    const x = M + i * (sw + gap);
    s.addShape(pres.ShapeType.rect, {
      x, y: 4.86, w: sw, h: 1.32, fill: { color: c === NAVY ? BLUE_LT : ORANGE_LT }, line: { color: c, width: 1 },
    });
    s.addText(b, { x: x + 0.16, y: 4.86, w: 1.16, h: 1.32, isTextBox: true, margin: 0, valign: 'middle', fontFace: F, fontSize: 24, bold: true, color: c });
    s.addText(l, { x: x + 1.38, y: 4.86, w: sw - 1.56, h: 1.32, isTextBox: true, margin: 0, valign: 'middle', fontFace: F, fontSize: 10.5, color: INK2, lineSpacingMultiple: 1.12 });
  });
  cite(s, [{ text: 'E1: 4 bộ × 4 mức nhãn × 3 mô hình × 3 seed = 144 lượt. Val/test giữ nguyên khi giảm nhãn, nên các cột so sánh được trực tiếp với nhau.' }], 6.34);
  s.addNotes('Chỉ vào panel MIT và HUST trước, đó là chỗ câu chuyện rõ nhất. Nói thêm: vật lý là một prior, nó bù nhiều nhất ở nơi mô hình chỉ-dữ-liệu đói nhãn nhất; ở TJU thì 30 % nhãn đã đủ cho MLP nên ràng buộc trở thành thiên kiến.');
}

// ═══════════════════════════ 11 · đã làm (3/3) — hai phép kiểm chứng
{
  const s = slide('Công việc đã làm (3/3) — hai phép kiểm chứng khắt khe');
  s.addText('(a)   Cân bằng ngân sách tinh chỉnh — E12, 384 lượt', {
    x: M, y: TOP - 0.06, w: 7.3, h: 0.3, isTextBox: true, margin: 0, fontFace: F, fontSize: 13.5, bold: true, color: NAVY,
  });
  s.addImage({ path: IMG('fig_E12_fair.png'), x: M, y: TOP + 0.3, w: 7.3, h: 3.01 });
  s.addText([
    { text: 'Khi cho MLP hưởng ĐÚNG lưới siêu tham số mà PINN được hưởng, tỉ số PINN/MLP trung bình đi từ ', options: { color: INK2 } },
    { text: '0.72× lên 0.90×', options: { bold: true, color: ORANGE } },
    { text: ' — một phần lợi thế của PINN biến mất thật. PINN vẫn thắng ', options: { color: INK2 } },
    { text: '7/8 ô', options: { bold: true, color: NAVY } },
    { text: ', tỉ số 0.75–1.01×. Riêng MIT @ 30 %, baseline tự nó cải thiện 0.0103 → 0.0048 chỉ nhờ dropout 0.1.', options: { color: INK2 } },
  ], { x: M, y: 4.5, w: 7.3, h: 1.5, isTextBox: true, margin: 0, fontFace: F, fontSize: 12, lineSpacingMultiple: 1.16 });

  s.addText('(b)   Đem sang bộ dữ liệu khác — E5', {
    x: 8.22, y: TOP - 0.06, w: 4.6, h: 0.3, isTextBox: true, margin: 0, fontFace: F, fontSize: 13.5, bold: true, color: NAVY,
  });
  boxed(s, 8.22, TOP + 0.3, 4.6, 4.72, 'Thích nghi miền không nhãn', null);
  s.addText('Zero-shot thất bại với mọi mô hình — đặc trưng đường sạc phụ thuộc protocol nên đầu vào bộ đích nằm ngoài phân bố. Nhưng vì loss vật lý không cần nhãn, chúng chạy được ngay trên cell của bộ đích.', {
    x: 8.42, y: TOP + 0.88, w: 4.2, h: 1.1, isTextBox: true, margin: 0, fontFace: F, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.14,
  });
  const t5 = simpleTable(s, 8.42, TOP + 1.98, [1.5, 0.95, 0.95, 0.8], [
    ['Cặp chuyển', 'MLP', 'PINN', ''],
    ['XJTU → TJU', '2.944', '0.041', '72×'],
    ['HUST → MIT', '0.369', '0.027', '13×'],
    ['TJU → XJTU', '0.329', '0.103', '3.2×'],
    ['MIT → HUST', '0.062', '0.059', '1.05×'],
  ], { size: 11.5, headSize: 11, rowH: 0.34, aligns: ['left', 'right', 'right', 'right'] });
  s.addText('MAE trên cell test của bộ đích, không cell đích nào mang nhãn (k = 0).', {
    x: 8.42, y: t5 + 0.16, w: 4.2, h: 0.5, isTextBox: true, margin: 0, fontFace: F, fontSize: 10.5, color: GREY, lineSpacingMultiple: 1.1,
  });
  cite(s, [{ text: 'E12: 4 bộ × 8 biến thể × 2 mô hình × 2 mức nhãn × 3 seed. E5: nguồn 70 % nhãn, 2 500 bước cố định, không chọn mô hình theo nhãn đích.' }], 6.12);
  s.addNotes('Con số 0.72 lên 0.90 là chỗ nên chủ động nói ra trước khi bị hỏi — nó cho thấy mình tự kiểm chứng chứ không chọn số đẹp. Bên phải là chỗ vật lý trả lại giá trị thật khi đổi bộ dữ liệu.');
}

// ═════════════════════════════════════════════════ 12 · khó khăn gặp phải
{
  const s = slide('Khó khăn gặp phải');
  const rows = [
    ['Phần dư dạng đạo hàm khuếch đại nhiễu 1000×', 'Với h = 1, mẫu số Δt = 0.001 biến dao động 10⁻³ của mạng thành phần dư cỡ 1. MAE xấu gấp đôi MLP thuần.', 'Viết lại phần dư ở dạng Euler', 'Đã xử lý', GREEN],
    ['Dấu tương quan đặc trưng–SOH đổi theo hoá học', 'CV Q: −0.94 ở XJTU nhưng +0.41 ở MIT; hai bộ cùng NCM cũng lệch (CC Q: −0.18 so với +0.95).', 'Bỏ mọi ràng buộc dấu trên đặc trưng', 'Đã xử lý', GREEN],
    ['So sánh ban đầu không công bằng', 'Baseline MLP chưa được tinh chỉnh ngang với PINN nên lợi thế bị thổi phồng.', 'Chạy thêm E12; kết quả tự hạ 0.72× → 0.90×', 'Đã xử lý', GREEN],
    ['Zero-shot chuyển miền thất bại hoàn toàn', 'Sai 2–3 bậc độ lớn. Có thể không khả thi giữa hai họ hoá học với bộ đặc trưng này.', 'Mới khai thác được dạng thích nghi miền không nhãn', 'Chưa xong', ORANGE],
    ['Các chỉ số đánh giá mâu thuẫn nhau', 'TJU @ 30 %: MLP thắng theo MAE (0.0088 / 0.0099) nhưng PINN thắng theo cell tệ nhất, vi phạm đơn điệu (0.20 / 0.68) và sai số EOL (24.5 / 29.8 chu kỳ).', 'Chưa chốt được chỉ số chính để báo cáo', 'Cần ý kiến Thầy', ORANGE],
    ['Chạy trên CPU nên chỉ đủ 3 seed', 'Chưa đủ để kết luận thống kê, mới chỉ có trung bình và độ lệch chuẩn.', 'Chuyển sang GPU trong hai tuần tới', 'Kế hoạch', GREY],
  ];
  let y = TOP - 0.08;
  rows.forEach(([t, d, fix, st, c], i) => {
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.05, w: 0.135, h: 0.135, fill: { color: NAVY }, line: { color: BLUE, width: 1 } });
    s.addText(t, { x: M + 0.3, y, w: 5.28, h: 0.3, isTextBox: true, margin: 0, fontFace: F, fontSize: 12.5, bold: true, color: INK });
    s.addText(d, { x: M + 0.3, y: y + 0.3, w: 5.28, h: 0.56, isTextBox: true, margin: 0, fontFace: F, fontSize: 11, color: GREY, lineSpacingMultiple: 1.1 });
    s.addText([{ text: 'Hướng xử lý: ', options: { bold: true, color: c === GREY ? GREY : c } }, { text: fix, options: { color: INK2 } }], {
      x: 6.22, y: y + 0.02, w: 5.3, h: 0.66, isTextBox: true, margin: 0, valign: 'top', fontFace: F, fontSize: 11.5, lineSpacingMultiple: 1.1,
    });
    s.addShape(pres.ShapeType.rect, {
      x: 11.65, y: y + 0.02, w: 1.18, h: 0.3,
      fill: { color: c === GREEN ? GREEN_LT : (c === ORANGE ? ORANGE_LT : 'F2F2F2') }, line: { color: c, width: 1 },
    });
    s.addText(st, { x: 11.65, y: y + 0.02, w: 1.18, h: 0.3, isTextBox: true, margin: 0, align: 'center', valign: 'middle', fontFace: F, fontSize: 9.5, bold: true, color: c });
    y += 0.93;
  });
  s.addNotes('Nếu hết giờ thì chỉ nói ba dòng đầu — đó là ba khó khăn đã tự giải quyết được. Dòng "chỉ số mâu thuẫn" là chỗ nên chủ động xin ý kiến Thầy vì nó quyết định cách viết mục kết quả.');
}

// ═════════════════════════════════════ 13 · công việc hai tuần tiếp theo
{
  const s = slide('Công việc hai tuần tiếp theo');
  const weeks = [
    ['Tuần 1 — làm cho kết quả hiện có đủ tin cậy', [
      ['Chuyển toàn bộ sang GPU.', 'Tăng 3 → 10 seed; kiểm định thống kê ghép cặp theo cell để có khoảng tin cậy thay vì chỉ trung bình.'],
      ['Mở rộng lưới ngân sách công bằng.', 'E12 hiện chỉ cân bằng ở 30 % và 70 % — bổ sung mức 10 % và 50 %, thêm learning rate và số bước vào lưới.'],
      ['Chốt bộ chỉ số chính để báo cáo.', 'Chọn giữa MAE trung bình và sai số dự báo EOL, rồi viết lại mục kết quả theo chỉ số đã chốt.'],
    ]],
    ['Tuần 2 — mở rộng đóng góp và dựng prototype', [
      ['Quét số cell đích cần đo nhãn.', 'k = 0 / 1 / 3 / 5 / 10 cho bài toán thích nghi miền, để trả lời "dây chuyền mới cần đo bao nhiêu cell".'],
      ['Sửa đầu ra đơn điệu.', 'Đơn điệu dọc quỹ đạo thật, không chỉ đơn điệu theo biến t như hiện nay.'],
      ['Dựng demo prototype.', 'Suy luận online trên một cell, không cần biết tuổi thọ tương lai — đúng điều kiện BMS chạy thật.'],
    ]],
  ];
  weeks.forEach(([head, items], wi) => {
    const x = M + wi * 6.43;
    s.addShape(pres.ShapeType.rect, { x, y: TOP, w: 5.87, h: 0.42, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(head, { x: x + 0.16, y: TOP, w: 5.55, h: 0.42, isTextBox: true, margin: 0, valign: 'middle', fontFace: F, fontSize: 13.5, bold: true, color: WHITE });
    let y = TOP + 0.64;
    items.forEach(([a, b], i) => {
      s.addShape(pres.ShapeType.ellipse, { x, y, w: 0.32, h: 0.32, fill: { color: BLUE_LT }, line: { color: NAVY, width: 1 } });
      s.addText(String(wi * 3 + i + 1), { x, y, w: 0.32, h: 0.32, isTextBox: true, margin: 0, align: 'center', valign: 'middle', fontFace: F, fontSize: 12, bold: true, color: NAVY });
      s.addText(a, { x: x + 0.46, y: y - 0.02, w: 5.4, h: 0.3, isTextBox: true, margin: 0, fontFace: F, fontSize: 13, bold: true, color: INK });
      s.addText(b, { x: x + 0.46, y: y + 0.3, w: 5.4, h: 0.76, isTextBox: true, margin: 0, fontFace: F, fontSize: 11.5, color: INK2, lineSpacingMultiple: 1.14 });
      y += 1.24;
    });
  });
  chipLine(s, M, 5.62, 12.3, 'Cần hỗ trợ',
    'Xin Thầy cho ý kiến về chỉ số chính để báo cáo — MAE trung bình hay sai số dự báo EOL? Hai chỉ số cho hai kết luận ngược nhau trên TJU.',
    { fill: ORANGE_LT, edge: ORANGE, ink: ORANGE });
  s.addNotes('Kết bằng câu hỏi xin ý kiến để buổi họp có đầu ra chứ không chỉ là báo cáo một chiều.');
}

// ═════════════════════════════════════════════════════════ 14 · kết luận
{
  const s = slide('Kết luận');
  let y = TOP + 0.12;
  const pts = [
    ['Cơ chế, chứ không phải con số, là thứ cần chứng minh.', 'PINN4SOH báo cáo rằng ràng buộc vật lý giúp giảm nhu cầu dữ liệu, nhưng đọc mã nguồn cho thấy loss vật lý của họ phụ thuộc nhãn và pipeline có rò rỉ chuẩn hoá — nên cơ chế chưa được chứng minh.'],
    ['Vật lý chỉ thay được nhãn khi nó không cần nhãn.', 'Ba loss đề xuất tính trên cặp chu kỳ của cùng một cell mà không dùng y, nên áp được lên cả cell chưa đo dung lượng. Ở mức 10 % nhãn, cách này thắng baseline trên cả bốn bộ dữ liệu.'],
    ['Phần thắng phải đo sau khi baseline được tinh chỉnh ngang.', 'Cân bằng ngân sách tinh chỉnh kéo tỉ số PINN/MLP từ 0.72× lên 0.90×. Đây mới là con số nên dùng khi báo cáo, và PINN vẫn thắng 7/8 ô.'],
    ['Giá trị lớn nhất nằm ở chỗ đổi bộ dữ liệu.', 'Zero-shot thất bại với mọi mô hình, nhưng vì loss vật lý không cần nhãn nên chúng chạy được ngay trên cell của dây chuyền mới — XJTU → TJU cải thiện 72 lần so với baseline.'],
  ];
  pts.forEach(([a, b]) => {
    bullet(s, M, y, 12.3, a, '');
    s.addText(b, { x: M + 0.3, y: y + 0.3, w: 12.0, h: 0.72, isTextBox: true, margin: 0, fontFace: F, fontSize: 12.5, color: INK2, lineSpacingMultiple: 1.14 });
    y += 1.28;
  });
  cite(s, [{ text: 'Toàn bộ số liệu sinh tự động từ results/*.csv qua 1 227 lượt huấn luyện, 3 seed, chạy trên CPU. Báo cáo đầy đủ 19 trang kèm mã nguồn đã nộp cùng slide này.' }], 6.30);
  s.addNotes('Bốn ý này là thứ muốn Thầy nhớ lại nếu chỉ nhớ một slide. Ý thứ ba là ý thể hiện tính khoa học rõ nhất.');
}

// ══════════════════════════════════════════════ 15 · tài liệu tham khảo
{
  const s = slide('Tài liệu tham khảo');
  const refs = [
    ['Ma, G., Xu, S., Jiang, B., et al. (2022).', 'Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning.', 'Energy & Environmental Science, 15(10), 4083–4094.'],
    ['Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019).', 'Physics-informed neural networks.', 'Journal of Computational Physics, 378, 686–707.'],
    ['dos Reis, G., Strange, C., Yadav, M., Li, S. (2021).', 'Lithium-ion battery data and where to find it.', 'Energy and AI, 5, 100081.'],
    ['Severson, K. A., Attia, P. M., Jin, N., et al. (2019).', 'Data-driven prediction of battery cycle life before capacity degradation.', 'Nature Energy, 4(5), 383–391.'],
    ['Su, L., Xu, Y., Dong, Z. (2024).', 'State-of-health estimation of lithium-ion batteries: A comprehensive literature review from cell to pack levels.', 'Energy Conversion and Economics, 5(4), 224–242.'],
    ['Wang, F., Zhai, Z., Zhao, Z., Di, Y., Chen, X. (2024).', 'Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis.', 'Nature Communications, 15, 4332.'],
    ['Ziyin, L., Hartwig, T., Ueda, M. (2020).', 'Neural networks fail to learn periodic functions and how to fix it.', 'NeurIPS, 33, 1583–1594.'],
    ['Zhu, J., Wang, Y., Huang, Y., et al. (2022).', 'Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation.', 'Nature Communications, 13, 2261.'],
  ];
  let y = TOP - 0.18;
  refs.forEach(([who, what, where]) => {
    s.addText(who, { x: M, y, w: 12.3, h: 0.25, isTextBox: true, margin: 0, fontFace: F, fontSize: 11, color: BLUE });
    s.addText(what, { x: M, y: y + 0.23, w: 12.3, h: 0.25, isTextBox: true, margin: 0, fontFace: F, fontSize: 11, color: INK });
    s.addText(where, { x: M, y: y + 0.45, w: 12.3, h: 0.25, isTextBox: true, margin: 0, fontFace: F, fontSize: 10.5, italic: true, color: GREY });
    y += 0.74;
  });
  s.addNotes('Tám tài liệu, xếp theo họ tác giả. Bốn bộ dữ liệu tương ứng bốn bài: Wang 2024 XJTU, Zhu 2022 TJU, Severson 2019 MIT, Ma 2022 HUST.');
}

// ══════════════════════════════════════════════════════ 16 · cảm ơn
{
  const s = pres.addSlide();
  s.background = { path: A('bia-bg.png') };
  s.addImage({ path: A('vnu.png'), x: 5.20, y: 0.34, w: 1.42, h: 0.70 });
  s.addImage({ path: A('bk.png'), x: 6.92, y: 0.30, w: 0.68, h: 0.78 });
  s.addText('Xin cảm ơn Thầy và các bạn', {
    x: 1.0, y: 3.2, w: W - 2.0, h: 0.7, isTextBox: true, margin: 0, align: 'center',
    fontFace: F, fontSize: 27, bold: true, color: INK,
  });
  s.addImage({ path: A('footer.png'), x: 4.17, y: 6.62, w: 5.0, h: 0.44 });
  s.addNotes('Cảm ơn Thầy. Em sẵn sàng nhận câu hỏi. Nếu Thầy muốn xem chi tiết thì em có báo cáo 19 trang, sơ đồ pipeline đầy đủ và các hình dự phòng E7, E9, E10, E11.');
}

pres.writeFile({ fileName: path.join(__dirname, 'bao-cao-2-tuan-BK.pptx') })
  .then((f) => console.log('đã ghi:', f));
