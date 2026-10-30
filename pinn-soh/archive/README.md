# Lưu trữ

Sản phẩm đã nộp của đồ án và các script giai đoạn đầu. Không còn được `run.py` gọi tới; giữ nguyên như lúc nộp.

| Thư mục | Nội dung |
|---|---|
| `thesis/do-an/` | báo cáo đồ án (`bao-cao-do-an.pdf`, 75 trang), mã LaTeX `main.tex` và hình `hinh/` |
| `thesis/make_thesis.py`, `thesis/thesis_bk.py` | script dựng báo cáo từ `results/` (khuôn đồ án Bách Khoa) |
| `thesis/de-cuong-E31.md` | đề cương E31 (so với SambaMixer trên bộ NASA PCoE) |
| `slides/` | slide thuyết trình và báo cáo tiến độ (`.pptx`, `.pdf`), script dựng slide, kịch bản thuyết trình |
| `early-report/` | báo cáo dự đoán SOH bản đầu (Markdown, PDF) |
| `legacy/` | pipeline v1: thí nghiệm E1–E12, báo cáo HTML/LaTeX cũ, các script kiểm toán (xem `legacy/README.md`) |

Báo cáo đồ án có phần bộ NASA PCoE (E24, E31 và một phần E26, E29, E30, E32). Mã của phần này đã được bỏ khỏi
thư viện để mã nguồn chỉ còn bốn bộ dữ liệu chính, nên các script dựng báo cáo và slide ở đây không chạy được với mã
hiện tại. Để dựng lại đúng báo cáo, slide hay chạy lại pipeline v1, dùng commit `2d66f75` (lúc nộp báo cáo):

```bash
git worktree add ../pinn-soh-do-an 2d66f75
cd ../pinn-soh-do-an/pinn-soh
python run.py thesis              # sau khi có runs/ và results/ (xem README ở commit đó)
```
