# Gói Tái Lập Xác Nhận Độc Lập: `R2_matched_128` vs `A_ref_128` (`0 bps` & `10 bps`)

Thư mục này đóng gói hoàn chỉnh hồ sơ xuất bản và kiểm chứng độc lập cho **hai thí nghiệm xác nhận** so sánh chiến lược huấn luyện đa kịch bản theo kỳ vọng đều **`R2_matched_128` (`R2_DECAY`)** với nhánh đối chứng một kịch bản danh định **`A_ref_128` (`A_DECAY`)** trên bài toán phòng hộ quyền chọn mua ATM bán khống (`20` bước, `S0 = K = 1.0`, `T = 0.40`):

1. **Thí nghiệm không phí (`cost_s = 0.0`):** [robust_r2_matched_confirmatory/](../robust_r2_matched_confirmatory/) — **Slurm Job `16911`** (`24` seed `[4001..4024]`, `48` lượt huấn luyện, `03:50:53`).
2. **Thí nghiệm có chi phí giao dịch `10 bps` (`cost_s = 0.0010`):** [robust_r2_matched_cost_confirmatory/](../robust_r2_matched_cost_confirmatory/) — **Slurm Job `16914`** (`24` seed `[5001..5024]`, `48` lượt huấn luyện, `03:44:55`), đưa phí `10 bps` vào cả **huấn luyện, chọn bản lưu `best_val` và kiểm tra** đồng thời khắc phục triệt để lỗi chia sẻ nhiễu `Val(s)/Train(s+1)` của nghiên cứu B3 cũ từ lớp dẫn xuất khóa SHA-256.

> [!NOTE]
> Toàn bộ dữ liệu thô, bản lưu `.pkl`, thiết kế khóa, mã nguồn thực thi và bảng băm gốc trong hai thư mục thí nghiệm **được giữ nguyên văn từng byte**. Các hiệu đính phương pháp luận và ranh giới diễn giải trước xuất bản được lập riêng tại [PUBLICATION_ERRATA.md](PUBLICATION_ERRATA.md).

---

## 1. Câu Hỏi Nghiên Cứu & Kết Luận Chính

1. **Câu hỏi nghiên cứu:** Huấn luyện đa kịch bản `R2_matched_128` có giảm sai số phòng hộ ròng tại `σ = 0.30` (`H1a: ΔHE_030 < 0`), đồng thời không làm suy giảm `OCE` phòng hộ tại `σ = 0.20` vượt quá biên `ε = +0.00010` (`H1b: ΔOCE_020 < +0.00010`), so với `A_ref_128` có cùng ngân sách tối ưu (`31,200` bước, `3,993,600` lượt xử lý đường) ở mức `α = 0.025` một phía (Berger IUT)?
2. **Kết quả đã xác nhận trên `N = 24` hạt giống mỗi thí nghiệm (`PASS_CONFIRMATORY_IUT`):**
   - **Ở `0 bps` (`Job 16911`, `[4001..4024]`):** `ΔHE_030 = -0.004562` (`95% CI [-0.005108, -0.004016]`, `p = 5.68e-15`), `ΔOCE_020 = +0.0000311` (`95% CI [+0.0000148, +0.0000474]`, `p = 4.37e-09`), **`p_IUT = 4.37e-09`**.
   - **Ở `10 bps` (`Job 16914`, `[5001..5024]`):** `ΔHE_030 = -0.003547` (`95% CI [-0.004489, -0.002605]`, `p = 3.40e-08`), `ΔOCE_020 = +0.0000001` (`95% CI [-0.0000359, +0.0000361]`, `p = 3.75e-06`), **`p_IUT = 3.75e-06`** (nhạy bỏ ngoại lệ `5024` cho `p_IUT = 4.46e-10`).
   - **Cơ chế dưới `10 bps`:** Cả hai nhánh đều giao dịch `> 99.97%` bước (`immobility_rate < 1e-5` khoảng `0.01%–0.02%`); `R2_DECAY` giảm `turnover` và `mean_cost` nhờ **giảm quy mô mỗi lệnh hoạt động** (`-1.82%` đến `-3.06%`) và hạ đỉnh vị thế (`max_abs_delta`), **không phải** cơ chế bất động hoàn toàn.
   - **Giới hạn & đánh đổi:** Tại vùng biến động thấp `fixed_015`, `R2_DECAY` kém hơn `A_DECAY` trên `HE` và `CVaR95` ở cả hai mức phí. Kết luận chỉ có giá trị trong bộ giả lập `20` bước hiện tại với hai mức phí `0` và `10 bps`. Xem tổng hợp đầy đủ tại [R2_MATCHED_FINAL_SYNTHESIS.md](R2_MATCHED_FINAL_SYNTHESIS.md).

---

## 2. Bản Đồ Hồ Sơ Đóng Gói

- [r2_matched_confirmatory_package/](.) — Hồ sơ tổng hợp, đính chính, phụ thuộc, môi trường và kiểm tra:
  - [R2_MATCHED_FINAL_SYNTHESIS.md](R2_MATCHED_FINAL_SYNTHESIS.md) — Báo cáo tổng hợp kết quả hai thí nghiệm `0 bps` & `10 bps`.
  - [PUBLICATION_ERRATA.md](PUBLICATION_ERRATA.md) — Bảng kiểm 6 hạng mục CSV/JSON và 6 quy tắc đính chính diễn giải (`a`–`f`).
  - [DEPENDENCIES.md](DEPENDENCIES.md) & [`deps_snapshot/`](deps_snapshot/) — Ảnh chụp nguyên trạng 17 mô-đun nội bộ phụ thuộc (`core/*` khớp `origin/main` `a32e663`, và 12 tệp mở rộng `robust_world.py`, `r3_*`, `bc_*`, `diag_*`, `lr_*`), nguồn, giấy phép GPLv3 thượng nguồn `deephedging-jax`.
  - [ENVIRONMENT.md](ENVIRONMENT.md) & [environment_server_freeze.txt](environment_server_freeze.txt) — Phần cứng Slurm `dgx02`, phiên bản Python/JAX/NumPy và hướng dẫn cài đặt.
  - [MANIFEST.json](MANIFEST.json), [FILE_INDEX.md](FILE_INDEX.md), [SHA256SUMS.txt](SHA256SUMS.txt) — Danh mục và mã băm SHA-256 của toàn bộ tệp đóng gói.
  - [verify_r2_matched_package.py](verify_r2_matched_package.py) — Kịch bản kiểm tra tự động 3 cấp độ.
- [robust_r2_matched_confirmatory/](../robust_r2_matched_confirmatory/) — Toàn bộ mã, thiết kế khóa, CSV/JSON chưa làm tròn, biểu đồ, nhật ký Slurm `16911`, hồ sơ phản biện `@gng5` và toàn bộ `96` bản lưu (`48 best_val.pkl` + `48 final_with_opt.pkl`) cho phí `0.0`.
- [robust_r2_matched_cost_confirmatory/](../robust_r2_matched_cost_confirmatory/) — Toàn bộ mã, thiết kế khóa, kiểm toán kế toán phí & độc lập seed, CSV/JSON chưa làm tròn, biểu đồ, nhật ký Slurm `16914`, hồ sơ phản biện `@gng5` và toàn bộ `96` bản lưu (`48 best_val.pkl` + `48 final_with_opt.pkl`) cho phí `10 bps`.

---

## 3. Cách Chạy Kiểm Tra & Ba Cấp Độ Tái Lập

```bash
# Kiểm tra Mức 1 (SHA-256, schema CSV/JSON, thống kê Berger IUT) và Mức 2 (tái nạp bản lưu nếu có JAX):
python r2_matched_confirmatory_package/verify_r2_matched_package.py --level auto
```

| Cấp độ | Phạm vi | Tình trạng kiểm chứng thực tế |
| :--- | :--- | :--- |
| **Mức 1** | Tái kiểm mã băm `SHA256SUMS.txt`, cấu trúc dữ liệu & tái tính thống kê từ CSV gốc | **PASS** *(chạy trực tiếp bằng `verify_r2_matched_package.py --level auto`)* |
| **Mức 2** | Tái nạp `best_val.pkl` (`96` tệp) & `final_with_opt.pkl` (`96` tệp) bằng JAX và đối chiếu số liệu | **PASS** *(chạy xác minh trực tiếp trên `Windows JAX 0.12.0.1`)* |
| **Mức 3** | Huấn luyện lại từ đầu toàn bộ `96` lượt (`3,993,600` đường/lượt) | **CHƯA CHẠY LẠI KHI ĐÓNG GÓI** *(đã hoàn tất gốc ở Jobs `16911` & `16914`)* |
