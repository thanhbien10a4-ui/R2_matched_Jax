# Báo Cáo Tổng Hợp Hai Thí Nghiệm Xác Nhận `R2_matched_128` vs `A_ref_128` (`R2_MATCHED_FINAL_SYNTHESIS.md`)

Tài liệu tổng hợp kết quả và ranh giới khoa học của hai thí nghiệm xác nhận độc lập đối xứng hoàn toàn, so sánh chiến lược huấn luyện đa kịch bản kỳ vọng đều **`R2_matched_128` (`R2_DECAY`)** trên 4 kịch bản `("fixed_015", "fixed_020", "fixed_025", "rs_015_025")` với nhánh đối chứng danh định **`A_ref_128` (`A_DECAY`)** trên `fixed_020` dưới cùng ngân sách (`31,200` bước cập nhật, `3,993,600` lượt xử lý đường/lượt, `Cosine LR Decay`, `6,762` tham số policy khởi tạo giống hệt từng bit).

---

## 1. Bảng Tổng Kết Song Song Job `16911` (`0 bps`) và Job `16914` (`10 bps`)

> [!NOTE]
> Theo [PUBLICATION_ERRATA.md](PUBLICATION_ERRATA.md), hai thí nghiệm sử dụng **hai tập hạt giống độc lập khác nhau** (`[4001..4024]` cho `0 bps` và `[5001..5024]` cho `10 bps`). Bảng dưới đây trình bày **đối chiếu mô tả** song song giữa hai thí nghiệm, không khẳng định toàn bộ khác biệt giữa hai cột là tác động nhân quả thuần túy của chi phí giao dịch.

| Chỉ tiêu (`best_val`, `N = 24`) | Job `16911` — Phí `cost_s = 0.0` (`[4001..4024]`) | Job `16914` — Phí `cost_s = 0.0010` (`10 bps`, `[5001..5024]`) |
| :--- | :--- | :--- |
| **Thời gian chạy Slurm (`8 CPU / 0 GPU`)** | `COMPLETED 0:0`, `03:50:53`, `MaxRSS = 1,220,600K` | `COMPLETED 0:0`, `03:44:55`, `MaxRSS = 851,780K` |
| **Hoàn tất / Tái nạp bản lưu (`48/48`)** | `24/24` seed sạch; `max reload diff = 7.45e-09` | `24/24` seed sạch; `max reload diff = 0.0` |
| **Điều kiện chính A (`H1a`: `ΔHE_030 < 0`)** | Mean **`-0.004562 ± 0.001293`** (`-11.7%`), `95% CI [-0.005108, -0.004016]`, `t = -17.283`, **`p = 5.68e-15`**, `24/24 < 0` | Mean **`-0.003547 ± 0.002231`** (`-10.4%`), `95% CI [-0.004489, -0.002605]`, `t = -7.787`, **`p = 3.40e-08`**, `23/24 < 0` (seed `5004` `+0.000368`) |
| **Điều kiện chính B (`H1b`: `ΔOCE_020 < +0.00010`)** | Mean **`+0.0000311 ± 0.0000385`** (`+0.31 bp`), `95% CI [+0.0000148, +0.0000474]`, `t = -8.760`, **`p = 4.37e-09`**, `22/24 < +ε` (`4020, 4023` vượt `+ε`) | Mean **`+0.0000001 ± 0.0000852`** (`+0.001 bp`), `95% CI [-0.0000359, +0.0000361]`, `t = -5.745`, **`p = 3.75e-06`**, `24/24 < +ε` |
| **Phán quyết Berger IUT (`α = 0.025`)** | **`PASS_CONFIRMATORY_IUT`** (`p_IUT = 4.37e-09`) | **`PASS_CONFIRMATORY_IUT`** (`p_IUT = 3.75e-06`) |
| **Độ nhạy `ep200` (không chọn `best_val`)** | `ΔHE_030 = -0.004901` (`p = 3.05e-14`); `ΔOCE_020 = +0.0000377` (`p = 7.31e-07`, `21/24 < +ε`) -> **PASS** | `ΔHE_030 = -0.003877` (`p = 5.16e-09`); `ΔOCE_020 = -0.0000071` (`p = 7.61e-07`, `24/24 < +ε`) -> **PASS** |
| **Độ nhạy loại ngoại lệ (hậu nghiệm)** | `0` LOO flips; tối đa LOO vẫn `< +ε` | Bỏ `5024` (`n=23`): `p_IUT = 4.46e-10`; bỏ `5004 & 5024`: `p_IUT = 5.68e-10` |
| **Đánh đổi bất lợi (`fixed_015`)** | `ΔHE_015 = +0.000695` (`23/24 > 0`); `ΔOCE_015 = +0.000102`; `ΔCVaR95 = +0.001097` | `ΔHE_015 = +0.000438` (`22/24 > 0`); `ΔOCE_015 = +0.000034`; `ΔCVaR95 = +0.000723` |

---

## 2. Cơ Chế Thực Sự Dưới Chi Phí Giao Dịch `10 bps`

1. **Không phải cơ chế bất động (no-trade band):** Do kiến trúc MLP `softplus` là trơn, ở `10 bps` cả `A_DECAY` lẫn `R2_DECAY` vẫn giao dịch gần như ở toàn bộ `20` bước (`19.996–19.998 / 20`, `immobility_rate |a| < 1e-5` chỉ `0.01%–0.02%`).
2. **Giảm biên độ lệnh (`trade_size_active`) và kiềm chế đỉnh vị thế (`max_abs_delta`):** Tiếp xúc với các kịch bản biến động cao (`fixed_025`, `rs_015_025`) khiến `R2_DECAY` co nhỏ kích thước mỗi lệnh (`-1.82%` đến `-3.06%`) và hạ đỉnh vị thế (`max_abs_delta` giảm `-0.106` đến `-0.272`, ở `23/24` seed), từ đó hạ `turnover` và tiết kiệm chi phí giao dịch (`ΔCost = -1.97e-05` đến `-5.42e-05`, dù vẫn có `5–8/24` hạt giống cá biệt `R2_DECAY` tốn phí hơn `A_DECAY`).
3. **Cân bằng tại `σ = 0.20`:** Khoản tiết kiệm chi phí `-3.01e-05` (`-0.30 bp`) tại `fixed_020` giúp bù đắp gần như toàn bộ phần bất lợi nhỏ từng xuất hiện ở `0 bps`, kéo `ΔOCE_020` ròng trung bình về `+0.001 bp`.

---

## 3. Ba Cấp Độ Tái Lập & Trạng Thái Kiểm Chứng Hiện Tại

| Cấp độ tái lập | Nội dung kiểm chứng | Trạng thái xác nhận trong gói |
| :--- | :--- | :--- |
| **Mức 1 — Tái tính bảng & thống kê từ dữ liệu thô** | Kiểm tra SHA-256 toàn bộ tệp gói & bản gốc, schema CSV/JSON, toàn vẹn `24/24` seed, tái tính Berger IUT (`best_val`, `ep200`, độ nhạy ngoại lệ) và cân bằng kế toán phí (`gross - cost = net`) | **ĐÃ CHẠY & ĐẠT (`PASS`)** via [verify_r2_matched_package.py](verify_r2_matched_package.py) |
| **Mức 2 — Tái nạp 192 bản lưu & đánh giá lại JAX** | Tái nạp `best_val.pkl` (`48 + 48 = 96` tệp) và kiểm tra `final_with_opt.pkl` (`ep200`, `48 + 48 = 96` tệp) trên dữ liệu tái sinh từ khóa `conf_keys.py` / `cost_keys.py`, so sánh từng cặp số với CSV/JSON đã xuất | **ĐÃ CHẠY & ĐẠT (`PASS`)** *(kiểm tra đóng gói trên Windows cục bộ `JAX 0.12.0.1`; bản lưu gốc tạo và hậu kiểm tại cụm HUST `user04` `JAX 0.4.34`)* |
| **Mức 3 — Huấn luyện lại từ đầu (`96` lượt × `200` epoch)** | Thực thi lại đầy đủ `slurm_confirmatory_n24.sbatch` (`Job 16911`, `~3.85h`) và `slurm_cost_confirmatory_n24.sbatch` (`Job 16914`, `~3.75h`) trên CPU `8` luồng | **KHÔNG CHẠY LẠI KHI ĐÓNG GÓI** *(chỉ chạy thực tế ở Slurm Jobs `16911` & `16914`)* |

---

## 4. Phạm Vi Giới Hạn Kết Luận

1. Các kết luận trên chỉ có hiệu lực cho cặp phương pháp `R2_matched_128` (`R2_DECAY`, trung bình đều 4 kịch bản) vs `A_ref_128` (`A_DECAY`) trong bài toán phòng hộ quyền chọn mua châu Âu ATM (`S0 = K = 1.0`, `T = 0.40`, `20` bước, không lãi suất/drift) ở hai mức phí tuyến tính `cost_s = 0.0` và `cost_s = 0.0010` (`10 bps`).
2. Không ngoại suy từ thí nghiệm này sang mô hình biến động ngẫu nhiên Heston (được tách riêng sang hướng phát triển mới `B1`/`B2`), kiến trúc `NTBN`, hay mức phí lớn hơn (`25–50 bps`).
3. Không dùng thiết kế sạch `R2_COST10_CONF_V1` để hồi tố hợp thức hóa kết quả của thí nghiệm `B3` (`b3_cost_confirmatory_10seeds`).
