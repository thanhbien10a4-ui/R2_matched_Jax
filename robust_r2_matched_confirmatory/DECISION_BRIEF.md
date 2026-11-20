# Tóm Tắt Quyết Định Trước Khi Chạy Xác Nhận (`DECISION_BRIEF.md`)

> [!IMPORTANT]
> **Trạng thái hiện tại:** Đã hoàn tất đóng hồ sơ đính chính, chuẩn bị thiết kế, tính cỡ mẫu, mã nguồn, kiểm thử đơn vị và phản biện `@gng5`. **CHƯA chạy huấn luyện xác nhận, CHƯA mô phỏng/mở dữ liệu kiểm tra `[4001..4040]`, CHƯA mở `HOLD_EVAL`, CHƯA nộp Slurm (`0` job xác nhận đang chạy), và CHƯA đẩy GitHub.**

---

## 1. Câu Hỏi Cần Trả Lời Ở Pha Xác Nhận

> *"So với `A_ref_128`, dưới cùng ngân sách (`31,200` bước, `3,993,600` lượt đường) và cùng quy trình ổn định (`Cosine LR Decay`, `Adam eps=1e-7`, `cost_s=0.0`), `R2_matched_128` (huấn luyện kỳ vọng đều 4 kịch bản) có **giảm sai số phòng hộ `HE = np.std(gains, ddof=0)` tại `σ = 0.30`** (`H1a: μ_A < 0`), đồng thời **không làm tăng `hedged_oce_y_nom` tại `σ = 0.20` vượt quá biên `ε = +0.00010`** (`H1b: μ_B < +0.00010`) trên dữ liệu độc lập chưa từng thấy hay không?"*

---

## 2. Những Gì Đã Đóng Gói Sẵn Sàng

1. **Đính chính hồ sơ phát triển ([DEVELOPMENT_ERRATA.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DEVELOPMENT_ERRATA.md)):** Sửa rõ 5 điểm `1a–1e` mà không ghi đè tệp gốc (`LR_OSCILLATION_181_200.csv` 24 dòng với `0.000107–0.000269` chỉ thuộc `DECAY`; `+0.000236`/`+0.000426` ở `fixed_030` là `OCE` còn `HE` là `+0.000993`/`+0.000826`; bản chất `delete20_jk_se_osc` trên 20 nhóm đường của `oce_vals_by_ep`; `PAUSE_COORDINATOR` thực tế tồn tại `32` bytes tại `.../experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR` với dấu `Nov 16 10:17`; tách biệt heuristic `0.5ε`).
2. **Thiết kế khóa xác nhận & Cỡ mẫu ([R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md), [SAMPLE_SIZE_SENSITIVITY.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/SAMPLE_SIZE_SENSITIVITY.csv)):**
   - Tổng hợp tách lớp 4 giai đoạn phát triển (`BC`, `DIAG`, `LR_STAB` — dưới `DECAY best_val`, `ΔHE_030 = -0.004812 ± 0.000466`, `ΔOCE_020 = -0.000002 ± 0.000047`, `n=3`).
   - Kiểm định đồng thời theo **Berger IUT ở `α = 0.025` một phía** (`FWER <= 0.025`, khớp 1–1 cận trên KTC 95% hai phía), nâng tập kiểm tra `conf_test` lên **`20,000` đường/kịch bản** (`6` kịch bản) và giữ nguyên `val` `5,000` đường để bảo toàn cơ chế `best_val` kèm đối chiếu `ep200`.
3. **Cách ly PRNG & An toàn dữ liệu ([DATA_AND_SEED_PLAN.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DATA_AND_SEED_PLAN.json), [conf_keys.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_keys.py)):** `2,167` khóa lịch sử vs `861` khóa mới (`[4001..4040]` + smoke `3999`, `21` khóa/seed): **`0` va chạm, `0` lỗi tràn**, `HOLD_EVAL` vẫn niêm phong.
4. **Mã thực thi, phân tích & mẫu Slurm ([conf_design.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_design.py), [conf_stats.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_stats.py), [conf_run.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py), [slurm_confirmatory_template.sbatch](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/slurm_confirmatory_template.sbatch)):** Đã tích hợp đủ 6 bản vá từ phản biện `@gng5` (`GNG5_DESIGN_REVIEW.md`, `AAS 93.5/100`), vượt qua toàn bộ kiểm thử đơn vị và chế độ `dry_run`.

---

## 3. Lựa Chọn Đề Xuất Cho Người Dùng Chốt Trước Khi Nộp Slurm

| Phương án | Dải Seed | Thời gian CPU (`8 CPU, 0 GPU`) | Lực chung `R1–R5` | Lực `R6` (stress biên) | Lực `R7` (phương sai nặng `sd_B=1.1e-4`) | Lực `R8` (cực đoan) | Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`N = 24` (Khuyến nghị chính)** | `[4001..4024]` (`48` runs) | **`~3.86h`** (`--time=05:30:00`) | **`>= 99.9%`** | **`97.3%`** | **`92.7%`** | `62.0%` | Đủ lực >90% ngay cả ở `R7`; khuyến nghị khóa cứng từ đầu. |
| `N = 16` (Ngân sách thấp) | `[4001..4016]` (`32` runs) | `~2.57h` (`--time=03:30:00`) | `>= 97.8%` | `86.9%` | `77.3%` (`< 80%`) | `36.8%` | Chỉ dùng nếu bị giới hạn cứng thời gian job; **cấm** nhìn kết quả rồi tăng lên `24`. |

**Bước tiếp theo chờ người dùng phê duyệt:** Chốt duy nhất **`N = 24`** (hoặc `N = 16`) và biên `cost_s = 0.0` trong bản khóa thiết kế cuối cùng trước khi cho phép nộp [slurm_confirmatory_template.sbatch](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/slurm_confirmatory_template.sbatch).
