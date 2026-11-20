# Hồ Sơ Kiểm Toán Xác Nhận Độc Lập (`CONFIRMATORY_AUDIT.md`)

- **Báo cáo chính:** [CONFIRMATORY_FINAL_REPORT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/CONFIRMATORY_FINAL_REPORT.md)
- **Thiết kế khóa:** [R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md)
- **Bản ghi phản hồi thô của `@gng5` (hậu nghiệm):** [logs/gng5_postrun_audit_text.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/gng5_postrun_audit_text.md)
- **Bảng mã băm toàn bộ hồ sơ sau chạy:** [POSTRUN_SHA256SUMS.txt](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/POSTRUN_SHA256SUMS.txt)

> [!IMPORTANT]
> **PHÁN QUYẾT KIỂM TOÁN HẬU NGHIỆM (`@gng5`):** **`PASS` (`SOVEREIGN_PASSED` / `PASS_CONFIRMATORY_IUT` — AAS `98.5 / 100`, `5/5` cổng T1–T5 Đạt)**.
>
> **Công bố trung thực phương thức vận hành `@gng5`:** Cả phiên tiền kiểm thiết kế (`conversation_id = 4d45219f-1066-4a7f-aa51-d687e49e6010`, `416.50s`) lẫn phiên kiểm toán hậu nghiệm (`conversation_id = 52ee7f5f-526d-4754-9c31-627c022718da`, `297.59s`) đều thực thi bằng `/home/user04/.local/bin/agy --model gemini-4-argon-high` ở chế độ **CHỈ ĐỌC (`READ-ONLY`)** dưới dạng **một phiên mô hình đơn lẻ (`single-conversation session`)**, **KHÔNG phải hệ nhiều tác nhân chạy song song**.

---

## 1. Xác Minh Toàn Bộ Bất Biến Từ Nhật Ký & Dữ Liệu Thô

| Hạng mục kiểm toán | Kết quả xác minh trực tiếp | Bằng chứng |
| :--- | :--- | :--- |
| **Thực thi Slurm Job `16911`** | `COMPLETED`, ExitCode `0:0`, node `dgx02`, `8` CPUs, `0` GPU, Elapsed `03:50:53` (`13,836.36s`), `MaxRSS = 1,220,600K` | `sacct -j 16911`, [slurm_conf_n24_16911.log](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/slurm_conf_n24_16911.log) |
| **Bất biến SHA-256 trước/trong/sau chạy** | Khớp **17/17 tệp** thiết kế khóa, cấu hình, mã thực thi/thống kê và mô-đun thượng nguồn | [DESIGN_LOCK_SHA256.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DESIGN_LOCK_SHA256.json), [CONFIRMATORY_STATS_SUMMARY.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/CONFIRMATORY_STATS_SUMMARY.json) |
| **Hoàn tất 48 lượt & Reload Parity** | `24/24` seed `[4001..4024]` `COMPLETED`, `failed_seeds = []`; `48/48` `best_val.pkl` có `val_nom_reload_diff <= 7.451e-09` (`36/48` bằng `0.0`); `96/96` `.pkl` (`> 27 KB`) & `48/48` manifest | [SEED_LEDGER.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/SEED_LEDGER.json), [POSTRUN_AUDIT_TABLES.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/POSTRUN_AUDIT_TABLES.json) |
| **Tái tính độc lập IUT (`α = 0.025`)** | Khớp từng bit giữa `CONFIRMATORY_STATS_SUMMARY.json` và tái tính từ 288 dòng `PAIRED_CONFIRMATORY.csv`; `best_val` đạt `p_IUT = 4.37e-09`, `ep200` đạt `p_IUT = 7.31e-07`, `0` LOO flips | [STATS_RECOMPUTED_CLI.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/STATS_RECOMPUTED_CLI.json) |
| **Cô lập dữ liệu & Bảo toàn hạ tầng** | `PAUSE_COORDINATOR` nguyên vẹn tại `.../experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR` (`32 bytes`, `Nov 16 10:17`); `[4025..4040]`, `HOLD_EVAL` và `R2/R3` cũ giữ kín nguyên vẹn; chưa đẩy GitHub | Kiểm tra máy chủ & `@gng5` |

---

## 2. Xử Lý 4 Ghi Chú Bổ Sung Sau Kiểm Toán `@gng5`

Mặc dù không có bất kỳ sai sót số học hay vi phạm khóa thiết kế nào, tất cả 4 điểm bổ sung làm rõ của `@gng5` đã được cập nhật ngay vào [CONFIRMATORY_FINAL_REPORT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/CONFIRMATORY_FINAL_REPORT.md):
1. Tách bạch rõ phần đánh đổi tại `fixed_015`: đảo dấu ở `OCE` là do `y_nom` vs `y_scen`, còn mức tăng `+0.000695` ở `HE` (`23/24` seed) và `+0.001097` ở `CVaR95` (`23/24` seed) là bất biến với `y` và phản ánh đánh đổi thực sự của policy đa kịch bản tại biến động thấp.
2. Ghi rõ tỷ lệ theo hạt giống của `turnover` (`23/24` seed giảm trên cả 6 kịch bản ở `best_val`, ngoại lệ `4010`; và `23/24` ở `ep200`, ngoại lệ `4023`) và `max_abs_delta` (`24/24` ở 5 kịch bản đầu, `23/24` ở `fixed_030` trừ `4005`).
3. Bổ sung KTC `95%` đầy đủ vào cột `ΔOCE y_scen` trong Bảng 4 của báo cáo chính.
4. Đối chiếu trực tiếp bộ tham số thực đo `(mean_A = -0.004562, sd_A = 0.001293, mean_B = +0.000031, sd_B = 0.000039, rho_AB = -0.2012)` với bảng công suất thiết kế `R1–R8`.
