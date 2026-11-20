# Đính Chính Hồ Sơ Phát Triển (`DEVELOPMENT_ERRATA.md`)

> [!IMPORTANT]
> Tài liệu này lập bản đính chính độc lập cho chuỗi thí nghiệm phát triển trước pha xác nhận (`robust_r3_dev/`, `robust_r3_budget_control/`, `robust_r3_selection_diagnostics/`, `robust_r3_lr_stability/` và `LR_VERIFICATION_PACKAGE.zip`). **Không một tệp báo cáo, CSV, JSON, mã nguồn hay nhật ký gốc nào được chỉnh sửa hoặc ghi đè.**

---

## Mục (a): `LR_OSCILLATION_181_200.csv` — 24 dòng và phạm vi của khoảng `0.000107–0.000269`

- **Bằng chứng trực tiếp:** [LR_OSCILLATION_181_200.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/results/run/LR_OSCILLATION_181_200.csv) gồm đúng **24 dòng dữ liệu** (+ 1 dòng header = 25 dòng tệp): `3 seed (3401, 3402, 3403) × 4 cấu hình (A_FIXED, A_DECAY, R2_FIXED, R2_DECAY) × 2 bộ chẩn đoán (diag1_20k_020, diag2_20k_020)`.
- **Đính chính diễn đạt (liên quan [LR_STABILITY_AUDIT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/LR_STABILITY_AUDIT.md#L39)):**
  - Cụm *"Toàn bộ `12/12` run có `below_05eps_threshold = False` (`0.000107..0.000269`)"* dễ gây hiểu nhầm rằng mọi cấu hình `FIXED` lẫn `DECAY` đều nằm trong `0.000107–0.000269`.
  - **Số liệu chính xác trên cả `24/24` dòng (`12` cặp `(seed, config)` trên `2` bộ `diag`):**
    - **`DECAY` (`12` dòng: `A_DECAY` & `R2_DECAY`):** `osc_oce_181_200` nằm trong **`0.00010673 – 0.00026933`** (làm tròn `0.000107–0.000269`), cụ thể `A_DECAY`: `0.00010673–0.00018394` và `R2_DECAY`: `0.00013286–0.00026933`.
    - **`FIXED` (`12` dòng: `A_FIXED` & `R2_FIXED`):** `osc_oce_181_200` nằm trong **`0.00057222 – 0.00156031`** (làm tròn `0.000572–0.001560`), cụ thể `A_FIXED`: `0.00075640–0.00156031` và `R2_FIXED`: `0.00057222–0.00084475`.
    - Mệnh đề `"12/12 run có below_05eps_threshold = False"` vẫn đúng vì toàn bộ `24/24` dòng đều `> 0.00005` (`0.5ε`), nhưng **khoảng `0.000107–0.000269` chỉ thuộc nhóm `DECAY`**, không phải toàn bộ `FIXED` và `DECAY`.

---

## Mục (b): Làm rõ tên chỉ số cho `+0.000236` (nhánh `A`) và `+0.000426` (nhánh `R2`) tại `fixed_030`

- **Bằng chứng trực tiếp:** [LR_DEV_EVAL_FULL_PRECISION.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/results/run/LR_DEV_EVAL_FULL_PRECISION.csv) (`best_val`, `scenario = fixed_030`, 3 seed `3401, 3402, 3403`).
- **Đính chính tên chỉ số:** Các mức chênh lệch trung bình 3 seed dưới `DECAY` so với `FIXED` tại `fixed_030`:
  - Nhánh `A_ref_128`: `(0.0001394649 + 0.0002347345 + 0.0003341256) / 3 = +0.000236108` (`+0.000236`) là **chỉ số `hedged_oce_y_nom` (OCE phòng hộ tại `y_nom`)** — **KHÔNG phải** `he_net_ddof0` (sai số phòng hộ `HE`).
  - Nhánh `R2_matched_128`: `(+0.0005267726 + 0.0008230092 - 0.0000718513) / 3 = +0.000425977` (`+0.000426`) cũng là **chỉ số `hedged_oce_y_nom` (OCE phòng hộ tại `y_nom`)**.
- **Số liệu song song của `he_net_ddof0` (`HE = np.std(gains, ddof=0)`) tại `fixed_030` (`best_val`, `DECAY - FIXED`):**
  - Nhánh `A_ref_128`: `seed 3401 = +0.000465877`, `3402 = +0.001152856`, `3403 = +0.001358915` -> trung bình **`+0.000992549` (`+0.000993`)**.
  - Nhánh `R2_matched_128`: `seed 3401 = +0.000852858`, `3402 = +0.000849948`, `3403 = +0.000774084` -> trung bình **`+0.000825630` (`+0.000826`)**.
  - Lưu ý: Dù cả `A` lẫn `R2` đều tăng nhẹ `HE` khi chuyển từ `FIXED` sang `DECAY` tại `fixed_030`, hiệu đối xứng `R2_DECAY - A_DECAY` trên `HE` tại `fixed_030` vẫn đạt **`-0.004812`** (`-0.004291, -0.004957, -0.005189`), và trên `hedged_oce_y_nom` đạt **`-0.001251`**. Mọi báo cáo tiếp theo phải ghi rõ `OCE` (`hedged_oce_y_nom`) tách biệt với `HE` (`he_net_ddof0`).

---

## Mục (c): Bản chất thực của `delete20_jk_se_osc` trong [run_lr_stability.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/run_lr_stability.py#L101-L119)

- **Kiểm tra trực tiếp mã nguồn (`delete20_jk_se_osc`, dòng 101–119):**
  - Input là ma trận `oce_vals_by_ep` kích thước `(20, 20000)`: chứa `roll["hedged_loss"]` theo từng đường tại `y_nom` đóng băng của `20` epoch cuối (`181..200`) trên cùng `20,000` đường mô phỏng.
  - Thuật toán xáo trộn chỉ số đường mô phỏng (cột `20,000` bằng `perm_seed`), chia thành **`n_groups = 20` nhóm đường mô phỏng** (mỗi nhóm `1,000` đường), mỗi lần bỏ đồng thời một nhóm đường khỏi **tất cả 20 epoch**, tính lại trung bình cộng tuyến tính `ep_means = np.mean(arr[:, mask], axis=1)` trên `19,000` đường còn lại cho từng epoch `t = 1..20`, rồi lấy `max(ep_means) - min(ep_means)`, cuối cùng tính sai số chuẩn jackknife delete-20 qua 20 nhóm đường.
- **Đính chính cách diễn giải:**
  1. `delete20_jk_se_osc` là **jackknife trên 20 nhóm đường mô phỏng** (để đo nhiễu Monte Carlo theo đường của đại lượng `max - min` trên một lần chạy cố định), **KHÔNG phải** jackknife bỏ bớt từng checkpoint trong 20 checkpoint (`181..200`).
  2. Hạn chế phương pháp luận: (i) chỉ đánh giá độ bất định lấy mẫu Monte Carlo của đường đánh giá trên **một** quỹ đạo huấn luyện, không đo phương sai giữa các lần khởi tạo/seed; (ii) thống kê mục tiêu là `max - min` — một thống kê bậc không trơn có thể làm ước lượng jackknife delete-20 nhạy hơn so với các hàm trơn; và (iii) khi bộ suy diễn chính sách có `normalize_batch_mean=True`, trạng thái đầu vào từng đường có sự liên kết chéo nhỏ giữa các đường trong batch.
  3. Ngoài ra, trong [LR_STABILITY_REPORT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/LR_STABILITY_REPORT.md), mục tiêu chuẩn hóa câu chữ là không đồng nhất `20 nhóm đường` với `20 checkpoint`.

---

## Mục (d): Bằng chứng thực tế về `PAUSE_COORDINATOR` & Đính chính câu `"không tồn tại"`

- **Đính chính [LR_STABILITY_AUDIT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/LR_STABILITY_AUDIT.md#L29):**
  - Dòng 29 của `LR_STABILITY_AUDIT.md` viết *"PAUSE_COORDINATOR không tồn tại / không bị sửa"*. Nhận định *"không tồn tại"* là **KHÔNG chính xác** do nhầm đường dẫn kiểm tra trong lúc lập báo cáo.
- **Bằng chứng xác minh thực tế đã ghi nhận trong quá trình kiểm tra đường dẫn và đóng gói `LR_VERIFICATION_PACKAGE.zip`:**
  - **Đường dẫn thực của tệp:** `/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR`
  - **Kích thước & dấu thời gian gốc:** `32` bytes, `mtime = 2026-11-16 10:17:35.802798856 +0700` (`Mon 16 Nov 2026 10:17:35 AM +07`).
  - **Nguyên nhân nhầm lẫn kỹ thuật trước đó:** Các bước kiểm tra ngày `2026-11-19` lúc `22:12` và `22:23 +07` đã chạy `ls` nhầm ở thư mục cha `/home/user04/deephedging_jax_training_optimization/PAUSE_COORDINATOR*` thay vì `.../experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR`.
  - **Các mốc đã kiểm tra trực tiếp trên máy chủ ngày `2026-11-19`:**
    - Đã ghi nhận nguyên vẹn lúc `20:01:31 +07` (`32` bytes, `Nov 16 10:17`).
    - Đã xác minh lại nguyên vẹn trong khoảng `22:33–22:39 +07` (`32` bytes, `Nov 16 10:17`).
    - Khoảng trống quan sát: **Không có lệnh kiểm tra riêng nào chạy giữa `20:02` và `22:32 +07`**, do đó không tuyên bố giám sát liên tục theo từng giây, chỉ xác nhận dấu thời gian `mtime` không đổi kể từ `2026-11-16 10:17:35 +07` trước và sau các thao tác.
  - **Mức độ tuân thủ ràng buộc:** Tác tử **không** tạo, sửa, hay xóa `PAUSE_COORDINATOR` và **không** can thiệp các job Slurm khác (`Job 16909 asrtts_g`).

---

## Mục (e): Tách bạch Heuristic `0.5ε = 0.00005` & Quy tắc không gán xác suất nhân quả cảm tính

1. **Vai trò thực sự của `0.5ε = 0.00005`:**
   - Ngưỡng `0.5ε = 0.00005` được đặt trong pha ổn định tốc độ học chỉ nhằm làm **mốc tham chiếu kỹ thuật phát triển** cho độ biên dao động cuối kỳ `osc_oce_181_200`.
   - Nó **KHÔNG phải là điều kiện tiên quyết khoa học** để quyết định cho phép hay bác bỏ việc chuyển sang giai đoạn thiết kế xác nhận. Thực tế, `DECAY` đã cắt giảm `osc_oce_181_200` ~`73%–85%`, triệt tiêu hoàn toàn spike ở `ep200`, giữ `0/3` seed `R2_DECAY` vượt biên `+0.00010` ở `σ = 0.20`, và giữ `dHE_030 = -0.004812`. Phép kiểm chứng xác nhận thực sự nằm ở **kiểm định cỡ mẫu đủ lực (`N = 24` seed mới) trên dữ liệu chưa từng thấy**, không phải việc chạm một heuristic `0.5ε` trên 3 seed phát triển.
2. **Kỷ luật xác suất & nhân quả:**
   - Trong toàn bộ hồ sơ xác nhận, **tuyệt đối không gán phần trăm quy kết nhân quả** (ví dụ *"x% do MC noise, y% do chọn checkpoint"*) hay **"xác suất thành công"** khi không có mô hình sinh xác suất được định nghĩa tường minh.
   - Mọi phân tích lực thống kê (`SAMPLE_SIZE_SENSITIVITY.csv`) chỉ được trình bày như **độ nhạy có điều kiện theo từng kịch bản giả định `(μ_A, sd_A, μ_B, sd_B, ρ_AB)`**, phân biệt rạch ròi giữa kịch bản gốc phát triển và kịch bản stress bất lợi.
