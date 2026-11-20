# Dự Thảo Thiết Kế Thí Nghiệm Xác Nhận Độc Lập: `R2_matched_128` vs `A_ref_128` (`R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md`)

> [!CAUTION]
> **TRẠNG THÁI TÀI LIỆU:** **BẢN DỰ THẢO THIẾT KẾ CHƯA CHẠY THỰC NGHIỆM (`PRE-RUN DESIGN DRAFT`).** Chưa nộp Slurm xác nhận, chưa mở/mô phỏng dữ liệu kiểm tra `[4001..4040]`, và chưa mở `HOLD_EVAL`.

---

## 1. Câu Hỏi Xác Nhận & Định Vị Khoa Học

**Câu hỏi nghiên cứu chính:**
> *"Khi `R2_matched_128` và `A_ref_128` được cân bằng chặt chẽ về ngân sách huấn luyện (`31,200` bước cập nhật, `3,993,600` lượt xử lý đường) và cùng dùng quy trình tối ưu ổn định (`Cosine LR Decay`, `Adam eps=1e-7`, `cost_s=0.0`), liệu việc huấn luyện theo kỳ vọng trên 4 kịch bản biến động có làm giảm sai số phòng hộ (`HE`) tại `σ = 0.30` đồng thời không làm tăng OCE phòng hộ (`hedged_oce_y_nom`) tại `σ = 0.20` vượt quá biên `ε = +0.00010` trên dữ liệu độc lập chưa từng thấy hay không?"*

**Phân biệt rạch ròi ba thành phần từ giai đoạn phát triển:**
1. **`R3` (Minimax mượt có ràng buộc — ĐÃ DỪNG MỞ RỘNG):** Trong `robust_r3_dev/`, hai cấu hình `R3_cfg1` và `R3_cfg2` đã dừng mở rộng sau khi làm rõ chênh lệch ngân sách và cơ chế tối ưu minimax. Nghiên cứu xác nhận này **KHÔNG** chạy `R3`.
2. **`R2_matched_128` (Huấn luyện kỳ vọng đều trên 4 kịch bản):** Hàm mục tiêu huấn luyện là **trung bình cộng đều (`1/4`)** tổn thất OCE trên 4 kịch bản `TRAINING_SCENARIOS = ("fixed_015", "fixed_020", "fixed_025", "rs_015_025")`, với 4 cặp tham số `(y_k, y0_k)` tách rời theo kịch bản (trong đó `k = 1` là `fixed_020`, tức `y_nom = y[1]`) và `6,762` tham số policy chia sẻ chung (`6,770` tham số tổng).
3. **`Cosine LR Decay` (Thay đổi tối ưu áp dụng ĐỐI XỨNG cho cả `A_ref_128` lẫn `R2_matched_128`):** Giữ nguyên `lr = 1e-3` trong `100` epoch đầu (`15,600` bước), sau đó giảm theo đường cosine xuống `1e-4` ở cuối epoch `200` (`31,200` bước). Đây là bản nâng cấp ổn định hóa cho **cả hai** nhánh so với `FIXED`, đã cắt `73%–85%` dao động cuối kỳ và xóa spike tại `ep200` trong `robust_r3_lr_stability/`.

---

## 2. Tổng Hợp Tách Lớp Các Tầng Phát Triển (`Δ = R2_matched_128 - A_ref_128`)

Số liệu tổng hợp trực tiếp từ [dev_strata_summary.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/dev_strata_summary.json) (giữ nguyên tách lớp theo từng giai đoạn, không gộp lẫn):

| Tầng dữ liệu (`Stage`) | Dải Seed (`n`) | Lịch LR | Checkpoint | Số đường Test | `ΔHE_030` (`Mean ± SD`) `[Min, Max]` | `ΔOCE_020` (`Mean ± SD`) `[Min, Max]` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BC_old_pilot` | `3101..3103` (`n=3`) | `FIXED` | `best_val` | `5,000` | `-0.008676 ± 0.008552` `[-0.018546, -0.003501]` | `-0.000079 ± 0.000177` `[-0.000273, +0.000074]` |
| `BC_new_dev` | `3201..3203` (`n=3`) | `FIXED` | `best_val` | `5,000` | `-0.003942 ± 0.002275` `[-0.006557, -0.002418]` | `-0.000027 ± 0.000069` `[-0.000107, +0.000014]` |
| `DIAG_val_diag_a_20k` | `3101..3203` (`n=6`) | `FIXED` | `best_val` | `20,000` | `-0.005706 ± 0.004925` `[-0.015396, -0.002218]` | `-0.000041 ± 0.000140` `[-0.000294, +0.000060]` |
| `DIAG_val_diag_a_20k` | `3101..3203` (`n=6`) | `FIXED` | `ep200` | `20,000` | `-0.005235 ± 0.004787` `[-0.014860, -0.002529]` | `+0.000042 ± 0.000200` `[-0.000259, +0.000225]` |
| `LR_STAB` (`FIXED`) | `3401..3403` (`n=3`) | `FIXED` | `best_val` | `5,000` | `-0.004645 ± 0.000038` `[-0.004678, -0.004604]` | `+0.000012 ± 0.000045` `[-0.000025, +0.000062]` |
| `LR_STAB` (`FIXED`) | `3401..3403` (`n=3`) | `FIXED` | `ep200` | `5,000` | `-0.003451 ± 0.001424` `[-0.004494, -0.001828]` | `+0.000209 ± 0.000291` `[+0.000012, +0.000543]` |
| **`LR_STAB` (`DECAY`)** | **`3401..3403` (`n=3`)** | **`DECAY`** | **`best_val`** | **`5,000`** | **`-0.004812 ± 0.000466` `[-0.005189, -0.004291]`** | **`-0.000002 ± 0.000047` `[-0.000040, +0.000051]`** |
| `LR_STAB` (`DECAY`) | `3401..3403` (`n=3`) | `DECAY` | `ep200` | `5,000` | `-0.004489 ± 0.000844` `[-0.005075, -0.003521]` | `+0.000043 ± 0.000055` `[-0.000019, +0.000084]` |

**Ý nghĩa đối với thiết kế xác nhận:**
- Dưới `FIXED`, chuyển từ `best_val` sang `ep200` từng làm tăng mạnh phương sai `ΔOCE_020` (`SD = 0.000200..0.000291`). Dưới `Cosine LR Decay`, `best_val` đạt `ΔOCE_020 = -0.000002 ± 0.000047` (cả 3 seed nằm sát 0, rất xa `+0.00010`), đồng thời `ep200` cũng không còn spike vượt ngưỡng (`+0.000043 ± 0.000055`, max `+0.000084`).
- Phân tích trong `robust_r3_selection_diagnostics/` (`DIAG_PART1`) cũng chỉ ra rằng với `5,000` đường kiểm tra (`b0..b3`), độ lệch chuẩn lấy mẫu Monte Carlo kinh nghiệm của `ΔOCE_020` trung bình tới `0.000041` (`max = 0.000079`), còn trên `20,000` đường (`diag_a` vs `diag_c`), `JK20 SE` giảm xuống trung bình `0.000027` (`|diag_a - diag_c| = 0.000009`). Do đó thiết kế xác nhận **nâng `conf_test` lên `20,000` đường/kịch bản**.

---

## 3. Khóa Quy Trình Huấn Luyện, Dữ Liệu & Định Nghĩa Chỉ Số ([conf_design.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_design.py))

1. **Hai nhánh đối sánh duy nhất (`1:1` ghép cặp theo `seed_label`):**
   - `A_DECAY` (`A_ref_128`, `k_oce = 1`, `6,762` policy + `2` OCE = `6,764` params): huấn luyện trên `4` sub-batch độc lập `fixed_020` (`4 × 32 = 128` đường/bước). Chọn `best_val` = epoch có `val_hedged_per_scen["fixed_020"]` nhỏ nhất trên `5,000` đường.
   - `R2_DECAY` (`R2_matched_128`, `k_oce = 4`, `6,762` policy + `8` OCE = `6,770` params): huấn luyện trên `4` kịch bản `(fixed_015, fixed_020, fixed_025, rs_015_025)` (`4 × 32 = 128` đường/bước). Chọn `best_val` = epoch có trung bình cộng đều `4` kịch bản `val` (`4 × 5,000`) nhỏ nhất.
2. **Bình đẳng & cố định tham số:**
   - Dùng chung vector khởi tạo `init_policy` trong từng seed (sai khác `0.0`), dùng chung tập `val` `5k` để giữ nguyên quy tắc chọn đã khảo sát ở `LR_STABILITY` và `BUDGET_CONTROL`.
   - `Epochs = 200`, `Steps/Epoch = 156`, tổng **`31,200` bước cập nhật** và **`3,993,600` lượt đường**.
   - `Adam(beta_1=0.9, beta_2=0.999, eps=1e-7)`, `Cosine LR Decay` (`1e-3` bước `0..15,600`, cosine xuống `1e-4` bước `15,600..31,200`). Môi trường `cost_s = 0.0`, `steps = 20`, `normalize_batch_mean = True`.
3. **Dữ liệu & cách ly PRNG ([conf_keys.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_keys.py), [DATA_AND_SEED_PLAN.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DATA_AND_SEED_PLAN.json)):**
   - Không gian tên SHA-256 `R2_MATCHED_CONF_V1` tách biệt hoàn toàn với mọi `2,167` khóa lịch sử (`0` va chạm, `0` tràn `u32`). Dải seed dự trữ xác nhận: `[4001..4040]`; seed smoke kỹ thuật cách ly: `3999`. `HOLD_EVAL` tiếp tục giữ niêm phong hoàn toàn.
   - Tập `conf_test` (`20,000` đường/kịch bản × `6` kịch bản, dùng chung chuỗi nhiễu `conf_test_shared_dW` cho ghép cặp chuẩn) **chỉ được sinh SAU KHI quá trình huấn luyện và chọn `best_val` của cả hai nhánh đã đóng niêm phong**.
4. **Định nghĩa chính xác các chỉ số ([r3_metrics.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_dev/r3_metrics.py)):**
   - **`HE` (`he_net_ddof0`):** `np.std(gains, ddof=0)` với `gains = payoff + pnl - cost` — tức **độ lệch chuẩn tổng thể quanh trung bình của thu nhập sau phòng hộ** (**KHÔNG phải** RMSE quanh `0`).
   - **`OCE` (`hedged_oce_y_nom`):** Giá trị `mean(hedged_loss)` tại `y_nom` **đóng băng từ quá trình huấn luyện** (`y[0]` cho `A_DECAY`, `y[1]` cho `R2_DECAY`), **KHÔNG phải** tối ưu lại `y` kiểu oracle trên tập test và **KHÔNG phải** mixture OCE trên cả 4 kịch bản.
   - **`hedged_oce_y_scen`:** Giá trị phụ chẩn đoán nội miền khi thay `y_k` theo kịch bản (chỉ tính cho 4 kịch bản trong `TRAINING_SCENARIOS`, không dùng cho quyết định chính).

---

## 4. Giả Thuyết Chính, Quy Tắc Quyết Định & Kiểm Soát Sai Lầm Loại I ([conf_stats.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_stats.py))

Đặt cặp chênh lệch trên mỗi seed độc lập `i = 1..N` (`best_val`, đánh giá trên `20,000` đường `conf_test`):
- `d_HE,i = HE_{R2_DECAY}(fixed_030, i) - HE_{A_DECAY}(fixed_030, i)`
- `d_OCE,i = OCE_{R2_DECAY}(fixed_020, i) - OCE_{A_DECAY}(fixed_020, i)`

**Hệ đồng giả thuyết (Co-Primary Endpoints) ở mức một phía `α = 0.025`:**
- **H1a (Superiority tại `σ = 0.30`):** `H0a: μ_A >= 0` đối chọi `H1a: μ_A < 0`.
  - Bác bỏ `H0a` khi `t_A = mean(d_HE) / SE(d_HE) < -t_{0.975, N-1}` (`p_A < 0.025`), tương đương **cận trên khoảng tin cậy 95% hai phía `< 0.0`**.
- **H1b (Non-inferiority tại `σ = 0.20` với biên `ε = +0.00010`):** `H0b: μ_B >= ε` đối chọi `H1b: μ_B < ε`.
  - Bác bỏ `H0b` khi `t_B = (mean(d_OCE) - ε) / SE(d_OCE) < -t_{0.975, N-1}` (`p_B < 0.025`), tương đương **cận trên khoảng tin cậy 95% hai phía `< +0.00010`**.
- **Quy tắc kết luận chung — Berger Intersection-Union Test (IUT):**
  - Chỉ kết luận **ĐẠT XÁC NHẬN (`PASS_CONFIRMATORY_IUT`)** khi và chỉ khi **CẢ HAI** `H0a` và `H0b` đồng thời bị bác bỏ ở `α = 0.025` (nếu một trong hai không đạt thì kết luận **`FAIL_CONFIRMATORY_IUT`**).
  - Theo nguyên lý IUT của Berger (1982), vì vùng bác bỏ là giao của hai biến cố, tỷ lệ sai lầm loại I chung cả họ (`FWER`) được chặn chặt `<= max(α_A, α_B) = 0.025` trên toàn bộ không gian giả thuyết không `(H0a ∪ H0b)` **mà không cần chia Bonferroni `α/2`**. Mức `α = 0.025` một phía khớp chặt với quy ước KTC 95% hai phía.
- **Kiểm định nhạy bổ trợ (không thay thế quy tắc chính IUT):**
  - Sign test nhị thức chính xác và Wilcoxon signed-rank được báo cáo để kiểm tra độ vững trước lệch phân phối (skewness/outliers), nhưng ghi rõ chúng kiểm định giả thuyết không khác (`median = bound` hoặc `đối xứng quanh bound`) chứ không kiểm định kỳ vọng `μ`.
  - Kiểm tra ảnh hưởng loại-từng-seed (`Leave-One-Out`), hệ số bất đối xứng (`skewness`), và báo cáo các kịch bản phụ (`fixed_015`, `fixed_025`, `rs_015_025`, `regime_switch_015_028`), `ep200`, `CVaR95`, `turnover`.
- **Xử lý trường hợp thiếu seed, không hội tụ hoặc `NaN`/`Inf`:**
  - Bộ phân tích lập sổ cái hoàn chỉnh trước khi tính `t`-test; nếu thiếu bất kỳ seed nào trong tập đã khóa hoặc có `NaN`/`Inf`, hệ thống gắn cờ **`INCOMPLETE_OR_FAILED_AUDIT`** và **chặn phán quyết `PASS`** (cấm lọc bỏ thầm lặng các seed thất bại).

---

## 5. Phân Tích Cỡ Mẫu & Độ Nhạy Giả Định ([SAMPLE_SIZE_SENSITIVITY.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/SAMPLE_SIZE_SENSITIVITY.csv))

Do giai đoạn phát triển `LR_STABILITY` (`DECAY`) chỉ có `n = 3` seed (`df = 2`), ước lượng độ lệch chuẩn giữa các seed (`sd_B = 0.000047`) có **khoảng tin cậy rất rộng**. Vì vậy, chúng tôi tính toán lực thống kê chuẩn (`exact noncentral t` từng nhánh + cận dưới Fréchet + mô phỏng Monte Carlo nhị biến `B = 60,000`) qua **8 kịch bản giả định (`R1–R8`)** tại `α_one_sided = 0.025`:

| Kịch bản giả định (`α = 0.025` 1 phía / `CI 95%` 2 phía) | `(μ_A, sd_A)` | `(μ_B, sd_B, ρ)` | Lực chung `N = 16` | **Lực chung `N = 24` (Chính)** | Lực chung `N = 36` | `N` cần cho `90%` ở `H1b` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `R1_LR_DECAY_PointEst` (điểm ước lượng lạc quan `n=3`) | `(-0.0048, 0.0008)` | `(0.0, 0.000050, 0)` | `1.000` | **`1.000`** | `1.000` | `5` |
| `R2_Realistic_Base` (trượt nhẹ `+0.10 bp`, `sd_B=6.5e-5`) | `(-0.0038, 0.0012)` | `(+0.00001, 0.000065, 0)` | `0.999` | **`1.000`** | `1.000` | `8` |
| `R3_Conservative_Design` (`+0.20 bp`, `sd_B=7.5e-5 ≈ 1.6× DECAY`) | `(-0.0030, 0.0015)` | `(+0.00002, 0.000075, 0)` | `0.979` | **`0.999`** | `1.000` | `12` |
| `R4_NegCorr_Conservative` (tương quan nghịch `ρ = -0.30`) | `(-0.0030, 0.0015)` | `(+0.00002, 0.000075, -0.3)` | `0.978` | **`0.999`** | `1.000` | `12` |
| `R5_PosCorr_Conservative` (tương quan thuận `ρ = +0.40`) | `(-0.0030, 0.0015)` | `(+0.00002, 0.000075, +0.4)` | `0.978` | **`0.999`** | `1.000` | `12` |
| `R6_HighDrift_Stress` (còn biên `0.70 bp`, `sd_B=8.5e-5`) | `(-0.0025, 0.0018)` | `(+0.00003, 0.000085, 0)` | `0.869` | **`0.973`** | `0.998` | `18` |
| `R7_HeavyVariance_FIXEDlike` (`sd_B=0.000110`, gần `FIXED` `DIAG_20k`) | `(-0.0030, 0.0025)` | `(+0.00002, 0.000110, 0)` | `0.773` | **`0.927`** | `0.989` | `22` |
| `R8_Worst_40pctMargin_FIXEDsd` (`μ_B=+0.40 bp`, `sd_B=0.000120`, `ρ=-0.3`) | `(-0.0020, 0.0025)` | `(+0.00004, 0.000120, -0.3)` | `0.368` | **`0.620`** | `0.827` | `44` |

*(Ghi chú: Ở `R1–R5`, yếu tố chi phối lực thống kê gần như hoàn toàn là `H1b` — do `H1a` có hiệu ứng rất lớn `-0.003..-0.0048` nên đạt lực `> 0.999` chỉ từ `N >= 8..10`; nếu giả định trường hợp cực đoan `R8` xảy ra với `sd_B = 0.000120` và trôi tới `+0.40 bp`, `N = 24` chỉ đạt `62.0%` và phải cần `N = 44` mới đạt `90%`.)*

**Đề xuất cỡ mẫu & Kỷ luật chống Optional Stopping (để người dùng / cố vấn chốt DUY NHẤT 1 giá trị `N` trước khi chạy):**
- **Khuyến nghị chính: Khóa cứng `N = 24` seed mới (`[4001..4024]`, `48` runs, `~3.86h` trên 8 CPU):** Đạt lực chung `>= 92.7%` ngay cả ở kịch bản phương sai nặng `R7` (`sd_B = 0.000110`, tiệm cận `FIXED`), vượt xa `97%` ở `R1–R6` (`R6 = 97.3%`), đủ bậc tự do (`df = 23`, `t_crit = 2.069`) cho độ tin cậy cận trên KTC 95% và Sign/Wilcoxon, đồng thời khớp vừa vặn giới hạn 1 job Slurm `--time=05:30:00`.
- **Phương án dự phòng ngân sách thấp (chỉ chọn trước khi chạy nếu có giới hạn cụm cứng): `N = 16` seed (`[4001..4016]`, `32` runs, `~2.57h` CPU):** Phù hợp nếu hạ tầng Slurm giới hạn cứng thời gian dưới `3.5 giờ`, vẫn giữ lực `>= 97.8%` dưới `R1–R5` và `86.9%` dưới `R6`, nhưng phải **chấp nhận hạn chế lực tụt xuống `77.3%` dưới `R7` (`< 80%`) và `36.8%` dưới `R8`**.
- **QUY TẮC CỨNG CHỐNG OPTIONAL STOPPING:** Phải khóa **một con số `N` duy nhất** trước khi nộp job xác nhận (`R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md`). **Tuyệt đối cấm** chạy `N = 16`, mở kết quả xem `p`-value rồi mới chạy bổ sung thêm `8` seed thành `N = 24` dưới cùng ngưỡng `α = 0.025`, vì hành vi tăng mẫu theo kết quả (unadjusted sequential peeking) làm phình sai lầm loại I thực tế vượt quá `0.025`. Dải `[4025..4040]` được để trống hoàn toàn cho các nghiên cứu nối tiếp độc lập nếu cần trong tương lai.

---

## 6. Rủi Ro, Lệch Có Thể Xảy Ra & Biên Giới Hợp Lệ

1. **Giới hạn của môi trường không phí (`cost_s = 0.0`):** `LR_STABILITY` cho thấy `DECAY` làm `turnover` tăng `+3.11% .. +4.00%`. Thí nghiệm xác nhận khóa `cost_s = 0.0` để trả lời câu hỏi nhân quả công bằng nhất với chuỗi `R2/R3` trước đó; kết luận xác nhận **chỉ có hiệu lực trong môi trường `cost_s = 0.0`**, không tự động ngoại suy sang thị trường có phí giao dịch dương.
2. **Suy giảm nhẹ `σ = 0.30` dưới `DECAY` vs `FIXED`:** Cả `A_DECAY` và `R2_DECAY` đều cho `OCE` và `HE` tại `fixed_030` cao hơn nhẹ so với `FIXED`, dù hiệu `R2 - A` vẫn âm (`-0.004812` trên `HE`). Mọi phân tích phải báo cáo giá trị tuyệt đối từng nhánh song hành với chênh lệch ghép cặp.
3. **Tính tái lặp `val_020` vs `val 4-kịch bản`:** `A_DECAY` chọn `best_val` trên `5k` 1 kịch bản còn `R2_DECAY` chọn trên `4 × 5k`; điều này có thể khiến `best_epoch` của hai nhánh lệch nhau (cộng thêm dao động nhỏ `0.000107–0.000269`). Báo cáo luôn đối chiếu cả `ep200` (không có nhiễu chọn tập validation) như độ nhạy bắt buộc.
4. **Phụ thuộc `normalize_batch_mean=True`:** Các mẫu `conf_test` dùng `normalize_batch_mean=True` trên tập `20,000` đường cố định theo quy chuẩn gốc, tạo tương quan nhẹ xuyên đường trong lúc suy diễn.
