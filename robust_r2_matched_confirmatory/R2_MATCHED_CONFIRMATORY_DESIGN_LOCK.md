# Thiết Kế Khóa Chính Thức Thí Nghiệm Xác Nhận (`R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md`)

> [!IMPORTANT]
> **TRẠNG THÁI CHÍNH THỨC: `DESIGN LOCKED — PRE-EXECUTION` (`N = 24`, Seeds `[4001..4024]`, `cost_s = 0.0`).**
> Tài liệu này chính thức chuyển bản dự thảo [R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md) thành **bản khóa cố định (`DESIGN LOCK`)** được người dùng phê duyệt. Kể từ thời điểm khóa, **nghiêm cấm** thay đổi siêu tham số, tăng `N`, mở rộng tìm kiếm cấu hình, hay dừng sớm dựa trên kết quả từng phần. `HOLD_EVAL` tiếp tục giữ niêm phong hoàn toàn.

---

## 1. Phạm Vi Khóa Chính Thức (48 Lượt Chạy = `24` Seed × `2` Nhánh)

1. **Cỡ mẫu & danh sách seed cố định duy nhất:** **`N = 24`** cặp độc lập với danh sách seed nguyên cố định:
   `[4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024]`.
   - Dải `[4025..4040]` được giữ trống và **bị khóa ngoài** trong suốt thí nghiệm này (`resolve_locked_confirmatory_seeds` ném lỗi `ValueError` nếu lát chạy vượt quá `4024`). Seed `3999` chỉ dành cho kiểm tra kỹ thuật cách ly, không nằm trong phân tích.
2. **Hai cấu hình đối xứng & chi phí giao dịch:**
   - Nhánh đối chứng **`A_DECAY` (`A_ref_128`)**: `k_oce = 1`, `6,762` tham số policy + `2` tham số OCE (`y[0], y0[0]`) = `6,764` tham số; huấn luyện trên 4 sub-pool `fixed_020` (`4 × 5,000` đường, batch `4 × 32 = 128`).
   - Nhánh thử nghiệm **`R2_DECAY` (`R2_matched_128`)**: `k_oce = 4`, cùng `6,762` tham số policy khởi tạo giống hệt từng bit với `A_DECAY` + `8` tham số OCE (`y[0..3], y0[0..3]`) = `6,770` tham số; huấn luyện theo **kỳ vọng đều `1/4`** trên 4 kịch bản `("fixed_015", "fixed_020", "fixed_025", "rs_015_025")` (`4 × 5,000` đường, batch `4 × 32 = 128`).
   - **Môi trường cố định:** **`cost_s = 0.0`**, `steps = 20`, `drift = 0.0`, `normalize_batch_mean = True`. Hiệu lực kết luận giới hạn trong môi trường `cost_s = 0.0`.
   - **Ngân sách & tối ưu hóa đối xứng:** `200` epoch × `156` bước/ep = **`31,200` bước cập nhật** (**`3,993,600` lượt xử lý đường/lượt chạy**); `Adam(beta_1=0.9, beta_2=0.999, eps=1e-7)`; `Cosine LR Decay` (`1e-3` bước `0..15,600`, cosine xuống `1e-4` bước `15,600..31,200`). Tổng cộng **`48` lượt huấn luyện** (`24 × 2`).
3. **Quy trình chọn bản lưu (`best_val`) & Đóng niêm phong trước kiểm tra:**
   - Tại cuối mỗi epoch `1..200`, `A_DECAY` đánh giá trên `1 × 5,000` đường validation `fixed_020` và chọn `best_val.pkl` tại epoch có `val_020` thấp nhất. `R2_DECAY` đánh giá trên `4 × 5,000` đường validation (`4` kịch bản huấn luyện) và chọn `best_val.pkl` tại epoch có trung bình cộng đều 4 kịch bản thấp nhất. Bản lưu `final_with_opt.pkl` ở epoch `200` (`ep200`) được giữ làm kiểm tra độ nhạy phụ (không dùng làm phán quyết chính).
   - Xác minh tái nạp bản lưu `best_val.pkl` khớp bộ nhớ `|reload - mem| < 1e-6` trước khi sinh dữ liệu kiểm tra.
   - Tập kiểm tra `conf_test` (**`20,000` đường/kịch bản** × `6` kịch bản, dùng chung chuỗi `conf_test_shared_dW` trong mỗi seed) **chỉ được sinh SAU KHI `best_val.pkl` và `ep200` của cả hai nhánh đã chốt xong**.

---

## 2. Hai Điều Kiện Thành Công Chính, `α = 0.025` & Kế Hoạch Phân Tích

Trên `best_val.pkl` tại `conf_test` (`20,000` đường/kịch bản), cặp chênh lệch `Δ_i = R2_DECAY_i - A_DECAY_i` (`i = 1..24`):
- **Chỉ số 1 — `HE` (`he_net_ddof0`):** `np.std(gains, ddof=0)` với `gains = payoff + pnl - cost` (độ lệch chuẩn tổng thể quanh trung bình thu nhập ròng, không phải RMSE quanh `0`).
- **Chỉ số 2 — `OCE` (`hedged_oce_y_nom`):** `mean(hedged_loss)` tại `y_nom` **đóng băng từ huấn luyện** (`y[0]` của `A_DECAY`, `y[1]` của `R2_DECAY`).

**Quy tắc quyết định đồng thời — Berger Intersection-Union Test (IUT) ở `α = 0.025` một phía:**
1. **Điều kiện chính A (`H1a`, Superiority tại `σ = 0.30`):** `H0a: μ(ΔHE_030) >= 0` vs `H1a: μ(ΔHE_030) < 0`. Bác bỏ `H0a` khi `t_A = mean(ΔHE_030) / SE < -t_{0.975, 23}` (`p_A < 0.025`, tức cận trên KTC 95% hai phía `< 0.0`).
2. **Điều kiện chính B (`H1b`, Non-inferiority tại `σ = 0.20`, biên `ε = +0.00010`):** `H0b: μ(ΔOCE_020) >= +0.00010` vs `H1b: μ(ΔOCE_020) < +0.00010`. Bác bỏ `H0b` khi `t_B = (mean(ΔOCE_020) - 0.00010) / SE < -t_{0.975, 23}` (`p_B < 0.025`, tức cận trên KTC 95% hai phía `< +0.00010`).
3. **Phán quyết chung:** Chỉ kết luận **`PASS_CONFIRMATORY_IUT`** khi và chỉ khi **(i) đạt CẢ HAI điều kiện A và B ở `α = 0.025`** VÀ **(ii) `24/24` seed `[4001..4024]` đều hoàn tất sạch (`complete_and_clean_gate == True`)**. Nếu một hoặc cả hai điều kiện không đạt, kết luận đúng theo thiết kế (`FAIL_CONFIRMATORY_IUT`), tuyệt đối không thay thế bằng chỉ tiêu phụ thuận lợi (`fixed_015`, `fixed_025`, `rs_015_025`, `regime_switch_015_028`, `ep200`, `CVaR95`, `turnover`, `hedged_oce_y_scen`).
4. **Kiểm định & chẩn đoán đi kèm ([conf_stats.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_stats.py)):** Exact Sign test, exact Wilcoxon signed-rank DP (`m <= 50`), độ lệch `skewness`/`excess_kurtosis`, ảnh hưởng `Leave-One-Out`, toàn bộ giá trị tuyệt đối từng nhánh + hiệu ghép cặp ở 6 kịch bản (`best_val` & `ep200`).

---

## 3. Bảng Công Suất Thống Kê Đầy Đủ (`R1–R8`, `α = 0.025` Một Phía) & Công Thức Tính

> [!WARNING]
> **Lưu ý thuật ngữ khoa học:** Các giá trị dưới đây là **công suất thống kê có điều kiện (`statistical power`)** giả định tham số tổng thể `(μ_A, sd_A, μ_B, sd_B, ρ_AB)` đúng như kịch bản; **KHÔNG gọi đây là "xác suất thành công của phương pháp"**.

**Công thức tính công suất trong [scripts/sample_size_and_power.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/scripts/sample_size_and_power.py#L96-L159):**
- **Công suất biên từng nhánh (Exact Noncentral-$t$, tích phân Simpson `600` đoạn):** Với cỡ mẫu $N$, ngưỡng bác bỏ một phía tại $\alpha = 0.025$ là $c = -t_{0.975, N-1}$ ($\text{df} = N - 1$). Dưới phân phối chuẩn giả định $\Delta_i \sim \mathcal{N}(\mu, \sigma^2)$:
  - $P_A = P(T_A < c)$ với $T_A \sim t_{N-1}(\delta_A)$, tham số lệch $\delta_A = \frac{\sqrt{N}\,(0 - \mu_A)}{\text{sd}_A}$ (vì bác bỏ về đuôi dưới $\Delta HE_{030} < 0$, trong hàm `nct_power_lower_tail(n, -mu_a, sd_a, alpha)` dùng hiệu $\Delta_{\text{margin}} = -\mu_A$).
  - $P_B = P(T_B < c)$ với $T_B \sim t_{N-1}(\delta_B)$, $\delta_B = \frac{\sqrt{N}\,(\varepsilon_B - \mu_B)}{\text{sd}_B}$ với $\varepsilon_B = 0.00010$.
- **Công suất chung IUT ($\text{Rej}_A \cap \text{Rej}_B$):**
  - **Cận dưới Fréchet (phi tham số theo tương quan):** $P_{\text{Frechet}} = \max(0, P_A + P_B - 1)$.
  - **Mô phỏng Monte Carlo nhị biến ($B = 60,000$ lần lặp, hạt giống cố định):** Sinh $B = 60,000$ mẫu cỡ $N$ từ $\mathcal{N}_2\!\left(\begin{pmatrix}\mu_A \\ \mu_B\end{pmatrix}, \begin{pmatrix}\text{sd}_A^2 & \rho_{AB}\text{sd}_A\text{sd}_B \\ \rho_{AB}\text{sd}_A\text{sd}_B & \text{sd}_B^2\end{pmatrix}\right)$, tính $T_A = \frac{\bar{\Delta}_A}{s_A / \sqrt{N}}$, $T_B = \frac{\bar{\Delta}_B - \varepsilon_B}{s_B / \sqrt{N}}$, và lấy $P_{\text{joint, MC}} = \frac{1}{B}\sum_{b=1}^{B} \mathbb{I}(T_{A,b} < c \;\land\; T_{B,b} < c)$.

**Bảng đầy đủ cả 8 kịch bản `R1–R8` tại `N = 24` và `α = 0.025` (trích nguyên văn từ [SAMPLE_SIZE_SENSITIVITY.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/SAMPLE_SIZE_SENSITIVITY.csv)):**

| Kịch bản | Giả định `(μ_A, sd_A, μ_B, sd_B, ρ_AB)` (với `ε_B = 0.00010`) | Biên còn `ε_B - μ_B` | `P_A` (`nct`) | `P_B` (`nct`) | Cận dưới Fréchet (`N=24`) | **Công suất chung MC (`N=24`)** | `N` cần cho `90%` (`H1b`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `R1_LR_DECAY_PointEst` | `(-0.0048, 0.0008, 0.00000, 0.000050, 0.0)` | `0.00010` | `1.00000` | `1.00000` | `1.00000` | **`1.00000`** | `5` |
| `R2_Realistic_Base` | `(-0.0038, 0.0012, 0.00001, 0.000065, 0.0)` | `0.00009` | `1.00000` | `1.00000` | `1.00000` | **`1.00000`** | `8` |
| `R3_Conservative_Design` | `(-0.0030, 0.0015, 0.00002, 0.000075, 0.0)` | `0.00008` | `1.00000` | `0.99881` | `0.99881` | **`0.99893`** | `12` |
| `R4_NegCorr_Conservative` | `(-0.0030, 0.0015, 0.00002, 0.000075, -0.3)` | `0.00008` | `1.00000` | `0.99881` | `0.99881` | **`0.99877`** | `12` |
| `R5_PosCorr_Conservative` | `(-0.0030, 0.0015, 0.00002, 0.000075, +0.4)` | `0.00008` | `1.00000` | `0.99881` | `0.99881` | **`0.99868`** | `12` |
| `R6_HighDrift_Stress` | `(-0.0025, 0.0018, 0.00003, 0.000085, 0.0)` | `0.00007` | `1.00000` | `0.97135` | `0.97134` | **`0.97255`** | `18` |
| `R7_HeavyVariance_FIXEDlike` | `(-0.0030, 0.0025, 0.00002, 0.000110, 0.0)` | `0.00008` | `0.99987` | `0.92656` | `0.92643` | **`0.92713`** | `22` |
| **`R8_Worst_40pctMargin_FIXEDsd`** | `(-0.0020, 0.0025, 0.00004, 0.000120, -0.3)` | `0.00006` | `0.96334` | `0.65038` | `0.61373` | **`0.62015`** (`~62%`) | `44` |

**Hạn chế công bố trước chạy:** Với `N = 24`, nếu trạng thái thực tế rơi vào kịch bản stress cực đoan **`R8`** (mất 40% biên `μ_B = +0.40 bp`, `sd_B = 0.000120`, `ρ_AB = -0.3`), công suất chung **chỉ đạt ~`62.0%`** (`0.62015` MC / `0.61373` cận dưới Fréchet; phải cần tới `N = 44` mới đạt `90%`).

---

## 4. Chính Sách Tiếp Tục (`--seed-offset`) & Phân Biệt Lỗi Hạ Tầng vs Thất Bại Thuật Toán

Thực thi chặt chẽ trong `resolve_locked_confirmatory_seeds` và `classify_run_exception` ([conf_run.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py)):
1. **`--seed-offset` chỉ giới hạn trong `[4001..4024]`:** Nếu `--seed-offset + --n-seeds > 24`, chương trình từ chối chạy ngay (không bao giờ lấn sang `[4025..4040]`).
2. **Phân biệt hai loại thất bại trong `SEED_LEDGER.json` và `CONFIRMATORY_STATS_SUMMARY.json`:**
   - **`ALGORITHM_FAILURE`:** Lỗi số học/thuật toán (`NaN`, `Inf`, `non-finite`, sai khác reload parity). Seed bất lợi bị đánh dấu vĩnh viễn là thất bại, ghi dòng `NaN` trong `PAIRED_CONFIRMATORY.csv`, **cấm bỏ qua hay thay thế bằng seed khác**.
   - **`INFRA_ERROR`:** Lỗi hạ tầng / OS / I/O / ngắt thời gian Slurm giữa chừng. Chỉ những seed chưa chạy hoặc lỗi hạ tầng trong `[4001..4024]` mới được chạy tiếp bằng `--seed-offset`.
3. **Cấm kết luận thành công từ tập con:** Hàm `analyze_confirmatory_pairs` luôn kiểm đếm đối chiếu với đủ `planned_seeds = [4001..4024]`. Nếu bất kỳ seed nào thiếu hoặc mang trạng thái thất bại chưa xử lý theo thiết kế, phán quyết bị khóa cứng là **`INCOMPLETE_OR_FAILED_AUDIT`** (`confirmatory_pass = False`).
