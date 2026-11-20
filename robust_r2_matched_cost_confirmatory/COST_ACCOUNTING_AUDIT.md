# Kiểm Toán Kế Toán Chi Phí Giao Dịch (`COST_ACCOUNTING_AUDIT.md`)

- **Phạm vi kiểm toán:** Cơ chế tính phí tỷ lệ `c_s = 0.0010` (`10 bps`) trong toàn bộ chuỗi mô phỏng thị trường ([robust_world.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/scripts/robust_deep_hedging_phase2/robust_world.py)), cuộn chính sách & mất mát OCE ([r3_models.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_dev/r3_models.py), [bc_models.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_budget_control/bc_models.py)), trình bọc và chỉ số ([cost_world_and_metrics.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/cost_world_and_metrics.py)), và bộ kiểm thử ([test_cost_conf_suite.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/tests/test_cost_conf_suite.py)).
- **Trạng thái đối với hồ sơ cũ:** Không sửa bất kỳ tệp nào trong `robust_r3_dev/`, `robust_r3_lr_stability/`, `robust_r2_matched_confirmatory/`, `b3_cost_confirmatory_10seeds/`, hoặc `scripts/`. Lưu ý rằng `run_r3_pilot.py::build_scenario_world` cố định `cost_s=0.0`, nên thí nghiệm mới dùng trình bọc riêng `build_cost_scenario_world(..., cost_s=0.0010)`.

---

## 1. Dòng Thời Gian Vị Thế, Giao Dịch & Phí (`j = 0 .. 19` và Đáo Hạn `j = 20`)

Trong [robust_world.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/scripts/robust_deep_hedging_phase2/robust_world.py#L125-L155) (`steps = 20`, `dt = 0.02`, `T = 0.40`, `S_0 = 1.0`, `K = 1.0`) và `rollout_policy_paths` ([r3_models.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_dev/r3_models.py#L114-L149)):

1. **Khởi tạo trước bước `j = 0`:** $\delta_{-1} = 0$, $\text{pnl} = 0$, $\text{cost} = 0$, $\text{first}_0 = 1.0$.
2. **Tại mỗi bước tái cân bằng `j = 0, 1, ..., 19`:**
   - Đầu vào mạng (khi `normalize_batch_mean = True`): `[feature_price_j, delta_{j-1}, time_left_j]` (chuẩn hóa feature không làm thay đổi `market["hedges"]`, `market["cost"]` hay `market["payoff"]`).
   - Lượng giao dịch tại bước $j$:
     $$a_j = \text{softclip}(\text{out}_j + \text{idelta} \cdot \text{first}_j,\ \text{lb}=-5.0,\ \text{ub}=+5.0) = \delta_j - \delta_{j-1}.$$
   - Vị thế sau giao dịch: $\delta_j = \delta_{j-1} + a_j$. Giao dịch mở vị thế ban đầu tại $j = 0$ ($\delta_{-1}=0 \to \delta_0 = a_0$) **chịu phí** $|a_0| \cdot S_0 \cdot c_s$.
   - Phí giao dịch từng bước: $\text{cost\_dS}_j = S_j \cdot c_s \implies \text{cost} = \sum_{j=0}^{19} |a_j| S_j c_s$.
   - Lãi/lỗ cộng dồn giao ngay tính đến đáo hạn: $\text{pnl} = \sum_{j=0}^{19} a_j (S_T - S_j)$.
3. **Tại đáo hạn `j = 20` ($S_T = S_{20}$):**
   - Quyền chọn mua ATM bán khống thanh toán tiền mặt: $\text{payoff} = -(S_T - K)^+ \le 0$.
   - **Không có giao dịch tất toán vị thế tại $j = 20$** (phí tất toán tại $T$ bằng `0`). Đảo chiều hoặc đóng vị thế trước đáo hạn ($j \le 19$) chịu phí tỷ lệ thuận trên $|a_j| S_j$.

---

## 2. Xác Minh Đồng Nhất Kế Toán Gross vs Net & Không Trừ Phí Hai Lần

- **Thu nhập trước phí (Gross Gains):** $G_{\text{gross}} = \text{payoff} + \text{pnl}$.
- **Thu nhập ròng sau phí (Net Gains):** $G_{\text{net}} = \text{payoff} + \text{pnl} - \text{cost} = G_{\text{gross}} - \text{cost}$.
- **Mất mát OCE phòng hộ (`hedged_loss`):** Đánh giá hàm utility `exp2` ($\lambda = 10$) **trực tiếp trên `gains = G_net`** cùng bù trừ vô hướng `y`:
  - Không cộng lại `cost` ở bất kỳ tầng loss hay metrics nào sau đó.
  - Khi đặt `cost_s = 0.0`, `market["cost"] == 0` và $G_{\text{net}} \equiv G_{\text{gross}}$ chuẩn xác từng bit trong sai số dấu phẩy động `float32`.
  - Mọi phép đánh giá kiểm tra ràng buộc `| (mean_gross_gains - mean_cost) - mean_gains | < 1e-5` (ngưỡng làm tròn cộng `float32` trên `20,000` đường).

---

## 3. Chứng Minh Phí Đi Xuyên Cả 3 Giai Đoạn (`Train`, `best_val`, `conf_test`)

| Giai đoạn | Nguồn dữ liệu & Tham số `cost_s` | Đóng góp vào mục tiêu / lựa chọn | Kiểm chứng đơn vị (`test_cost_conf_suite.py`) |
| :--- | :--- | :--- | :--- |
| **1. Huấn luyện (`Train`)** | `build_cost_train_and_val_for_seed(..., cost_s=0.0010)` | `aref_128_step_loss` & `r2_matched_step_loss` gọi `rollout_policy_paths` dùng `market["cost"] = S_j * 0.0010`; ghi `train_cost_means > 0` | `test_3`: gradient theo tham số `policy` khác rõ (`max_grad_diff > 1e-5`) so với `cost_s = 0.0` |
| **2. Chọn bản lưu (`best_val`)** | `val_batches` tạo với `cost_s = 0.0010` | `train_cost_config` tối thiểu hóa `hedged_loss` sau phí (`gains = G_net`); ghi `val_cost_mean_020 > 0` | `test_3`: kiểm tra `val_cost_mean_020_at_best > 0` trên chạy thử 2 epoch |
| **3. Kiểm tra (`conf_test`)** | `build_cost_test_for_seed(..., cost_s=0.0010)` (chỉ tạo SAU khi đóng `best_val` & `ep200`) | Đánh giá đồng thời `he_net_ddof0`, `hedged_oce_y_nom`, `mean_gross_gains`, `mean_gains`, `mean_cost`, `CVaR95`, `turnover`, `immobility_rate` | `test_1`: kiểm chứng bằng tay đường `0` giao dịch, mở giữ nguyên `j=0`, đảo chiều dấu `delta`, và đối chiếu từng bước với NumPy thuần |

---

## 4. Định Nghĩa Tách Bạch `Turnover` vs Tần Suất Giao Dịch vs Quy Mô Lệnh

Để không lặp lại ngộ nhận quy đồng `turnover` thành tần suất giao dịch:
1. **`turnover`:** $\frac{1}{M}\sum_{i=1}^M \sum_{j=0}^{19} |a_{i,j}|$ — tổng khối lượng giao dịch tuyệt đối trung bình trên mỗi quỹ đạo.
2. **`active_steps_gt1e5`:** $\frac{1}{M}\sum_{i=1}^M \sum_{j=0}^{19} \mathbb{I}(|a_{i,j}| \ge 10^{-5})$ — số bước có giao dịch thực chất trên 20 bước.
3. **`active_trade_rate` & `immobility_rate`:**
   - `active_trade_rate` = $\text{active\_steps\_gt1e5} / 20$.
   - **`immobility_rate` (chính):** $1 - \text{active\_trade\_rate} = \text{tỷ lệ ô } (i, j) \text{ có } |a_{i,j}| < 10^{-5}$ (kèm hai ngưỡng chẩn đoán phụ `< 1e-4` và `< 1e-3` để kiểm tra xem MLP `softplus` đứng yên thật dưới `1e-5` hay chỉ giao dịch vi mô `[1e-5, 1e-3)`).
4. **Quy mô giao dịch trung bình khi hoạt động (ghi tách cả hai công thức):**
   - **`mean_trade_size_active_pooled`:** $\frac{\sum_{i,j} |a_{i,j}| \mathbb{I}(|a_{i,j}| \ge 10^{-5})}{\sum_{i,j} \mathbb{I}(|a_{i,j}| \ge 10^{-5})}$ (tỷ số gộp trên toàn bộ bước hoạt động).
   - **`mean_trade_size_active_pathwise`:** trung bình theo quỹ đạo $\frac{1}{|\mathcal{I}_+|} \sum_{i \in \mathcal{I}_+} \frac{\sum_j |a_{i,j}| \mathbb{I}(|a_{i,j}| \ge 10^{-5})}{\sum_j \mathbb{I}(|a_{i,j}| \ge 10^{-5})}$ trên các quỹ đạo $\mathcal{I}_+$ có $\ge 1$ bước hoạt động (kèm `zero_active_path_frac`).
5. **Chỉ số kiểm tra vị thế cực đoan:** `mean_abs_delta`, `max_abs_delta`, `p99_abs_delta`, `frac_steps_abs_delta_gt2` ($|\delta| > 2.0$), `frac_steps_abs_delta_ge45` ($|\delta| \ge 4.5$ gần ngưỡng cắt mềm `±5.0`).
