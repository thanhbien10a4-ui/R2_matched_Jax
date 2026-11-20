# Kiểm Toán Độc Lập Hạt Giống & Khắc Phục Lỗi Chia Sẻ Nhiễu Kiểu B3 (`SEED_INDEPENDENCE_AUDIT.md`)

---

## 1. Tóm Tắt Vấn Đề Ở B3 & Thiết Kế Khắc Phục Từ Gốc

Theo [ERRATA_AND_EPISTEMIC_CORRECTIONS.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/b3_cost_confirmatory_10seeds/ERRATA_AND_EPISTEMIC_CORRECTIONS.md), thí nghiệm B3 trước đây (`seeds [701..710]`) tạo dữ liệu bằng `seed` (train `50,000` đường) và `seed + 1` (val `5,000` đường) thông qua hàm gọi `np.random.seed(c.seed)` toàn cục. Hệ quả là đối với 9 hạt giống liên tiếp (`701..709`), tập `Val(s)` trùng khít từng bit nhiễu `dW` với `5,000` đường đầu tiên của `Train(s + 1)`.

**Thiết kế mới trong [cost_keys.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/cost_keys.py) và [cost_world_and_metrics.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/cost_world_and_metrics.py) triệt tiêu vấn đề này trực tiếp ở lớp khóa phân cấp:**

1. **Dẫn xuất khóa 31-bit bằng SHA-256 theo tuple đầy đủ:**
   $$\text{key} = \text{SHA256}(\texttt{"R2\_COST10\_CONF\_V1"} \mid \text{stage} \mid \text{seed\_label} \mid \text{purpose} \mid \text{scenario} \mid \text{component}) \ \& \ \texttt{0x7FFFFFFF}$$
   (nếu bằng `0` thì gán `1`). Mọi khóa đều thỏa `key > 0` và `key + 54321 < 2^32`.
2. **Bảo toàn trạng thái RNG toàn cục `np.random`:**
   - Do `robust_world.py` dùng `np.random.seed(c.seed)`, trình bọc `build_cost_scenario_world` **lưu (`np.random.get_state()`) trước và khôi phục (`np.random.set_state(...)`) ngay sau** mỗi lần gọi `simulate_robust_world`. Nhờ vậy thứ tự sinh batch hay mã khác tuyệt đối không làm lệch luồng ngẫu nhiên.
   - Hoán vị minibatch trong [cost_train.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/cost_train.py) dùng bộ sinh cục bộ `np.random.default_rng([perm_key, ep])`, hoàn toàn tách biệt khỏi `np.random` toàn cục.

---

## 2. Cây Phân Chia Nguồn Ngẫu Nhiên & Cấu Trúc Ghép Cặp

Trong mỗi hạt giống `s` (chỉ có 24 hạt giống chính `[5001..5024]`, seed khói `5999`, và dải dự phòng niêm phong `[5025..5040]`):

- **Dùng chung có chủ đích TRONG CÙNG MỘT HẠT GIỐNG `s` (để ghép cặp công bằng `A_DECAY` vs `R2_DECAY`):**
  1. `init_policy`: cùng khóa JAX `PRNGKey` khởi tạo 6,762 tham số policy cho cả `A_DECAY` (`k_oce=1`) và `R2_DECAY` (`k_oce=4`), sai khác bit ban đầu `= 0.0`.
  2. `train["fixed_020"]` (`5,000` đường) & `train_perm["fixed_020"]`: đóng góp `1/4` minibatch mỗi bước của cả `A_DECAY` và `R2_DECAY` (`3/4` còn lại của `A_DECAY` dùng `fixed_020_sub0/2/3`; `3/4` còn lại của `R2_DECAY` dùng `fixed_015`, `fixed_025`, `rs_015_025`).
  3. `val["fixed_020"]` (`5,000` đường): dùng để đánh giá nhánh `A_DECAY` và là 1 trong 4 kịch bản trung bình đều để chọn `best_val` của `R2_DECAY`.
  4. `conf_test_shared_dW` (`20,000` đường, chỉ sinh **SAU KHI** `best_val.pkl` và `final_with_opt.pkl` đã chốt): dùng chung cho cả `A_DECAY` và `R2_DECAY` trên 6 kịch bản đánh giá trong cùng hạt giống `s`.
- **Lưu ý thống kê về `conf_test_shared_dW`:** Việc dùng chung `conf_test_shared_dW` giữa 6 kịch bản *bên trong cùng một hạt giống `s`* giúp giảm phương sai ghép cặp trong từng kịch bản nhưng tạo tương quan giữa 6 kịch bản trong cùng một hạt giống; mọi suy luận xác nhận chính đều tách riêng từng kịch bản và lấy đơn vị độc lập là **24 hạt giống khác nhau `[5001..5024]`**.
- **Cách ly tuyệt đối GIỮA các tập và GIỮA các hạt giống:**
  - `Train` $\cap$ `Val` $\cap$ `conf_test` có khóa rời nhau hoàn toàn.
  - Mọi cặp hạt giống `s1 != s2` có khóa `init`, `train`, `val`, `perm`, `conf_test` rời nhau hoàn toàn (không bao giờ còn hiện tượng `Val(s)` trùng `Train(s+1)`).

---

## 3. Kết Quả Kiểm Toán Xung Đột Toàn Kho & Ý Nghĩa Thực Chất

Kiểm tra trực tiếp trên toàn bộ bảng khóa lịch sử bởi `cost_keys.py::verify_cost_key_isolation()`:
- **Tổng số nguyên lịch sử đã rà:** **`3,061`** khóa/nhãn (bao gồm toàn bộ `2,167` khóa R3/BC/DIAG/LR/HOLD_EVAL, toàn bộ `492` khóa `R2_MATCHED_CONF_V1` `[3999, 4001..4040]`, các khóa/offset B3 `[701..710, 1700..1719]`, `rough_*`, và chuỗi thăm dò v3 `[42..46, 501..603]`).
- **Số nguyên mới được kiểm:** `861` (`41` hạt giống `[5999, 5001..5040]` × `20` khóa/seed + `1` nhãn).
- **Xung đột nội bộ / xung đột lịch sử / xung đột `Val(s) vs Train(s+1)` / xung đột `+54321`:** **`0 / 0 / 0 / 0`**.

> [!NOTE]
> **Giới hạn suy diễn của kiểm toán khóa băm:** Việc toàn bộ khóa số nguyên 31-bit hoàn toàn rời nhau **chỉ chứng minh không có tái sử dụng hay trùng lặp hạt giống PRNG**; nó **không tự thân là chứng minh toán học cho "độc lập tuyệt đối 100%" hay "sai lệch 0%"** giữa hai luồng MT19937 khác hạt giống. Tính trực giao thực nghiệm giữa `Val(5001)` vs `Train(5002)` và giữa các tập trên đường Brownian-bridge vuông góc với $S_T$ (do `simulate_robust_world` sắp thứ tự quỹ đạo theo $S_T$) được kiểm trực tiếp trong [test_cost_conf_suite.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/tests/test_cost_conf_suite.py) (`|corr_bridge| < 0.015`, trái ngược với mức trùng bit của B3).
