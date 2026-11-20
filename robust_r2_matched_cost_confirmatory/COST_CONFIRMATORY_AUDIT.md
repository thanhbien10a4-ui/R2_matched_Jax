# Biên Bản Kiểm Toán Đầy Đủ Quá Trình Thực Thi & Phản Biện `@gng5` (`COST_CONFIRMATORY_AUDIT.md`)

> [!NOTE]
> **Báo cáo trung thực cơ chế vận hành `@gng5`:** Cả hai lượt phản biện `@gng5` (tiền nghiệm `gng5_pre_audit` và hậu nghiệm `gng5_post_audit`) đều được thực thi bằng **một phiên mô hình CLI đơn lẻ** (`gemini-4-argon-high` qua `/home/user04/.local/bin/agy --model gemini-4-argon-high --output-format json`) chạy trực tiếp trên thư mục `robust_r2_matched_cost_confirmatory/` của máy chủ `user04` — **không phải hệ thống nhiều tác nhân chạy song song**, dù văn phong đầu ra của mô hình tự đặt tiêu đề theo persona nội bộ của nó.

---

## 1. Tóm Tắt Chuỗi Kiểm Toán Tiền Nghiệm (`Pre-Run`)

- **Nhật ký gốc:** [gng5_pre_audit_raw.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/logs/gng5_pre_audit_raw.json) | [gng5_pre_audit_text.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/logs/gng5_pre_audit_text.md)
- **Phán quyết tiền nghiệm ban đầu của `@gng5`:** `CONDITIONAL_GO`, chỉ ra **4 hạng mục cần vá trước khi khóa `SOURCE_SHA256SUMS.txt` và nộp Slurm** (đã vá và kiểm thử lại toàn bộ 100% trước khi nộp Job `16914`):
  1. **`[BLOCKER / HIGH]` Nghẽn tốc độ vòng lặp Python trong `cost_train.py`:** Chuyển từ vòng lặp `for step in range(156)` sang `@jax.jit` + `jax.lax.scan`, rút thời gian ước tính `N=24` từ `~12.5–18h` xuống thực đo **`03:44:55`** (`< 05:30:00`).
  2. **`[MEDIUM]` Phân loại ngoại lệ & dọn dòng lỗi:** Thay `classify_run_exception` (từng khớp nhầm `"inf"` trong `"meminfo"`) bằng `classify_cost_run_exception` dùng `isinstance(exc, (FloatingPointError, AssertionError))` + ranh giới từ `\b(nan|inf|-inf|\+inf|infinity|non-finite|nonfinite)\b`, đồng thời lọc bỏ dòng dở dang của seed lỗi trước khi chèn dòng `NaN`.
  3. **`[MEDIUM]` Tích lũy khi resume `--seed-offset`:** Cố định thư mục `results/confirmatory_n24` và ghi nhận `reload_checks` + `dataset_hashes` ngay trong `SEED_LEDGER.json` sau mỗi hạt giống.
  4. **`[LOW]` Kiểm tra `assert` phí dương, đầy đủ bảng `R1..R9` & kiểm tra hash tự động trong `.sbatch`:** Bổ sung `assert all(v > 0.0 ...)`, hạt giống CRN đồng bộ cho `cost_power_table.py`, hiển thị đủ 9 kịch bản `R1..R9` trong `COST_CONFIRMATORY_DESIGN_LOCK.md`, và thêm `sha256sum -c SOURCE_SHA256SUMS.txt` vào [slurm_cost_confirmatory_n24.sbatch](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/slurm_cost_confirmatory_n24.sbatch).

---

## 2. Xác Minh Kiểm Thử Đơn Vị & Chạy Thử Cô Lập Trước Slurm

- **Kiểm thử máy chủ & cục bộ ([test_cost_conf_suite.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/tests/test_cost_conf_suite.py)):** Cả **4/4 nhóm kiểm thử đạt (`0` failed, `0` skipped)** trên cả Windows local và Linux `.venv` máy chủ trước khi nộp Slurm:
  1. `test_1_cost_accounting_short_trajectories_and_metrics`: Đường `0` giao dịch (`actions==0, cost==0`), mở giữ `a_0=+0.6` (`cost=0.0006`, không phí tất toán `j=20`), quỹ đạo đảo dấu vs NumPy từng bước (`gross - cost == net`, `c_s=0.0` phục hồi chính xác), và đối chiếu tay `immobility_rate`, `mean_trade_size_active_pooled/pathwise`, `zero_active_path_frac`.
  2. `test_2_prng_isolation_global_state_and_b3_flaw_elimination`: Cách ly `3,061` khóa cũ + `861` khóa mới (`0` xung đột), bảo toàn `np.random.get_state()`, bất biến thứ tự gọi, và trực giao cầu Brownian `|corr_bridge| < 0.015` giữa `Val(5001)` và `Train(5002)` cũng như các luồng smoke `5999`.
  3. `test_3_cost_in_training_val_test_and_2ep_smoke_parity`: Khởi tạo policy bit-identical (`diff == 0.0`), `cost=0.0010` làm thay đổi gradient policy (`grad_diff > 1e-5`) và tạo phí dương xuyên `Train`/`Val` trong vòng chạy 2 epoch đúng `312` bước (`39,936` đường).
  4. `test_4_iut_stats_and_seed_offset_guards`: Quyết định IUT (`PASS`, `FAIL` khi `OCE_020 >= +1e-4`, `INCOMPLETE` khi `NaN`/thiếu), khóa `--seed-offset` không vượt `5024`, bảo toàn `COMPLETED`/`ALGORITHM_FAILURE` khi resume, và không phân loại nhầm `"meminfo"` thành lỗi thuật toán.
- **Chạy thử kỹ thuật cô lập trên seed `5999`:** Hoàn tất sạch `wall = 15.58s` trên `user04`; dữ liệu kiểm tra `[5001..5040]` giữ nguyên chưa tạo trước khi nộp Slurm.

---

## 3. Kiểm Toán Hậu Nghiệm (`Post-Run`) & Đối Soát Phán Quyết `@gng5`

- **Nhật ký gốc `@gng5` hậu nghiệm:** [gng5_post_audit_raw.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/logs/gng5_post_audit_raw.json) | [gng5_post_audit_text.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/logs/gng5_post_audit_text.md)
- **Phán quyết hậu nghiệm của `@gng5`:** **`CONFIRMED_PASS` (`PASS_CONFIRMATORY_IUT`)**, `0` lỗi `CRITICAL`, xác nhận toàn vẹn `T1–T5`:
  - Kiểm chứng lại độc lập trên CSV thô & script riêng:
    - Tái nạp `48/48` `best_val.pkl` trên máy chủ đạt **`max_diff = 0.0`** trên cả `hedged_loss` lẫn `val_cost_mean_020`.
    - Độc lập tính toán lại loại bỏ ngoại lệ `5024` (`n=23`, `p_IUT = 4.464e-10`), loại `5004` (`p_IUT = 4.693e-06`), và loại cả hai (`n=22`, `p_IUT = 5.684e-10`): trùng khớp kết luận `@gng5` rằng phán quyết **không phụ thuộc vào ngoại lệ `5024`**.
    - Khẳng định đúng ghi chú `LOW-01` & `MED-01`: `23/23` tệp trong `SOURCE_SHA256SUMS.txt` và `22/22` khóa trong JSON khớp `100%`; các đánh đổi bất lợi (`fixed_015`, `5004`, `5013`, và `5–8/24` seed tăng `cost`/`turnover`) được công bố đầy đủ trong Mục 3–5 của [COST_CONFIRMATORY_FINAL_REPORT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/COST_CONFIRMATORY_FINAL_REPORT.md).

---

## 4. Xác Nhận Bảo Toàn Môi Trường & Ranh Giới Cuối

- **`PAUSE_COORDINATOR`:** Còn nguyên `32 bytes` (`Nov 16 10:17`) tại `/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR` — không bị tạo, sửa, hay xóa.
- **Tập niêm phong:** `HOLD_EVAL`, `[4001..4040]`, và `[5025..5040]` không bị mở hay sinh thêm dữ liệu; các thư mục `robust_r3_*`, `robust_r2_matched_confirmatory/`, `b3_cost_confirmatory_10seeds/`, `core/`, `scripts/` hoàn toàn không bị chỉnh sửa.
- **Tác vụ khác & GitHub:** Không can thiệp tác vụ Slurm khác (`squeue -u user04` rỗng khi kết thúc); không đẩy mã hay dữ liệu lên GitHub.
