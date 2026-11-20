# META
{
  "conversation_id": "4d45219f-1066-4a7f-aa51-d687e49e6010",
  "status": "SUCCESS",
  "duration_seconds": 416.495525744,
  "num_turns": 1,
  "usage": {
    "input_tokens": 281945,
    "output_tokens": 33472,
    "thinking_tokens": 22371,
    "cache_read_tokens": 481989,
    "total_tokens": 315417
  }
}

# RESPONSE

# BIÊN BẢN GIÁM ĐỐC THẨM ĐỘC LẬP (`@gng5` — READ-ONLY PRE-RUN DESIGN AUDIT)

- **Định danh phiên kiểm toán:** `@gng5` (chạy bằng một phiên mô hình đơn lẻ Gemini 4 Argon High, chế độ **CHỈ ĐỌC / READ-ONLY 100%**).
- **Đối tượng kiểm toán:** Hồ sơ thiết kế, mã nguồn, phân tích cỡ mẫu và bộ kiểm thử xác nhận độc lập tại [robust_r2_matched_confirmatory](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory).
- **Xác nhận tuân thủ ràng buộc cứng:**
  - **Không** tạo, sửa, ghi đè hay xóa bất kỳ tệp nào.
  - **Không** nộp Slurm job (`sbatch`), **không** huấn luyện hay mô phỏng tập kiểm tra `[4001..4040]`, **không** mở `HOLD_EVAL`.
  - **Không** chạm vào `PAUSE_COORDINATOR` hay job đang chờ `16909`.

---

## 1. Phán Quyết Thiết Kế

> **PHÁN QUYẾT: `CONDITIONAL PASS` (ĐẠT CÓ ĐIỀU KIỆN — CHO PHÉP CHỐT THIẾT KẾ SAU KHI KHẮC PHỤC 6 ĐIỂM KỸ THUẬT & TÀI LIỆU Ở MỤC 3 TRƯỚC KHI KHÓA SHA-256 VÀ CHẠY XÁC NHẬN)**
> - **Chỉ số Alignment Audit Score (AAS):** `93.5 / 100` (T1 Soundness: Đạt; T2 Counterexample Resistance: Đạt có điều kiện; T3 Anti-Sycophancy: Đạt; T4 Symbolic Reproducibility: Đạt; T5 Merkle/PRNG Isolation: Đạt tuyệt đối).

---

## 2. Các Điểm Mạnh Đã Khóa Chặt

1. **Đính chính trung thực, minh bạch các sai sót diễn đạt của giai đoạn phát triển ([DEVELOPMENT_ERRATA.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DEVELOPMENT_ERRATA.md)):**
   - **Mục 1a ([DEVELOPMENT_ERRATA.md#L8-L17](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DEVELOPMENT_ERRATA.md#L8-L17)):** Đối chiếu khớp 100% với 24 dòng dữ liệu của [LR_OSCILLATION_181_200.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/results/run/LR_OSCILLATION_181_200.csv#L1-L25) (`DECAY`: `0.00010673–0.00026933` với `A_DECAY` `0.00010673–0.00018394`, `R2_DECAY` `0.00013286–0.00026933`; `FIXED`: `0.00057222–0.00156031` với `A_FIXED` `0.00075640–0.00156031`, `R2_FIXED` `0.00057222–0.00084475`).
   - **Mục 1b ([DEVELOPMENT_ERRATA.md#L20-L30](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DEVELOPMENT_ERRATA.md#L20-L30)):** Đối chiếu khớp tuyệt đối với [LR_DEV_EVAL_FULL_PRECISION.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/results/run/LR_DEV_EVAL_FULL_PRECISION.csv#L6-L101), tách bạch rõ `+0.000236` (`A`) và `+0.000426` (`R2`) tại `fixed_030` là `hedged_oce_y_nom` (`OCE`), đồng thời công bố đầy đủ mức tăng `he_net_ddof0` (`HE`): `+0.000993` ở `A` và `+0.000826` ở `R2` (trong khi hiệu ghép cặp `R2_DECAY - A_DECAY` trên `HE` vẫn đạt `-0.004812`).
   - **Mục 1c, 1d, 1e ([DEVELOPMENT_ERRATA.md#L33-L68](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DEVELOPMENT_ERRATA.md#L33-L68)):** Đính chính rõ bản chất `delete20_jk_se_osc` là jackknife trên 20 nhóm đường mô phỏng (không phải 20 checkpoint); đính chính đường dẫn thực và khoảng trống quan sát `20:02–22:32 +07` của `PAUSE_COORDINATOR`; tách bạch mốc tham chiếu kỹ thuật `0.5ε = 0.00005` và cấm gán xác suất nhân quả cảm tính.
2. **Tách lớp nghiêm ngặt dữ liệu phát triển & Định nghĩa chỉ số chuẩn xác ([R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md), [dev_strata_summary.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/dev_strata_summary.json)):**
   - Giữ tách lớp hoàn toàn giữa `BC_old_pilot`, `BC_new_dev`, `DIAG_val_diag_a/c_20k`, `DIAG_val_diag_b0_5k` và `LR_STAB` (`FIXED` vs `DECAY`, `best_val` vs `ep200`), không gộp lẫn các thiết kế khác ngân sách hoặc khác lịch LR.
   - Phân biệt rạch ròi 3 khái niệm: `R3` (minimax mượt có ràng buộc — đã dừng), `R2_matched_128` (kỳ vọng đều `1/4` trên 4 kịch bản với 4 cặp `(y_k, y0_k)` tách rời), và `Cosine LR Decay` (áp dụng đối xứng cho cả `A_DECAY` và `R2_DECAY`).
   - Khóa đúng định nghĩa `HE = np.std(gains, ddof=0)` (độ lệch chuẩn tổng thể quanh trung bình của `gains`, không phải RMSE quanh `0`) và `hedged_oce_y_nom` (đánh giá tại `y_nom` đóng băng từ huấn luyện: `y[0]` cho `A_DECAY`, `y[1]` cho `R2_DECAY`, không tối ưu lại `y` trên tập test), khớp với [r3_metrics.py#L44-L61](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_dev/r3_metrics.py#L44-L61).
   - Nâng kích thước `conf_test` từ `5,000` lên `20,000` đường/kịch bản để triệt tiêu phương sai lấy mẫu Monte Carlo nội hạt giống, và khóa rõ biên giới hợp lệ `cost_s = 0.0`.
3. **Cách ly PRNG mật mã học tuyệt đối & Bảo mật tập kiểm tra ([conf_keys.py](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_keys.py), [DATA_AND_SEED_PLAN.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DATA_AND_SEED_PLAN.json)):**
   - Kiểm tra toàn bộ `2,167` khóa lịch sử (`2,092` khóa đến Selection Diagnostics + `75` khóa từ LR Stability `[3401..3403]`) đối chiếu với `861` khóa mới (`41` seed = `3999` smoke + `[4001..4040]` × `21` khóa/seed dưới namespace `R2_MATCHED_CONF_V1`): **`0` va chạm nội bộ, `0` va chạm lịch sử, `0` lỗi tràn `u32`**.
   - Tập `conf_test` chỉ được khởi tạo sau khi quá trình huấn luyện và chọn `best_val` của cả hai nhánh đã đóng niêm phong ([conf_run.py#L260-L262](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py#L260-L262)). `HOLD_EVAL` và `[4001..4040]` hoàn toàn chưa được mở hay mô phỏng.
4. **Độ chính xác toán học của hàm phân phối Student-$t$ và Noncentral-$t$ ([scripts/sample_size_and_power.py](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/scripts/sample_size_and_power.py#L31-L160)):**
   - Thuật toán phân số liên tục Lentz `_betacf`, hàm beta không đầy đủ `_betainc`, `t_cdf`, `t_ppf` (tìm kiếm nhị phân 85 bước, sai số $< 10^{-23}$) và tích phân Simpson 600 đoạn cho `nct_power_lower_tail` hoàn toàn chuẩn xác về mặt giải tích và khớp với mô phỏng Monte Carlo nhị biến `B = 60,000` trong [SAMPLE_SIZE_SENSITIVITY.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/SAMPLE_SIZE_SENSITIVITY.csv) tới $\pm 0.001$.
5. **Dự toán tài nguyên bám sát thực nghiệm Job `16908` ([RESOURCE_ESTIMATE.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/RESOURCE_ESTIMATE.md), [slurm_confirmatory_template.sbatch](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/slurm_confirmatory_template.sbatch)):**
   - Thời gian trung bình từ [slurm_lr_run_16908.log](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/logs/slurm_lr_run_16908.log#L16-L31) được tính chính xác (`A_DECAY`: `(262.61+263.49+263.96)/3 = 263.35s`; `R2_DECAY`: `(302.38+300.02+301.23)/3 = 301.21s`; tổng `564.56s/seed` + rollout `20k` $\approx 9.65\text{ phút/seed}$), hoàn toàn không ngoại suy GPU.

---

## 3. Danh Sách Lỗi, Lỗ Hổng Kỹ Thuật, Mâu Thuẫn Số Liệu / Điểm Yếu Thống Kê & Cách Khắc Phục Bắt Buộc

Trước khi khóa mã SHA-256 và chạy xác nhận, cần khắc phục **6 điểm cụ thể** dưới đây:

### (3.1) Mâu thuẫn mô tả công thức nội bộ của `delete20_jk_se_osc` trong bản đính chính
- **Vị trí:** [DEVELOPMENT_ERRATA.md#L36-L37](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DEVELOPMENT_ERRATA.md#L36-L37) đối chiếu với [run_lr_stability.py#L101-L120](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/run_lr_stability.py#L101-L120) và [run_lr_stability.py#L219](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/run_lr_stability.py#L219).
- **Chi tiết lỗi:** Mặc dù kết luận phương pháp luận của Mục 1c (jackknife trên 20 nhóm đường `20 × 1,000` chứ không phải 20 checkpoint `181..200`) là hoàn toàn đúng, dòng 36–37 của `DEVELOPMENT_ERRATA.md` lại ghi tên biến đầu vào là `gains_20ep_x_paths` và viết rằng hàm tính lại công thức phi tuyến `mean_gain + (1/lambda) * log(mean(exp(-lambda * r)))` trên 19,000 đường còn lại. Trên thực tế, trong `run_lr_stability.py:101-120, 219`, tham số đầu vào là `oce_vals_by_ep` chứa mảng `roll["hedged_loss"]` (`-u(gains, y_nom)` theo từng đường tại `y_nom` đóng băng của mỗi epoch) và dòng 116 chỉ tính trung bình cộng tuyến tính `ep_means = np.mean(arr[:, mask], axis=1)`.
- **Cách khắc phục bắt buộc:** Sửa lại câu mô tả tại `DEVELOPMENT_ERRATA.md:36-37` cho khớp chính xác với tên biến `oce_vals_by_ep` (`roll["hedged_loss"]`) và phép lấy trung bình `np.mean(arr[:, mask], axis=1)` trong `run_lr_stability.py`.

### (3.2) Ba điểm lỗi/hạn chế trong hàm kiểm định phi tham số `wilcoxon_lower_tail_p`
- **Vị trí:** [conf_stats.py#L63-L114](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_stats.py#L63-L114).
- **Chi tiết lỗi:**
  1. **Nhầm thuật ngữ quy ước xử lý điểm `0` (dòng 68):** Docstring ghi `"Drops exact zeros per Pratt-zero-excluded convention"`, nhưng dòng 71 (`nz = arr[arr != 0.0]`) loại bỏ các phần tử bằng `0` *trước khi xếp hạng* — đây là quy ước **Wilcoxon zero-exclusion (`wilcox`)**, còn quy ước **Pratt (1959)** giữ các điểm `0` khi gán hạng rồi mới bỏ hạng của điểm `0`.
  2. **Giới hạn `exact_max_m: int = 16` không cần thiết khiến $N = 24$ bị đẩy sang xấp xỉ chuẩn (dòng 64, 80–98):** Chú thích dòng 81 ghi `# Exact enumeration over 2^m sign assignments`, nhưng thuật toán ở dòng 83–91 thực chất là quy hoạch động (0-1 knapsack dynamic programming) có độ phức tạp chỉ $O(m^3)$ (`max_sum = m(m+1)/2`). Với $m = 24$, `max_sum = 300`, vòng lặp chỉ thực hiện `24 × 300 = 7,200` phép cộng số nguyên (tốn `< 0.5 ms`), và với $m = 40$ chỉ thực hiện `32,800` phép cộng. Việc chặn `exact_max_m = 16` làm mất tính chính xác tuyệt đối ở cỡ mẫu chính $N = 24$.
  3. **Dấu hiệu chỉnh liên tục (`cc`) trong nhánh xấp xỉ chuẩn cho kiểm định một phía đuôi trái (dòng 105–107):** Dòng 105 viết `cc = 0.5 if w_plus < mean_w else (-0.5 if w_plus > mean_w else 0.0)` (hiệu chỉnh hướng về trung bình dùng cho kiểm định hai phía). Tuy nhiên, để xấp xỉ hàm phân phối tích lũy đuôi trái một phía $P(W^+ \le w_{\text{obs}}^+)$ (bao gồm cả khối lượng xác suất tại $W^+ = w_{\text{obs}}^+$), cận tích phân chuẩn luôn phải là $w_{\text{obs}}^+ + 0.5$ (`cc = +0.5`) bất kể $w_{\text{obs}}^+$ nằm bên trái hay bên phải $\mu_W$.
- **Cách khắc phục bắt buộc:** Trong `conf_stats.py`, đổi docstring dòng 68 thành `"Wilcoxon zero-excluded convention"`, nâng `exact_max_m: int = 50` để $N \in [16..40]$ luôn tính $p$-value chính xác bằng quy hoạch động khi không có đồng hạng, và sửa dòng 105 thành `cc = 0.5` cho hàm CDF đuôi trái.

### (3.3) Thiếu lưu vết SHA-256 của mã nguồn (cục bộ + thượng nguồn) và thông tin môi trường trong `CONFIRMATORY_STATS_SUMMARY.json`
- **Vị trí:** [conf_run.py#L150-L156](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py#L150-L156), [conf_run.py#L381-L395](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py#L381-L395), và [slurm_confirmatory_template.sbatch#L46](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/slurm_confirmatory_template.sbatch#L46).
- **Chi tiết lỗ hổng:**
  1. `conf_run.py` nhập và thực thi trực tiếp hàm huấn luyện, mô hình và chỉ số từ các thư mục thượng nguồn (`lr_train.py`, `lr_schedule.py`, `lr_keys.py`, `bc_models.py`, `r3_models.py`, `r3_metrics.py`, `r3_train.py`, `r3_keys.py`) cũng như `scripts/sample_size_and_power.py` và `tests/test_conf_suite.py`, nhưng `tracked` trong `run_dry_run_check()` (`conf_run.py:150-156`) và lệnh `sha256sum` trong `slurm_confirmatory_template.sbatch:46` chưa băm các tệp thượng nguồn này.
  2. Khi chạy `execute_confirmatory`, tệp kết quả chính `CONFIRMATORY_STATS_SUMMARY.json` (`conf_run.py:381-395`) không lưu `source_sha256`, `locked_spec`, `prng_isolation_audit` và thông tin môi trường (`python`, `jax_version`, `devices`, `platform`) như đã làm trong `LR_DATA_AND_SEED_MANIFEST.json` của giai đoạn trước.
- **Cách khắc phục bắt buộc:** Mở rộng danh sách `tracked` trong `conf_run.py` và `slurm_confirmatory_template.sbatch` để băm cả các tệp thượng nguồn được import, đồng thời nhúng `source_sha256`, `spec`, `isolation`, và `environment` trực tiếp vào `CONFIRMATORY_STATS_SUMMARY.json`.

### (3.4) Lỗ hổng chịu lỗi khi gặp `NaN`/timeout giữa chừng và bất đồng bộ với phương án chia lát seed trong `RESOURCE_ESTIMATE.md`
- **Vị trí:** [conf_run.py#L198-L200](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py#L198-L200), [conf_run.py#L231-L363](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py#L231-L363), [lr_train.py#L210](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r3_lr_stability/lr_train.py#L210) và [RESOURCE_ESTIMATE.md#L48](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/RESOURCE_ESTIMATE.md#L48).
- **Chi tiết lỗ hổng:**
  1. Trong `conf_run.py:231-363`, các bảng `RESULTS_CONFIRMATORY_FULL_PRECISION.csv` và `PAIRED_CONFIRMATORY.csv` chỉ được ghi xuống đĩa *sau khi* hoàn tất toàn bộ vòng lặp `for s in seeds:`. Nếu một seed bất kỳ bị `NaN`/`Inf` trong quá trình huấn luyện, `verify_finite_tree(carry[0])` (`lr_train.py:210`) sẽ ném `RuntimeError` làm sập toàn bộ tiến trình ngay lập tức; kết quả là không có tệp CSV hay JSON nào được xuất ra, và cổng sổ cái `INCOMPLETE_OR_FAILED_AUDIT` trong `conf_stats.py` không bao giờ được gọi.
  2. `RESOURCE_ESTIMATE.md:48` viết rằng có thể tách thành các lát seed rời rạc (`2` job $\times$ `12` seed hoặc `3` job $\times$ `8` seed) rồi hợp nhất `PAIRED_CONFIRMATORY.csv` để chạy `conf_stats.py`. Tuy nhiên, `conf_run.py:200` luôn cắt từ đầu danh sách `seeds = list(CONF_RESERVED_SEEDS[:n_seeds])` (không có tham số chọn lát seed hoặc chế độ đánh giá lại từ checkpoint), và `conf_stats.py` chưa có hàm `main()` CLI để đọc `PAIRED_CONFIRMATORY.csv` độc lập.
- **Cách khắc phục bắt buộc:**
  - Trong `conf_run.py`, bọc khối huấn luyện/đánh giá từng seed bằng cơ chế ghi nhận lỗi (nếu một seed ném ngoại lệ ` non-finite`, ghi dòng `NaN` cho seed đó vào `paired_rows` và flush CSV từng seed để `analyze_confirmatory_pairs` luôn xuất được biên bản `INCOMPLETE_OR_FAILED_AUDIT`).
  - Bổ sung CLI entrypoint cho `conf_stats.py` (hoặc thêm tùy chọn `--seed-offset` / `--analyze-only` trong `conf_run.py`) để khớp với tuyên bố tại `RESOURCE_ESTIMATE.md:48`.

### (3.5) Thiếu thống kê tổng hợp giá trị tuyệt đối từng nhánh (`A_DECAY`, `R2_DECAY`) và `delta_oce_y_scen_R2_minus_A` trong `CONFIRMATORY_STATS_SUMMARY.json`
- **Vị trí:** [conf_stats.py#L336-L370](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_stats.py#L336-L370) đối chiếu với [R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md#L107](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md#L107) và [conf_run.py#L328-L343](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py#L328-L343).
- **Chi tiết:** Mục 6.2 của bản thiết kế (`R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md:107`) cam kết *"Mọi phân tích phải báo cáo giá trị tuyệt đối từng nhánh song hành với chênh lệch ghép cặp"* để theo dõi hiện tượng suy giảm nhẹ tại `fixed_030` dưới `DECAY`. Mặc dù `PAIRED_CONFIRMATORY.csv` đã lưu các cột tuyệt đối (`A_he_net_ddof0`, `R2_he_net_ddof0`, `A_hedged_oce_y_nom`, `R2_hedged_oce_y_nom`, `A_cvar95`, `R2_cvar95`, `A_turnover`, `R2_turnover`) và `delta_oce_y_scen_R2_minus_A`, hàm `analyze_confirmatory_pairs` (`conf_stats.py:341-346`) mới chỉ tóm tắt 4 cột hiệu `delta_*`.
- **Cách khắc phục bắt buộc:** Bổ sung các khóa tuyệt đối từng nhánh và `delta_oce_y_scen_R2_minus_A` vào `secondary_summaries` trong `conf_stats.py:341-370`.

### (3.6) Quy tắc khóa cứng một giá trị `N` duy nhất trước khi chạy (chống Optional Stopping) và kiểm tra sạch thư mục `checkpoints/`
- **Vị trí:** [R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md#L86-L100](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md#L86-L100) và [tests/test_conf_suite.py#L58-L61](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/tests/test_conf_suite.py#L58-L61).
- **Chi tiết:**
  1. Tài liệu hiện đang để mở hai lựa chọn `N = 24` (chính) và `N = 16` (dự phòng). Trước khi chuyển từ `DRAFT` sang `DESIGN LOCK`, bắt buộc phải chốt **một con số `N` cố định duy nhất** và ghi rõ quy tắc **cấm dừng tùy ý / mở rộng tuần tự không hiệu chỉnh alpha** (tuyệt đối không được chạy `N = 16`, nhìn $p$-value rồi mới chạy thêm 8 seed lên `N = 24`, vì hành vi đó làm tỷ lệ sai lầm loại I thực tế vượt $\alpha = 0.025$).
  2. Trong `tests/test_conf_suite.py:58-61`, bài kiểm thử mới chỉ quét thư mục `results/` để tìm `confirmatory_*`, cần bổ sung quét cả thư mục `checkpoints/` để đảm bảo tính đối xứng.
  3. *(Ghi chú nhỏ về trình bày)*: Bảng mục 5 của `R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md:86-95` liệt kê 7 kịch bản (`R1–R4, R6–R8`), bỏ qua dòng `R5_PosCorr_Conservative` đang có trong `SAMPLE_SIZE_SENSITIVITY.csv:26-31` dù chú thích dòng 96 vẫn nhắc tới `R1–R5`; nên thêm dòng `R5` vào bảng Markdown cho đầy đủ 8/8 kịch bản.

---

## 4. Đánh Giá Riêng: Lựa Chọn Cỡ Mẫu (`N = 24` vs `N = 16`) & Tính Hợp Lệ Của Berger IUT ở $\alpha = 0.025$

### 4.1. Tính hợp lệ toán học và thống kê của Berger Intersection-Union Test (IUT) ở $\alpha = 0.025$ một phía
1. **Kiểm soát chặt FWER $\le 0.025$ không cần hiệu chỉnh bội (Berger, 1982):**
   - Thí nghiệm xác nhận đặt hai giả thuyết đồng chính (co-primary endpoints):
     - $H_{0a}: \mu(\Delta \text{HE}_{030}) \ge 0$ đối chọi $H_{1a}: \mu(\Delta \text{HE}_{030}) < 0$ (Superiority tại $\sigma = 0.30$).
     - $H_{0b}: \mu(\Delta \text{OCE}_{020}) \ge +0.00010$ đối chọi $H_{1b}: \mu(\Delta \text{OCE}_{020}) < +0.00010$ (Non-inferiority tại $\sigma = 0.20$).
   - Giả thuyết không tổng quát là hợp $H_0 = H_{0a} \cup H_{0b}$ và miền bác bỏ tổng quát là giao $R = R_a \cap R_b$. Với mọi tham số $\theta \in H_0$, ít nhất một trong hai giả thuyết thành phần đúng ($i \in \{a, b\}$), do đó:
     $$P_\theta(R_a \cap R_b) \le P_\theta(R_i) \le \alpha = 0.025.$$
   - Bất đẳng thức này đúng với **mọi cấu trúc tương quan bất kỳ** giữa hai thống kê kiểm định $T_a$ và $T_b$. Do đó, việc `conf_test` dùng chung chuỗi nhiễu chuẩn `conf_test_shared_dW` giữa 6 kịch bản trong cùng một seed (tạo tương quan chéo giữa `fixed_030` và `fixed_020`) **hoàn toàn không làm suy giảm tính hợp lệ của Berger IUT**.
2. **Tại sao $\alpha = 0.025$ một phía là lựa chọn chuẩn xác nhất:**
   - Trong kiểm định không thua kém (non-inferiority) và đồng giả thuyết xác nhận, việc chọn $\alpha = 0.025$ một phía (thay vì $\alpha = 0.05$ một phía) tương đương chính xác 1-1 với tiêu chuẩn **cận trên của khoảng tin cậy $95\%$ hai phía nằm hoàn toàn dưới ngưỡng** ($\text{UCL}_{95\%}(\Delta \text{HE}_{030}) < 0$ và $\text{UCL}_{95\%}(\Delta \text{OCE}_{020}) < +0.00010$).
   - Hơn nữa, vì nhánh $H_{1a}$ có kích thước hiệu ứng chuẩn hóa rất lớn ($|\mu_A| / \text{sd}_A \approx 1.2\dots 6.0$, lực kiểm định biên $P(R_a) \approx 0.9999\dots 1.0000$ tại $N = 24$ trên `R1–R7`), độ bảo thủ thường gặp của IUT khi cả hai nhánh đều có lực trung bình gần như triệt tiêu: xác suất sai lầm loại I thực tế tại biên của $H_{0b}$ đạt sát đúng $\alpha = 0.025$, và lực đồng thời $P(R_a \cap R_b)$ gần như bằng đúng lực biên của $H_{1b}$.

### 4.2. So sánh định lượng `N = 24` vs `N = 16` và khuyến nghị chốt
1. **Tại sao không được tin tưởng tuyệt đối vào $\text{sd}_B = 0.000047$ từ 3 seed phát triển:**
   - Ở `LR_STABILITY` (`DECAY`), cỡ mẫu $n = 3$ chỉ có $\text{df} = 2$ bậc tự do; cận trên khoảng tin cậy $95\%$ một phía của độ lệch chuẩn tổng thể $\sigma_B$ dưới phân phối chuẩn lên tới $\hat{s}_B \sqrt{2 / \chi^2_{0.05, 2}} \approx 4.41 \times \hat{s}_B \approx 0.000207$. Dù việc nâng tập test từ `5,000` lên `20,000` đường giúp giảm phương sai lấy mẫu Monte Carlo nội seed, độ lệch chuẩn thực giữa các seed $\sigma_B$ hoàn toàn có thể rơi vào vùng `0.000085 – 0.000110` (kịch bản stress `R6`, `R7`).
2. **Sự khác biệt quyết định tại vùng stress `R6` và `R7`:**
   - Ở các kịch bản thuận lợi đến bảo thủ vừa (`R1–R5`, $\sigma_B \le 0.000075$, $\mu_B \le +0.00002$), cả `N = 16` (lực chung $\ge 97.8\%$) và `N = 24` (lực chung $\ge 99.9\%$) đều đủ lực.
   - Tuy nhiên, khi chuyển sang **kịch bản stress `R6`** ($\mu_B = +0.00003$, còn biên $0.70\text{ bp}$, $\sigma_B = 0.000085$) và **`R7`** ($\mu_B = +0.00002$, phương sai nặng $\sigma_B = 0.000110$ tiệm cận `FIXED` trên `20k` đường):
     - **`N = 16` bị hụt lực dưới ngưỡng 90% và 80%:** đạt `86.9%` ở `R6` và tụt xuống **`77.3%` ở `R7`** (`< 80%` — ngưỡng tối thiểu chuẩn của thí nghiệm xác nhận).
     - **`N = 24` giữ vững lực thống kê trên 90%:** đạt **`97.3%` ở `R6`** và **`92.7%` ở `R7`** (do yêu cầu cỡ mẫu để đạt lực $90\%$ ở $H_{1b}$ lần lượt là $N = 18$ cho `R6` và $N = 22$ cho `R7`).
   - Ngoài ra, so với `N = 16` ($\text{df} = 15, t_{0.975, 15} = 2.131$), cỡ mẫu `N = 24` ($\text{df} = 23, t_{0.975, 23} = 2.069$) giúp thu hẹp bán kính khoảng tin cậy $95\%$ tới **$26.2\%$**, cải thiện độ chuẩn tiệm cận theo Định lý Giới hạn Trung tâm trước khả năng phân phối $\Delta \text{OCE}_{020}$ bị lệch (skewness) do chọn `argmin` trên tập validation, và giảm hẳn nguy cơ lật phán quyết khi kiểm tra Leave-One-Out (LOO).
   - *(Lưu ý trung thực về kịch bản cực đoan `R8`)*: Nếu đồng thời xảy ra $\mu_B = +0.00004$ (mất $40\%$ biên $\varepsilon$) và $\sigma_B = 0.000120$ kèm $\rho = -0.3$, thì `N = 24` chỉ đạt lực `62.0%` (`N = 16` chỉ đạt `36.8%`, và cần tới `N = 44` mới đạt `90%`). Đây là rủi ro biên đã được báo cáo minh bạch trong mục 5 của bản dự thảo.
3. **Kết luận về cỡ mẫu:**
   - Với chi phí tính toán chỉ `~3.86 giờ` trên `8 CPU` (`0 GPU`), nằm gọn trong một job Slurm `--time=05:30:00`, **khuyến nghị khóa cứng `N = 24` (`[4001..4024]`) làm cỡ mẫu xác nhận duy nhất**, không sử dụng `N = 16` trừ khi hạ tầng Slurm áp đặt giới hạn cứng không thể vượt quá `3.5 giờ` và không thể chạy chia lát.

---
*Chữ ký kiểm toán:* `Gemini-4-Argon-High Sovereign Epistemic Gate (@gng5 Single-Session Native Read-Only Audit)`
