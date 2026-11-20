# META
{
  "conversation_id": "52ee7f5f-526d-4754-9c31-627c022718da",
  "status": "SUCCESS",
  "duration_seconds": 297.589293522,
  "num_turns": 1,
  "usage": {
    "input_tokens": 225536,
    "output_tokens": 25272,
    "thinking_tokens": 6416,
    "cache_read_tokens": 1888963,
    "total_tokens": 250808
  }
}

# RESPONSE

# BIÊN BẢN KIỂM TOÁN HẬU NGHIỆM TỐI CAO (`@gng5` POST-HOC AUDIT — SLURM JOB `16911`)

- **Định danh kiểm toán:** `@gng5` — *Sovereign Epistemic Apex Gate (Chuẩn T1–T5, Chế độ Chỉ Đọc Nghiêm Ngặt / Strictly Read-Only)*
- **Đối tượng kiểm toán:** Thực nghiệm xác nhận độc lập `R2_matched_128` (`R2_DECAY`) vs `A_ref_128` (`A_DECAY`), Slurm Job **`16911`** (`N = 24`, seeds `[4001..4024]`, `48` lượt huấn luyện) tại thư mục [robust_r2_matched_confirmatory](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory).
- **Trạng thái tuân thủ ràng buộc cứng:** **TUYỆT ĐỐI (`100%`)** — Không tạo, sửa, ghi đè hay xóa bất kỳ tệp nào (kể cả tệp tạm/`.pyc`); không nộp Slurm job; không chạm vào `[4025..4040]`, `HOLD_EVAL`, hay [PAUSE_COORDINATOR](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR).

---

## 1. Phán Quyết Hậu Nghiệm (`POST-HOC VERDICT`)

> [!IMPORTANT]
> **PHÁN QUYẾT CHÍNH THỨC: `PASS` (`SOVEREIGN_PASSED` / `PASS_CONFIRMATORY_IUT` — Điểm Kiểm Toán AAS: `98.5 / 100`)**
>
> Thí nghiệm xác nhận độc lập trong Slurm Job `16911` đã tuân thủ **100% bất biến khóa thiết kế trước chạy** ([R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md) & [DESIGN_LOCK_SHA256.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DESIGN_LOCK_SHA256.json)), hoàn tất sạch **`24/24` hạt giống (`48/48` lượt chạy)**, và vượt qua đồng thời cả hai điều kiện chính theo quy tắc **Berger Intersection-Union Test (IUT) ở mức ý nghĩa một phía $\alpha = 0.025$** với $p_{\text{IUT}} = \max(p_A, p_B) = 4.37 \times 10^{-9} \ll 0.025$ (và giữ nguyên kết luận `PASS` trên cả kiểm tra độ nhạy `ep200` với $p_{\text{IUT, ep200}} = 7.31 \times 10^{-7} \ll 0.025$ cùng `0` lần lật phán quyết Leave-One-Out).
>
> Hồ sơ tổng kết [CONFIRMATORY_FINAL_REPORT.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/CONFIRMATORY_FINAL_REPORT.md) khớp chính xác từng con số với dữ liệu thô trên máy chủ; chỉ có **4 điểm ghi chú làm rõ nhỏ (non-blocking clarifications)** nêu tại Mục 4 bên dưới.

| Cổng Chuẩn `@gng5` | Tiêu chí Thẩm Định | Kết Quả | Bằng Chứng Xác Minh Trực Tiếp Trên Máy Chủ |
| :--- | :--- | :--- | :--- |
| **T1: Soundness** | Tính đúng đắn thống kê & quy tắc Berger IUT | **ĐẠT (`PASS`)** | $t_A(23) = -17.2833$ ($p_A = 5.68 \times 10^{-15} < 0.025$); $t_B(23) = -8.7600$ ($p_B = 4.37 \times 10^{-9} < 0.025$). |
| **T2: Counterexample Resistance** | Kháng nhiễu chọn bản lưu (`ep200`) & ngoại lai (LOO) | **ĐẠT (`PASS`)** | `ep200` đạt cả $H_{1a}$ ($t = -15.9726$) và $H_{1b}$ ($t = -6.4304$); Leave-One-Out có `0/24` lần lật phán quyết trên cả `best_val` và `ep200`. |
| **T3: Anti-Sycophancy** | Công bố trung thực seed & kịch bản bất lợi | **ĐẠT (`PASS`)** | Công bố rõ `4020`, `4023` (và `4009` ở `ep200`) vượt $+\varepsilon$ ở $\sigma=0.20$; đánh đổi bất lợi tại `fixed_015`; và giới hạn `cost_s = 0.0`. |
| **T4: Symbolic Reproducibility** | Khớp tái nạp `.pkl` & tái tính thống kê độc lập từ CSV | **ĐẠT (`PASS`)** | `48/48` `best_val.pkl` có `val_nom_reload_diff <= 7.45e-09 < 1e-6`; `CONFIRMATORY_STATS_SUMMARY.json["stats_summary"] == STATS_RECOMPUTED_CLI.json` từng bit. |
| **T5: Merkle Immutability** | Khớp mã băm SHA-256 khóa trước/trong/sau chạy | **ĐẠT (`PASS`)** | Khớp **17/17** mã băm SHA-256 giữa khóa trước chạy, log Slurm Job `16911`, JSON kết quả, và tệp thực tế trên đĩa hiện tại. |

---

## 2. Kiểm Toán Tính Khớp Số Liệu & Tuân Thủ Khóa Thiết Kế

### 2.1. Kiểm tra bất biến `17/17` mã băm SHA-256 & nhật ký Slurm Job `16911`
- **Đối chiếu 4 nguồn độc lập:** (i) [DESIGN_LOCK_SHA256.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DESIGN_LOCK_SHA256.json#L31-L49), (ii) khối `sha256sum` đầu & trong [slurm_conf_n24_16911.log](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/slurm_conf_n24_16911.log#L5-L22) (cùng dry-run tại dòng 181–199, 363–381), (iii) trường `source_sha256` trong [CONFIRMATORY_STATS_SUMMARY.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/CONFIRMATORY_STATS_SUMMARY.json#L34-L52), và (iv) băm SHA-256 tính trực tiếp từ đĩa máy chủ tại thời điểm kiểm toán:
  - **Kết quả:** **Khớp tuyệt đối `17/17` tệp (`100%`)**, bao gồm [R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md) (`2595caf8...`), [conf_keys.py](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_keys.py) (`9281e532...`), [conf_design.py](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_design.py) (`a1e4f1bc...`), [conf_stats.py](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_stats.py) (`05f71843...`), [conf_run.py](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py) (`9374623a...`), [DATA_AND_SEED_PLAN.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DATA_AND_SEED_PLAN.json) (`796adea1...`), cùng toàn bộ các mô-đun huấn luyện/mô hình/độ đo thượng nguồn.
- **Xác minh thực thi Slurm Job `16911` (`sacct -j 16911` & [slurm_confirmatory_n24.sbatch](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/slurm_confirmatory_n24.sbatch)):**
  - Chạy một mạch duy nhất trên node `dgx02` (`Partition=defq`, `AllocCPUS=8`, `0 GPU`), bắt đầu `2026-11-19T23:32:23`, kết thúc `2026-11-20T03:23:16`, trạng thái `COMPLETED` (`ExitCode 0:0`), `Elapsed = 03:50:53` (`13,836.356s` wall-clock trong Python), `MaxRSS = 1,220,600K` (`sacct`) / `1,217,332 KB` (`ru_maxrss`).
  - Toàn bộ `8/8` unit tests tiền thực thi (`test_conf_suite.py`) và kiểm tra cô lập khóa PRNG (`2,167` khóa lịch sử vs `861` khóa mới, `0` va chạm) đều `PASS` trước khi bước vào `execute_confirmatory` với `N_SEEDS=24, SEED_OFFSET=0`. Không có bất kỳ lệnh nộp lại `sbatch` hay sửa đổi tham số nào giữa chừng.

### 2.2. Kiểm toán sổ cái `24/24` seed, `reload_checks`, thư mục `checkpoints/`, `PAUSE_COORDINATOR` & `HOLD_EVAL`
1. **Sổ cái `24/24` seed ([SEED_LEDGER.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/SEED_LEDGER.json)):** Đủ `24` hạt giống `[4001..4024]`, tất cả mang trạng thái `"COMPLETED"`, `failed_seeds = []`. Thời gian huấn luyện trung bình: `A_DECAY = 264.725s/seed`, `R2_DECAY = 302.364s/seed`.
2. **Kiểm tra tái nạp bản lưu (`reload_checks` trong [CONFIRMATORY_STATS_SUMMARY.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/CONFIRMATORY_STATS_SUMMARY.json#L189-L430)):** Đủ `48/48` bản lưu `best_val.pkl` được nạp lại từ đĩa và đối chiếu `val_nom` với bộ nhớ trước khi sinh `conf_test`. Sai lệch tối đa `max_reload_diff = 7.450580596923828e-09` (`< 1e-6`), trong đó `36/48` lượt có sai lệch `0.0` tuyệt đối và `12/48` lượt lệch `3.725e-09` hoặc `7.451e-09` (mức lượng tử hóa `float32`).
3. **Hiện trạng thư mục `checkpoints/confirmatory_n24/`:** Kiểm tra trực tiếp trên hệ thống tệp xác nhận có đúng **`24` thư mục hạt giống** (`seed_4001` .. `seed_4024`), **`96` tệp `.pkl`** (`48` `best_val.pkl` + `48` `final_with_opt.pkl`, kích thước nhỏ nhất `27,910 bytes`), và **`48` tệp `checkpoint_manifest.json`**.
4. **Hiện trạng [PAUSE_COORDINATOR](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR), `[4025..4040]` và `HOLD_EVAL`:**
   - [PAUSE_COORDINATOR](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR) tồn tại nguyên vẹn (`size = 32 bytes`, dấu thời gian `2026-11-16 10:17:35 +0700` không đổi).
   - Không có bất kỳ tệp, thư mục hay dòng dữ liệu nào liên quan đến `[4025..4040]` hoặc `HOLD_EVAL` được sinh ra hay mở ra.

### 2.3. Tái tính độc lập Berger IUT ở $\alpha = 0.025$ trên `best_val` và `ep200`
Kiểm toán viên đã tái tính độc lập trực tiếp từ `288` dòng của [PAIRED_CONFIRMATORY.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/PAIRED_CONFIRMATORY.csv) và `576` dòng của [RESULTS_CONFIRMATORY_FULL_PRECISION.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/RESULTS_CONFIRMATORY_FULL_PRECISION.csv), đối chiếu với [CONFIRMATORY_STATS_SUMMARY.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/CONFIRMATORY_STATS_SUMMARY.json), [STATS_RECOMPUTED_CLI.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/STATS_RECOMPUTED_CLI.json), [POSTRUN_AUDIT_TABLES.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/POSTRUN_AUDIT_TABLES.json) và [SECONDARY_SUMMARY_TABLE.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/SECONDARY_SUMMARY_TABLE.csv):

| Hạng mục Kiểm Định ($N=24$, $\text{df}=23$, $t_{0.975,23} = 2.0686576$) | Mean ($\bar{\Delta}$) | SD ($s$, `ddof=1`) | SE ($s/\sqrt{24}$) | KTC 95% Hai Phía (`[CI_low, CI_high]`) | Thống kê $t(23)$ | $p$-value 1 phía | LOO Range / Flips | Sign / Wilcoxon Exact | Phán Quyết |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chính $H_{1a}$ (`best_val`):** $\Delta\text{HE}_{030} < 0.0$ | `-0.00456155` | `0.00129298` | `0.00026393` | `[-0.00510752, -0.00401557]` | `-17.283292` | `5.677e-15` | `[-0.004708, -0.004404]` (`0` flips) | `24/24 < 0` ($p=5.96\text{e-}08$); $W^+=0$ ($p=5.96\text{e-}08$) | **PASS** |
| **Chính $H_{1b}$ (`best_val`):** $\Delta\text{OCE}_{020} < +0.00010$ | `+0.00003109` | `0.00003853` | `0.00000787` | `[+0.00001482, +0.00004737]` | `-8.760037` | `4.367e-09` | `[+0.000028, +0.000035]` (`0` flips) | `22/24 < +ε` ($p=1.79\text{e-}05$); $W^+=3$ ($p=2.98\text{e-}07$) | **PASS** |
| **Độ nhạy $H_{1a}$ (`ep200`):** $\Delta\text{HE}_{030} < 0.0$ | `-0.00490101` | `0.00150320` | `0.00030684` | `[-0.00553576, -0.00426627]` | `-15.972564` | `3.046e-14` | `[-0.005074, -0.004788]` (`0` flips) | `24/24 < 0` ($p=5.96\text{e-}08$); $W^+=0$ ($p=5.96\text{e-}08$) | **PASS** |
| **Độ nhạy $H_{1b}$ (`ep200`):** $\Delta\text{OCE}_{020} < +0.00010$ | `+0.00003771` | `0.00004745` | `0.00000969` | `[+0.00001767, +0.00005775]` | `-6.430406` | `7.306e-07` | `[+0.000033, +0.000041]` (`0` flips) | `21/24 < +ε` ($p=1.39\text{e-}04$); $W^+=12$ ($p=4.17\text{e-}06$) | **PASS** |

- **Đối chiếu từng ô trong Bảng 2, Bảng 3 (24 seed $\times$ 8 cột số liệu) và Bảng 4 (6 kịch bản $\times$ 6 chỉ số) của [CONFIRMATORY_FINAL_REPORT.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/CONFIRMATORY_FINAL_REPORT.md):** Toàn bộ `100%` các giá trị làm tròn trong báo cáo khớp chính xác tới chữ số cuối cùng (`sai số làm tròn <= 0.5e-6`) so với dữ liệu gốc trong [PAIRED_CONFIRMATORY.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/PAIRED_CONFIRMATORY.csv), [SECONDARY_SUMMARY_TABLE.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/SECONDARY_SUMMARY_TABLE.csv) và [EXTRA_POSITION_SUMMARY.json](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/EXTRA_POSITION_SUMMARY.json).

---

## 3. Đánh Giá Trung Thực Các Điểm Bất Lợi và Giới Hạn

### 3.1. Các hạt giống bất lợi tại $\sigma = 0.20$ (`4020`, `4023`, `4009`) và bản chất sự đánh đổi danh nghĩa
- **Quan sát thực nghiệm:**
  - Tại `best_val`, có **`19/24` hạt giống** mang $\Delta\text{OCE}_{020} > 0$ (chỉ `5/24` hạt giống âm là `4004`, `4010`, `4012`, `4021`, `4022`), và có **`2/24` hạt giống cá biệt vượt ngưỡng $+\varepsilon = +0.00010$ (`+1.00 bp`)**:
    - **Seed `4020`:** $\Delta\text{OCE}_{020} = +0.00011248$ (`+1.125 bp` ở `best_val`, `A_best_ep=200 / R2_best_ep=199`) và $+0.00013831$ (`+1.383 bp` ở `ep200`), đổi lại đạt mức giảm lỗi phòng hộ stress rất lớn $\Delta\text{HE}_{030} = -0.005507$ (`-55.07 bp`).
    - **Seed `4023`:** $\Delta\text{OCE}_{020} = +0.00010930$ (`+1.093 bp` ở `best_val`, `A_best_ep=151 / R2_best_ep=200`) và $+0.00010018$ (`+1.002 bp` ở `ep200`), đổi lại giảm mạnh $\Delta\text{HE}_{030} = -0.006169$ (`-61.69 bp`).
  - Tại `ep200` (khi bỏ chọn lọc `best_val`), xuất hiện thêm **Seed `4009`** vượt ngưỡng: từ $\Delta\text{OCE}_{020} = +0.00004952$ (`+0.495 bp` ở `best_val`) tăng lên $+0.00011347$ (`+1.135 bp` ở `ep200`), gắn liền với việc tham số ngưỡng học được `learned_y_nom` của `R2_DECAY` ở `ep200` lệch dương so với `A_DECAY` ($\Delta y_{\text{nom}} = +0.000516$).
- **Đánh giá khách quan:**
  1. Giả thuyết không thua kém đã khóa $H_{1b}: \mu(\Delta\text{OCE}_{020}) < +0.00010$ là giả thuyết về **kỳ vọng tổng thể (population mean)** chứ không phải ràng buộc tất định từng hạt giống ($\max_i \Delta\text{OCE}_{020,i} < +0.00010$). Với trung bình mẫu $+0.311\text{ bp}$ và độ lệch chuẩn giữa các hạt giống $s = 0.385\text{ bp}$, biên $+1.00\text{ bp}$ nằm cách trung bình $+1.79$ độ lệch chuẩn mẫu, nên việc xuất hiện `2/24` (`8.3%`) hạt giống ở đuôi phải vượt $+1.00\text{ bp}$ hoàn toàn tương thích với phân phối chuẩn có trung bình $\mu \approx +0.31\text{ bp}$ (kể cả kiểm định dấu phi tham số cho trung vị $< +1.00\text{ bp}$ vẫn đạt $22/24$, $p = 1.79 \times 10^{-5}$, và Wilcoxon $W^+ = 3$, $p = 2.98 \times 10^{-7}$).
  2. Tuy nhiên, cần khẳng định trung thực: **`R2_DECAY` chịu một chi phí đánh đổi danh nghĩa dương có thực tại $\sigma = 0.20$** (KTC 95% `[+0.148, +0.474] bp` nằm hoàn toàn trên `0.0`), và tồn tại tương quan âm nhẹ giữa hai mục tiêu chính ($\hat{\rho}(\Delta\text{HE}_{030}, \Delta\text{OCE}_{020}) = -0.2012$ ở `best_val`): các hạt giống giảm mạnh $\text{HE}_{030}$ nhất (`4020`, `4023`) có xu hướng trả giá $\text{OCE}_{020}$ cao hơn.

### 3.2. Đánh giá vùng bất lợi hệ thống tại biến động thấp `fixed_015` ($\sigma = 0.15$)
- Tại kịch bản nội miền biến động thấp `fixed_015` (`best_val`), `R2_DECAY` kém hơn `A_DECAY` một cách nhất quán:
  - $\Delta\text{HE}_{015} = +0.000695$ (`95% CI [+0.000585, +0.000805]`, **`23/24` seed dương** ở `best_val`, trừ `4010`; **`24/24` seed dương** ở `ep200`).
  - $\Delta\text{CVaR95}_{015} = +0.001097$ (`95% CI [+0.000869, +0.001324]`, **`23/24` seed dương** ở `best_val`).
  - $\Delta\text{OCE}_{015}(y_{\text{nom}}) = +0.000102$ (`95% CI [+0.000076, +0.000128]`, **`22/24` seed dương**, `14/24` seed vượt $+0.00010$).
- **Phân tách cơ chế:** Khi thay ngưỡng cố định $y_{\text{nom}}$ (học cho $\sigma = 0.20$) bằng ngưỡng riêng của kịch bản $y_{\text{scen}}$ (học cho $\sigma = 0.15$), chỉ số chẩn đoán $\Delta\text{OCE}_{015}(y_{\text{scen}})$ đảo chiều sang âm (`-0.000648`, `95% CI [-0.000674, -0.000623]`, `24/24` seed âm). **Tuy nhiên**, vì hai chỉ số `he_net_ddof0` (`HE`) và `cvar95` (`CVaR95`) được tính trực tiếp từ phân phối lỗ/lãi phòng hộ ròng mà **không phụ thuộc vào tham số vô hướng $y$**, sự gia tăng $+6.95\text{ bp}$ trên `HE` và $+10.97\text{ bp}$ trên `CVaR95` tại `fixed_015` là **sự đánh đổi cấu trúc thực chất của mạng chính sách `R2_DECAY`** (do phải phân bổ dung lượng biểu diễn sang các chế độ biến động cao $\sigma = 0.25$ và chuyển chế độ `rs_015_025`), chứ không đơn thuần là hệ quả của việc cố định $y_{\text{nom}}$.

### 3.3. Giới hạn phạm vi `cost_s = 0.0`
- Toàn bộ `48` lượt huấn luyện và đánh giá đều đặt `cost_s = 0.0` (không phí giao dịch tỷ lệ).
- Dù `R2_DECAY` giảm khối lượng giao dịch (`turnover`) trung bình trên cả `6/6` kịch bản từ `-0.1076` (`fixed_015`) đến `-0.2126` (`fixed_030`), đây chỉ là thuộc tính cơ học thuận lợi của nghiệm tối ưu không phí giao dịch. Khi `cost_s > 0`, đạo hàm của hàm mục tiêu theo hành động phòng hộ $\delta_t$ thay đổi trực tiếp trong quá trình huấn luyện; do đó **hiệu lực của phán quyết `PASS_CONFIRMATORY_IUT` bị giới hạn nghiêm ngặt trong môi trường `cost_s = 0.0`**.

---

## 4. Danh Sách Sai Sót / Thiếu Sót Trong `CONFIRMATORY_FINAL_REPORT.md` Cần Lưu Ý

Kiểm toán đối chiếu từng dòng xác nhận **[CONFIRMATORY_FINAL_REPORT.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/CONFIRMATORY_FINAL_REPORT.md) KHÔNG có bất kỳ sai sót số học, lỗi làm tròn hay kết luận vượt quá phạm vi khóa thiết kế nào**. Tuy nhiên, để hồ sơ lưu trữ cuối cùng đạt độ chặt chẽ tối đa (không bắt buộc chạy lại bất kỳ phép tính nào), có **4 điểm diễn đạt / chi tiết bổ sung nhỏ** nên ghi nhận:

1. **Làm rõ bản chất đánh đổi tại `fixed_015` (Mục 4, dòng 97 của [CONFIRMATORY_FINAL_REPORT.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/CONFIRMATORY_FINAL_REPORT.md#L97)):**
   - Báo cáo hiện giải thích việc `hedged_oce_y_nom` tăng `+0.000102` ở `fixed_015` là *"do đánh giá với `y_nom` học ở $\sigma = 0.20$; nếu dùng `y_scen` tại `0.15`, `ΔOCE_y_scen = -0.000648`"*. Nên nói rõ thêm rằng nhận xét đó chỉ giải thích sự đảo dấu của `OCE`, còn mức tăng `ΔHE = +0.000695` (`23/24` seed) và `ΔCVaR95 = +0.001097` (`23/24` seed) ở `fixed_015` là **bất biến với $y$** và phản ánh sự đánh đổi thực sự của chính sách `policy` đa kịch bản tại vùng biến động thấp.
2. **Ghi chú cụ thể hạt giống ngoại lệ của `turnover` và `max_abs_delta` (Mục 4, dòng 96 của [CONFIRMATORY_FINAL_REPORT.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/CONFIRMATORY_FINAL_REPORT.md#L96)):**
   - Khi nhận xét `R2_DECAY` giảm `turnover` và `max_abs_delta` trung bình trên toàn bộ `6/6` kịch bản, nên ghi chú thêm tỷ lệ theo seed để đồng nhất với cách báo cáo của `HE` và `OCE`: tại `best_val`, `turnover` giảm trên **`23/24` seed ở cả 6 kịch bản** (ngoại lệ duy nhất là **seed `4010`** có $\Delta\text{Turnover} > 0$ từ `+0.0640` đến `+0.2292` do `A_DECAY` chọn bản lưu rất sớm tại `best_ep = 42` so với `R2_DECAY` tại `best_ep = 194`; ở `ep200`, seed `4010` lại giảm turnover và ngoại lệ duy nhất chuyển sang **seed `4023`** với mức tăng nhỏ `+0.0067` đến `+0.0151`). Với `max_abs_delta` tại `best_val`, `24/24` seed giảm ở 5 kịch bản đầu và `23/24` seed giảm ở `fixed_030` (ngoại lệ nhẹ ở **seed `4005`**).
3. **Đồng nhất khoảng tin cậy 95% cho hai cột cuối trong Bảng 4 (dòng 86–93 của [CONFIRMATORY_FINAL_REPORT.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/CONFIRMATORY_FINAL_REPORT.md#L86-L93)):**
   - Trong Bảng 4, hai cột `ΔOCE y_scen` và `Δmax_abs_delta` hiện chỉ ghi giá trị `Mean` trong khi 4 cột còn lại ghi `Mean [95% CI]`. Khoảng tin cậy 95% của `ΔOCE y_scen` đã có sẵn trong [SECONDARY_SUMMARY_TABLE.csv](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/SECONDARY_SUMMARY_TABLE.csv) (`fixed_015`: `[-0.000674, -0.000623]`, `fixed_020`: `[+0.000015, +0.000047]`, `fixed_025`: `[-0.001155, -0.001058]`, `rs_015_025`: `[-0.000020, +0.000013]`).
4. **Liên hệ thực nghiệm với bảng công suất thiết kế (`R1–R8` trong [R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md](file:///home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md#L55-L66)):**
   - Tham số thực đo của mẫu xác nhận $N = 24$ tại `best_val` là $(\bar{\Delta}_A, s_A, \bar{\Delta}_B, s_B, \hat{\rho}_{AB}) = (-0.004562, 0.001293, +0.000031, 0.000039, -0.2012)$. So với 8 kịch bản tiền thiết kế `R1–R8`, độ lệch chuẩn thực tế $s_B = 0.000039$ thấp hơn cả kịch bản cơ sở `R1` (`0.000050`), giúp biên an toàn thực tế $(\varepsilon_B - \bar{\Delta}_B) / \text{SE}_B = 8.76\,\text{SE}$ vượt xa kịch bản stress `R8`, giải thích vì sao cỡ mẫu khóa $N = 24$ đạt độ phân giải thống kê dứt khoát mà không rơi vào vùng thiếu lực thống kê.

---
*Chữ ký kiểm toán:* `Gemini-4-Argon-High Sovereign Epistemic Gate (@gng5 Single-Session Read-Only Post-Hoc Audit — Job 16911 Verified)`
