# Báo Cáo Tổng Kết Thí Nghiệm Xác Nhận Có Chi Phí Giao Dịch 10 bps: `R2_matched_128` vs `A_ref_128` (`COST_CONFIRMATORY_FINAL_REPORT.md`)

- **Thiết kế khóa chính thức:** [COST_CONFIRMATORY_DESIGN_LOCK.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/COST_CONFIRMATORY_DESIGN_LOCK.md)
- **Kiểm toán kế toán phí & độc lập hạt giống:** [COST_ACCOUNTING_AUDIT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/COST_ACCOUNTING_AUDIT.md) | [SEED_INDEPENDENCE_AUDIT.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/SEED_INDEPENDENCE_AUDIT.md)
- **Thực thi Slurm chính thức:** **Job `16914`** (`r2_cost10_n24`, phân vùng `defq`, node `dgx02`, `8 CPU, 0 GPU`), trạng thái **`COMPLETED` (`ExitCode 0:0`)**, thời lượng **`03:44:55`** (`13,475.24s` wall-clock Python; `A_DECAY` mean `256.35s ± 1.29s`, `R2_DECAY` mean `295.37s ± 1.19s`), **`MaxRSS = 851,780K`** (`sacct`) — nhật ký: [slurm_cost10_n24_16914.log](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/logs/slurm_cost10_n24_16914.log).
- **Quy mô thực thi:** **`N = 24`** hạt giống độc lập mới **`[5001..5024]`** × `2` nhánh đối xứng (`A_DECAY` = `A_ref_128`, `R2_DECAY` = `R2_matched_128`) = **đúng `48` lượt huấn luyện** (`31,200` bước/lượt, `3,993,600` lượt xử lý đường/lượt, `Cosine LR Decay`, `Adam eps=1e-7`, **`cost_s = 0.0010` (`10 bps`) trong huấn luyện, chọn `best_val`, và kiểm tra**), đánh giá trên **`20,000` đường/kịch bản** (`6` kịch bản, sinh sau khi chốt xong `best_val` & `ep200`). `HOLD_EVAL`, `[4001..4040]`, và `[5025..5040]` được bảo toàn hoàn toàn.

> [!IMPORTANT]
> **PHÁN QUYẾT XÁC NHẬN CHÍNH (`PASS_CONFIRMATORY_IUT`):**
> Trên cả `24/24` hạt giống hợp lệ (`0` thiếu, `0` `NaN`/`Inf`, `0` lỗi hạ tầng/thuật toán, tái nạp `48/48` `best_val.pkl` khớp từng bit `reload diff = 0.0`), thí nghiệm đạt đồng thời **cả hai điều kiện thành công theo quy tắc Berger Intersection-Union Test (IUT) ở `α = 0.025` một phía** (**`p_IUT = max(p_A, p_B) = 3.745 × 10⁻⁶ << 0.025`**):
> 1. **Điều kiện chính A (`H1a` — Giảm `HE` ròng tại `σ = 0.30`, `best_val`):** `ΔHE_030 = -0.003547 ± 0.002231` (`-10.41%` so với `A_DECAY = 0.034087`), **`95% CI = [-0.004489, -0.002605]`** (`< 0.0`), `t(23) = -7.7871`, **`p = 3.396 × 10⁻⁸`**, **`23/24` seed giảm** (`1/24` seed `5004` dương nhẹ `+0.000368`).
> 2. **Điều kiện chính B (`H1b` — Không thua kém `OCE` ròng tại `σ = 0.20`, biên `ε = +0.00010`, `best_val`):** `ΔOCE_020 = +0.00000010 ± 0.00008518` (`+0.001 bp`), **`95% CI = [-0.00003586, +0.00003607]`** (cận trên `+0.00003607 < +0.00010`), `t(23) = -5.7455`, **`p = 3.745 × 10⁻⁶`**, **`24/24` seed đều thấp hơn ngưỡng `+0.00010`** (`max = +0.00007310` tại seed `5019`).
>
> **Phạm vi hiệu lực:** Kết luận chỉ áp dụng cho `R2_matched_128` (kỳ vọng đều `1/4`) vs `A_ref_128` dưới chi phí giao dịch tỷ lệ thuận `c_s = 0.0010` (`10 bps`) trong họ mô hình thị trường `20` bước hiện tại; không ngoại suy sang `NTBN`, `R3 minimax`, hay mức phí cao hơn.

---

## 1. Xác Minh Toàn Vẹn Thực Thi, Niêm Phong Mã & Kế Toán Chi Phí `10 bps`

| Hạng mục kiểm chứng | Kết quả thực đo trực tiếp | Nguồn đối chiếu |
| :--- | :--- | :--- |
| **Hoàn tất 48 lượt & Sổ cái seed** | **`24/24` seed `COMPLETED`**, `failed_seeds = []`, `missing_seeds = []`, `non_finite_seeds = []`, `duplicate_seeds = []` | [SEED_LEDGER.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/SEED_LEDGER.json) |
| **Ngân sách & đối xứng tham số** | `actual_steps_unique = [31200]`, `actual_paths_unique = [3993600]` cho cả `48/48` lượt; `policy_total = 6,762` (sai khác init `0.0`) | [POSTRUN_AUDIT_TABLES.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/POSTRUN_AUDIT_TABLES.json) |
| **Niêm phong mã băm SHA-256** | `23/23` tệp trong [SOURCE_SHA256SUMS.txt](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/SOURCE_SHA256SUMS.txt) `OK`; `22/22` khóa chung khớp `COST_CONFIRMATORY_STATS_SUMMARY.json` | [POSTRUN_SHA256SUMS.txt](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/POSTRUN_SHA256SUMS.txt) |
| **Tái nạp `48/48` `best_val.pkl` & `96` `.pkl`** | `48/48` `best_val.pkl` + `48/48` `final_with_opt.pkl` nguyên vẹn trên máy chủ; sai số tái nạp `max = 0.0` | Kiểm chứng máy chủ & [COST_CONFIRMATORY_STATS_SUMMARY.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/COST_CONFIRMATORY_STATS_SUMMARY.json) |
| **Cách ly tập dữ liệu & sửa lỗi B3** | `408/408` mã băm (`24 × 17` thế giới) duy nhất (`0` trùng, `0` chồng chéo `Val(s)/Train(s+1)`); `Val(5001)=c3926366e42d0df2` $\neq$ `Train(5002)=5eed83775b92f59c` | [POSTRUN_AUDIT_TABLES.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/POSTRUN_AUDIT_TABLES.json) |
| **Phí dương xuyên 3 giai đoạn & không trừ 2 lần** | `Train` `mean(cost_step) ≈ 0.001000`; `Val` `[0.001183, 0.001343]`; `0/576` dòng test bằng `0`; `max \|(gross - cost) - net\| = 3.84 × 10⁻¹¹` | [RESULTS_COST_CONFIRMATORY_FULL_PRECISION.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/RESULTS_COST_CONFIRMATORY_FULL_PRECISION.csv) |
| **Tính lại thống kê độc lập từ CSV** | Khớp từng bit giữa JSON tổng kết và `cost_stats.py --paired-csv PAIRED_COST_CONFIRMATORY.csv` | [STATS_RECOMPUTED_CLI.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/STATS_RECOMPUTED_CLI.json) |
| **Bảo toàn `PAUSE_COORDINATOR`, `HOLD_EVAL`, `[5025..5040]`** | `PAUSE_COORDINATOR` `32 bytes` (`Nov 16 10:17`); `0` tệp/thư mục cho `[5025..5040]`; `HOLD_EVAL` chưa từng mở | Kiểm tra trực tiếp máy chủ |

---

## 2. Kết Quả Hai Điều Kiện Thành Công Chính & Kiểm Tra Độ Nhạy

Số liệu từ [PAIRED_COST_CONFIRMATORY.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/PAIRED_COST_CONFIRMATORY.csv) và [POSTRUN_AUDIT_TABLES.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/POSTRUN_AUDIT_TABLES.json):

| Tiêu chí | `A_DECAY` (`Mean ± SD`) | `R2_DECAY` (`Mean ± SD`) | Chênh lệch `Δ = R2 - A` (`Mean ± SD`) | KTC `95%` hai phía (`[low, high]`) | `t(23)` / `p` 1 phía (`α=0.025`) | Độ vững (`LOO` / Sign / Wilcoxon chính xác) | Kết luận |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`H1a`: `HE` ròng tại `fixed_030`** (`best_val`, `< 0`) | `0.034087 ± 0.002713` | `0.030540 ± 0.002930` | **`-0.003547 ± 0.002231`** (`med -0.003497`) | **`[-0.004489, -0.002605]`** | `t = -7.7871`, **`p = 3.40e-08`** | `0` LOO flips (`[-0.003717, -0.003193]`); `23/24 < 0` (`Sign p = 1.49e-06`, `Wilcoxon W+=1.0, p = 1.19e-07`) | **ĐẠT (`PASS`)** |
| **`H1b`: `OCE` ròng tại `fixed_020`** (`best_val`, `< +0.00010`) | `0.053083 ± 0.000179` | `0.053083 ± 0.000178` | **`+0.00000010 ± 0.000085`** (`med +0.0000157`) | **`[-0.00003586, +0.00003607]`** (`UCL +0.00003607`) | `t = -5.7455`, **`p = 3.75e-06`** | `0` LOO flips (`[-3.07e-6, +1.56e-5]`); `24/24 < +ε` (`Sign p = 5.96e-08`, `Wilcoxon W+=0.0, p = 5.96e-08`) | **ĐẠT (`PASS`)** |
| **Nhạy `ep200`: `ΔHE_030`** (`< 0`) | `0.034669 ± 0.002823` | `0.030792 ± 0.002871` | `-0.003877 ± 0.002189` (`med -0.003496`) | `[-0.004801, -0.002953]` | `t = -8.6786`, `p = 5.16e-09` | `0` LOO flips (`[-0.004047, -0.003568]`); `23/24 < 0` (`Sign p = 1.49e-06`, `W+=1.0, p = 1.19e-07`) | **ĐẠT (`PASS`)** |
| **Nhạy `ep200`: `ΔOCE_020`** (`< +0.00010`) | `0.053101 ± 0.000175` | `0.053093 ± 0.000176` | `-0.00000714 ± 0.00008185` (`med +0.0000027`) | `[-0.00004170, +0.00002742]` | `t = -6.4128`, `p = 7.61e-07` | `0` LOO flips (`[-1.11e-5, +6.46e-6]`); `24/24 < +ε` (`Sign p = 5.96e-08`, `W+=0.0, p = 5.96e-08`) | **ĐẠT (`PASS`)** |

**Phân tích độ lệch `skewness`/`kurtosis` và kiểm tra nhạy loại bỏ ngoại lệ (`5024` & `5004`):**
Trên mẫu đủ `N = 24`, cả hai biến `ΔHE_030` (`skew = -1.770`, `kurt = +5.715`) và `ΔOCE_020` (`skew = -3.222`, `kurt = +10.972`) có đuôi trái nhọn chủ yếu do hạt giống `5024` mang lại lợi thế rất lớn cho `R2_DECAY` (`ΔHE_030 = -0.011686`, `ΔOCE_020 = -0.000356`), còn ở chiều ngược lại seed `5004` là điểm bất lợi duy nhất cho `ΔHE_030` (`+0.000368`). Kiểm chứng trực tiếp khi loại bỏ các ngoại lệ:

| Tập mẫu nhạy (`best_val`) | `ΔHE_030` (`Mean ± SD`, `skew / kurt`) | `t` / `p_A` (`H1a`) | `ΔOCE_020` (`Mean ± SD`, `skew / kurt`) | `t` / `p_B` (`H1b`) | `p_IUT` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Đủ `24` seed (Gốc)** | `-0.0035470 ± 0.0022315` (`-1.77 / +5.71`) | `-7.7871` / `3.40e-08` | `+0.104e-06 ± 8.518e-05` (`-3.22 / +10.97`) | `-5.7455` / `3.75e-06` | **`3.75e-06`** |
| **Loại `5024` (`n = 23`)** | `-0.0031932 ± 0.0014367` (`+0.74 / +0.06`) | `-10.6589` / `1.87e-10` | `+1.557e-05 ± 3.983e-05` (`-1.15 / +1.92`) | `-10.1668` / `4.46e-10` | **`4.46e-10`** |
| **Loại `5004` (`n = 23`)** | `-0.0037173 ± 0.0021163` (`-2.22 / +6.96`) | `-8.4238` / `1.24e-08` | `-2.613e-06 ± 8.602e-05` (`-3.19 / +10.64`) | `-5.7207` / `4.69e-06` | **`4.69e-06`** |
| **Loại cả `5004 & 5024` (`n = 22`)** | `-0.0033551 ± 0.0012373` (`+0.45 / -0.45`) | `-12.7183` / `1.24e-11` | `+1.343e-05 ± 3.939e-05` (`-1.16 / +2.02`) | `-10.3076` / `5.68e-10` | **`5.68e-10`** |

> Ngoại lệ `5024` làm phình `sd_ddof1` lên hơn gấp đôi (`3.98e-05 -> 8.52e-05` ở `H1b`), nên sự có mặt của `5024` thực chất làm $|t|$ *nhỏ đi* (kiểm định bảo thủ hơn). Khi bỏ `5024`, phân phối gần chuẩn trở lại và `p_IUT` còn mạnh hơn gần bốn bậc độ lớn (`4.46e-10`), chứng tỏ kết luận `PASS_CONFIRMATORY_IUT` hoàn toàn không phụ thuộc vào ngoại lệ `5024`.

![Primary Paired Results at 10 bps](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/plots/fig1_cost10_primary_paired_n24.png)
![Sensitivity best_val vs ep200 and LOO](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/plots/fig4_cost10_best_val_vs_ep200_and_loo.png)

---

## 3. Công Bố Đầy Đủ Toàn Bộ 24 Hạt Giống `[5001..5024]` (`best_val` & `ep200`)

| Seed | `A / R2` `best_ep` | `A HE_030` | `R2 HE_030` | `ΔHE_030` (`best`) | `A OCE_020` | `R2 OCE_020` | `ΔOCE_020` (`best`) | `ΔCost_020` (`best`) | `ΔHE_030` (`ep200`) | `ΔOCE_020` (`ep200`) | Ghi chú nổi bật |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `5001` | `200 / 200` | `0.036657` | `0.032518` | `-0.004139` | `0.053027` | `0.053009` | `-0.000018` | `+0.000002` | `-0.004139` | `-0.000018` | `ΔCost_020 > 0` |
| `5002` | `194 / 184` | `0.034826` | `0.031539` | `-0.003287` | `0.053395` | `0.053384` | `-0.000011` | `+0.000029` | `-0.003152` | `-0.000011` | `ΔCost_020 > 0` |
| `5003` | `172 / 200` | `0.036401` | `0.031153` | `-0.005248` | `0.052892` | `0.052938` | `+0.000046` | `-0.000068` | `-0.005425` | `+0.000041` | `—` |
| **`5004`** | `188 / 198` | `0.023624` | `0.023992` | **`+0.000368`** | `0.052668` | `0.052730` | `+0.000063` | `-0.000123` | `-0.000051` | `+0.000017` | **Duy nhất `ΔHE_030 > 0` (`best`)** |
| `5005` | `148 / 198` | `0.033096` | `0.029548` | `-0.003547` | `0.053085` | `0.053132` | `+0.000047` | `-0.000058` | `-0.004562` | `+0.000001` | `—` |
| `5006` | `197 / 198` | `0.036641` | `0.033671` | `-0.002970` | `0.053076` | `0.053108` | `+0.000032` | `-0.000023` | `-0.003336` | `-0.000003` | `—` |
| `5007` | `199 / 200` | `0.033229` | `0.030720` | `-0.002509` | `0.053093` | `0.053123` | `+0.000030` | `+0.000008` | `-0.002904` | `-0.000057` | `ΔCost_020 > 0` |
| `5008` | `196 / 196` | `0.032142` | `0.028769` | `-0.003373` | `0.053223` | `0.053244` | `+0.000021` | `-0.000006` | `-0.003274` | `+0.000025` | `—` |
| `5009` | `159 / 192` | `0.034216` | `0.030009` | `-0.004207` | `0.053086` | `0.053149` | `+0.000063` | `-0.000039` | `-0.005182` | `+0.000062` | `—` |
| `5010` | `133 / 177` | `0.037216` | `0.032158` | `-0.005059` | `0.053178` | `0.053187` | `+0.000009` | `-0.000075` | `-0.005517` | `-0.000032` | `—` |
| `5011` | `115 / 156` | `0.034694` | `0.033388` | `-0.001306` | `0.053258` | `0.053249` | `-0.000009` | `-0.000015` | `-0.002671` | `+0.000054` | `ΔCost_020 > 0` (`ep200`) |
| `5012` | `164 / 200` | `0.033675` | `0.030926` | `-0.002749` | `0.053281` | `0.053290` | `+0.000009` | `+0.000019` | `-0.004088` | `-0.000020` | `ΔCost_020 > 0` |
| **`5013`** | `188 / 158` | `0.032911` | `0.032100` | `-0.000811` | `0.053101` | `0.053096` | `-0.000005` | `-0.000008` | **`+0.000027`** | `-0.000021` | **Duy nhất `ΔHE_030 > 0` (`ep200`)** |
| `5014` | `200 / 192` | `0.033441` | `0.029726` | `-0.003715` | `0.053108` | `0.053152` | `+0.000044` | `-0.000083` | `-0.003129` | **`+0.000084`** | `ΔOCE_020` max ở `ep200` (`< +ε`) |
| `5015` | `198 / 200` | `0.031640` | `0.026587` | `-0.005053` | `0.053234` | `0.053128` | `-0.000106` | `-0.000018` | `-0.005104` | `-0.000122` | `—` |
| `5016` | `190 / 180` | `0.035340` | `0.032267` | `-0.003073` | `0.052923` | `0.052934` | `+0.000011` | `-0.000095` | `-0.002832` | `+0.000005` | `—` |
| `5017` | `158 / 190` | `0.036289` | `0.035054` | `-0.001235` | `0.052845` | `0.052859` | `+0.000014` | `+0.000014` | `-0.002139` | `+0.000005` | `ΔCost_020 > 0` |
| `5018` | `65 / 183` | `0.036579` | `0.031780` | `-0.004799` | `0.053236` | `0.053254` | `+0.000018` | `-0.000078` | `-0.007082` | `+0.000048` | `—` |
| **`5019`** | `197 / 200` | `0.034987` | `0.031171` | `-0.003816` | `0.053167` | `0.053240` | **`+0.000073`** | `-0.000031` | `-0.003929` | `+0.000055` | `ΔOCE_020` max ở `best` (`< +ε`) |
| `5020` | `192 / 196` | `0.034538` | `0.030693` | `-0.003845` | `0.053196` | `0.053252` | `+0.000056` | `-0.000014` | `-0.004108` | `+0.000048` | `—` |
| `5021` | `196 / 200` | `0.033142` | `0.029696` | `-0.003446` | `0.052740` | `0.052693` | `-0.000047` | `-0.000010` | `-0.003571` | `-0.000052` | `—` |
| `5022` | `195 / 196` | `0.035561` | `0.031837` | `-0.003723` | `0.052916` | `0.052941` | `+0.000026` | `-0.000036` | `-0.003422` | `+0.000070` | `—` |
| `5023` | `194 / 199` | `0.033743` | `0.031843` | `-0.001900` | `0.053009` | `0.053001` | `-0.000007` | `-0.000045` | `-0.002470` | `-0.000027` | `—` |
| **`5024`** | `197 / 195` | `0.033504` | `0.021819` | **`-0.011686`** | `0.053255` | `0.052899` | **`-0.000356`** | `+0.000033` | `-0.010987` | `-0.000320` | **Ngoại lệ cực thuận `R2`; `ΔCost_020 > 0`** |

---

## 4. Toàn Bộ Chỉ Tiêu Phụ, Thu Nhập Gross vs Net & Cơ Chế Giao Dịch Trên 6 Kịch Bản (`best_val`, `N = 24`)

Trích từ [SECONDARY_COST_SUMMARY_TABLE.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/SECONDARY_COST_SUMMARY_TABLE.csv) và [POSTRUN_AUDIT_TABLES.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/results/confirmatory_n24/POSTRUN_AUDIT_TABLES.json):

### 4.1. Phòng Hộ Ròng (`HE`, `OCE`, `CVaR95`), Thu Nhập Trước Phí (`Gross`) vs Sau Phí (`Net`) & Chi Phí (`Cost`)

| Kịch bản (`best_val`) | `ΔHE_net` `[95% CI]` | `ΔOCE_y_nom` `[95% CI]` | `ΔCVaR95_net` `[95% CI]` | `ΔGross_Gains` `[95% CI]` | `ΔMean_Cost` `[95% CI]` | `ΔNet_Gains` (`= ΔGross - ΔCost`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`fixed_015`** | **`+0.000438`** `[+0.000119, +0.000758]` | `+0.000034` `[-0.000001, +0.000070]` | **`+0.000723`** `[+0.000231, +0.001215]` | `+2.07e-06` `[-2.70e-06, +6.83e-06]` | **`-0.000020`** `[-0.000036, -0.000004]` | **`+0.000022`** `[+0.000006, +0.000037]` |
| **`fixed_020`** | `+0.000156` `[-0.000166, +0.000478]` | **`+0.0000001`** `[-0.000036, +0.000036]` | **`-0.000526`** `[-0.000879, -0.000173]` | `+1.66e-06` `[-7.05e-06, +1.04e-05]` | **`-0.000030`** `[-0.000047, -0.000013]` | **`+0.000032`** `[+0.000012, +0.000052]` |
| **`fixed_025`** | **`-0.001205`** `[-0.001641, -0.000770]` | **`-0.000229`** `[-0.000293, -0.000165]` | **`-0.002334`** `[-0.002933, -0.001735]` | `-6.36e-07` `[-1.83e-05, +1.70e-05]` | **`-0.000042`** `[-0.000061, -0.000022]` | **`+0.000041`** `[+0.000016, +0.000066]` |
| **`rs_015_025`** | `+0.000098` `[-0.000182, +0.000379]` | `-0.000008` `[-0.000044, +0.000028]` | **`-0.000703`** `[-0.001059, -0.000347]` | `+3.57e-06` `[-4.82e-06, +1.20e-05]` | **`-0.000028`** `[-0.000047, -0.000010]` | **`+0.000032`** `[+0.000015, +0.000049]` |
| **`regime_switch_015_028`** | `-0.000245` `[-0.000514, +0.000023]` | **`-0.000064`** `[-0.000105, -0.000024]` | **`-0.001316`** `[-0.001766, -0.000867]` | `+3.88e-06` `[-5.98e-06, +1.37e-05]` | **`-0.000031`** `[-0.000050, -0.000012]` | **`+0.000035`** `[+0.000018, +0.000053]` |
| **`fixed_030`** | **`-0.003547`** `[-0.004489, -0.002605]` | **`-0.000818`** `[-0.001003, -0.000632]` | **`-0.004760`** `[-0.005815, -0.003705]` | `-4.91e-06` `[-3.67e-05, +2.69e-05]` | **`-0.000054`** `[-0.000078, -0.000030]` | **`+0.000049`** `[+0.000016, +0.000082]` |

### 4.2. Cơ Chế Giao Dịch: `Turnover`, Tần Suất Bất Động, Quy Mô Lệnh & Vị Thế Cực Đoan (`best_val`)

| Kịch bản | `Turnover` (`A / R2 / Δ`) | `Active Steps` (`/20`) | `Immobility` (`<1e-5`, `A / R2`) | `Immobility` (`<1e-4 / <1e-3`, `A` vs `R2`) | `Trade Size Active` (Pooled & Pathwise `A / R2 / Δ`) | `max_abs_delta` (`A / R2 / Δ`) | `p99_abs_delta` (`A / R2 / Δ`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`fixed_015`** | `1.0585 / 1.0393` (`-0.0192`) | `19.9960 / 19.9959` | `0.020% / 0.021%` | `0.19% / 1.94%` vs `0.20% / 2.01%` | `0.0529 / 0.0520` (`-0.00096`) | `1.624 / 1.517` (`-0.106`) | `0.995 / 0.967` (`-0.028`) |
| **`fixed_020`** | `1.2464 / 1.2176` (`-0.0288`) | `19.9972 / 19.9970` | `0.014% / 0.015%` | `0.15% / 1.49%` vs `0.15% / 1.54%` | `0.0623 / 0.0609` (`-0.00144`) | `2.027 / 1.865` (`-0.161`) | `1.152 / 1.105` (`-0.047`) |
| **`fixed_025`** | `1.4351 / 1.3961` (`-0.0390`) | `19.9975 / 19.9976` | `0.012% / 0.012%` | `0.12% / 1.20%` vs `0.13% / 1.26%` | `0.0718 / 0.0698` (`-0.00195`) | `2.461 / 2.250` (`-0.211`) | `1.311 / 1.243` (`-0.068`) |
| **`rs_015_025`** | `1.2364 / 1.2092` (`-0.0272`) | `19.9969 / 19.9967` | `0.016% / 0.016%` | `0.15% / 1.57%` vs `0.16% / 1.63%` | `0.0618 / 0.0605` (`-0.00136`) | `2.003 / 1.847` (`-0.156`) | `1.098 / 1.058` (`-0.040`) |
| **`regime_switch_015_028`** | `1.2898 / 1.2600` (`-0.0298`) | `19.9969 / 19.9968` | `0.015% / 0.016%` | `0.15% / 1.51%` vs `0.16% / 1.56%` | `0.0645 / 0.0630` (`-0.00149`) | `2.137 / 1.969` (`-0.168`) | `1.136 / 1.091` (`-0.045`) |
| **`fixed_030`** | `1.6243 / 1.5746` (`-0.0497`) | `19.9979 / 19.9978` | `0.010% / 0.011%` | `0.10% / 1.02%` vs `0.11% / 1.08%` | `0.0812 / 0.0787` (`-0.00248`) | `2.940 / 2.667` (`-0.272`) | `1.471 / 1.381` (`-0.091`) |

*(Lưu ý: Tỷ lệ quỹ đạo không có bước nào hoạt động `zero_active_path_frac = 0.0` và tỷ lệ `|delta| >= 4.5` gần biên cắt `±5.0` cũng bằng `0.0` trên toàn bộ `6/6` kịch bản; `frac_steps_abs_delta_gt2` tại `fixed_030` giảm từ `0.0802%` xuống `0.0392%`.)*

![All 6 Scenarios Summary](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/plots/fig2_cost10_6scenarios_best_val.png)
![Mechanism Turnover vs Immobility](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/plots/fig3_cost10_mechanism_gross_net_immobility.png)

---

## 5. Đối Chiếu Với `cost_s = 0.0` (`Job 16911`) & Công Bố Trung Thực Các Mặt Bất Lợi

1. **So sánh trực tiếp giữa hai thí nghiệm xác nhận (`Job 16911` `[4001..4024]` ở `0 bps` vs `Job 16914` `[5001..5024]` ở `10 bps`):**
   - **Phí `10 bps` làm giảm mạnh khối lượng giao dịch tuyệt đối của cả hai nhánh (~33%–38%):** Ví dụ tại `fixed_020`, `turnover` giảm từ `1.9242 -> 1.2464` (`A_DECAY`) và `1.7820 -> 1.2176` (`R2_DECAY`).
   - **Tác động lên `H1a` (`ΔHE_030`):** Biên lợi thế giảm từ `-0.004562` (`0 bps`) xuống **`-0.003547` (`10 bps`)**, đồng thời độ lệch chuẩn liên hạt giống tăng từ `0.001293` lên `0.002231`.
   - **Tác động lên `H1b` (`ΔOCE_020`):** Tại `0 bps`, `R2_DECAY` chịu phạt trung bình `+3.11e-05` (`+0.31 bp`, `2/24` seed vượt `+1e-4`); còn tại `10 bps`, do `R2_DECAY` tiết kiệm trung bình `3.01e-05` tiền phí so với `A_DECAY`, mức suy giảm `ΔOCE_020` thu hẹp về gần `0` (**`+1.04e-07`**, `0/24` seed vượt `+1e-4`), giúp `H1b` có biên an toàn trung bình rộng hơn dù `sd_ddof1` tăng từ `3.9e-05` lên `8.5e-05`.
2. **Mặt bất lợi cần công bố đầy đủ:**
   - **Tại vùng biến động thấp `fixed_015`:** `R2_DECAY` **kém hơn** `A_DECAY` về `HE` ròng (`+0.000438`, `22/24` seed dương), `CVaR95` (`+0.000723`, `19/24` seed dương), `VaR95` (`+0.000969`, `22/24` seed dương), và `OCE y_nom` (`+3.43e-05`, `18/24` seed dương, `4/24` seed vượt `+1e-4` tại `best_val`).
   - **Hạt giống bất lợi:** Seed `5004` là ngoại lệ `1/24` có `ΔHE_030 > 0` tại `best_val` (`+0.000368`); seed `5013` là ngoại lệ `1/24` ở `ep200` (`+0.000027`); và có `5–8/24` hạt giống (`5001, 5002, 5007, 5012, 5017`, cộng thêm `5024, 5015, 5021` tùy kịch bản) mà `R2_DECAY` phát sinh chi phí giao dịch/turnover cao hơn `A_DECAY`.

---

## 6. Trả Lời Trực Tiếp Sáu Câu Hỏi Kết Luận Của Nhiệm Vụ

1. **Câu hỏi 1 — Khi có chi phí giao dịch `10 bps`, `R2_matched_128` có tiếp tục vượt `A_ref_128` theo thiết kế khóa không?**
   **Có — ĐẠT chuẩn xác nhận `PASS_CONFIRMATORY_IUT` (`p_IUT = 3.745 × 10⁻⁶ << 0.025`, `24/24` seed sạch).** `R2_DECAY` giảm `HE` ròng tại `σ = 0.30` trung bình `-0.003547` (`[-0.004489, -0.002605]`, `p = 3.40e-08`), đồng thời giữ `ΔOCE_020` tại `+0.00000010` (cận trên 95% `+0.00003607 << +0.00010`, `p = 3.75e-06`), giữ nguyên kết luận ở cả `ep200` lẫn phép thử bỏ ngoại lệ `5024`.
2. **Câu hỏi 2 — Phần giảm `turnover`/`cost` đến từ giảm tần suất giao dịch hay giảm biên độ lệnh?**
   **Đến từ giảm biên độ lệnh (trade size) và kiềm chế vị thế cực đoan ở đuôi, KHÔNG phải giảm tần suất giao dịch (bất động).** Cả hai nhánh MLP `softplus` giao dịch trên `19.996–19.998 / 20` bước (`immobility_rate` `< 1e-5` chỉ khoảng `0.010%–0.021%`, ngay cả `< 1e-3` cũng chỉ `1.0%–2.0%`, không khác biệt giữa hai nhánh). Thay vào đó, `R2_DECAY` giảm `mean_trade_size_active` từ `-1.82%` (`fixed_015`) tới `-3.06%` (`fixed_030`) và giảm `max_abs_delta` `-10.6%` tại `fixed_030` (`2.940 -> 2.667`).
3. **Câu hỏi 3 — Phí `10 bps` làm thay đổi hiệu quả phòng hộ ròng và mức đánh đổi tại `σ = 0.20` như thế nào so với `cost_s = 0.0`?**
   So với `0 bps` (`Job 16911`), lợi thế tuyệt đối ở `σ = 0.30` thu hẹp nhẹ (`-0.004562 -> -0.003547`, tức `-11.7% -> -10.4%`) và phương sai liên hạt giống tăng (`0.001293 -> 0.002231`); bù lại, tại `σ = 0.20`, khoản tiết kiệm phí giao dịch `-0.000030` (`-0.30 bp`) giúp cân bằng hoàn toàn phần đánh đổi danh nghĩa `+0.31 bp` từng thấy ở `0 bps`, đưa `ΔOCE_020` trung bình về `+0.001 bp` (`0/24` seed vượt `+1.0 bp`).
4. **Câu hỏi 4 — Độ ổn định qua 24 hạt giống (`best_val` vs `ep200`, LOO, ngoại lệ `5004`/`5024`) như thế nào?**
   Rất ổn định về phán quyết IUT: `0/24` lần lật phán quyết LOO, `23/24` seed giảm `HE_030`, `24/24` seed đạt `OCE_020 < +ε`, Sign & Wilcoxon chính xác đều `p <= 1.49e-06`, `ep200` thậm chí mạnh hơn `best_val` (`p_IUT = 7.61e-07`), và loại bỏ ngoại lệ thuận lợi `5024` làm `p_IUT` giảm mạnh xuống `4.46e-10`. Tuy nhiên tính cá biệt từng seed vẫn tồn tại (`1/24` thua `HE_030` và `5–8/24` seed tăng `cost`/`turnover`).
5. **Câu hỏi 5 — Vấn đề phụ thuộc nguồn ngẫu nhiên từng phát hiện ở B3 đã được xử lý triệt để ở thiết kế mới chưa?**
   **Đã triệt tiêu ở tầng khóa hạt giống và kiểm chứng thực nghiệm:** khóa SHA-256 `R2_COST10_CONF_V1` rời nhau tuyệt đối trên `3,061` khóa lịch sử + `861` khóa mới (`0` trùng lặp `Val(s)/Train(s+1)`), cô lập `np.random` toàn cục (`try...finally set_state`), `408/408` mã băm tập dữ liệu độc nhất không giao nhau, và kiểm thử trực giao cầu Brownian đạt `|corr_bridge| < 0.015`.
6. **Câu hỏi 6 — Có cần thí nghiệm tiếp theo nào nữa không, hay kết luận đủ vững trong phạm vi `10 bps`?**
   Trong phạm vi câu hỏi nghiên cứu hiện tại (`R2_matched_128` vs `A_ref_128`, chi phí tuyến tính `c_s = 0.0010` / `10 bps`, mô hình Black-Scholes `20` bước), **bằng chứng xác nhận đã hoàn chỉnh và đủ vững để khép lại không cần mở rộng `N`**. Chỉ cần mở thí nghiệm mới khi đặt câu hỏi khác về phương pháp khác (ví dụ so sánh với kiến trúc dải không giao dịch thực thụ `NTBN` dưới chuẩn khóa sạch `R2_COST10_CONF_V1`, hoặc mức phí cao hơn `25–50 bps`).
