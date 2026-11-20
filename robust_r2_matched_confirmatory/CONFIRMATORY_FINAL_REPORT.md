# Báo Cáo Tổng Kết Thí Nghiệm Xác Nhận Độc Lập: `R2_matched_128` vs `A_ref_128` (`CONFIRMATORY_FINAL_REPORT.md`)

- **Thiết kế khóa chính thức:** [R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md)
- **Đính chính hồ sơ phát triển:** [DEVELOPMENT_ERRATA.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DEVELOPMENT_ERRATA.md)
- **Thực thi Slurm chính thức:** **Job `16911`** (`r2_conf_n24`, `defq`, node `dgx02`, `8 CPU, 0 GPU`), trạng thái **`COMPLETED` (`ExitCode 0:0`)**, thời lượng **`03:50:53`** (`13,836.36s` wall-clock Python), **`MaxRSS = 1,220,600K`** (`sacct`) / `1,217,332 KB` (Python) — nhật ký: [slurm_conf_n24_16911.log](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/slurm_conf_n24_16911.log).
- **Quy mô thực thi:** **`N = 24`** hạt giống độc lập chưa từng thấy **`[4001..4024]`** × `2` nhánh đối xứng (`A_DECAY`, `R2_DECAY`) = **đúng `48` lượt huấn luyện** (`31,200` bước/lượt, `3,993,600` đường xử lý/lượt, `Cosine LR Decay`, `Adam eps=1e-7`, **`cost_s = 0.0`**), đánh giá trên **`20,000` đường/kịch bản** (`6` kịch bản, sinh sau khi chốt xong `best_val`). `HOLD_EVAL` và `[4025..4040]` giữ kín hoàn toàn.

> [!IMPORTANT]
> **PHÁN QUYẾT XÁC NHẬN CHÍNH (`PASS_CONFIRMATORY_IUT`):**
> Trên cả `24/24` hạt giống hợp lệ (`0` thiếu, `0` `NaN`/`Inf`, `0` lỗi thuật toán, tái nạp bản lưu khớp `< 7.45e-09`), thí nghiệm đạt đồng thời **cả hai điều kiện thành công theo quy tắc Berger Intersection-Union Test (IUT) ở mức `α = 0.025` một phía** (`p_IUT = max(p_A, p_B) = 4.37 × 10⁻⁹ << 0.025`):
> 1. **Điều kiện chính A (`H1a` — Giảm `HE` tại `σ = 0.30`, `best_val`):** `ΔHE_030 = -0.004562` (`-11.7%`), **`95% CI = [-0.005108, -0.004016]`** (cận trên `< 0.0`), `t(23) = -17.283`, `p = 5.68 × 10⁻¹⁵`, **`24/24` seed đều giảm**.
> 2. **Điều kiện chính B (`H1b` — Không thua kém `OCE` tại `σ = 0.20`, biên `ε = +0.00010`, `best_val`):** `ΔOCE_020 = +0.00003109` (`+0.31 bp`), **`95% CI = [+0.00001482, +0.00004737]`** (cận trên `+0.00004737 < +0.00010`), `t(23) = -8.760`, `p = 4.37 × 10⁻⁹` (có `2/24` seed cá biệt vượt `+0.00010`: `4020`, `4023`).
>
> **Giới hạn hợp lệ:** Kết luận này chỉ áp dụng trong môi trường không phí giao dịch **`cost_s = 0.0`** và thiết kế `R2_matched_128` vs `A_ref_128` đã khóa.

---

## 1. Xác Minh Toàn Vẹn Hạ Tầng, Mã Băm & Tái Nạp Bản Lưu

| Hạng mục kiểm chứng | Kết quả thực đo | Nguồn đối chiếu |
| :--- | :--- | :--- |
| **Hoàn tất 48 lượt & Sổ cái seed** | **`24/24` seed `COMPLETED`**, `failed_seeds = []`, `missing_seeds = []`, `non_finite_seeds = []` | [SEED_LEDGER.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/SEED_LEDGER.json) |
| **Khớp mã băm `SHA-256` khóa** | Khớp **17/17 tệp** giữa [DESIGN_LOCK_SHA256.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DESIGN_LOCK_SHA256.json) và [CONFIRMATORY_STATS_SUMMARY.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/CONFIRMATORY_STATS_SUMMARY.json) | [slurm_conf_n24_16911.log](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/slurm_conf_n24_16911.log) |
| **Tái nạp `48/48` `best_val.pkl` & `96/96` `.pkl`** | `48/48` `val_nom` reload parity `<= 7.451 × 10⁻⁹` (`< 1e-6`); `96/96` `.pkl` (`> 1 KB`) + `48/48` `checkpoint_manifest.json` nguyên vẹn | [POSTRUN_AUDIT_TABLES.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/POSTRUN_AUDIT_TABLES.json) |
| **Tính lại thống kê độc lập từ CSV** | Khớp từng bit giữa `CONFIRMATORY_STATS_SUMMARY.json` và `conf_stats.py --paired-csv PAIRED_CONFIRMATORY.csv` | [STATS_RECOMPUTED_CLI.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/STATS_RECOMPUTED_CLI.json) |
| **Thời gian & tài nguyên CPU** | `A_DECAY` mean `264.72s`, `R2_DECAY` mean `302.36s`, tổng `03:50:53` (`~3.85h` vs ước lượng `3.86h`) | `sacct -j 16911` |
| **Bảo toàn `PAUSE_COORDINATOR` & `HOLD_EVAL`** | `PAUSE_COORDINATOR` giữ nguyên `32 bytes` (`Nov 16 10:17`); `HOLD_EVAL` và `[4025..4040]` chưa bao giờ mở | Kiểm tra trực tiếp máy chủ |

---

## 2. Kết Quả Hai Điều Kiện Chính (Berger IUT, `best_val`, `20,000` Đường Test)

Số liệu chuẩn hóa từ [STATS_RECOMPUTED_CLI.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/STATS_RECOMPUTED_CLI.json) và [SECONDARY_SUMMARY_TABLE.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/SECONDARY_SUMMARY_TABLE.csv):

| Tiêu chí | `A_DECAY` (`Mean ± SD`) | `R2_DECAY` (`Mean ± SD`) | Chênh lệch `Δ = R2 - A` (`Mean ± SD`) | KTC `95%` hai phía (`CI_low, CI_high`) | `t(23)` / `p` 1 phía (`α=0.025`) | Độ vững (LOO / Sign / Wilcoxon chính xác) | Kết luận thành phần |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`H1a`: `HE` (`he_net_ddof0`) tại `fixed_030`** (`ngưỡng < 0`) | `0.038996 ± 0.001723` | `0.034435 ± 0.001274` | **`-0.004562 ± 0.001293`** (`med -0.004473`) | **`[-0.005108, -0.004016]`** | `t = -17.283`, **`p = 5.68e-15`** | `0` LOO flips (`[-0.004708, -0.004404]`); `24/24 < 0` (`Sign p = 5.96e-08`, `Wilcoxon W+=0, p = 5.96e-08`) | **ĐẠT (`PASS`)** |
| **`H1b`: `OCE` (`hedged_oce_y_nom`) tại `fixed_020`** (`ngưỡng < +0.00010`) | `0.051631 ± 0.000088` | `0.051662 ± 0.000085` | **`+0.000031 ± 0.000039`** (`med +0.000029`) | **`[+0.000015, +0.000047]`** (`UCL +0.00004737`) | `t = -8.760`, **`p = 4.37e-09`** | `0` LOO flips (`[+0.000028, +0.000035]`); `22/24 < +ε` (`Sign p = 1.79e-05`, `Wilcoxon W+=3, p = 2.98e-07`) | **ĐẠT (`PASS`)** |

**Kiểm tra độ nhạy tại `ep200` (loại bỏ hoàn toàn bước chọn `best_val` trên tập validation):**
- **`ep200` `ΔHE_030`:** Mean **`-0.004901 ± 0.001503`**, **`95% CI = [-0.005536, -0.004266]`**, `t(23) = -15.973`, `p = 3.05 × 10⁻¹⁴` (`24/24 < 0`, `0` LOO flips) -> **ĐẠT**.
- **`ep200` `ΔOCE_020`:** Mean **`+0.00003771 ± 0.00004745`**, **`95% CI = [+0.00001767, +0.00005775]`**, `t(23) = -6.430`, `p = 7.31 × 10⁻⁷` (`21/24 < +0.00010`, `0` LOO flips) -> **ĐẠT**.
- **Nhận xét khoa học & liên hệ bảng công suất thiết kế (`R1–R8`):** Tồn tại một mức đánh đổi trung bình thực sự nhưng rất nhỏ ở nominal `σ = 0.20` (`+0.31 bp` ở `best_val`, KTC 95% nằm hoàn toàn dương `[+0.15, +0.47] bp`), song vẫn thấp hơn rõ rệt so với biên không thua kém `ε = +1.00 bp` (`+0.00010`) trên cả `best_val` lẫn `ep200`. Với bộ tham số thực đo `(mean_A = -0.004562, sd_A = 0.001293, mean_B = +0.000031, sd_B = 0.000039, rho_AB = -0.2012)`, độ lệch chuẩn thực tế `sd_B = 0.000039` thậm chí nhỏ hơn mốc cơ sở `R1` (`0.000050`), tạo độ tách `8.76 SE` khỏi ngưỡng `+0.00010` và cho thấy cỡ mẫu `N = 24` kết hợp `20,000` đường đánh giá đạt độ phân giải dứt khoát (không rơi vào kịch bản thiếu lực `R8`).

![Primary Paired Confirmatory Results](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/plots/fig1_confirmatory_primary_paired_n24.png)

---

## 3. Công Bố Đầy Đủ Toàn Bộ 24 Hạt Giống & Các Trường Hợp Bất Lợi

Bảng đầy đủ toàn bộ 24 seed từ [POSTRUN_AUDIT_TABLES.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/POSTRUN_AUDIT_TABLES.json) và [PAIRED_CONFIRMATORY.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/PAIRED_CONFIRMATORY.csv) (đánh dấu rõ các seed cá biệt vượt `+0.00010` ở `σ = 0.20`):

| Seed | `A` / `R2` `best_ep` | `A HE_030` | `R2 HE_030` | `ΔHE_030` (`best`) | `A OCE_020` | `R2 OCE_020` | `ΔOCE_020` (`best`) | `ΔHE_030` (`ep200`) | `ΔOCE_020` (`ep200`) | Ghi chú bất lợi (`σ=0.20`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `4001` | `195 / 188` | `0.038917` | `0.034421` | `-0.004495` | `0.051496` | `0.051540` | `+0.000044` | `-0.005289` | `+0.000013` | `—` |
| `4002` | `132 / 116` | `0.039379` | `0.035559` | `-0.003820` | `0.051655` | `0.051702` | `+0.000047` | `-0.003756` | `+0.000010` | `—` |
| `4003` | `196 / 147` | `0.038235` | `0.034434` | `-0.003801` | `0.051566` | `0.051620` | `+0.000054` | `-0.003455` | `+0.000017` | `—` |
| `4004` | `76 / 194` | `0.041676` | `0.036111` | `-0.005566` | `0.051562` | `0.051550` | `-0.000013` | `-0.007512` | `-0.000013` | `—` |
| `4005` | `141 / 170` | `0.039881` | `0.036157` | `-0.003724` | `0.051844` | `0.051857` | `+0.000013` | `-0.004478` | `+0.000092` | `—` |
| `4006` | `193 / 193` | `0.043346` | `0.035152` | `-0.008194` | `0.051529` | `0.051554` | `+0.000025` | `-0.007322` | `+0.000056` | `—` |
| `4007` | `105 / 187` | `0.038519` | `0.033984` | `-0.004535` | `0.051719` | `0.051783` | `+0.000064` | `-0.006705` | `+0.000081` | `—` |
| `4008` | `186 / 189` | `0.038601` | `0.034163` | `-0.004438` | `0.051645` | `0.051679` | `+0.000034` | `-0.004278` | `+0.000048` | `—` |
| `4009` | `165 / 197` | `0.038446` | `0.033495` | `-0.004951` | `0.051623` | `0.051673` | `+0.000050` | `-0.005263` | **`+0.000113`** | Vượt `+ε` ở `ep200` |
| `4010` | `42 / 194` | `0.036790` | `0.033682` | `-0.003108` | `0.051734` | `0.051678` | `-0.000055` | `-0.004502` | `+0.000036` | `—` |
| `4011` | `150 / 158` | `0.041242` | `0.037611` | `-0.003631` | `0.051601` | `0.051615` | `+0.000014` | `-0.004194` | `+0.000011` | `—` |
| `4012` | `191 / 195` | `0.041087` | `0.035731` | `-0.005356` | `0.051586` | `0.051582` | `-0.000004` | `-0.005155` | `-0.000032` | `—` |
| `4013` | `146 / 134` | `0.037883` | `0.033600` | `-0.004282` | `0.051806` | `0.051817` | `+0.000011` | `-0.004800` | `-0.000034` | `—` |
| `4014` | `199 / 160` | `0.038235` | `0.034382` | `-0.003852` | `0.051572` | `0.051604` | `+0.000032` | `-0.003067` | `+0.000098` | `—` |
| `4015` | `188 / 195` | `0.040195` | `0.034336` | `-0.005859` | `0.051593` | `0.051610` | `+0.000016` | `-0.005747` | `+0.000038` | `—` |
| `4016` | `198 / 180` | `0.036861` | `0.032779` | `-0.004081` | `0.051678` | `0.051747` | `+0.000069` | `-0.004475` | `-0.000009` | `—` |
| `4017` | `200 / 174` | `0.035513` | `0.034319` | `-0.001194` | `0.051771` | `0.051790` | `+0.000018` | `-0.000915` | `+0.000030` | `ΔHE_030` nhỏ nhất (`-11.9 bp`) |
| `4018` | `162 / 162` | `0.038778` | `0.034256` | `-0.004521` | `0.051618` | `0.051627` | `+0.000009` | `-0.004771` | `+0.000001` | `—` |
| `4019` | `137 / 197` | `0.039267` | `0.034363` | `-0.004905` | `0.051576` | `0.051650` | `+0.000074` | `-0.006035` | `+0.000080` | `—` |
| **`4020`** | `200 / 199` | `0.037068` | `0.031561` | `-0.005507` | `0.051577` | `0.051689` | **`+0.00011248`** | `-0.005154` | **`+0.00013831`** | **Vượt `+ε` (`best` & `ep200`)** |
| `4021` | `186 / 145` | `0.040015` | `0.035564` | `-0.004451` | `0.051605` | `0.051603` | `-0.000002` | `-0.003047` | `-0.000008` | `—` |
| `4022` | `61 / 184` | `0.038600` | `0.033309` | `-0.005291` | `0.051625` | `0.051611` | `-0.000014` | `-0.006412` | `+0.000018` | `—` |
| **`4023`** | `151 / 200` | `0.039191` | `0.033023` | `-0.006169` | `0.051538` | `0.051647` | **`+0.00010930`** | `-0.006810` | **`+0.00010018`** | **Vượt `+ε` (`best` & `ep200`)** |
| `4024` | `145 / 182` | `0.038184` | `0.034437` | `-0.003747` | `0.051622` | `0.051660` | `+0.000038` | `-0.004481` | `+0.000019` | `—` |

---

## 4. Toàn Bộ Chỉ Tiêu Phụ Trên Cả 6 Kịch Bản (`best_val`, `N = 24`)

Tổng hợp đầy đủ từ [SECONDARY_SUMMARY_TABLE.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/SECONDARY_SUMMARY_TABLE.csv) và [EXTRA_POSITION_SUMMARY.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/results/confirmatory_n24/EXTRA_POSITION_SUMMARY.json) (giá trị âm `Δ < 0` là `R2_DECAY` tốt/thấp hơn `A_DECAY`):

| Kịch bản (`best_val`) | `ΔHE` (`Mean` `[95% CI]`) | `ΔOCE y_nom` (`Mean` `[95% CI]`) | `ΔOCE y_scen` (`Mean` `[95% CI]`, chẩn đoán) | `ΔCVaR95` (`Mean` `[95% CI]`) | `ΔTurnover` (`Mean` `[95% CI]`) | `Δmax_abs_delta` (`Mean`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`fixed_015`** *(vùng thấp)* | **`+0.000695`** `[+0.000585, +0.000805]` | **`+0.000102`** `[+0.000076, +0.000128]` | `-0.000648` `[-0.000674, -0.000623]` | **`+0.001097`** `[+0.000869, +0.001324]` | `-0.1076` `[-0.1368, -0.0784]` | `-0.0968` |
| **`fixed_020`** *(nominal)* | `+0.000040` `[-0.000036, +0.000116]` | **`+0.000031`** `[+0.000015, +0.000047]` | `+0.000031` `[+0.000015, +0.000047]` | `-0.000260` `[-0.000459, -0.000061]` | `-0.1422` `[-0.1825, -0.1020]` | `-0.1379` |
| **`fixed_025`** *(nội miền cao)* | `-0.001973` `[-0.002219, -0.001727]` | `-0.000296` `[-0.000345, -0.000248]` | `-0.001107` `[-0.001155, -0.001058]` | `-0.002074` `[-0.002406, -0.001742]` | `-0.1772` `[-0.2290, -0.1255]` | `-0.1849` |
| **`rs_015_025`** *(nội miền chuyển chế độ)* | `-0.000146` `[-0.000217, -0.000075]` | `+0.000007` `[-0.000009, +0.000023]` | `-0.000003` `[-0.000020, +0.000013]` | `-0.000609` `[-0.000824, -0.000393]` | `-0.1400` `[-0.1793, -0.1006]` | `-0.1488` |
| **`regime_switch_015_028`** *(OOD)* | `-0.000800` `[-0.000913, -0.000687]` | `-0.000091` `[-0.000116, -0.000066]` | `—` | `-0.001349` `[-0.001621, -0.001077]` | `-0.1497` `[-0.1922, -0.1072]` | `-0.1668` |
| **`fixed_030`** *(OOD stress chính)* | **`-0.004562`** `[-0.005108, -0.004016]` | `-0.001039` `[-0.001175, -0.000903]` | `—` | `-0.004377` `[-0.004972, -0.003782]` | `-0.2126` `[-0.2762, -0.1490]` | `-0.2205` |

**Các phát hiện phụ đáng chú ý (công bố trung thực cả mặt thuận lợi lẫn bất lợi):**
1. **Giảm `turnover` và vị thế cực đoan trên toàn bộ `6/6` kịch bản:** Khi so trực tiếp `R2_DECAY` vs `A_DECAY` ở `cost_s = 0.0`, `R2_DECAY` giảm khối lượng giao dịch trung bình từ `-0.1076` (`fixed_015`) đến `-0.2126` (`fixed_030`, giảm `~8.1%`). Theo tỷ lệ hạt giống tại `best_val`, `turnover` giảm ở **`23/24` seed trên cả 6 kịch bản** (ngoại lệ duy nhất là **seed `4010`** có `Δturnover > 0` từ `+0.0640` đến `+0.2292` gắn với `A_DECAY` chọn `best_ep = 42` sớm hơn nhiều so với `R2_DECAY` `best_ep = 194`; ở `ep200`, `4010` lại giảm turnover và ngoại lệ duy nhất chuyển sang **seed `4023`** `+0.0067..+0.0151`). Tương tự, `max_abs_delta` giảm trên `24/24` seed ở 5 kịch bản đầu và `23/24` seed ở `fixed_030` (ngoại lệ nhẹ ở **seed `4005`**).
2. **Đánh đổi bất lợi tại vùng biến động thấp `fixed_015`:** Khi môi trường rơi xuống `σ = 0.15`, `R2_DECAY` kém hơn `A_DECAY` trên cả `HE` (`+0.000695`, **`23/24` seed dương** ở `best_val`, `24/24` ở `ep200`), `CVaR95` (`+0.001097`, **`23/24` seed dương**), lẫn `hedged_oce_y_nom` (`+0.000102`, `22/24` seed dương, `14/24 > +ε`). Lưu ý phân tách cơ chế: việc dùng `y_scen` thay `y_nom` chỉ làm đảo dấu phần `OCE` (`ΔOCE_y_scen = -0.000648`), **còn `HE` và `CVaR95` vốn bất biến với tham số vô hướng `y`** — do đó mức tăng `+6.95 bp` ở `HE` và `+10.97 bp` ở `CVaR95` là đánh đổi cấu trúc thực chất của chính sách đa kịch bản tại vùng biến động thấp. Ngược lại, từ `σ = 0.25`, `rs_015_025`, `regime_switch_015_028` đến `fixed_030`, `R2_DECAY` vượt trội rõ rệt trên cả `HE`, `CVaR95`, `OCE`, lẫn `turnover`.

![All 6 Scenarios Summary](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/plots/fig2_confirmatory_6scenarios_best_val.png)
![Sensitivity best_val vs ep200 and LOO](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/plots/fig3_confirmatory_best_val_vs_ep200_and_loo.png)
