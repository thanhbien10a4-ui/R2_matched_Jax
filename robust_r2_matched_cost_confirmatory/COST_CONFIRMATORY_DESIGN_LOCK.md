# Thiết Kế Khóa Thí Nghiệm Xác Nhận Có Chi Phí Giao Dịch 10 bps (`COST_CONFIRMATORY_DESIGN_LOCK.md`)

> [!IMPORTANT]
> **TRẠNG THÁI CHÍNH THỨC: `DESIGN LOCKED — PRE-EXECUTION (ĐÃ QUA PHẢN BIỆN @gng5 TIỀN NGHIỆM)`** (`N = 24`, Seeds `[5001..5024]`, `cost_s = 0.0010` / `10 bps`).
> Phiên phản biện tiền nghiệm đơn lẻ `@gng5` (`gemini-4-argon-high`) đã được thực hiện trước khi khóa ([gng5_pre_audit_text.md](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/logs/gng5_pre_audit_text.md)); toàn bộ phát hiện `[BLOCKER]` (`jax.lax.scan`), `[MEDIUM]` (phân loại `classify_cost_run_exception`, xóa dòng dở dang khi lỗi, lưu `reload_checks`/`dataset_hashes` trong `SEED_LEDGER.json`), và `[LOW]` đã được vá và kiểm thử xong trước khi chốt `SOURCE_SHA256SUMS.txt`. Kể từ khi khóa, **nghiêm cấm** thay đổi siêu tham số, tăng `N`, mở rộng kịch bản phí, hoặc dừng sớm theo kết quả. `HOLD_EVAL` và `[5025..5040]` tiếp tục giữ kín tuyệt đối.

---

## 1. Câu Hỏi Nghiên Cứu, Bản Chất & Phạm Vi Ràng Buộc

**Câu hỏi nghiên cứu chính:**
> *“Khi đưa chi phí giao dịch vào cả huấn luyện lẫn đánh giá, huấn luyện đa kịch bản `R2_matched_128` có tiếp tục giảm sai số phòng hộ ròng ở `σ = 0.30`, đồng thời giữ mức suy giảm `OCE` tại `σ = 0.20` trong biên đã định trước (`ε = +0.00010`), so với `A_ref_128` có cùng ngân sách?”*

**Phân biệt rõ ranh giới phương pháp:**
1. **`R2_matched_128` (`R2_DECAY`):** Tối ưu kỳ vọng với trọng số đều `1/4` trên 4 kịch bản `("fixed_015", "fixed_020", "fixed_025", "rs_015_025")`, **không phải R3 minimax có ràng buộc**.
2. **Thí nghiệm mới hoàn toàn ở `c_s = 0.0010` (`10 bps`):** Phí giao dịch tỷ lệ thuận đi xuyên cả **huấn luyện, chọn bản lưu `best_val`, và kiểm tra `conf_test`**. Đây **không phải** phép trừ phí hậu nghiệm lên kết quả `cost_s = 0.0` của Job `16911`.
3. **Không đánh đồng giảm `turnover` với giảm tần suất giao dịch:** Công bố tách bạch cả `turnover`, `active_steps_gt1e5`, `immobility_rate` (`|a| < 1e-5`), lẫn quy mô lệnh khi hoạt động (pooled & pathwise).
4. **Cách ly so với B3:** Thí nghiệm này không có nhánh B3 và không hồi tố thay đổi kết luận B3; nó triệt tiêu lỗi `Val(s)/Train(s+1)` ngay từ kiến trúc khóa `R2_COST10_CONF_V1` và kiểm định riêng câu hỏi về `R2_matched_128` vs `A_ref_128` dưới `10 bps`.

---

## 2. Cấu Hình Đối Xứng, Dữ Liệu & Quy Trình Thực Thi

| Tham số khóa | `A_DECAY` (`A_ref_128`) | `R2_DECAY` (`R2_matched_128`) |
| :--- | :--- | :--- |
| **Danh sách seed chính (`N = 24`)** | `[5001..5024]` (`[5025..5040]` niêm phong ngoài; `5999` chỉ smoke; `[4001..4040]` & `HOLD_EVAL` không đụng) | Giống hệt `A_DECAY` (ghép cặp 1:1 theo seed) |
| **Chi phí giao dịch `cost_s`** | `0.0010` (`10 bps`) tại **Train, Val (`best_val`), Test** | `0.0010` (`10 bps`) tại **Train, Val (`best_val`), Test** |
| **Kiến trúc policy & tham số** | MLP `5 × 40` `softplus` + `idelta` (`6,762` tham số), `clip[-5, +5]` | Giống hệt từng bit khi khởi tạo (`6,762` tham số policy) |
| **Tham số OCE phụ trợ** | `k_oce = 1` (`y[0], y0[0]`, tổng `6,764` params) | `k_oce = 4` (`y[0..3], y0[0..3]`, tổng `6,770` params) |
| **Dữ liệu huấn luyện** | `4 × 5,000 = 20,000` đường `fixed_020` | `4 × 5,000 = 20,000` đường trên 4 kịch bản huấn luyện |
| **Ngân sách & Tối ưu** | `200` ep × `156` bước (`4×32=128` đường/bước) = **`31,200` bước**, **`3,993,600` lượt xử lý**; `Adam(β1=0.9, β2=0.999, ε=1e-7)` + `Cosine LR Decay (1e-3 → 1e-4)` | Đối xứng 100% với `A_DECAY` |
| **Chọn bản lưu (`best_val`)** | Epoch có `val_fixed_020` (`5,000` đường, sau phí `10 bps`) thấp nhất | Epoch có trung bình cộng 4 `val` huấn luyện (`4 × 5,000`, sau phí `10 bps`) thấp nhất |
| **Kiểm tra (`conf_test`)** | `20,000` đường/kịch bản × `6` kịch bản (`cost_s = 0.0010`), chỉ tạo **SAU KHI** `best_val` & `ep200` chốt và reload parity `< 1e-6` đạt | Chung chuỗi `conf_test_shared_dW` trong mỗi seed `s` |

---

## 3. Hai Điều Kiện Thành Công Chính, `α = 0.025` & Phán Quyết Berger IUT

Trên `best_val.pkl` tại `conf_test` (`20,000` đường/kịch bản, sau phí `10 bps`), đặt `Δ_i = R2_DECAY_i - A_DECAY_i` (`i = 1..24`):
1. **Điều kiện chính A (`H1a` — Giảm `HE` ròng tại `σ = 0.30`):** `H0a: μ(ΔHE_030) >= 0` vs `H1a: μ(ΔHE_030) < 0` trên `he_net_ddof0 = std(payoff + pnl - cost, ddof=0)`. Đạt khi `p_A < 0.025` (tức cận trên KTC 95% hai phía `< 0.0`).
2. **Điều kiện chính B (`H1b` — Không thua kém `OCE` ròng tại `σ = 0.20`, biên `ε = +0.00010`):** `H0b: μ(ΔOCE_020) >= +0.00010` vs `H1b: μ(ΔOCE_020) < +0.00010` trên `hedged_oce_y_nom` (dùng `y[0]` của `A` và `y[1]` của `R2` đóng băng từ huấn luyện). Đạt khi `p_B < 0.025` (tức cận trên KTC 95% hai phía `< +0.00010`).
3. **Phán quyết chung:** Chỉ kết luận **`PASS_CONFIRMATORY_IUT`** khi đạt **đồng thời A & B tại `α = 0.025`** VÀ **`24/24` seed `[5001..5024]` hoàn tất sạch (`complete_and_clean_gate == True`)**. Nếu không, kết luận **`FAIL_CONFIRMATORY_IUT`** hoặc **`INCOMPLETE_OR_FAILED_AUDIT`**. Tuyệt đối không thay thế bằng `ep200`, `CVaR95`, `turnover`, `gross_gains`, hay kịch bản phụ.

---

## 4. Độ Nhạy Công Suất Thống Kê Trước Chạy (`SAMPLE_SIZE_SENSITIVITY_COST.csv`) & Giới Hạn Ở `10 bps`

> [!WARNING]
> **Cảnh báo trung thực trước khi chạy `c_s = 0.0010`:** Các kịch bản `R1–R8` dưới đây kế thừa giả định từ môi trường không phí (`c_s = 0.0`), trong khi hiệu ứng và độ lệch chuẩn thực tế của `ΔHE_030` và `ΔOCE_020` **dưới mức phí `10 bps` chưa từng được đo trước thí nghiệm này**. Kịch bản bổ sung **`R9_Cost10_Unmeasured_FeeStress`** minh họa rủi ro nếu phí `10 bps` thu hẹp biên còn lại của `H1b` hoặc làm tăng phương sai liên hạt giống. **Công suất thống kê là đại lượng có điều kiện dưới giả định tham số, KHÔNG phải "xác suất thành công của phương pháp".**

Trích từ [SAMPLE_SIZE_SENSITIVITY_COST.csv](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/SAMPLE_SIZE_SENSITIVITY_COST.csv) (`N = 24`, `α = 0.025`, `ε = 0.00010`, MC `B = 60,000`):

| Kịch bản | Giả định `(μ_A, sd_A, μ_B, sd_B, ρ_AB)` | `P_A` (`nct`) | `P_B` (`nct`) | Cận Fréchet | **Công suất chung MC (`N=24`)** | `N` cho `90%` (`H1b`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `R1_ZeroCost_PointEst` | `(-0.0048, 0.0008, 0.00000, 0.000050, 0.0)` | `1.00000` | `1.00000` | `1.00000` | **`1.00000`** | `5` |
| `R2_Realistic_Base` | `(-0.0038, 0.0012, 0.00001, 0.000065, 0.0)` | `1.00000` | `1.00000` | `1.00000` | **`1.00000`** | `8` |
| `R3_Conservative_Design` | `(-0.0030, 0.0015, 0.00002, 0.000075, 0.0)` | `1.00000` | `0.99881` | `0.99881` | **`0.99885`** | `12` |
| `R4_NegCorr_Conservative` | `(-0.0030, 0.0015, 0.00002, 0.000075, -0.3)` | `1.00000` | `0.99881` | `0.99881` | **`0.99903`** | `12` |
| `R5_PosCorr_Conservative` | `(-0.0030, 0.0015, 0.00002, 0.000075, +0.4)` | `1.00000` | `0.99881` | `0.99881` | **`0.99880`** | `12` |
| `R6_HighDrift_Stress` | `(-0.0025, 0.0018, 0.00003, 0.000085, 0.0)` | `1.00000` | `0.97135` | `0.97134` | **`0.97283`** | `18` |
| `R7_HeavyVariance_FIXEDlike` | `(-0.0030, 0.0025, 0.00002, 0.000110, 0.0)` | `0.99987` | `0.92656` | `0.92643` | **`0.92832`** | `22` |
| **`R8_Worst_40pctMargin_FIXEDsd`** | `(-0.0020, 0.0025, 0.00004, 0.000120, -0.3)` | `0.96334` | `0.65038` | `0.61373` | **`0.62285`** | `44` |
| **`R9_Cost10_Unmeasured_FeeStress`** | `(-0.0020, 0.0025, 0.00006, 0.000150, -0.3)` | `0.96334` | `0.23955` | `0.20290` | **`0.22183`** | `150` |

---

## 5. Chính Sách Tiếp Tục (`--seed-offset`) & Phân Biệt Lỗi

- `--seed-offset` chỉ chạy trong **`[5001..5024]`** (từ chối mọi lát chạy vượt `5024`).
- Khi chạy tiếp sau gián đoạn hạ tầng (`INFRA_ERROR`), mã tự động đọc `SEED_LEDGER.json` & CSV hiện có để **bỏ qua các seed `COMPLETED`, giữ nguyên các seed `ALGORITHM_FAILURE` (`NaN`/`Inf`/reload mismatch — tuyệt đối không bỏ hay thay seed bất lợi)**, và chỉ chạy lại các seed thiếu hoặc lỗi hạ tầng.
- Nếu còn bất kỳ seed nào thiếu hoặc `ALGORITHM_FAILURE`, cổng `complete_and_clean_gate` khóa cứng phán quyết ở `INCOMPLETE_OR_FAILED_AUDIT`.
