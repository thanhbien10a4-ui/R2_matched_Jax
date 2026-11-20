# Đính Chính Xuất Bản & Rà Soát Tiền Công Bố (`PUBLICATION_ERRATA.md`)

Tài liệu này rà soát toàn bộ số liệu gốc chưa làm tròn của hai thí nghiệm xác nhận ([robust_r2_matched_confirmatory/](../robust_r2_matched_confirmatory/) — **Job `16911`**, `cost_s = 0.0` và [robust_r2_matched_cost_confirmatory/](../robust_r2_matched_cost_confirmatory/) — **Job `16914`**, `cost_s = 0.0010` / `10 bps`) và **xác lập các quy tắc diễn giải bắt buộc trước khi công bố lên GitHub**, mà **không chỉnh sửa bất kỳ byte nào** trong dữ liệu, thiết kế khóa, mã nguồn đã chạy và bảng băm gốc.

---

## 1. Bảng Rà Soát Sáu Hạng Mục Từ CSV/JSON Chưa Làm Tròn

| Hạng mục kiểm chứng | Job `16911` (`cost_s = 0.0`, `[4001..4024]`) | Job `16914` (`cost_s = 0.0010`, `[5001..5024]`) | Trạng thái kiểm khớp |
| :--- | :--- | :--- | :---: |
| **1. Đủ seed, lượt chạy, kịch bản & bản lưu** | `24/24` seed, `48/48` lượt, `6` kịch bản, `576` dòng đánh giá, `288` dòng ghép cặp; **`48` `best_val.pkl` + `48` `final_with_opt.pkl`** (`96` `.pkl`) | `24/24` seed, `48/48` lượt, `6` kịch bản, `576` dòng đánh giá, `288` dòng ghép cặp; **`48` `best_val.pkl` + `48` `final_with_opt.pkl`** (`96` `.pkl`) | **KHỚP 100%** |
| **2. Hai điều kiện đồng chính, biên `ε`, kiểm định 1 phía & `p_IUT`** | `H1a`: `< 0` (`p_A = 5.6767e-15`); `H1b`: `< +0.00010` (`p_B = 4.3666e-09`); **`p_IUT = 4.3666e-09`** (`PASS_CONFIRMATORY_IUT`) | `H1a`: `< 0` (`p_A = 3.3962e-08`); `H1b`: `< +0.00010` (`p_B = 3.7454e-06`); **`p_IUT = 3.7454e-06`** (`PASS_CONFIRMATORY_IUT`) | **KHỚP 100%** |
| **3. KTC 95%, độ lệch chuẩn & số seed thuận/bất lợi (`best_val`)** | `ΔHE_030 = -0.0045615 ± 0.0012930`, `95% CI [-0.0051075, -0.0040156]` (`24/24 < 0`); `ΔOCE_020 = +0.00003109 ± 0.00003853`, `95% CI [+0.00001482, +0.00004737]` (`22/24 < +ε`, bất lợi `4020, 4023`) | `ΔHE_030 = -0.0035470 ± 0.0022315`, `95% CI [-0.0044893, -0.0026047]` (`23/24 < 0`, bất lợi `5004`); `ΔOCE_020 = +0.00000010 ± 0.00008518`, `95% CI [-0.00003586, +0.00003607]` (`24/24 < +ε`) | **KHỚP 100%** |
| **4. Tổng bước cập nhật & tổng lượt xử lý đường** | `200 ep × 156 bước = 31,200` bước/lượt; `31,200 × 128 = 3,993,600` lượt đường/lượt (`48 × 3,993,600 = 191,692,800` toàn job) | Đếm runtime xác thực trực tiếp: `actual_steps_unique = [31200]`, `actual_paths_unique = [3993600]` cho `48/48` lượt | **KHỚP 100%** |
| **5. Công thức phí, giao dịch mở đầu & tất toán** | `cost_s = 0.0`; giao dịch `j = 0..19` không phí; thanh toán tiền mặt `-(S_20 - K)^+` tại `j = 20` | `cost = sum_{j=0}^{19} \|a_j\| S_j (0.0010)` (tính phí cho giao dịch mở đầu `j = 0`, không phí tất toán `j = 20`); `max \|(gross - cost) - net\| = 3.84e-11` | **KHỚP 100%** |
| **6. Tham số chính sách vs tổng tham số được học** | `policy = 6,762`; tổng tham số học: `A_DECAY = 6,764` (`+ 2` OCE `y[0], y0[0]`), `R2_DECAY = 6,770` (`+ 8` OCE `y[0..3], y0[0..3]`) | `policy = 6,762` (sai khác khởi tạo `0.0`); tổng tham số học: `A_DECAY = 6,764`, `R2_DECAY = 6,770` | **KHỚP 100%** |

---

## 2. Sáu Quy Tắc Đính Chính Diễn Giải Bắt Buộc (`a` – `f`)

### `a.` So sánh giữa Job `16911` (`0 bps`) và Job `16914` (`10 bps`) là so sánh mô tả giữa hai nhóm seed độc lập
- Job `16911` dùng cohort `[4001..4024]`; Job `16914` dùng cohort khác biệt hoàn toàn `[5001..5024]`.
- Mọi đối chiếu thay đổi của `turnover`, `OCE` hay `HE` giữa mức `0 bps` và `10 bps` **chỉ là so sánh mô tả giữa hai thí nghiệm độc lập** (khác cả hạt giống lẫn phiên bản JAX trên đường chạy lại, dù cùng chạy trên CPU `dgx02` `JAX 0.4.34`). Không tuyên bố toàn bộ chênh lệch quan sát được giữa hai job là tác động nhân quả thuần túy của riêng chi phí giao dịch.

### `b.` Cách ly nguồn ngẫu nhiên: phân bổ luồng khóa là bằng chứng chính, mã băm và tương quan chỉ là kiểm tra bổ trợ
- Bằng chứng chính cho cách ly dữ liệu là **cơ chế dẫn xuất khóa 31-bit SHA-256 theo tuple đầy đủ** (`stage × seed × purpose × scenario × component`), tách riêng `train` / `val` / `conf_test` và không chia sẻ giữa bất kỳ hai seed nào, kèm bảo toàn trạng thái `np.random` toàn cục (`set_state`) và hoán vị minibatch cục bộ (`default_rng`), khắc phục tận gốc lỗi trùng `Val(s)/Train(s+1)` của B3.
- Số lượng `408/408` mã băm tập dữ liệu rời nhau và kiểm thử tương quan cầu Brownian (`|corr_bridge| < 0.015`) chỉ là **kiểm tra hậu kiểm/bổ trợ xác nhận không có lỗi lập trình hay trùng luồng**, tự thân chúng không phải phép chứng minh toán học cho "độc lập tuyệt đối 100%".

### `c.` Quy tắc xử lý ngoại lệ: toàn bộ `24/24` hạt giống nằm trong phân tích chính
- Tất cả `24/24` hạt giống đều được giữ nguyên trong phán quyết xác nhận chính thức, bao gồm **`5004`** (`ΔHE_030 = +0.000368`, điểm bất lợi duy nhất của `H1a` ở `best_val`), **`5024`** (`ΔHE_030 = -0.011686`, `ΔOCE_020 = -0.000356`, ngoại lệ thuận lợi mạnh trong Job `16914`), và **`4020`, `4023`** (hai seed vượt `+0.00010` trên `OCE_020` ở Job `16911`).
- Các phép tính loại `5024`, `5004` (cho thấy khi bỏ `5024` độ lệch chuẩn thu hẹp hơn nửa và `p_IUT` giảm về `4.46 × 10⁻¹⁰`) **chỉ được báo cáo như phân tích độ nhạy hậu nghiệm**, không thay thế mẫu chính thức `N = 24`.

### `d.` Phân biệt rõ `6,762` tham số chính sách (policy) với tổng số tham số được học
- Hai nhánh chỉ đối xứng từng bit trên **`6,762` tham số mạng chính sách (`policy`: `6,761` lớp Dense + `1` `idelta`)**.
- Do `R2_DECAY` ước lượng rủi ro OCE riêng cho `k_oce = 4` kịch bản huấn luyện còn `A_DECAY` chỉ dùng `k_oce = 1` kịch bản danh định, **tổng số tham số được học thực tế khác nhau**: **`6,764`** (`A_DECAY`) vs **`6,770`** (`R2_DECAY`). Tuyệt đối không tuyên bố "tổng tham số mô hình bằng nhau" khi thực chất mới chỉ kiểm tra bằng nhau ở phần `policy`.

### `e.` Diễn đạt chính xác trạng thái các dải hạt giống & dữ liệu niêm phong theo từng job
- Không dùng câu mơ hồ *“[4001..4040] không bị mở”* cho toàn bộ lịch sử:
  - Dải **`[4001..4024]` ĐÃ ĐƯỢC SỬ DỤNG** để chạy và đánh giá trong **Job `16911`** (`cost_s = 0.0`).
  - Trong **Job `16914`** (`cost_s = 0.0010`), **`[4001..4040]` không bị mở lại, không bị sinh lại và không bị dùng lại**.
  - Dải dự phòng **`[4025..4040]`** (thuộc bộ khóa `0 bps`), **`[5025..5040]`** (thuộc bộ khóa `10 bps`), và tập **`HOLD_EVAL`** của các nghiên cứu R3 trước đó **chưa từng được mở hay sinh dữ liệu kiểm tra** trong cả hai job.

### `f.` Ranh giới suy diễn thống kê và cơ chế giao dịch
1. **Không thua kém (`H1b`) $\neq$ Tương đương hay không suy giảm:** Đạt ngưỡng `< +0.00010` tại `σ = 0.20` nghĩa là bác bỏ giả thuyết suy giảm vượt quá `1.00 bp`; điều đó không chứng minh hai nhánh tương đương hệt nhau ở `σ = 0.20` (chẳng hạn tại `cost_s = 0.0`, `ΔOCE_020 = +0.31 bp` dương có ý nghĩa thống kê với KTC 95% `[+0.15, +0.47] bp`).
2. **Tỷ lệ seed thuận lợi (`23/24`, `24/24`) $\neq$ Xác suất thắng tương lai:** Đây là tần suất mẫu thực nghiệm trên `24` cặp hạt giống đã quan sát, không phải xác suất dự báo cho một hạt giống bất kỳ trong tương lai.
3. **`p_IUT` $\neq$ Xác suất kết luận sai:** `p_IUT = 4.37e-09` (`0 bps`) và `3.75e-06` (`10 bps`) là giá trị p-value quan sát theo giả thuyết không `H0a ∪ H0b`, không phải xác suất hậu nghiệm rằng `PASS` là sai.
4. **Giảm `turnover` $\neq$ Giảm tần suất giao dịch:** Dữ liệu `10 bps` cho thấy cả `A_DECAY` và `R2_DECAY` đều giao dịch `~19.996–19.998 / 20` bước (`immobility_rate < 1e-5` chỉ `~0.01%–0.02%`). Nguyên nhân giảm `turnover` và `mean_cost` ở `R2_DECAY` là **giảm quy mô lệnh khi giao dịch (`trade_size_active` giảm `-1.82%` tới `-3.06%`) và hạ đỉnh vị thế (`max_abs_delta`)**, không phải đứng yên bỏ bước giao dịch.
5. **Bất lợi cấu trúc tại `fixed_015` & phạm vi hiệu lực:** Ở cả `0 bps` lẫn `10 bps`, `R2_DECAY` đều tăng sai số phòng hộ và `CVaR95` tại vùng biến động thấp `σ = 0.15`. Kết quả chỉ có giá trị trong bộ giả lập `20` bước (`S0 = K = 1.0`, `T = 0.40`, `Black–Scholes / Regime-Switch`, quyền chọn mua ATM bán khống, phí tuyến tính `0` hoặc `10 bps`).
