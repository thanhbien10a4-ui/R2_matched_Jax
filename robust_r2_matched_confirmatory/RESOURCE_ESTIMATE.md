# Ước Lượng Tài Nguyên Cho Thí Nghiệm Xác Nhận (`RESOURCE_ESTIMATE.md`)

> [!IMPORTANT]
> Mọi con số thời lượng và bộ nhớ dưới đây được suy ra **chỉ từ phép đo CPU thực tế của Slurm Job `16908`** (`8 CPU cores, 0 GPU`, `robust_r3_lr_stability/`). **Tuyệt đối không ngoại suy hay phỏng đoán thời gian chạy GPU từ dữ liệu CPU**, vì chưa có phép đo chuẩn trực tiếp trên GPU cho mã này.

---

## 1. Dữ liệu Đo Đạc Cơ Sở Từ Slurm Job `16908`

- **Cấu hình Job `16908`:** `1 node`, `8 CPUs`, `0 GPU`, môi trường `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`, `XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"`.
- **Kết quả thực đo (`COMPLETED`, ExitCode `0:0`):**
  - Tổng wall-clock (`12` lần chạy × `200` epoch + đánh giá `2 × 20k diag` trên `24` checkpoint + `5k dev_eval` 5 kịch bản): **`00:59:24` (`3560.83s`)**.
  - Bộ nhớ đỉnh (`MaxRSS`): **`867,376K`** theo `sacct` / **`864,128 KB`** (`~0.86 GB`) theo `resource.getrusage`.
  - Thời gian huấn luyện trung bình (`200` epoch × `156` bước = `31,200` bước):
    - `A_DECAY`: **`263.35s`** / run.
    - `R2_DECAY`: **`301.21s`** / run.
    - Cộng 1 cặp seed `(A_DECAY + R2_DECAY)`: **`~564.6s` (`~9.41 phút / seed`)**.

---

## 2. Chuyển Đổi Sang Tải Công Việc Xác Nhận

So với Job `16908`, thiết kế xác nhận [conf_run.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_run.py) thay đổi đúng các phần sau trên **cùng kiến trúc CPU 8 nhân (`0 GPU`)**:
1. **Giảm số cấu hình từ 4 xuống 2 (`A_DECAY`, `R2_DECAY`):** bỏ hoàn toàn `A_FIXED` và `R2_FIXED` -> chỉ chạy `2` run/seed.
2. **Tắt lưu 23 checkpoint phụ (`save_extra_ckpts=False`) và bỏ đánh giá `diag1/diag2 20k × 24 ep`:** mỗi run chỉ lưu đúng **`best_val.pkl`**, **`final_with_opt.pkl`** và **`checkpoint_manifest.json`** (`2` checkpoint `.pkl`/run). Điều này cắt giảm phần I/O & chi phí rollout `48 × 20k` mỗi seed của Job `16908`.
3. **Nâng tập kiểm tra `conf_test` lên `20,000` đường/kịch bản (`6` kịch bản, `best_val` + `ep200`):** tổng `12` lượt rollout `20k` × `2` nhánh (~`24` lượt rollout/seed) -> ước tính tốn **`~10–15s / seed`** (kết hợp sinh thế giới đường dẫn `20k`).
4. **Tổng chi phí ước tính mỗi seed:** `~564.6s + (10..15)s ≈ 575–580s` (**`~9.65 phút / seed`**).

---

## 3. Bảng Dự Toán Cho Hai Phương Án Cỡ Mẫu (`8 CPUs, 0 GPU`)

| Hạng mục | Phương án Khuyến nghị Chính (`N = 24` seed) | Phương án Ngân sách Thấp (`N = 16` seed) | Cơ sở |
| :--- | :--- | :--- | :--- |
| **Dải seed dự kiến** | `[4001..4024]` | `[4001..4016]` | [DATA_AND_SEED_PLAN.json](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/DATA_AND_SEED_PLAN.json) |
| **Số lần huấn luyện (`2` nhánh)** | `48` runs (`1,497,600` bước cập nhật tổng; `191,692,800` lượt xử lý đường) | `32` runs (`998,400` bước cập nhật tổng; `127,795,200` lượt xử lý đường) | `31,200` bước/run, `3,993,600` đường/run |
| **Thời gian chạy CPU ước tính (`8 CPU`)** | `~231.6 phút` (**`~3.86 giờ`**) | `~154.4 phút` (**`~2.57 giờ`**) | `~9.65 phút/seed` từ Job `16908` |
| **Giới hạn Slurm khuyến nghị (`--time`)** | **`05:30:00`** (cộng đệm `+42%` an toàn JIT/dao động CPU) | **`03:30:00`** (cộng đệm `+36%`) | [slurm_confirmatory_template.sbatch](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/slurm_confirmatory_template.sbatch) |
| **RAM yêu cầu (`--mem`)** | **`16G`** | **`16G`** | Đỉnh thực đo `0.86 GB` + dự phòng `20k` `6`-kịch bản |
| **Số tệp checkpoint `.pkl`** | `96` tệp (`48` `best_val.pkl` + `48` `final_with_opt.pkl`) | `64` tệp (`32` `best_val.pkl` + `32` `final_with_opt.pkl`) | `save_extra_ckpts=False` |
| **Ước lượng GPU** | **Chưa ước lượng (không ngoại suy từ CPU)** | **Chưa ước lượng (không ngoại suy từ CPU)** | Thiếu benchmark GPU trực tiếp |

---

## 4. Ghi Chú Khả Năng Chia Nhỏ Khi Cần Thiết

- Mẫu [slurm_confirmatory_template.sbatch](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/slurm_confirmatory_template.sbatch) cấu hình chạy một job đơn tuần tự với `--time=05:30:00` để đảm bảo cách ly tuần tự và nhật ký hợp nhất.
- Vì mỗi seed trong `CONF_RESERVED_SEEDS` sinh khóa PRNG, tập train/val, khởi tạo mạng và tập `conf_test` hoàn toàn **độc lập nhau theo `seed_label`** (mọi kết quả từng cặp `(seed, config)` không phụ thuộc seed khác), nên nếu sau này cần tăng tốc trên cụm nhiều node, có thể tách thành các lát seed rời rạc (ví dụ `2` job × `12` seed hoặc `3` job × `8` seed) rồi hợp nhất duy nhất tệp `PAIRED_CONFIRMATORY.csv` trước khi chạy [conf_stats.py](file:///C:/Users/Hi%20Windows%2011%20Home/Desktop/KET_NOI_USER04/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/conf_stats.py) — **nhưng mọi hạt giống nằm trong nhóm đã khóa ban đầu đều phải được giữ và báo cáo đủ**.
