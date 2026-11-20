# Môi Trường Thực Thi & Hướng Dẫn Cài Đặt (`ENVIRONMENT.md`)

## 1. Phần Cứng & Hệ Điều Hành Thực Thi Chính Thức (HUST Slurm CPU)

| Thành phần | Job `16911` (`cost_s = 0.0`, `[4001..4024]`) | Job `16914` (`cost_s = 0.0010`, `[5001..5024]`) |
| :--- | :--- | :--- |
| **Cụm / Phân vùng / Nút** | HUST Slurm `defq`, node `dgx02` | HUST Slurm `defq`, node `dgx02` |
| **Tài nguyên cấp phát** | `8 CPU`, `0 GPU` (`JAX_PLATFORMS=cpu`, `mem=16G`, `time=05:30:00`) | `8 CPU`, `0 GPU` (`JAX_PLATFORMS=cpu`, `mem=16G`, `time=05:30:00`) |
| **Hệ điều hành** | `Linux-5.4.0-124-generic-x86_64-with-glibc2.31` *(theo log Slurm trên `dgx02`)* | `Linux-5.4.0-124-generic-x86_64-with-glibc2.31` *(theo log Slurm trên `dgx02`)* |
| **Python & Thiết bị JAX** | `Python 3.10.21` (Clang 22.1.3), `JAX 0.4.34`, `[CpuDevice(id=0)]` (`TFRT_CPU_0`) | `Python 3.10.21` (Clang 22.1.3), `JAX 0.4.34`, `[CpuDevice(id=0)]` (`TFRT_CPU_0`) |
| **Thời gian chạy thực tế** | **`03:50:53`** (`13,836.36s`), `MaxRSS = 1,220,600K` | **`03:44:55`** (`13,475.24s`), `MaxRSS = 851,780K` |

## 2. Môi Trường Kiểm Thử & Đóng Gói Trên Windows Cục Bộ

- **Hệ điều hành:** `Windows-11-10.0.26300-SP0`
- **Python / JAX:** `Python 3.13.4`, `JAX 0.12.0.1` (`cpu:0`)
- Lưu ý tái lập số học giữa `JAX 0.4.34` (Linux `dgx02`) và `JAX 0.12.0.1` (Windows): Khi đánh giá lại các checkpoint `.pkl` bằng `float32`, sai lệch XLA/LLVM khác phiên bản có thể xuất hiện ở bậc `~1e-7 – 1e-6`, rất nhỏ hơn sai số ngưỡng tái nạp `< 1e-6` và các chênh lệch thực nghiệm (`1e-5 – 1e-3`).

## 3. Phiên Bản Thư Viện Cốt Lõi & Đóng Băng Đầy Đủ

Toàn bộ **148 gói** trong `.venv` trên máy chủ được lưu tại [`environment_server_freeze.txt`](environment_server_freeze.txt). Các thư viện cốt lõi trực tiếp tham gia chạy mã:

| Thư viện | Phiên bản trên máy chủ (Slurm) | Vai trò |
| :--- | :---: | :--- |
| `jax` / `jaxlib` | `0.4.34` / `0.4.34` | Rollout mạng chính sách `5×40 softplus`, `jax.lax.scan`, `jax.value_and_grad` |
| `numpy` | `1.26.4` | Mô phỏng đường giá, thống kê ghép cặp, Sign & Wilcoxon chính xác |
| `scipy` | `1.15.3` | Cài sẵn trong `.venv` *(thống kê IUT tự cài tích phân `betainc`/`t_cdf` chuẩn xác trong `conf_stats.py`)* |
| `matplotlib` | `3.10.9` | Tạo 3 hình của Job `16911` và 4 hình của Job `16914` |

## 4. Hướng Dẫn Cài Đặt Tối Thiểu Để Kiểm Tra Gói Độc Lập

```bash
# Tạo môi trường Python sạch (khuyến nghị Python >= 3.10)
python -m venv .venv_r2_verify
# Kích hoạt: source .venv_r2_verify/bin/activate (Linux/macOS) hoặc .venv_r2_verify\Scripts\activate (Windows)

# Mức 1 (kiểm tra toàn vẹn SHA-256, schema CSV/JSON, tái tính thống kê IUT & kế toán phí):
pip install "numpy>=1.26.4"

# Mức 2 & 3 (nếu muốn chạy đánh giá lại checkpoint JAX .pkl hoặc chạy huấn luyện):
pip install "jax[cpu]>=0.4.34" "matplotlib>=3.7"

# Chạy trình kiểm tra tự động ngay tại thư mục gốc gói:
python r2_matched_confirmatory_package/verify_r2_matched_package.py --level auto
```
