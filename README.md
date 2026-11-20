# JAX + NTBN 7 Notebook Groups Benchmark (Original Scale Edition)
**Hệ Thống Thí Nghiệm Đối Chuẩn Toàn Diện JAX Gốc vs JAX + NTBN Trên Toàn Bộ 7 Nhóm Bài Toán**  
*Chuẩn mực thẩm định tối cao: Astra Gemini Sovereign Epistemic Gate `@gng5` (T1–T5)*

---

## 1. TỔNG QUAN HỆ THỐNG
Bộ mã nguồn này hiện thực hóa toàn diện kiến trúc **No-Transaction Band Network (NTBN)** (Imaki et al. 2021) tích hợp trên nền tảng **JAX Deep Hedging** (Buehler et al. 2019 / NdQuan 2024), bảo toàn **100% quy mô dữ liệu, số bước, số epoch, hàm mục tiêu và cơ chế tài chính gốc** của toàn bộ 7 nhóm bài toán quyền chọn trong repository `deephedging-jax`.

### Hai Nhánh Đối Chuẩn Song Hành:
* **Nhánh A (Original JAX):** Mạng nơ-ron sâu chuẩn tắc Buehler với đầu vào $(\text{Price}, \Delta, \text{TimeLeft}, \dots)$ và hàm bão hòa mượt `softclip(a, lb, ub)`.
* **Nhánh B (JAX + NTBN):** Mạng dự đoán dải không giao dịch ngoại sinh $[b_l(t, I_t), b_u(t, I_t)]$ độc lập với vị thế $\Delta$, thực thi chính sách kẹp vị thế tối ưu $\delta_t = \text{clamp}(\delta_{t-1}, b_l, b_u)$, triệt tiêu vòng phản hồi tự hồi quy méo mó.

---

## 2. DANH MỤC 7 NHÓM BÀI TOÁN & 24 CẤU HÌNH GỐC

| STT | Nhóm Bài Toán | Tệp Notebook Gốc | Số Biến Thể | Trạng Thái Lý Thuyết NTBN | Đặc Tính Tài Chính & Khối Lượng |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `blackscholes` | `notebooks/blackscholes.ipynb` | 5 | `SUPPORTED_EMPIRICAL_ZERO_COST` | ATM Call chuẩn Black-Scholes, $50.000$ mẫu, 20 bước, phí $0\%$, $400 - 600$ epoch. |
| **2** | `trainer-bs` | `notebooks/trainer-bs.ipynb` | 2 | `SUPPORTED_EMPIRICAL_ZERO_COST` | Khảo sát bước lớn ($\Delta t = 0,02$) vs bước mịn ($\Delta t = 0,004$), $20.000$ epoch, batch $2.000$. |
| **3** | `trainer` | `notebooks/trainer.ipynb` | 2 | `SUPPORTED_THEORETICAL` | Phí giao dịch dương ($c_s > 0$). Biến thể `sv` có 2 công cụ phòng hộ ($n_{\text{inst}}=2$: Spot + ATM Call). |
| **4** | `trainer-american_digital` | `notebooks/trainer-american_digital.ipynb` | 5 | `SUPPORTED_EMPIRICAL_NONSMOOTH` | Quyền chọn American Digital có rào cản chạm $B=0,95$, payoff gián đoạn, $20.000$ mẫu, 100 bước. |
| **5** | `trainer-recurrent-barrier` | `notebooks/trainer-recurrent-barrier.ipynb` | 2 | `SUPPORTED_EMPIRICAL_NONSMOOTH` | Knock-out Barrier Put ($K=1,0, B=0,95$), 100 bước mịn ($\Delta t = 0,004$), $20.000$ mẫu, hàm hữu dụng mũ `exp@1.01`. |
| **6** | `trainer-recurrent-barrier_smalldt` | `notebooks/trainer-recurrent-barrier_smalldt.ipynb` | 2 | `SUPPORTED_EMPIRICAL_NONSMOOTH` | Knock-out Barrier Put, 20 bước chuẩn ($\Delta t = 0,02$), $20.000$ mẫu, hàm hữu dụng `cvar@1.01`. |
| **7** | `trainer-recurrent-fwdstart` | `notebooks/trainer-recurrent-fwdstart.ipynb` | 6 | `SUPPORTED_EMPIRICAL_PATH_DEPENDENT` | Forward-Start Put có giá thực hiện thiết lập lại tại bước 10 ($S_{10}$), $20.000$ mẫu, 20 bước, $2.000$ epoch. |

---

## 3. NGUYÊN TẮC ĐỐI CHUẨN CÔNG BẰNG (FAIR BENCHMARK PARITY)
1. **Dữ liệu đóng băng hoàn toàn:** Tập Train, Validation và Test được sinh bằng PRNG xác định với seed cố định và kiểm soát bằng mã băm SHA-256.
2. **Khóa thứ tự xáo trộn Minibatch:** Cả Nhánh A và Nhánh B sử dụng chung chuỗi hoán vị `perm = np.random.default_rng([c.seed, epoch]).permutation(n)` cho từng epoch.
3. **Tiêu chí chọn điểm lưu độc lập (Checkpoint Selection):** Cả hai nhánh khóa trước quy tắc chọn checkpoint theo **Validation Loss tối thiểu (`val_loss`)**. Tuyệt đối không dùng dữ liệu tập Test để chọn điểm lưu, chỉnh epoch hay dừng sớm.
4. **Cách ly tập Test:** Tập Test chỉ được giải mã và đánh giá duy nhất một lần sau khi checkpoint huấn luyện đã đóng băng.

---

## 4. HƯỚNG DẪN VẬN HÀNH DÒNG LỆNH (CLI INSTRUCTIONS)

### 4.1 Chạy một cấu hình đơn lẻ
```bash
# Chạy cả 2 nhánh A và B trên cấu hình Black-Scholes cơ sở (10 epoch thử nghiệm)
python experiments/jax_ntbn_7notebooks_original_scale/scripts/run_single_config.py \
    --config_id blackscholes__trainer_bs_delta \
    --branch both \
    --epochs 10 \
    --seed 42

# Chạy cấu hình đa tài sản (nInst=2) trong nhóm trainer
python experiments/jax_ntbn_7notebooks_original_scale/scripts/run_single_config.py \
    --config_id trainer__sv \
    --branch both \
    --epochs 20 \
    --seed 42
```

### 4.2 Chạy toàn bộ cấu hình trong một nhóm
```bash
# Chạy toàn bộ 5 biến thể của nhóm trainer-american_digital
python experiments/jax_ntbn_7notebooks_original_scale/scripts/run_group.py \
    --group_id trainer-american_digital \
    --branch both \
    --epochs 5 \
    --seed 42
```

### 4.3 Chạy đối chuẩn tổng thể (Master Benchmark)
```bash
# Chạy chế độ Smoke Test trên toàn bộ 24 cấu hình (2 epoch / cấu hình)
python experiments/jax_ntbn_7notebooks_original_scale/scripts/run_benchmark.py --smoke --seed 42

# Chạy toàn bộ quy mô gốc trên GPU (qua Slurm)
sbatch experiments/jax_ntbn_7notebooks_original_scale/scripts/slurm_job_template.sh
```

---

## 5. TRUY XUẤT NGUỒN GỐC & TÍNH BẤT BIẾN (MERKLE DAG PROVENANCE)
Mọi hiện vật gốc từ commit `f21c6e4` của `deephedging-jax` và 23 tệp mã nguồn mới đã được khắc niệm băm mật mã SHA-256 trong tệp `MANIFEST.json`.
