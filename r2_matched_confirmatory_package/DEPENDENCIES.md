# Phụ Thuộc Mã Nguồn, Phiên Bản, Giấy Phép & Khác Biệt So Với Kho Gốc (`DEPENDENCIES.md`)

Gói tái lập `R2_matched` bao gồm hai thư mục thí nghiệm xác nhận ([robust_r2_matched_confirmatory/](../robust_r2_matched_confirmatory/) và [robust_r2_matched_cost_confirmatory/](../robust_r2_matched_cost_confirmatory/)) cùng ảnh chụp nguyên trạng toàn bộ **17 tệp mô-đun nội bộ phụ thuộc bắc cầu** trong [`deps_snapshot/`](deps_snapshot/) để bảo đảm người dùng tải nhánh `codex/r2-matched-confirmatory-package` từ GitHub có thể tái kiểm tra độc lập mà không phụ thuộc vào các thư mục phát triển trung gian chưa đưa lên `main`.

---

## 1. Đối Chiếu Kho Gốc & Khác Biệt So Với `origin/main` (`a32e663`)

- **Kho lưu trữ Git:** `https://github.com/thanhbien10a4-ui/deephedging-jax-ntbn-7groups.git`
- **Commit nền (`origin/main`):** `a32e66386d4deb7841c66161ae05e5437537833f` (`feat(coordinator): add autonomous GPU coordinator with atomic checkpointing and auto-resume`)
- **Trạng thái đối chiếu:** Không sửa đổi bất kỳ tệp nào thuộc `core/` hoặc `adapters/` trên `main`.

| Tệp trong [`deps_snapshot/`](deps_snapshot/) | Kích thước (B) | `SHA-256` | Trạng thái so với `origin/main` (`a32e663`) | Vai trò thực tế được gọi |
| :--- | ---: | :--- | :--- | :--- |
| `core/metrics.py` | `3,769` | `fb18b3a2adbf0c515e3b96cd364030d9c3994244948db4d37557dfaada96da39` | Trùng khớp từng byte (`IDENTICAL`) | Mô-đun nền của `core/` |
| `core/model.py` | `13,547` | `bd53a7b8d7b639ccc7947fbd9436ffb2492befd05c51db9509a0a65b798aa040` | Trùng khớp từng byte (`IDENTICAL`) | `AgentConfigOriginal`, `GymConfigOriginal`, `softclip`, `_glorot` |
| `core/ntbn_primitives.py` | `2,139` | `d58863d49941b1ca50eaeafcb43dafec77f784d70342d611475ee6b87edec638` | Trùng khớp từng byte (`IDENTICAL`) | Nguyên thủy NTBN nền của `core/` |
| `core/objectives.py` | `2,398` | `73a3f9eb3a44ae2840d2053bc95d4b7b3a6a0c506623e65ebd0fdf672375c113` | Trùng khớp từng byte (`IDENTICAL`) | Định nghĩa `exp2` OCE utility |
| `core/train.py` | `10,754` | `db05981c8338b29087c871c50a2cea6540b6f01b37011246e728ce5bdfb22c62` | Trùng khớp từng byte (`IDENTICAL`) | Hàm `as_batch` chuyển `World` sang batch JAX |
| `scripts/robust_deep_hedging_phase2/robust_world.py` | `5,636` | `b25846bd3cf3e4b550cb50c72b1cb679846a549b04f2739d79d13e0bd44036a3` | Mô-đun mới chưa có trên `main` | Bộ mô phỏng BS/Regime-Switch kế thừa từ `deephedging/world.py` |
| `robust_r3_dev/r3_keys.py` | `7,920` | `b02bf07e0191039a3885f2bad47475a3d611a8ee4cd13d1f63f8fff3fa04815a` | Mô-đun mới chưa có trên `main` | Kịch bản `TRAINING_SCENARIOS`, `TEST_SCENARIOS` & lịch sử khóa R3 |
| `robust_r3_dev/r3_metrics.py` | `4,783` | `601795002691fef55c84da447fac1cd47b4da96584349cb6e75605f2a4552a2f` | Mô-đun mới chưa có trên `main` | Hàm `evaluate_rollout_metrics_r3` |
| `robust_r3_dev/r3_models.py` | `12,138` | `cf391002e10af3e785f5e1189ac5999189603840f39f2eeee31abe1732c0799e` | Mô-đun mới chưa có trên `main` | `make_canonical_configs`, `init_full_params`, `rollout_policy_paths`, `rollout_with_oce`, `r2_matched_step_loss` |
| `robust_r3_dev/r3_train.py` | `21,929` | `3b5dd6b8e349f877baaec0cb48cfeb444b0e3c2a85d5151b74e1a5d2609f7aa6` | Mô-đun mới chưa có trên `main` | `R3TrainConfig`, `adam_init`, `_take_batch`, `verify_finite_tree` |
| `robust_r3_dev/run_r3_pilot.py` | `16,619` | `6ac900b065c32cb6b36b9030f9b1b74762df84b3c6c0c74a7eb0a20803547268` | Mô-đun mới chưa có trên `main` | `SCENARIO_SPECS` & `build_scenario_world` (dùng trong Job 16911) |
| `robust_r3_budget_control/bc_keys.py` | `8,246` | `a046df3ac0e0df6bc8964229abfe9c6acd47b9b8bf140cab2f63f6e75b1c4f4f` | Mô-đun mới chưa có trên `main` | Thu thập khóa lịch sử nhóm budget control |
| `robust_r3_budget_control/bc_models.py` | `1,723` | `4f3c5155e21c1bf0dce84661b85935b5f3838b88641691cccf4e0f5eb7fc3f56` | Mô-đun mới chưa có trên `main` | `aref_128_step_loss` & hàm tiện ích OCE |
| `robust_r3_selection_diagnostics/diag_keys.py` | `8,610` | `28ab2dea4cc06ddb2a906ae234b0bb53424a2c9c13715be9d5bb078d51f5851e` | Mô-đun mới chưa có trên `main` | Thu thập khóa lịch sử chẩn đoán (`HOLD_EVAL` hash) |
| `robust_r3_lr_stability/lr_keys.py` | `5,728` | `586049076d5cfc3865842a31ae3fc9e14b890a96d0e5976be9b4e3a7d2c82d6b` | Mô-đun mới chưa có trên `main` | Thu thập khóa lịch sử `[3401..3403]` |
| `robust_r3_lr_stability/lr_schedule.py` | `2,215` | `7d856f2adf2fc89b7d7d8e9364f846bba5f9fad64cec5e4a5db593e56c0b519e` | Mô-đun mới chưa có trên `main` | Lịch `Cosine LR Decay` (`15,600 -> 31,200`, `1e-3 -> 1e-4`) |
| `robust_r3_lr_stability/lr_train.py` | `11,523` | `2759f098522467f5809cb8d00386ab2d6bad16589dcb93cbbc38fc540fcc39f3` | Mô-đun mới chưa có trên `main` | `train_lr_config` (Job 16911) & `adam_update_scheduled` |

---

## 2. Nguồn Gốc, Giấy Phép & Khác Biệt Với Kho Thượng Nguồn

1. **Kho `deephedging-jax` (thượng nguồn của bộ mô phỏng `world.py`):**
   - **URL nguồn:** `https://github.com/NdQuan127/deephedging-jax.git`
   - **Commit trên máy chủ HUST:** `f21c6e42f4cae79864df57b2893f39dd484f37ff`
   - **Giấy phép gốc:** **GNU General Public License v3.0 (GPL-3.0)** (tệp `LICENSE` tại thư mục gốc `deephedging-jax`).
   - **Khác biệt của `robust_world.py` so với `deephedging/world.py`:** `robust_world.py` chỉ giữ công cụ phòng hộ duy nhất (`spot`, `nInst = 1`, quyền chọn bán khống ATM Call `strike = 1.0`, `steps = 20`, `dt = 1/50`, `drift = 0.0`), bổ sung các chế độ biến động `sigma_mode` (`"fixed"`, `"uniform"`, `"regime_switch"`, `"static_path"`), và tách tham số `normalize_batch_mean` (mặc định `True` chuẩn hóa trung bình mẫu từng bước giống `world.py`). `robust_world.py` được đóng gói nguyên trạng trong `deps_snapshot/`, **không phụ thuộc runtime vào thư mục `deephedging-jax` khi chạy lại gói `R2_matched`**.
2. **Kho `deephedging-jax-ntbn-7groups` (kho hiện tại):**
   - Tệp `pyproject.toml` trên `main` chưa khai báo trường `license` và thư mục gốc chưa có tệp `LICENSE` riêng. Toàn bộ mã `core/` trong `deps_snapshot/` khớp 100% từng byte với `origin/main` (`a32e663`).
3. **Thư viện Python ngoài (bắt buộc để chạy JAX evaluation / training):**
   - **`jax` / `jaxlib` (`0.4.34` trên Slurm CPU; Apache-2.0)**
   - **`numpy` (`1.26.4` trên Slurm; BSD-3-Clause)**
   - **`matplotlib` (`3.10.9` trên Slurm; PSF-based)** *(chỉ cần khi tái sinh biểu đồ)*
   - Kiểm tra tái tính bảng & thống kê CSV/JSON (Mức 1) **chỉ cần thư viện chuẩn Python + `numpy`**.
