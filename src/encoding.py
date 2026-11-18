from __future__ import annotations

import h5py
import numpy as np
import torch


def load_events_from_h5(h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Đọc dữ liệu x, y, t, p từ events.h5 và tự động chuẩn hóa tên key + đơn vị micro-giây."""
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        x = f["x"][:].astype(np.int32)
        y = f["y"][:].astype(np.int32)

        # Tự động nhận diện tên trường thời gian và cực tính thực tế trong file
        t_key = next(k for k in ["ts", "timestamp", "timestamps", "event_timestamp", "t"] if k in keys)
        p_key = next(k for k in ["p", "pol", "polarity", "event_polarity"] if k in keys)

        t = f[t_key][:].astype(np.float64)
        p = f[p_key][:]

    # Nếu thời gian lưu ở đơn vị giây (float), quy đổi sang micro-giây (us)
    if len(t) > 1 and (t[-1] - t[0]) < 1000.0:
        t = t * 1_000_000.0

    return x, y, t, p


def encode_events_2ch_lif(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    N: int = 25,
    H: int = 260,
    W: int = 346,
    tau_us: float = 20000.0,     # Hằng số suy giảm màng 20 ms (20,000 us)
    v_th: float = 1.0,           # Ngưỡng phát xung LIF
    t_start_us: float | None = None
) -> np.ndarray:
    """
    Mã hóa luồng sự kiện 1 giây thành Tensor xung thưa kích thước (N, 2, H, W).
    - Kênh 0: Sự kiện ON (p > 0)
    - Kênh 1: Sự kiện OFF (p <= 0, bao gồm cả định dạng p=0 hoặc p=-1)
    """
    I_out = np.zeros((N, 2, H, W), dtype=np.uint8)
    if len(t) == 0:
        return I_out

    U = np.zeros((2, H, W), dtype=np.float32)  # Điện thế màng cho 2 kênh độc lập
    t0 = t[0] if t_start_us is None else t_start_us
    dt_us = 1_000_000.0 / N                    # 40,000 us (1/25 giây) cho mỗi bước
    decay_step = np.exp(-dt_us / tau_us)

    # Phân kênh: ON (p > 0) -> 0, OFF (p <= 0) -> 1
    c = np.where(p > 0, 0, 1).astype(np.int32)

    # Lọc các sự kiện nằm trong cửa sổ 1 giây [t0, t0 + 1s) và hợp lệ tọa độ
    bin_idx = np.floor((t - t0) / dt_us).astype(np.int32)
    valid = (bin_idx >= 0) & (bin_idx < N) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, t, c, bin_idx = x[valid], y[valid], t[valid], c[valid], bin_idx[valid]

    for n in range(N):
        # 1. Suy giảm điện thế màng từ bước trước
        U *= decay_step

        mask = (bin_idx == n)
        if not np.any(mask):
            continue

        xn, yn, tn, cn = x[mask], y[mask], t[mask], c[mask]
        t_window_end = t0 + (n + 1) * dt_us

        # 2. Tích lũy trọng số suy giảm theo độ trễ thời gian thực tới cuối cửa sổ 1/25s
        weights = np.exp(-(t_window_end - tn) / tau_us)
        np.add.at(U, (cn, yn, xn), weights)

        # 3. Chỉ xét phát xung tại các tọa độ (c, y, x) có event xuất hiện trong cửa sổ n
        active_mask = np.zeros((2, H, W), dtype=bool)
        active_mask[cn, yn, xn] = True

        spikes = (U >= v_th) & active_mask
        I_out[n, spikes] = 1

        # 4. Soft-reset điện thế màng tại các nơ-ron đã phát xung
        U[spikes] -= v_th

    return I_out


if __name__ == "__main__":
    h5_file = "/Users/trantung/Downloads/Field_dataset/Night_outdoor/night_outdoor_31st_dec/31_crack_40/31_crack_40_events.h5"

    # Bước 1: Đọc luồng sự kiện thô
    x, y, t, p = load_events_from_h5(h5_file)
    duration_sec = (t[-1] - t[0]) / 1e6 if len(t) > 1 else 0.0
    print(f"Tổng số events: {len(t):,} | Thời lượng chuỗi: {duration_sec:.2f}s")

    # Bước 2: Mã hóa 1 giây đầu tiên thành Tensor (25, 2, 260, 346)
    sparse_tensor_np = encode_events_2ch_lif(x, y, t, p, N=25, H=260, W=346)

    # Bước 3: Chuyển sang PyTorch Tensor chuẩn QKFormer: (B, T, C, H, W)
    qkformer_input = torch.from_numpy(sparse_tensor_np).unsqueeze(0).float()

    sparsity = 100.0 * (1.0 - np.count_nonzero(sparse_tensor_np) / sparse_tensor_np.size)
    print(f"Shape đầu vào QKFormer: {tuple(qkformer_input.shape)}")  # (1, 25, 2, 260, 346)
    print(f"Tỷ lệ thưa (Zero-Sparsity): {sparsity:.2f}%")