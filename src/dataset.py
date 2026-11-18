from __future__ import annotations

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from encoding import load_events_from_h5, encode_events_2ch_lif
except ImportError:
    from src.encoding import load_events_from_h5, encode_events_2ch_lif


class EvCivilSpikingDataset(Dataset):
    """
    PyTorch Dataset đọc trực tiếp từ dataset_splits.json cho tập ev-CIVIL.
    Trả về:
      - spikes: Tensor (N=25, C=2, H=260, W=346) kiểu float32
      - targets: List dài N=25, mỗi phần tử là dict {'boxes': Tensor[K, 4] (x1,y1,x2,y2), 'labels': Tensor[K]}
    """

    def __init__(
        self,
        split_json: str = "dataset_splits.json",
        split: str = "train",
        N: int = 25,
        H: int = 260,
        W: int = 346,
        tau_us: float = 40000.0,
        v_th: float = 1.2,
        clip_stride_sec: float | None = None,
        label_tolerance_us: float = 60000.0,
        augment: bool | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split phải là 'train', 'val' hoặc 'test', nhận được: '{split}'")
        if not os.path.exists(split_json):
            raise FileNotFoundError(f"Không tìm thấy tệp chia dữ liệu: {split_json}")

        self.split = split
        self.N = N
        self.H = H
        self.W = W
        self.tau_us = tau_us
        self.v_th = v_th
        self.dt_us = 1_000_000.0 / N  # 40,000 us (25 Hz)
        self.label_tolerance_us = label_tolerance_us

        # Train: trượt 0.5s + lật ngang ngẫu nhiên; Val/Test: cắt nối tiếp 1.0s không chồng lấn
        self.clip_stride_sec = clip_stride_sec if clip_stride_sec is not None else (0.5 if split == "train" else 1.0)
        self.augment = augment if augment is not None else (split == "train")

        self.samples: list[dict] = []
        self._load_split_sequences(split_json)

    def _load_split_sequences(self, split_json: str) -> None:
        with open(split_json, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        seq_list = splits_data.get(self.split, [])
        stride_us = self.clip_stride_sec * 1_000_000.0

        for seq_info in seq_list:
            h5_path = seq_info["h5_path"]
            npy_path = seq_info["npy_path"]
            if not (os.path.exists(h5_path) and os.path.exists(npy_path)):
                continue

            raw_labels = np.load(npy_path, allow_pickle=True)
            if raw_labels.ndim != 2 or raw_labels.shape[1] < 6 or len(raw_labels) == 0:
                continue

            labels = raw_labels[:, :6].astype(np.float64)
            if len(labels) > 1 and (labels[-1, 0] - labels[0, 0]) < 1000.0:
                labels[:, 0] *= 1_000_000.0

            t_min, t_max = labels[0, 0], labels[-1, 0]

            # Xử lý chuỗi ngắn hơn 0.5s nhưng vẫn có đủ nhãn hợp lệ
            if t_max - t_min < 500_000.0:
                if t_max - t_min >= 200_000.0:
                    self.samples.append({
                        "seq_id": seq_info.get("seq_id", os.path.basename(os.path.dirname(h5_path))),
                        "h5_path": h5_path,
                        "labels": labels,
                        "t0": float(t_min),
                    })
                continue

            curr_t0 = t_min
            while curr_t0 + 500_000.0 <= t_max:
                self.samples.append({
                    "seq_id": seq_info.get("seq_id", os.path.basename(os.path.dirname(h5_path))),
                    "h5_path": h5_path,
                    "labels": labels,
                    "t0": float(curr_t0),
                })
                curr_t0 += stride_us

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        item = self.samples[idx]
        t0 = item["t0"]
        labels = item["labels"]

        # 1. Đọc sự kiện và mã hóa thành Tensor xung thưa (25, 2, 260, 346)
        x, y, t, p = load_events_from_h5(item["h5_path"])
        spikes_np = encode_events_2ch_lif(
            x, y, t, p,
            N=self.N,
            H=self.H,
            W=self.W,
            tau_us=self.tau_us,
            v_th=self.v_th,
            t_start_us=t0,
        )
        spikes_tensor = torch.from_numpy(spikes_np).float()

        # Lật ngang ngẫu nhiên (không làm thay đổi cực tính ON/OFF)
        do_hflip = self.augment and (np.random.rand() < 0.5)
        if do_hflip:
            spikes_tensor = torch.flip(spikes_tensor, dims=[-1])

        # 2. Đồng bộ bounding boxes cho từng bước thời gian n in [0..24]
        unique_ts = np.unique(labels[:, 0])
        targets_per_step: list[dict[str, torch.Tensor]] = []

        for n in range(self.N):
            t_center = t0 + (n + 0.5) * self.dt_us
            nearest_idx = int(np.argmin(np.abs(unique_ts - t_center)))
            nearest_t = unique_ts[nearest_idx]

            if abs(nearest_t - t_center) <= self.label_tolerance_us:
                frame_rows = labels[labels[:, 0] == nearest_t]
                cls_ids = frame_rows[:, 1].astype(np.int64)

                x1 = np.clip(frame_rows[:, 2], 0, self.W)
                y1 = np.clip(frame_rows[:, 3], 0, self.H)
                x2 = np.clip(frame_rows[:, 2] + frame_rows[:, 4], 0, self.W)
                y2 = np.clip(frame_rows[:, 3] + frame_rows[:, 5], 0, self.H)

                if do_hflip:
                    x1_flipped = self.W - x2
                    x2_flipped = self.W - x1
                    x1, x2 = x1_flipped, x2_flipped

                valid_box = (x2 > x1 + 1.0) & (y2 > y1 + 1.0)
                boxes = np.stack([x1[valid_box], y1[valid_box], x2[valid_box], y2[valid_box]], axis=1).astype(np.float32)
                cls_ids = cls_ids[valid_box]
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                cls_ids = np.zeros((0,), dtype=np.int64)

            targets_per_step.append({
                "boxes": torch.from_numpy(boxes),
                "labels": torch.from_numpy(cls_ids),
            })

        return spikes_tensor, targets_per_step


def evcivil_collate_fn(batch):
    """Gom batch dữ liệu có số lượng bounding box thay đổi theo từng khung hình."""
    spikes_batch = torch.stack([item[0] for item in batch], dim=0)  # (B, 25, 2, 260, 346)
    targets_batch = [item[1] for item in batch]                     # List[B][25]
    return spikes_batch, targets_batch


def build_dataloaders(
    split_json: str = "dataset_splits.json",
    batch_size: int = 4,
    num_workers: int = 2,
    tau_us: float = 40000.0,
    v_th: float = 1.2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Khởi tạo đồng bộ 3 DataLoader (train, val, test) từ dataset_splits.json."""
    use_pin_memory = torch.cuda.is_available()

    train_ds = EvCivilSpikingDataset(split_json=split_json, split="train", tau_us=tau_us, v_th=v_th)
    val_ds = EvCivilSpikingDataset(split_json=split_json, split="val", tau_us=tau_us, v_th=v_th)
    test_ds = EvCivilSpikingDataset(split_json=split_json, split="test", tau_us=tau_us, v_th=v_th)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=evcivil_collate_fn,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=evcivil_collate_fn,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=evcivil_collate_fn,
        pin_memory=use_pin_memory,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(
        split_json="dataset_splits.json",
        batch_size=8,
        num_workers=0,
        tau_us=40000.0,
        v_th=1.2,
    )
    print(
        f"Số lượng clip 1 giây -> Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}"
    )

    if len(train_loader) > 0:
        spikes_b, targets_b = next(iter(train_loader))
        sparsity = 100.0 * (1.0 - torch.count_nonzero(spikes_b).item() / spikes_b.numel())
        print(f"Train Batch Shape: {tuple(spikes_b.shape)} | Sparsity: {sparsity:.2f}%")
        print(f"Mẫu target tại Batch 0, Frame 12: {targets_b[0][12]}")