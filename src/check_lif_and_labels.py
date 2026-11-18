from __future__ import annotations

import glob
import json
import os
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from encoding import load_events_from_h5, encode_events_2ch_lif
except ImportError:
    from src.encoding import load_events_from_h5, encode_events_2ch_lif


def load_evcivil_labels(npy_path: str) -> np.ndarray:
    """
    Đọc file nhãn .npy của ev-CIVIL định dạng (M, 6):
    Cột: [timestamp_us, class_id, x, y, w, h]
    """
    raw = np.load(npy_path, allow_pickle=True)
    if raw.ndim != 2 or raw.shape[1] < 6 or len(raw) == 0:
        return np.zeros((0, 6), dtype=np.float64)

    labels = raw[:, :6].astype(np.float64)
    # Quy đổi sang micro-giây nếu lưu dưới dạng giây
    if len(labels) > 1 and (labels[-1, 0] - labels[0, 0]) < 1000.0:
        labels[:, 0] *= 1_000_000.0
    return labels


def discover_and_split_dataset(
    root_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    save_split_json: str = "dataset_splits.json",
) -> dict[str, list[dict]]:
    """
    Quét toàn bộ chuỗi trong Field_dataset và chia phân tầng (Stratified Split)
    theo cấp độ Sequence (tránh rò rỉ dữ liệu giữa train / val / test).
    """
    # Tìm tất cả file sự kiện (*_events.h5 hoặc events.h5)
    all_h5 = sorted(glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True))
    event_h5_files = [
        p for p in all_h5
        if os.path.basename(p).endswith("_events.h5") or os.path.basename(p) == "events.h5"
    ]

    grouped_sequences: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for h5_path in event_h5_files:
        seq_dir = os.path.dirname(h5_path)
        npy_files = glob.glob(os.path.join(seq_dir, "*.npy"))
        if not npy_files:
            skipped += 1
            continue

        labels = load_evcivil_labels(npy_files[0])
        if len(labels) == 0:
            skipped += 1
            continue

        # Xác định nhóm điều kiện (VD: Night_outdoor / Day_outdoor) và lớp chính để chia phân tầng
        rel_path = os.path.relpath(seq_dir, root_dir)
        condition = rel_path.split(os.sep)[0] if os.sep in rel_path else "default"
        unique_cls, counts = np.unique(labels[:, 1].astype(int), return_counts=True)
        dominant_cls = int(unique_cls[np.argmax(counts)])

        strat_key = f"{condition}_cls{dominant_cls}"
        grouped_sequences[strat_key].append({
            "seq_id": rel_path,
            "h5_path": h5_path,
            "npy_path": npy_files[0],
            "condition": condition,
            "num_labels": int(len(labels)),
            "crack_count": int(np.sum(labels[:, 1] == 0)),
            "spalling_count": int(np.sum(labels[:, 1] == 1)),
        })

    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for strat_key, items in sorted(grouped_sequences.items()):
        rng.shuffle(items)
        n_total = len(items)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else 0

        if n_train + n_val >= n_total and n_total >= 3:
            n_val = 1
            n_train = n_total - 2

        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    # Lưu cấu hình chia tập ra file JSON để dataset.py tái sử dụng đồng bộ
    with open(save_split_json, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print(f"TỔNG HỢP CHIA DỮ LIỆU TỪ: {root_dir} (Bỏ qua {skipped} thư mục thiếu nhãn)")
    print("=" * 78)
    for split_name in ["train", "val", "test"]:
        seqs = splits[split_name]
        tot_lbl = sum(s["num_labels"] for s in seqs)
        tot_crack = sum(s["crack_count"] for s in seqs)
        tot_spall = sum(s["spalling_count"] for s in seqs)
        print(
            f"  - {split_name.upper():<6}: {len(seqs):>3} sequences | "
            f"Tổng nhãn: {tot_lbl:>5} (Crack: {tot_crack:>5}, Spalling: {tot_spall:>5})"
        )
    print(f"Đã lưu danh sách chia tập tại: {save_split_json}\n")
    return splits


def verify_train_labels(train_seqs: list[dict], H: int = 260, W: int = 346) -> list[dict]:
    """Kiểm tra tính hợp lệ của nhãn trên tập TRAIN (tọa độ BBox, class_id, đồng bộ thời gian)."""
    print("--- KIỂM TRA TOÀN VẸN NHÃN TRÊN TẬP TRAIN ---")
    valid_train_seqs = []
    invalid_boxes = 0
    total_boxes = 0
    class_counts: dict[int, int] = defaultdict(int)

    for seq in train_seqs:
        labels = load_evcivil_labels(seq["npy_path"])
        total_boxes += len(labels)

        for row in labels:
            _, cls_id, bx, by, bw, bh = row
            class_counts[int(cls_id)] += 1
            # Kiểm tra hộp giới hạn có nằm hợp lệ trong khung hình 346x260 không
            if bw <= 1.0 or bh <= 1.0 or bx >= W or by >= H or (bx + bw) <= 0 or (by + bh) <= 0:
                invalid_boxes += 1

        valid_train_seqs.append(seq)

    print(f"  + Số chuỗi hợp lệ trong TRAIN : {len(valid_train_seqs)}/{len(train_seqs)}")
    print(f"  + Tổng số BBox trong TRAIN    : {total_boxes} (Hộp lỗi kích thước: {invalid_boxes})")
    print(f"  + Phân bố lớp (Class ID)      : {dict(class_counts)} (0: crack, 1: spalling)\n")
    return valid_train_seqs


def evaluate_and_select_lif_params(
    train_seqs: list[dict],
    max_eval_seqs: int = 25,
    seed: int = 42,
    H: int = 260,
    W: int = 346,
    N: int = 25,
) -> tuple[float, float]:
    """
    Đánh giá lưới tham số (tau_us, v_th) trên các chuỗi đại diện của tập TRAIN
    và tự động chọn bộ tham số tối ưu cân bằng giữa SNR và Sparsity.
    """
    rng = random.Random(seed)
    eval_seqs = train_seqs if len(train_seqs) <= max_eval_seqs else rng.sample(train_seqs, max_eval_seqs)
    dt_us = 1_000_000.0 / N  # 40,000 us

    # Chuẩn bị các lát cắt kiểm thử từ tập TRAIN (mỗi chuỗi lấy 1 clip 1 giây quanh vùng có nhãn)
    eval_clips = []
    for seq in eval_seqs:
        labels = load_evcivil_labels(seq["npy_path"])
        x, y, t, p = load_events_from_h5(seq["h5_path"])
        if len(t) == 0 or len(labels) == 0:
            continue

        mid_label_t = labels[len(labels) // 2, 0]
        t0 = float(np.clip(mid_label_t - 500_000.0, t[0], max(t[0], t[-1] - 1_000_000.0)))
        eval_clips.append((seq["seq_id"], x, y, t, p, labels, t0))

    param_grid = [
        (20000.0, 0.6), (20000.0, 0.8), (20000.0, 1.0),
        (30000.0, 0.8), (30000.0, 1.0),
        (40000.0, 0.6), (40000.0, 0.8), (40000.0, 1.0), (40000.0, 1.2),
    ]

    print(f"--- ĐÁNH GIÁ THAM SỐ (tau_us, v_th) TRÊN {len(eval_clips)} CHUỖI TẬP TRAIN ---")
    print(
        f"{'tau(ms)':<8} | {'v_th':<5} | {'Sparsity(%)':<12} | {'Avg In-Box':<11} | "
        f"{'Density Ratio (SNR)':<19} | {'Score':<8}"
    )
    print("-" * 78)

    best_score = -1.0
    best_params = (40000.0, 1.0)
    vis_sample_data = None

    for tau_us, v_th in param_grid:
        sparsity_list = []
        snr_list = []
        inbox_list = []

        for clip_idx, (seq_id, x, y, t, p, labels, t0) in enumerate(eval_clips):
            spikes_2ch = encode_events_2ch_lif(
                x, y, t, p, N=N, H=H, W=W, tau_us=tau_us, v_th=v_th, t_start_us=t0
            )
            sparsity = 100.0 * (1.0 - np.count_nonzero(spikes_2ch) / spikes_2ch.size)
            sparsity_list.append(sparsity)

            # Kiểm tra các khung hình khớp nhãn trong clip 1 giây này
            unique_ts = np.unique(labels[:, 0])
            for n in range(N):
                t_center = t0 + (n + 0.5) * dt_us
                nearest_t = unique_ts[np.argmin(np.abs(unique_ts - t_center))]
                if abs(nearest_t - t_center) > dt_us:
                    continue

                boxes = labels[labels[:, 0] == nearest_t]
                bbox_mask = np.zeros((H, W), dtype=bool)
                for _, _, bx, by, bw, bh in boxes:
                    x1, y1 = max(0, int(round(bx))), max(0, int(round(by)))
                    x2, y2 = min(W, int(round(bx + bw))), min(H, int(round(by + bh)))
                    bbox_mask[y1:y2, x1:x2] = True

                box_area = int(np.sum(bbox_mask))
                if box_area == 0:
                    continue
                bg_area = max(1, H * W - box_area)

                active_any = (spikes_2ch[n, 0] > 0) | (spikes_2ch[n, 1] > 0)
                in_box = int(np.sum(active_any & bbox_mask))
                bg_noise = int(np.sum(active_any & (~bbox_mask)))

                snr = (in_box / box_area) / max(1e-6, (bg_noise / bg_area))
                snr_list.append(snr)
                inbox_list.append(in_box)

                # Lưu lại mẫu đầu tiên để vẽ minh họa trực quan
                if clip_idx == 0 and n == 12 and vis_sample_data is None:
                    vis_sample_data = (seq_id, x, y, t, p, boxes, t0, n)

        mean_sparsity = float(np.mean(sparsity_list))
        mean_snr = float(np.mean(snr_list)) if snr_list else 0.0
        mean_inbox = float(np.mean(inbox_list)) if inbox_list else 0.0

        # Hàm mục tiêu: Ưu tiên SNR cao, giữ Sparsity >= 94% và đảm bảo đủ xung nét vết nứt (>= 150)
        sparsity_factor = 1.0 if mean_sparsity >= 94.0 else (mean_sparsity / 94.0) ** 3
        continuity_factor = min(1.0, mean_inbox / 150.0)
        score = mean_snr * sparsity_factor * continuity_factor

        print(
            f"{tau_us/1000:<8.0f} | {v_th:<5.1f} | {mean_sparsity:<12.2f} | {mean_inbox:<11.1f} | "
            f"{mean_snr:<19.2f}x | {score:<8.3f}"
        )

        if score > best_score:
            best_score = score
            best_params = (tau_us, v_th)

    print("-" * 78)
    print(
        f">>> THAM SỐ TỐI ƯU ĐƯỢC CHỌN TỪ TẬP TRAIN: "
        f"tau_us = {best_params[0]:.0f} ({best_params[0]/1000:.0f} ms), v_th = {best_params[1]} "
        f"(Score = {best_score:.3f})\n"
    )

    # Vẽ ảnh kiểm tra chất lượng với bộ tham số tối ưu vừa chọn
    if vis_sample_data is not None:
        seq_id, x, y, t, p, boxes, t0, target_n = vis_sample_data
        spikes_best = encode_events_2ch_lif(
            x, y, t, p, N=N, H=H, W=W, tau_us=best_params[0], v_th=best_params[1], t_start_us=t0
        )
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        vis[spikes_best[target_n, 0] > 0] = [255, 65, 65]
        vis[spikes_best[target_n, 1] > 0] = [65, 145, 255]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(vis)
        for _, cls_id, bx, by, bw, bh in boxes:
            rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.0, edgecolor="#00FF66", facecolor="none")
            ax.add_patch(rect)
            cls_name = "crack" if int(cls_id) == 0 else "spalling"
            ax.text(bx, max(0, by - 5), f"{cls_name}", color="#00FF66", fontsize=9, weight="bold")

        ax.set_title(
            f"Train Sample ({os.path.basename(seq_id)}) | "
            f"tau={int(best_params[0]/1000)}ms, v_th={best_params[1]}"
        )
        ax.axis("off")
        plt.savefig("lif_best_train_check.png", dpi=150, bbox_inches="tight")
        print("Đã lưu ảnh kiểm tra mẫu Train với tham số tối ưu tại: lif_best_train_check.png")

    return best_params


def main():
    dataset_root = "/Users/trantung/Downloads/Field_dataset/"

    # 1. Quét và chia dữ liệu thành Train (70%) / Val (15%) / Test (15%)
    splits = discover_and_split_dataset(
        root_dir=dataset_root,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
        save_split_json="dataset_splits.json",
    )

    # 2. Chỉ sử dụng tập TRAIN để kiểm tra nhãn và chọn tham số LIF
    valid_train_seqs = verify_train_labels(splits["train"])
    best_tau_us, best_v_th = evaluate_and_select_lif_params(
        train_seqs=valid_train_seqs,
        max_eval_seqs=25,  # Đánh giá trên 25 chuỗi đại diện của tập Train (đổi thành None nếu muốn chạy toàn bộ)
        seed=42,
    )


if __name__ == "__main__":
    main()