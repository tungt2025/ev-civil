from __future__ import annotations

import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from torch.cuda.amp import autocast, GradScaler

# Giải pháp 1: Chống phân mảnh VRAM do PyTorch gây ra
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from dataset import build_dataloaders
    from model import QKFormerDetector
except ImportError:
    from src.dataset import build_dataloaders
    from src.model import QKFormerDetector


# ==============================================================================
# 1. HÀM MẤT MÁT ANCHOR-FREE (CIoU LOSS + IoU-AWARE FOCAL OBJECTNESS + CLS LOSS)
# ==============================================================================

def compute_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tính Complete IoU (CIoU) và IoU chuẩn giữa hai tập hộp giới hạn [x1, y1, x2, y2] tương ứng 1-1.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)

    w1 = torch.clamp(b1_x2 - b1_x1, min=eps)
    h1 = torch.clamp(b1_y2 - b1_y1, min=eps)
    w2 = torch.clamp(b2_x2 - b2_x1, min=eps)
    h2 = torch.clamp(b2_y2 - b2_y1, min=eps)
    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    iou = inter_area / union_area

    cx1, cy1 = 0.5 * (b1_x1 + b1_x2), 0.5 * (b1_y1 + b1_y2)
    cx2, cy2 = 0.5 * (b2_x1 + b2_x2), 0.5 * (b2_y1 + b2_y2)
    center_dist_sq = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)

    enc_x1 = torch.minimum(b1_x1, b2_x1)
    enc_y1 = torch.minimum(b1_y1, b2_y1)
    enc_x2 = torch.maximum(b1_x2, b2_x2)
    enc_y2 = torch.maximum(b1_y2, b2_y2)
    c_sq = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

    v = (4.0 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (center_dist_sq / c_sq + alpha * v)
    return ciou, iou


class SpikingDetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def forward(
        self,
        preds: torch.Tensor,
        anchors: torch.Tensor,
        targets: list[list[dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, T, M, _ = preds.shape
        device = preds.device

        gx = anchors[:, 0]
        gy = anchors[:, 1]
        stride = anchors[:, 2]
        ax = (gx + 0.5) * stride
        ay = (gy + 0.5) * stride

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_pos = 0

        for b in range(B):
            for t in range(T):
                gt_boxes = targets[b][t]["boxes"].to(device)
                gt_labels = targets[b][t]["labels"].to(device)
                K = gt_boxes.shape[0]

                pred_frame = preds[b, t]
                obj_target = torch.zeros((M,), device=device, dtype=torch.float32)

                if K > 0:
                    gt_cx = 0.5 * (gt_boxes[:, 0] + gt_boxes[:, 2])
                    gt_cy = 0.5 * (gt_boxes[:, 1] + gt_boxes[:, 3])
                    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
                    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
                    gt_max_size = torch.maximum(gt_w, gt_h)

                    in_box = (
                        (ax[:, None] >= gt_boxes[None, :, 0])
                        & (ax[:, None] <= gt_boxes[None, :, 2])
                        & (ay[:, None] >= gt_boxes[None, :, 1])
                        & (ay[:, None] <= gt_boxes[None, :, 3])
                    )
                    in_center = (
                        (torch.abs(ax[:, None] - gt_cx[None, :]) <= 1.5 * stride[:, None])
                        & (torch.abs(ay[:, None] - gt_cy[None, :]) <= 1.5 * stride[:, None])
                    )

                    is_p3 = (stride[:, None] == 8.0) & (gt_max_size[None, :] <= 80.0)
                    is_p4 = (stride[:, None] == 16.0) & (gt_max_size[None, :] > 32.0) & (gt_max_size[None, :] <= 160.0)
                    is_p5 = (stride[:, None] == 32.0) & (gt_max_size[None, :] > 96.0)
                    scale_ok = is_p3 | is_p4 | is_p5

                    match_matrix = (in_box | in_center) & scale_ok

                    for k in range(K):
                        if not torch.any(match_matrix[:, k]):
                            dist_sq = (ax - gt_cx[k]).pow(2) + (ay - gt_cy[k]).pow(2)
                            best_anchor = int(torch.argmin(dist_sq).item())
                            match_matrix[best_anchor, k] = True

                    gt_areas = (gt_w * gt_h)[None, :].expand(M, K)
                    cost_matrix = torch.where(match_matrix, gt_areas, torch.full_like(gt_areas, 1e12))
                    min_cost, matched_gt_idx = cost_matrix.min(dim=1)
                    pos_mask = min_cost < 1e11

                    num_pos = int(pos_mask.sum().item())
                    if num_pos > 0:
                        total_pos += num_pos
                        p_pos = pred_frame[pos_mask]
                        gt_idx_pos = matched_gt_idx[pos_mask]

                        gx_p = gx[pos_mask]
                        gy_p = gy[pos_mask]
                        st_p = stride[pos_mask]

                        cx = (gx_p + torch.sigmoid(p_pos[:, 0]) * 2.0 - 0.5) * st_p
                        cy = (gy_p + torch.sigmoid(p_pos[:, 1]) * 2.0 - 0.5) * st_p
                        bw = torch.exp(torch.clamp(p_pos[:, 2], -5.0, 5.0)) * st_p
                        bh = torch.exp(torch.clamp(p_pos[:, 3], -5.0, 5.0)) * st_p

                        pred_xyxy = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=1)
                        target_xyxy = gt_boxes[gt_idx_pos]

                        ciou, ious = compute_ciou(pred_xyxy, target_xyxy)
                        total_box_loss = total_box_loss + (1.0 - ciou).sum()

                        obj_target[pos_mask] = ious.detach().clamp(min=0.1, max=1.0)

                        cls_target = F.one_hot(gt_labels[gt_idx_pos], num_classes=self.num_classes).float()
                        total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                            p_pos[:, 5:], cls_target, reduction="sum"
                        )

                obj_logits = pred_frame[:, 4]
                bce_obj = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="none")
                p_t = torch.sigmoid(obj_logits) * obj_target + (1.0 - torch.sigmoid(obj_logits)) * (1.0 - obj_target)
                focal_weight = (1.0 - p_t).pow(1.5)
                total_obj_loss = total_obj_loss + (focal_weight * bce_obj).sum()

        norm_factor = float(max(1, total_pos))
        loss_box = (total_box_loss / norm_factor) * self.lambda_box
        loss_obj = (total_obj_loss / norm_factor) * self.lambda_obj
        loss_cls = (total_cls_loss / norm_factor) * self.lambda_cls
        loss_total = loss_box + loss_obj + loss_cls

        return loss_total, {
            "loss": float(loss_total.item()),
            "box": float(loss_box.item()),
            "obj": float(loss_obj.item()),
            "cls": float(loss_cls.item()),
            "pos_anchors": total_pos,
        }


# ==============================================================================
# 2. BỘ TÍNH ĐIỂM ĐÁNH GIÁ CHUẨN OBJECT DETECTION: mAP@0.5
# ==============================================================================

def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union

def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

@torch.no_grad()
def evaluate_map50(
    model: QKFormerDetector,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_thresh: float = 0.50,
    conf_thresh: float = 0.05,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    num_classes = model.num_classes

    preds_by_class: dict[int, list[tuple[int, float, np.ndarray]]] = {c: [] for c in range(num_classes)}
    gts_by_class: dict[int, dict[int, np.ndarray]] = {c: {} for c in range(num_classes)}
    gt_counts: dict[int, int] = {c: 0 for c in range(num_classes)}

    sops_stats_list = []
    frame_uid = 0

    for batch_idx, (spikes, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        spikes = spikes.to(device)
        
        # Thêm context autocast vào quá trình đánh giá để nhất quán và tiết kiệm bộ nhớ
        with autocast():
            preds, anchors = model(spikes)
        
        sops_stats_list.append(model.estimate_sops_and_energy(T=spikes.shape[1]))

        # Đảm bảo preds trở về dạng FP32 cho việc decode chính xác nếu nó ở FP16
        decoded = model.decode_predictions(preds.float(), anchors.float(), conf_thresh=conf_thresh, nms_thresh=0.45)
        B, T = spikes.shape[0], spikes.shape[1]

        for b in range(B):
            for t in range(T):
                gt_b = targets[b][t]["boxes"].numpy()
                gt_l = targets[b][t]["labels"].numpy()

                pred_b = decoded[b][t]["boxes"].cpu().numpy()
                pred_s = decoded[b][t]["scores"].cpu().numpy()
                pred_l = decoded[b][t]["labels"].cpu().numpy()

                for c in range(num_classes):
                    gt_c = gt_b[gt_l == c]
                    if len(gt_c) > 0:
                        gts_by_class[c][frame_uid] = gt_c
                        gt_counts[c] += len(gt_c)

                    mask_p = pred_l == c
                    for box_p, score_p in zip(pred_b[mask_p], pred_s[mask_p]):
                        preds_by_class[c].append((frame_uid, float(score_p), box_p))

                frame_uid += 1
                
        del spikes, preds, anchors, decoded

    ap_per_class = {}
    for c in range(num_classes):
        n_gt = gt_counts[c]
        c_preds = preds_by_class[c]
        if n_gt == 0 or len(c_preds) == 0:
            ap_per_class[c] = 0.0
            continue

        c_preds.sort(key=lambda x: x[1], reverse=True)
        matched_flags = {uid: np.zeros(len(boxes), dtype=bool) for uid, boxes in gts_by_class[c].items()}

        tp = np.zeros(len(c_preds), dtype=np.float32)
        fp = np.zeros(len(c_preds), dtype=np.float32)

        for i, (uid, _, box_p) in enumerate(c_preds):
            if uid not in gts_by_class[c]:
                fp[i] = 1.0
                continue
            gt_boxes = gts_by_class[c][uid]
            ious = box_iou_matrix(box_p[None, :], gt_boxes)[0]
            best_gt = int(np.argmax(ious))
            if ious[best_gt] >= iou_thresh and not matched_flags[uid][best_gt]:
                tp[i] = 1.0
                matched_flags[uid][best_gt] = True
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / float(n_gt)
        precisions = cum_tp / np.maximum(1e-7, cum_tp + cum_fp)
        ap_per_class[c] = compute_ap_from_pr(recalls, precisions)

    mean_ap = sum(ap_per_class.values()) / max(1, num_classes)
    avg_saving = float(np.mean([s["energy_saving_pct"] for s in sops_stats_list])) if sops_stats_list else 0.0
    avg_fr = float(np.mean([s["avg_spike_rate_pct"] for s in sops_stats_list])) if sops_stats_list else 0.0

    return {
        "mAP50": mean_ap * 100.0,
        "AP50_crack": ap_per_class.get(0, 0.0) * 100.0,
        "AP50_spalling": ap_per_class.get(1, 0.0) * 100.0,
        "spike_rate_pct": avg_fr,
        "energy_saving_pct": avg_saving,
    }


# ==============================================================================
# 3. VÒNG LẶP HUẤN LUYỆN CHÍNH (TRAINING LOOP)
# ==============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: QKFormerDetector,
    criterion: SpikingDetectionLoss,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,      # Giải pháp 2: Sử dụng Scaler để điều chỉnh gradient cho AMP
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 1,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()

    running = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    num_steps = 0
    t0 = time.time()

    total_steps = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    log_interval = 1 if (max_batches is not None and max_batches <= 10) else 25

    for step, (spikes, targets) in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break

        spikes = spikes.to(device)
        
        # Giải pháp 2: Automatic Mixed Precision (AMP) - Ép kiểu tự động sang FP16 giảm nửa RAM
        with autocast():
            preds, anchors = model(spikes)
            loss, metrics = criterion(preds, anchors, targets)
            loss_scaled = loss / float(grad_accum_steps)

        # Giải pháp 2: Tối ưu ngược với GradScaler
        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
            # Unscale gradient trước khi kẹp
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        for k in running:
            running[k] += metrics[k]
        num_steps += 1

        if (step + 1) % log_interval == 0 or (step + 1) == total_steps:
            elapsed = time.time() - t0
            print(
                f"  [Epoch {epoch:02d} | Step {step+1:>4}/{total_steps}] "
                f"Loss: {running['loss']/num_steps:.4f} "
                f"(Box: {running['box']/num_steps:.4f}, Obj: {running['obj']/num_steps:.4f}, "
                f"Cls: {running['cls']/num_steps:.4f}) | Pos: {metrics['pos_anchors']} | {elapsed:.1f}s"
            )
            
        # Giải pháp 3: Giải phóng bộ nhớ GPU của các biến tạm sau mỗi batch
        del spikes, targets, preds, anchors, loss, loss_scaled
        
    # Xóa cache rác sau mỗi epoch
    torch.cuda.empty_cache()
    gc.collect()

    denom = max(1, num_steps)
    return {k: v / denom for k, v in running.items()}


def resolve_lif_params(
    tau_us_cli: float | None,
    v_th_cli: float | None,
    params_json: str,
    default_tau_us: float = 40000.0,
    default_v_th: float = 1.2,
) -> tuple[float, float]:
    file_params = {}
    if params_json:
        params_path = os.path.abspath(params_json)
        if os.path.isfile(params_path):
            with open(params_path, "r", encoding="utf-8") as f:
                file_params = json.load(f)
            if not isinstance(file_params, dict):
                raise ValueError(f"File tham số phải chứa một JSON object: {params_path}")

    tau_us = tau_us_cli
    if tau_us is None:
        tau_us = file_params.get("tau_us", default_tau_us)

    v_th = v_th_cli
    if v_th is None:
        v_th = file_params.get("v_th", default_v_th)

    return float(tau_us), float(v_th)


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện QKFormerDetector trên tập ev-CIVIL")
    parser.add_argument("--split-json", type=str, default="dataset_splits.json", help="Đường dẫn file chia tập")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện")
    # Giải pháp 4: Mặc định giảm Batch Size xuống 2 để đảm bảo không bị OOM, bù đắp bằng grad_accum
    parser.add_argument("--batch-size", type=int, default=2, help="Kích thước batch (mỗi mẫu là 25 frames)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Số bước tích lũy gradient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Tốc độ học khởi tạo (AdamW)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Hệ số điều chuẩn Weight Decay")
    parser.add_argument("--num-workers", type=int, default=2, help="Số luồng đọc HDF5 song song")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/spikes",
        help="Thư mục lưu tensor spike đã mã hóa; dùng '' để tắt cache",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Chỉ mã hóa và lưu cache, không huấn luyện",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default="params.json",
        help="File JSON chứa tau_us và v_th, ví dụ params.json",
    )
    parser.add_argument(
        "--tau-us",
        type=float,
        default=None,
        help="Hằng số suy giảm LIF (us); ghi đè giá trị từ params.json",
    )
    parser.add_argument(
        "--v-th",
        type=float,
        default=None,
        help="Ngưỡng phát xung LIF; ghi đè giá trị từ params.json",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Thư mục lưu trọng số")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Đường dẫn checkpoint để tiếp tục huấn luyện, ví dụ checkpoints/best_model.pt",
    )
    parser.add_argument("--sanity-check", action="store_true", help="Chạy thử nhanh 5 batch để kiểm tra pipeline")
    args = parser.parse_args()
    tau_us, v_th = resolve_lif_params(
        tau_us_cli=args.tau_us,
        v_th_cli=args.v_th,
        params_json=args.params_json,
    )
    print(f"LIF params -> tau_us={tau_us:.0f}, v_th={v_th}")

    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()
    print(f"Khởi chạy huấn luyện QKFormerDetector trên thiết bị: {device}")

    cache_dir = args.cache_dir or None
    train_loader, val_loader, test_loader = build_dataloaders(
        split_json=args.split_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tau_us=tau_us,
        v_th=v_th,
        cache_dir=cache_dir,
    )
    if cache_dir:
        print(f"Đang tiền xử lý và lưu cache tensor spike tại: {os.path.abspath(cache_dir)}")
        for split_name, loader in (
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ):
            stats = loader.dataset.preprocess_cache()
            print(
                f"  {split_name}: đã có {stats['cached']} cache, "
                f"đã mã hóa {stats['encoded']} clip"
            )
        if args.preprocess_only:
            print("Tiền xử lý hoàn tất; không bắt đầu huấn luyện.")
            return
    print(
        f"Dữ liệu đã sẵn sàng -> Train: {len(train_loader.dataset)} clips | "
        f"Val: {len(val_loader.dataset)} clips | Test: {len(test_loader.dataset)} clips"
    )

    model = QKFormerDetector(num_classes=2, embed_dims=(64, 128, 256, 320), fpn_dim=128).to(device)
    criterion = SpikingDetectionLoss(num_classes=2, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Khởi tạo Scaler cho Mixed Precision
    scaler = GradScaler()

    epochs = 2 if args.sanity_check else args.epochs
    max_batches = 5 if args.sanity_check else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_map50 = -1.0
    history = []
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
        print(f"Đang resume từ checkpoint: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])

        resumed_epoch = int(resume_ckpt.get("epoch", 0))
        start_epoch = resumed_epoch + 1
        best_map50 = float(resume_ckpt.get("val_metrics", {}).get("mAP50", -1.0))

        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            scheduler.T_max = epochs
        if "scaler_state_dict" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

        history_path = os.path.join(args.save_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            best_map50 = max(
                (record["val"]["mAP50"] for record in history if "val" in record),
                default=best_map50,
            )

        if start_epoch > epochs:
            print(f"Checkpoint đã ở epoch {resumed_epoch}; không còn epoch nào để huấn luyện.")
    elif args.resume:
        print(f"Không tìm thấy checkpoint '{args.resume}', bắt đầu huấn luyện từ đầu.")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*76}\nEPOCH {epoch}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.6f})\n{'='*76}")
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            grad_accum_steps=args.grad_accum,
            max_batches=max_batches,
        )
        scheduler.step()

        val_metrics = evaluate_map50(
            model=model,
            dataloader=val_loader,
            device=device,
            iou_thresh=0.50,
            conf_thresh=0.05,
            max_batches=max_batches,
        )

        print(
            f"--> [VAL Epoch {epoch:02d}] mAP@0.5: {val_metrics['mAP50']:.2f}% "
            f"(Crack: {val_metrics['AP50_crack']:.2f}%, Spalling: {val_metrics['AP50_spalling']:.2f}%) | "
            f"Spike Rate: {val_metrics['spike_rate_pct']:.2f}% | "
            f"Energy Saving: {val_metrics['energy_saving_pct']:.2f}%"
        )

        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_record)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.save_dir, "last_model.pt"))
        if val_metrics["mAP50"] >= best_map50:
            best_map50 = val_metrics["mAP50"]
            torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pt"))
            print(f"    *** Đã cập nhật Best Model (mAP@0.5 = {best_map50:.2f}%) ***")

        with open(os.path.join(args.save_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    best_ckpt = torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_map50(
        model=model,
        dataloader=test_loader,
        device=device,
        iou_thresh=0.50,
        conf_thresh=0.05,
        max_batches=max_batches,
    )
    print("\n" + "=" * 76)
    print(
        f"KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST (Best Checkpoint Epoch {best_ckpt['epoch']}):\n"
        f"  - mAP@0.5       : {test_metrics['mAP50']:.2f}%\n"
        f"  - AP50 Crack    : {test_metrics['AP50_crack']:.2f}%\n"
        f"  - AP50 Spalling : {test_metrics['AP50_spalling']:.2f}%\n"
        f"  - Avg Spike Rate: {test_metrics['spike_rate_pct']:.2f}%\n"
        f"  - Energy Saving : {test_metrics['energy_saving_pct']:.2f}%"
    )
    print("=" * 76)

if __name__ == "__main__":
    main()