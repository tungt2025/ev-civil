from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. NƠ-RON XUNG LIF ĐA BƯỚC VỚI GRADIENT THAY THẾ (SURROGATE GRADIENT)
# ==============================================================================

class ATanSurrogate(torch.autograd.Function):
    """Hàm bậc thang Heaviside ở lượt đi (forward) và đạo hàm ATan trơn ở lượt về (backward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_x = (alpha / 2.0) / (1.0 + ((math.pi / 2.0) * alpha * x).pow(2)) * grad_output
        return grad_x, None


def atan_spike(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return ATanSurrogate.apply(x, alpha)


class MultiStepLIF(nn.Module):
    """
    Nơ-ron Leaky Integrate-and-Fire (LIF) xử lý tuần tự qua T bước thời gian.
    Đầu vào/ra có dạng (T, B, ...). Tự động ghi nhận tỷ lệ phát xung (firing rate) để tính SOPs.
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_th: float = 1.0,
        alpha: float = 2.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.decay = 1.0 - (1.0 / tau)
        self.v_th = v_th
        self.alpha = alpha
        self.detach_reset = detach_reset

        # Biến thống kê phục vụ tính toán năng lượng SOPs
        self.last_firing_rate: float = 0.0
        self.last_num_elements: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, ...)
        T = x.shape[0]
        u = torch.zeros_like(x[0])
        spikes = []

        for t in range(T):
            u = self.decay * u + x[t]
            s = atan_spike(u - self.v_th, self.alpha)
            s_reset = s.detach() if self.detach_reset else s
            u = u - s_reset * self.v_th
            spikes.append(s)

        out = torch.stack(spikes, dim=0)
        with torch.no_grad():
            self.last_num_elements = out.numel()
            self.last_firing_rate = float(out.mean().item())
        return out


# ==============================================================================
# 2. CÁC KHỐI CƠ BẢN CỦA QKFORMER (SPIKING CONV, Q-K ATTENTION, SPIKING MLP)
# ==============================================================================

class SpikingConv2d(nn.Module):
    """Khối tích chập không gian 2D thuần xung: Conv2d -> BatchNorm2d -> MultiStepLIF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        v_th: float = 1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = MultiStepLIF(v_th=v_th)
        self.last_in_firing_rate: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W)
        with torch.no_grad():
            self.last_in_firing_rate = float((x > 0).float().mean().item())

        T, B, C, H, W = x.shape
        y = self.conv(x.reshape(T * B, C, H, W))
        y = self.bn(y)
        _, C_out, H_out, W_out = y.shape
        y = y.reshape(T, B, C_out, H_out, W_out)
        return self.lif(y)


class TokenQKAttention(nn.Module):
    """
    Cơ chế Q-K Attention độ phức tạp tuyến tính O(N) của QKFormer:
    Loại bỏ hoàn toàn ma trận Value (V) và Softmax, chỉ dùng phép cộng và nhân mặt nạ xung nhị phân.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) phải chia hết cho num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIF()

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIF()

        # LIF tạo mặt nạ attention theo chiều token (với ngưỡng thấp hơn do tổng hợp theo head_dim)
        self.attn_lif = MultiStepLIF(v_th=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIF()

        self.last_in_firing_rate: float = 0.0
        self.last_attn_firing_rate: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W)
        with torch.no_grad():
            self.last_in_firing_rate = float((x > 0).float().mean().item())

        T, B, C, H, W = x.shape
        N = H * W
        x_flat = x.flatten(3).reshape(T * B, C, N)  # (T*B, C, N)

        q = self.q_bn(self.q_conv(x_flat)).reshape(T, B, C, N)
        q = self.q_lif(q)

        k = self.k_bn(self.k_conv(x_flat)).reshape(T, B, C, N)
        k = self.k_lif(k)

        # Tách đa đầu (Multi-head): (T, B, num_heads, head_dim, N)
        q_heads = q.reshape(T, B, self.num_heads, self.head_dim, N)
        k_heads = k.reshape(T, B, self.num_heads, self.head_dim, N)

        # Tổng hợp xung Q theo chiều đặc trưng đầu -> tạo mặt nạ xung chú ý theo không gian token
        q_sum = q_heads.sum(dim=3, keepdim=True)  # (T, B, num_heads, 1, N)
        attn_mask = self.attn_lif(q_sum)          # Nhị phân {0, 1}

        with torch.no_grad():
            self.last_attn_firing_rate = float(attn_mask.mean().item())

        # Nhân mặt nạ xung (tương đương phép AND logic trên phần cứng neuromorphic)
        x_attn = (attn_mask * k_heads).reshape(T * B, C, N)
        out = self.proj_bn(self.proj_conv(x_attn)).reshape(T, B, C, H, W)
        return self.proj_lif(out)


class SpikingMLP(nn.Module):
    """Mạng truyền thẳng theo kênh thuần xung (Channel-wise Spiking MLP)."""

    def __init__(self, in_features: int, hidden_features: int | None = None):
        super().__init__()
        hidden_features = hidden_features or (in_features * 2)
        self.fc1 = SpikingConv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.fc2 = SpikingConv2d(hidden_features, in_features, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class QKFormerBlock(nn.Module):
    """Khối biến áp xung QKFormer với kết nối thặng dư cộng xung (Spike-driven Residual)."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.attn = TokenQKAttention(dim=dim, num_heads=num_heads)
        self.mlp = SpikingMLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ==============================================================================
# 3. KIẾN TRÚC PHÂN CẤP QKFORMER BACKBONE + SPIKING FPN + DETECTION HEAD
# ==============================================================================

class QKFormerBackbone(nn.Module):
    """
    Backbone phân cấp 4 giai đoạn nhận tensor sự kiện thưa (T, B, 2, 260, 346)
    và xuất ra 3 bản đồ đặc trưng xung đa tỷ lệ (P3, P4, P5) ở các bước nhảy 8, 16, 32.
    """

    def __init__(self, in_channels: int = 2, embed_dims: tuple[int, int, int, int] = (64, 128, 256, 320)):
        super().__init__()
        c1, c2, c3, c4 = embed_dims

        # Stage 1 (Stride 4): 260x346 -> 130x173 -> 65x87
        self.stem1 = SpikingConv2d(in_channels, c1 // 2, kernel_size=3, stride=2, padding=1)
        self.stem2 = SpikingConv2d(c1 // 2, c1, kernel_size=3, stride=2, padding=1)
        self.stage1 = QKFormerBlock(dim=c1, num_heads=4)

        # Stage 2 (Stride 8 -> P3): 65x87 -> 33x44
        self.down2 = SpikingConv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.stage2 = QKFormerBlock(dim=c2, num_heads=4)

        # Stage 3 (Stride 16 -> P4): 33x44 -> 17x22 (2 khối QKFormer sâu hơn để bắt hình thái vết nứt)
        self.down3 = SpikingConv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Sequential(
            QKFormerBlock(dim=c3, num_heads=8),
            QKFormerBlock(dim=c3, num_heads=8),
        )

        # Stage 4 (Stride 32 -> P5): 17x22 -> 9x11
        self.down4 = SpikingConv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        self.stage4 = QKFormerBlock(dim=c4, num_heads=8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (T, B, 2, 260, 346)
        x1 = self.stage1(self.stem2(self.stem1(x)))   # (T, B, c1, 65, 87)
        p3 = self.stage2(self.down2(x1))              # (T, B, c2, 33, 44) - Stride 8
        p4 = self.stage3(self.down3(p3))              # (T, B, c3, 17, 22) - Stride 16
        p5 = self.stage4(self.down4(p4))              # (T, B, c4, 9, 11)  - Stride 32
        return p3, p4, p5


class SpikingFPN(nn.Module):
    """Tháp đặc trưng thuần xung kết hợp thông tin ngữ nghĩa từ trên xuống (Top-Down)."""

    def __init__(self, in_dims: tuple[int, int, int] = (128, 256, 320), fpn_dim: int = 128):
        super().__init__()
        c2, c3, c4 = in_dims
        self.lat_p3 = SpikingConv2d(c2, fpn_dim, kernel_size=1, stride=1, padding=0)
        self.lat_p4 = SpikingConv2d(c3, fpn_dim, kernel_size=1, stride=1, padding=0)
        self.lat_p5 = SpikingConv2d(c4, fpn_dim, kernel_size=1, stride=1, padding=0)

        self.smooth_p4 = SpikingConv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.smooth_p3 = SpikingConv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _upsample_spikes(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        T, B, C, H, W = x.shape
        up = F.interpolate(x.reshape(T * B, C, H, W), size=target_hw, mode="nearest")
        return up.reshape(T, B, C, target_hw[0], target_hw[1])

    def forward(
        self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = feats
        m5 = self.lat_p5(p5)
        m4 = self.smooth_p4(self.lat_p4(p4) + self._upsample_spikes(m5, (p4.shape[-2], p4.shape[-1])))
        m3 = self.smooth_p3(self.lat_p3(p3) + self._upsample_spikes(m4, (p3.shape[-2], p3.shape[-1])))
        return m3, m4, m5


class SpikingDetectionHead(nn.Module):
    """
    Đầu dò không neo (Anchor-free Decoupled Spiking Head) dự đoán tại từng bước thời gian t in [0..T-1].
    Lớp đọc ra cuối dùng tích hợp điện thế màng (không cắt ngưỡng nhị phân) để hồi quy tọa độ chính xác.
    """

    def __init__(self, in_dim: int = 128, num_classes: int = 2, strides: tuple[int, int, int] = (8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.stem = SpikingConv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.cls_conv = SpikingConv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.reg_conv = SpikingConv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)

        # Lớp đọc điện thế màng cuối (Membrane Readout)
        self.cls_pred = nn.Conv2d(in_dim, num_classes, kernel_size=1, stride=1, padding=0)
        self.reg_pred = nn.Conv2d(in_dim, 4, kernel_size=1, stride=1, padding=0)
        self.obj_pred = nn.Conv2d(in_dim, 1, kernel_size=1, stride=1, padding=0)

        # Khởi tạo bias cho lớp phân loại với xác suất tiên nghiệm thấp (tránh mất cân bằng nền/vật thể)
        prior_prob = 0.01
        bias_init = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_init)
        nn.init.constant_(self.obj_pred.bias, bias_init)

    def forward(
        self, fpn_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:
          - preds: Tensor (B, T, Total_Anchors, 5 + num_classes) gồm [tx, ty, tw, th, obj_logit, cls_logits...]
          - anchors: Tensor (Total_Anchors, 3) gồm [grid_x, grid_y, stride] trên device hiện tại
        """
        all_preds = []
        all_anchors = []
        device = fpn_feats[0].device

        for feat, stride in zip(fpn_feats, self.strides):
            T, B, C, H_l, W_l = feat.shape
            h_stem = self.stem(feat)

            cls_feat = self.cls_conv(h_stem).reshape(T * B, C, H_l, W_l)
            reg_feat = self.reg_conv(h_stem).reshape(T * B, C, H_l, W_l)

            cls_out = self.cls_pred(cls_feat).reshape(T, B, self.num_classes, H_l * W_l)
            reg_out = self.reg_pred(reg_feat).reshape(T, B, 4, H_l * W_l)
            obj_out = self.obj_pred(reg_feat).reshape(T, B, 1, H_l * W_l)

            # Ghép thành (T, B, 5 + num_classes, H_l * W_l) -> chuyển về (B, T, H_l * W_l, 5 + num_classes)
            level_pred = torch.cat([reg_out, obj_out, cls_out], dim=2).permute(1, 0, 3, 2)
            all_preds.append(level_pred)

            # Tạo lưới tọa độ tâm ô lưới (grid_x, grid_y, stride)
            gy, gx = torch.meshgrid(
                torch.arange(H_l, device=device, dtype=torch.float32),
                torch.arange(W_l, device=device, dtype=torch.float32),
                indexing="ij",
            )
            level_anchors = torch.stack(
                [gx.flatten(), gy.flatten(), torch.full((H_l * W_l,), float(stride), device=device)],
                dim=1,
            )
            all_anchors.append(level_anchors)

        preds = torch.cat(all_preds, dim=2)       # (B, T, 1925, 7)
        anchors = torch.cat(all_anchors, dim=0)   # (1925, 3)
        return preds, anchors


# ==============================================================================
# 4. MÔ HÌNH TỔNG THỂ QKFORMER-DETECTOR + BỘ ĐO NĂNG LƯỢNG SOPs
# ==============================================================================

class QKFormerDetector(nn.Module):
    """
    Mô hình phát hiện khuyết tật công trình (Crack & Spalling) trên dữ liệu sự kiện ev-CIVIL.
    Đầu vào: (B, T=25, C=2, H=260, W=346)
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dims: tuple[int, int, int, int] = (64, 128, 256, 320),
        fpn_dim: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = QKFormerBackbone(in_channels=2, embed_dims=embed_dims)
        self.neck = SpikingFPN(in_dims=(embed_dims[1], embed_dims[2], embed_dims[3]), fpn_dim=fpn_dim)
        self.head = SpikingDetectionHead(in_dim=fpn_dim, num_classes=num_classes, strides=(8, 16, 32))

    def forward(self, spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Chuyển (B, T, C, H, W) -> (T, B, C, H, W) chuẩn xử lý SNN đa bước
        x = spikes.permute(1, 0, 2, 3, 4).contiguous()
        p3, p4, p5 = self.backbone(x)
        m3, m4, m5 = self.neck((p3, p4, p5))
        preds, anchors = self.head((m3, m4, m5))
        return preds, anchors

    @torch.no_grad()
    def decode_predictions(
        self,
        preds: torch.Tensor,
        anchors: torch.Tensor,
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        img_hw: tuple[int, int] = (260, 346),
    ) -> list[list[dict[str, torch.Tensor]]]:
        """
        Giải mã đầu ra thô (B, T, M, 5 + num_classes) thành danh sách hộp giới hạn [x1, y1, x2, y2]
        đã qua lọc ngưỡng tin cậy và Non-Maximum Suppression (NMS) cho từng bước thời gian.
        """
        B, T, M, _ = preds.shape
        H, W = img_hw
        gx, gy, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]

        results: list[list[dict[str, torch.Tensor]]] = []
        for b in range(B):
            seq_results: list[dict[str, torch.Tensor]] = []
            for t in range(T):
                p = preds[b, t]  # (M, 5 + C)
                obj_prob = torch.sigmoid(p[:, 4])
                cls_probs = torch.sigmoid(p[:, 5:])
                max_cls_prob, cls_ids = cls_probs.max(dim=1)
                scores = obj_prob * max_cls_prob

                keep_mask = scores >= conf_thresh
                if not torch.any(keep_mask):
                    seq_results.append({
                        "boxes": torch.zeros((0, 4), device=preds.device),
                        "scores": torch.zeros((0,), device=preds.device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=preds.device),
                    })
                    continue

                p_kept = p[keep_mask]
                gx_k, gy_k, st_k = gx[keep_mask], gy[keep_mask], stride[keep_mask]
                scores_k = scores[keep_mask]
                cls_k = cls_ids[keep_mask]

                # Giải mã tọa độ tâm (cx, cy) và kích thước (w, h) theo công thức YOLOX anchor-free
                cx = (gx_k + torch.sigmoid(p_kept[:, 0]) * 2.0 - 0.5) * st_k
                cy = (gy_k + torch.sigmoid(p_kept[:, 1]) * 2.0 - 0.5) * st_k
                bw = torch.exp(torch.clamp(p_kept[:, 2], -5.0, 5.0)) * st_k
                bh = torch.exp(torch.clamp(p_kept[:, 3], -5.0, 5.0)) * st_k

                x1 = torch.clamp(cx - 0.5 * bw, 0.0, float(W))
                y1 = torch.clamp(cy - 0.5 * bh, 0.0, float(H))
                x2 = torch.clamp(cx + 0.5 * bw, 0.0, float(W))
                y2 = torch.clamp(cy + 0.5 * bh, 0.0, float(H))
                boxes = torch.stack([x1, y1, x2, y2], dim=1)

                keep_nms = self._batched_nms(boxes, scores_k, cls_k, nms_thresh)
                seq_results.append({
                    "boxes": boxes[keep_nms],
                    "scores": scores_k[keep_nms],
                    "labels": cls_k[keep_nms],
                })
            results.append(seq_results)
        return results

    @staticmethod
    def _batched_nms(
        boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thresh: float
    ) -> torch.Tensor:
        """NMS thuần PyTorch tương thích mọi nền tảng (MPS / CUDA / CPU)."""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        max_coord = boxes.max()
        offsets = labels.to(boxes.dtype) * (max_coord + 1.0)
        boxes_for_nms = boxes + offsets[:, None]

        x1, y1, x2, y2 = boxes_for_nms[:, 0], boxes_for_nms[:, 1], boxes_for_nms[:, 2], boxes_for_nms[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)

        keep = []
        while order.numel() > 0:
            i = int(order[0].item())
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            inter = torch.clamp(xx2 - xx1, min=0.0) * torch.clamp(yy2 - yy1, min=0.0)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
            order = rest[iou <= iou_thresh]

        return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

    @torch.no_grad()
    def estimate_sops_and_energy(self, T: int = 25) -> dict[str, float]:
        """
        Tính toán số phép tính Synaptic Operations (SOPs) và năng lượng tiêu thụ lý thuyết (mJ)
        dựa trên tỷ lệ phát xung thực tế của lần chạy forward gần nhất (chuẩn Horowitz 2014: 0.9pJ/AC vs 4.6pJ/MAC).
        """
        total_ann_macs = 0.0
        total_snn_sops = 0.0
        firing_rates = []

        for m in self.modules():
            if isinstance(m, SpikingConv2d):
                # Ước lượng MACs dựa trên số phần tử đầu ra của lớp LIF và kích thước nhân tích chập
                out_elems = m.lif.last_num_elements  # T * B * C_out * H_out * W_out
                k_h, k_w = m.conv.kernel_size
                c_in = m.conv.in_channels // m.conv.groups
                macs = float(out_elems) * k_h * k_w * c_in
                total_ann_macs += macs
                total_snn_sops += macs * m.last_in_firing_rate
                firing_rates.append(m.lif.last_firing_rate)

            elif isinstance(m, TokenQKAttention):
                out_elems = m.proj_lif.last_num_elements  # T * B * C * H * W
                c = m.dim
                # 3 phép chiếu tuyến tính 1x1 (Q, K, Proj)
                macs = 3.0 * float(out_elems) * c
                total_ann_macs += macs
                total_snn_sops += macs * m.last_in_firing_rate
                firing_rates.append(m.attn_lif.last_firing_rate)

        avg_fr = sum(firing_rates) / max(1, len(firing_rates))
        # Năng lượng lý thuyết trên tiến trình 45nm CMOS: 0.9 pJ cho mỗi phép cộng xung (AC), 4.6 pJ cho mỗi MAC ANN
        energy_snn_mj = (total_snn_sops * 0.9e-12) * 1e3
        energy_ann_mj = (total_ann_macs * 4.6e-12) * 1e3
        saving_ratio = (1.0 - (energy_snn_mj / max(1e-12, energy_ann_mj))) * 100.0

        return {
            "avg_spike_rate_pct": avg_fr * 100.0,
            "ann_gmacs": total_ann_macs / 1e9,
            "snn_gsops": total_snn_sops / 1e9,
            "energy_snn_mj": energy_snn_mj,
            "energy_ann_mj": energy_ann_mj,
            "energy_saving_pct": saving_ratio,
        }


# ==============================================================================
# 5. KIỂM THỬ KIẾN TRÚC VÀ ĐO NĂNG LƯỢNG SOPs
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Đang chạy kiểm thử QKFormerDetector trên thiết bị: {device}")

    model = QKFormerDetector(num_classes=2, embed_dims=(64, 128, 256, 320), fpn_dim=128).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Tổng số tham số mô hình: {total_params:.2f}M")

    # Tạo batch mô phỏng độ thưa 96.5% giống hệt đầu ra từ EvCivilSpikingDataset
    B, T, C, H, W = 2, 25, 2, 260, 346
    dummy_spikes = (torch.rand((B, T, C, H, W), device=device) > 0.965).float()

    # Kiểm tra lượt đi (Forward)
    preds, anchors = model(dummy_spikes)
    print(f"Input shape   : {tuple(dummy_spikes.shape)}")
    print(f"Preds shape   : {tuple(preds.shape)}  (B, T, 1925 anchors, 4 box + 1 obj + 2 cls)")
    print(f"Anchors shape : {tuple(anchors.shape)}")

    # Kiểm tra lượt về (Backward / Surrogate Gradient)
    loss = preds.pow(2).mean()
    loss.backward()
    grad_norm = model.backbone.stem1.conv.weight.grad.norm().item()
    print(f"Kiểm tra Surrogate Gradient tại lớp đầu vào (stem1 grad norm): {grad_norm:.6f}")

    # Thống kê năng lượng tiêu thụ SOPs
    stats = model.estimate_sops_and_energy(T=T)
    print("\n--- THỐNG KÊ TÍNH TOÁN & NĂNG LƯỢNG LÝ THUYẾT (1 CLIP 25 BƯỚC) ---")
    print(f"  + Tỷ lệ phát xung trung bình toàn mạng : {stats['avg_spike_rate_pct']:.2f}%")
    print(f"  + ANN tương đương (GMACs)             : {stats['ann_gmacs']:.2f} GMACs")
    print(f"  + SNN thực tế (GSOPs)                 : {stats['snn_gsops']:.2f} GSOPs")
    print(f"  + Năng lượng ANN vs SNN               : {stats['energy_ann_mj']:.2f} mJ vs {stats['energy_snn_mj']:.2f} mJ")
    print(f"  + Mức tiết kiệm năng lượng lý thuyết  : {stats['energy_saving_pct']:.2f}%")