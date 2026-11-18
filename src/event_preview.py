
import h5py
import numpy as np
import matplotlib.pyplot as plt

H, W = 260, 346  # Độ phân giải chuẩn của bộ dữ liệu ev-CIVIL

with h5py.File("/Users/trantung/Downloads/Field_dataset/Night_outdoor/night_outdoor_31st_dec/31_crack_40/31_crack_40_events.h5", "r") as f:
    print("Các keys có trong file:", list(f.keys()))
    
    x = f["x"][:]
    y = f["y"][:]
    
    # Tự động tìm key thời gian (thường là 'ts', 'timestamp' hoặc 't')
    t_key = next(k for k in ["ts", "timestamp", "timestamps",'event_timestamp', "t"] if k in f.keys())
    # Tự động tìm key cực tính (thường là 'p' hoặc 'pol')
    p_key = next(k for k in ["p", "pol", "polarity"] if k in f.keys())
    
    t = f[t_key][:]
    p = f[p_key][:]
    print(f"Đang sử dụng: t_key='{t_key}' (min={t[0]}, max={t[-1]}), p_key='{p_key}' (values={np.unique(p)})")

# Chọn một lát cắt thời gian delta_t = 41,666 us (41.666 ms) từ thời điểm bắt đầu
# Lưu ý: Nếu t lưu theo giây (float) thay vì micro-giây (int), đổi 41666 thành 0.041666
dt = 0.041666 if np.issubdtype(t.dtype, np.floating) else 41666
t_start = t[0]
t_end = t_start + dt
mask = (t >= t_start) & (t < t_end)

x_sub, y_sub, p_sub = x[mask], y[mask], p[mask]

# Khởi tạo khung ảnh nền đen (H, W, 3)
img = np.zeros((H, W, 3), dtype=np.uint8)

# Tô màu đỏ cho sự kiện ON (p > 0) và xanh dương cho sự kiện OFF (p <= 0)
on_mask = p_sub > 0
off_mask = ~on_mask

img[y_sub[on_mask], x_sub[on_mask]] = [255, 50, 50]    # Đỏ (ON)
img[y_sub[off_mask], x_sub[off_mask]] = [50, 120, 255] # Xanh (OFF)

plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.title(f"Event Frame ({len(x_sub)} events)")
plt.axis("off")
plt.savefig("event_preview.png", bbox_inches="tight")
plt.show()