
Bước 1 — Chạy kiểm tra thông luồng nhanh (2 epoch × 5 batch, mất ~15 giây):

```bash
python src/train.py --sanity-check --num-workers 0
```
Bước 2 — Tiền xử lý và cache tensor spike trước khi huấn luyện:

```bash
python src/train.py --preprocess-only --params-json params.json --cache-dir .cache/spikes
```

Bước 3 — Chạy huấn luyện đầy đủ trên toàn bộ tập ev-CIVIL:
```bash
python src/train.py --epochs 30 --batch-size 4 --grad-accum 2 --num-workers 4 --params-json params.json --cache-dir .cache/spikes
```

Có thể tiếp tục huấn luyện từ checkpoint (giá trị `--epochs` là tổng số epoch cần đạt):
```bash
python src/train.py --resume checkpoints/best_model.pt --epochs 50 --batch-size 4 --grad-accum 2 --num-workers 2
```