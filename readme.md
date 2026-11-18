
Bước 1 — Chạy kiểm tra thông luồng nhanh (2 epoch × 5 batch, mất ~15 giây):

```bash
python src/train.py --sanity-check --num-workers 0
```
Bước 2 — Chạy huấn luyện đầy đủ trên toàn bộ tập ev-CIVIL:
```bash
python src/train.py --epochs 30 --batch-size 4 --grad-accum 2 --num-workers 2
```