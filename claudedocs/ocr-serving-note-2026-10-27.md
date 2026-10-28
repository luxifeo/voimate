# Note: OCR Triton serving (`ocr-serving-1`) – 2026-10-27

## 1. GPU usage trước khi tắt

| PID | Container | GPU | VRAM |
|---|---|---|---|
| 3694082 | `ocr-serving-1` (compose `/home/hieunm/workspace/ocr`) | 0 | 19354 MiB |
| 3724922 | `vbpl_deploy-serving-1` | 0 | 3504 MiB |
| 3097121 | `tritonserver-reranker-h01` | 1 | 5826 MiB |
| 2590464 | `tritonserver-reranker-btc` | 1 | 5826 MiB |
| 2606389 | `mass-law-embed-btc` | 1 | 1580 MiB |

## 2. `ocr-serving-1`

- Image: `nvcr.io/nvidia/tritonserver:25.02-py3`
- Command: `tritonserver --model-store=/models`
- Mount: `/home/hieunm/workspace/ocr/models` → `/models`
- Restart policy: `unless-stopped`

### Models (10, ONNX Runtime, KIND_GPU)

| Model | Size | Instances | Input |
|---|---|---|---|
| detector-dbtext | 25M | 4 | `[-1,3,-1,-1]` |
| reader-parseq-vit-small-encoder | 56K | 4 | `[-1,-1,48]` |
| reader-parseq-vit-small-decoder | 28M | 4 | `[-1,-1,192]` |
| doclayout-imgsz1024 | 72M | 2 | batch 8, `[3,-1,-1]` |
| segmentation-table-line | 55M | 2 | `[-1,3,-1,-1]` |
| segmentation-table-space | 13M | 2 | `[-1,3,-1,-1]` |
| segmentation-table-span | – | 2 | `[-1,5,-1,-1]` |
| cropper-deeplabv3 | 152M | 2 | `[-1,3,384,384]` |
| classify-orientation-word | 43M | 2 | `[-1,3,256,256]` |
| classify-orientation-paddle | 6.5M | 2 | `[-1,3,224,224]` |

Tổng model trên đĩa ~400MB, nhưng process giữ ~19GB VRAM. Tổng 26 instance.

### Nguyên nhân VRAM cao

Chưa đo được VRAM theo từng model (Triton không có metric per-model).

[Inference] Nguyên nhân khả dĩ:
1. 26 instance, mỗi instance có CUDA context + cuDNN workspace + ORT memory arena riêng.
2. ORT BFC arena chỉ tăng, không tự trả bộ nhớ; model có H×W dynamic giữ peak memory của ảnh lớn nhất.
3. Nghi ngờ chính: `detector-dbtext`, `doclayout-imgsz1024`, `segmentation-table-*`.

## 3. Uptime & request (tính đến 2026-10-27 18:01 +07)

- Start: 2026-10-26 15:02 (+07), uptime ~27h, restart = 0.

### API (`ocr-api-1`, port 3001)

6 request `POST /convert`, tất cả 200:

| Time (log) | Source | Duration |
|---|---|---|
| 26/10 08:05 | 118.69.248.83 | 53.1s |
| 26/10 08:05 | 118.69.248.83 | 53.1s |
| 26/10 08:20 | 118.69.248.83 | 178.0s |
| 26/10 08:21 | 118.69.248.83 | 177.2s |
| 27/10 08:02 | 10.99.28.1 | 147.6s |
| 27/10 08:07 | 10.99.28.1 | 121.1s |

### Triton

Tổng 1712 inference request, 0 failure.

| Model | Requests | Avg latency |
|---|---|---|
| reader-parseq-vit-small-decoder | 544 | ~148 ms |
| reader-parseq-vit-small-encoder | 544 | ~17 ms |
| detector-dbtext | 222 | ~50 ms |
| doclayout-imgsz1024 | 88 | ~80 ms |
| cropper-deeplabv3 | 88 | ~13 ms |
| classify-orientation-word | 88 | ~3 ms |
| segmentation-table-line/space/span | 46 mỗi model | ~15 ms |
| classify-orientation-paddle | 0 | – |

[Inference] ~88 trang ảnh cho 6 job (cropper/classify/doclayout cùng 88 request).

## 4. Vấn đề bảo mật

Port 3001 (`ocr-api-1`) bind `0.0.0.0`, bị scan từ IP public (`/mcp`, `/sse`, `/api`, `/_next`, `/jsonrpc`...). `/convert` và `/docs` truy cập được từ ngoài.
→ Nên giới hạn mạng nội bộ hoặc đặt sau gateway có auth.

## 5. Action đã làm

- 2026-10-27 ~18:0x (+07): `docker stop ocr-serving-1` → `Exited (0)`.
- GPU 0: ~22.8GB → 3.5GB.
- `ocr-api-1` **vẫn đang chạy** → `/convert` sẽ lỗi do không kết nối được Triton.

### Bật lại

```bash
docker start ocr-serving-1
# hoặc
cd /home/hieunm/workspace/ocr && docker compose up -d serving
```

## 6. Đề xuất khi bật lại

[Unverified] Tham số ORT backend (cần kiểm tra lại docs Triton 25.02):

```protobuf
parameters { key: "arena_extend_strategy" value: { string_value: "1" } }
parameters { key: "gpu_mem_limit"         value: { string_value: "2147483648" } }
parameters { key: "memory.enable_memory_arena_shrinkage" value: { string_value: "gpu:0" } }
parameters { key: "cudnn_conv_algo_search" value: { string_value: "1" } }
```

- Giảm `instance_group.count` 4 → 1–2 cho `detector-dbtext`, `reader-parseq-*`.
- Cân nhắc bỏ `classify-orientation-paddle` (0 request).
- Bottleneck latency: `reader-parseq-vit-small-decoder`.
