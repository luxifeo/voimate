# LifeMate baseline: Gemma-4-E2B (ZAC 2026)

Pipeline không huấn luyện lại model:

```
audio (.wav) ──► Gemma-4-E2B-it (ASR, prompt có ngữ cảnh app) ──► transcript
                                                                    │
train.jsonl ──► bge-m3 embedding ──► top-4 ví dụ tương tự (few-shot) ─┤
                                                                    ▼
             Gemma-4-E2B-it (thinking + native tool calling, vLLM) ◄──► zaic_tools ToolEnv
                     │  env.validate() trước env.call(): call sai schema được trả lỗi cho model, không vào log
                     ▼
             submission.jsonl (tool_calls + state_hash)
```

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `lifemate/data.py` | đọc split, catalog, import `zaic_tools` của từng split |
| `lifemate/asr.py` | ASR qua vLLM (audio base64), WER/CER |
| `lifemate/retriever.py` | bge-m3 few-shot retrieval trên train |
| `lifemate/agent.py` | system prompt, chuyển schema tool sang format Gemma, vòng lặp agent |
| `lifemate/pipeline.py` | chạy cả split, ghi submission, chấm điểm train |
| `lifemate/scorer.py` | metric local theo `evaluation.md` |
| `scripts/serve_vllm.sh` | chạy vLLM (Docker) cho dev |
| `scripts/run_asr.py` | ASR cả split, cache `outputs/asr/<split>_ctx.json` |
| `scripts/run_agent.py` | agent từ transcript cache + chấm điểm (train) |
| `scripts/predict.py` | end-to-end một split (dùng trong Docker nộp bài) |
| `Dockerfile`, `docker/entrypoint.sh` | image nộp bài (bản nháp) |
| `docker/vllm.Dockerfile` | image vLLM + thư viện audio cho dev |

## Chạy local

```bash
uv sync
unzip "train+public_test.zip" -x "__MACOSX/*" -d data
HF_HUB_DISABLE_XET=1 uv run python scripts/download_models.py   # gemma-4-E2B-it + bge-m3 vào HF cache (xet bị treo trên máy này)

docker build -f docker/vllm.Dockerfile -t lifemate-vllm:0.32.0 docker
CUDA_COMPAT=1 scripts/serve_vllm.sh 0 8011        # CUDA_COMPAT=1 chỉ cần với driver < CUDA 13 (máy này: 535)

export PYTHONIOENCODING=utf-8
uv run python scripts/run_asr.py --split train
uv run python scripts/run_asr.py --split public_test
uv run python scripts/run_agent.py --split train --dev 300 --thinking --name v3          # đánh giá
uv run python scripts/run_agent.py --split public_test --thinking --name v3              # submission
```

Kết quả nằm ở `outputs/runs/<split>/<name>/` (`submission.jsonl`, `traces.jsonl`, `metrics.json`, `scores.jsonl`).

Kiểm tra schema file nộp bằng tool của BTC:

```bash
cd data/train+public_test
PYTHONPATH=public_test python3 -m zaic_tools --split public_test --root . --validate ../../outputs/runs/public_test/v3/submission.jsonl
```

## Docker nộp bài (bản nháp)

Hướng dẫn đóng gói chính thức của BTC chưa có; đường dẫn mount và biến môi trường dưới đây là giả định.

```bash
docker build -t lifemate-submission:baseline .
docker run --rm --gpus '"device=0"' \
  -v $PWD/data/train+public_test:/data -v $PWD/outputs/docker:/output \
  -e SPLIT=public_test lifemate-submission:baseline
```

Trên máy có driver cũ hơn CUDA 13 (như máy dev, driver 535), thêm lúc chạy, không đưa vào Dockerfile:
`-e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib`.
