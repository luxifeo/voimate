# LifeMate: gipformer ASR + Qwen3.5-4B agent (ZAC 2026)

Pipeline không huấn luyện lại model:

```
audio (.wav) ──► gipformer 1.5 (68M, sherpa-onnx, CPU) ──► transcript
                                                              │
train.jsonl ──► bge-m3 embedding ──► top-8 ví dụ tương tự (few-shot) ─┤
train.jsonl ──► tần suất điền tham số tùy chọn ──► ghi chú trong schema tool ─┤
                                                              ▼
             Qwen3.5-4B (thinking + tool calling, vLLM) ◄──► zaic_tools ToolEnv
                     │  bảng ngày có nhãn tuần trong system prompt
                     │  env.validate() trước env.call(): call sai schema được trả lỗi cho model, không vào log
                     ▼
             submission.jsonl (tool_calls + state_hash)
```

Cấu hình agent là `c1_combo` trong [claudedocs/llm-hypotheses-2026-10-28.md](claudedocs/llm-hypotheses-2026-10-28.md).

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `lifemate/data.py` | đọc split, catalog, import `zaic_tools` của từng split |
| `lifemate/asr.py` | ASR gipformer (sherpa-onnx) và Gemma qua vLLM, WER/CER |
| `lifemate/retriever.py` | bge-m3 few-shot retrieval trên train |
| `lifemate/agent.py` | system prompt, chuyển schema tool sang format tool calling, vòng lặp agent, các switch thí nghiệm |
| `lifemate/pipeline.py` | chạy cả split, ghi submission, chấm điểm train, ghi chú tham số tùy chọn |
| `lifemate/scorer.py` | metric local theo `evaluation.md` |
| `scripts/serve_vllm.sh` | chạy vLLM (Docker) cho dev |
| `scripts/run_asr.py` | ASR cả split, cache `outputs/asr/<split>_gip15.json` |
| `scripts/run_agent.py` | agent từ transcript cache + chấm điểm (train) |
| `scripts/compare_runs.py` | so sánh các run theo cặp (Δ ± SE, loại lỗi) |
| `scripts/predict.py` | end-to-end một split (dùng trong Docker nộp bài) |
| `Dockerfile`, `docker/entrypoint.sh`, `docker/fetch_models.py` | image nộp bài |
| `docker/vllm.Dockerfile` | image vLLM + thư viện audio cho dev |
| `lab/app.py` | UI xem lại và chỉnh tham số các LLM call đã log |

## Chạy local

```bash
uv sync
unzip "train+public_test.zip" -x "__MACOSX/*" -d data
HF_HUB_DISABLE_XET=1 uv run python scripts/download_models.py   # Qwen3.5-4B, bge-m3, gipformer vào HF cache (xet bị treo trên máy này)

docker build -f docker/vllm.Dockerfile -t lifemate-vllm:0.32.0 docker
MODEL=Qwen/Qwen3.5-4B SERVED_NAME=qwen3.5-4b TOOL_PARSER=qwen3_coder REASONING_PARSER=qwen3 \
  MM_LIMIT='{"image": 0, "video": 0}' CUDA_COMPAT=1 scripts/serve_vllm.sh 0 8011   # CUDA_COMPAT=1 chỉ cần với driver < CUDA 13 (máy này: 535)

export PYTHONIOENCODING=utf-8
uv run python scripts/run_asr.py --split train --engine gipformer
uv run python scripts/run_asr.py --split public_test --engine gipformer
COMBO="--model qwen3.5-4b --thinking --max-tokens 4096 --fewshot 8 --arg-guide --date-table weeks --asr-cache _gip15"
uv run python scripts/run_agent.py --split train --dev 300 $COMBO --name c1_combo_asr          # đánh giá
uv run python scripts/run_agent.py --split public_test $COMBO --name c1_combo_gip15            # submission
```

Kết quả nằm ở `outputs/runs/<split>/<name>/` (`submission.jsonl`, `traces.jsonl`, `metrics.json`, `scores.jsonl`, `llm_calls.jsonl`).

Kiểm tra schema file nộp bằng tool của BTC:

```bash
cd data/train+public_test
PYTHONPATH=public_test python3 -m zaic_tools --split public_test --root . --validate ../../outputs/runs/public_test/c1_combo_gip15/submission.jsonl
```

## Docker nộp bài

Hướng dẫn đóng gói chính thức của BTC chưa có; đường dẫn mount và biến môi trường dưới đây là giả định.
Weights (Qwen3.5-4B, bge-m3, gipformer 1.5) được tải và pin revision lúc build; container chạy offline.

```bash
docker build -t lifemate-submission:combo .
docker run --rm --gpus '"device=0"' \
  -v $PWD/data/train+public_test:/data -v $PWD/outputs/docker:/output \
  -e SPLIT=public_test lifemate-submission:combo
```

Trên máy có driver cũ hơn CUDA 13 (như máy dev, driver 535), thêm lúc chạy, không đưa vào Dockerfile:
`-e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib`.
