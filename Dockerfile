# LifeMate submission image (draft): vLLM serving Gemma-4-E2B-it (ASR + tool-calling agent) and bge-m3 few-shot retrieval.
#   docker build -t lifemate-submission .
#   docker run --gpus '"device=0"' -v <data_root>:/data -v <out_dir>:/output -e SPLIT=private_test lifemate-submission
# <data_root> holds <split>/ (audio, <split>.jsonl, catalog, environment, zaic_tools); output: /output/submission.jsonl
FROM vllm/vllm-openai:v0.32.0

ENV HF_HOME=/opt/hf \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

# vllm[audio] extras (not shipped in the base image) and sentence-transformers; --no-deps keeps the image's
# transformers/torch versions (they already satisfy sentence-transformers' requirements)
RUN pip install --no-cache-dir av scipy soundfile soxr "mistral_common[audio]" librosa \
 && pip install --no-cache-dir --no-deps sentence-transformers==6.2.0

# model weights baked in at build time (the container runs offline)
RUN HF_HUB_DISABLE_XET=1 python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('google/gemma-4-E2B-it'); \
snapshot_download('BAAI/bge-m3', allow_patterns=['*.json', 'pytorch_model.bin', 'sentencepiece.bpe.model', '1_Pooling/*'])"
ENV HF_HUB_OFFLINE=1

WORKDIR /app
COPY lifemate/ /app/lifemate/
COPY scripts/predict.py /app/scripts/predict.py
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
# labelled train rows for few-shot demonstrations and their precomputed bge-m3 embeddings
COPY data/train+public_test/train/train.jsonl /app/assets/train.jsonl
COPY outputs/index/train_bge-m3_1545.npy /app/assets/index/train_bge-m3_1545.npy

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
