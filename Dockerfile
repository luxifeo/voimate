# LifeMate submission image: gipformer 1.5 ASR (CPU, sherpa-onnx) -> bge-m3 few-shot retrieval -> Qwen3.5-4B
# tool-calling agent served by vLLM.
#   docker build -t lifemate-submission:combo .
#   docker run --gpus '"device=0"' -v <data_root>:/data -v <out_dir>:/output -e SPLIT=private_test lifemate-submission:combo
# <data_root> holds <split>/ (audio, <split>.jsonl, catalog, environment, zaic_tools); output: /output/submission.jsonl
# Parameters (counted from the checkpoints): Qwen3.5-4B 4.66B (LM 4.21B + vision 0.33B + MTP 0.12B), bge-m3 0.57B,
# gipformer 0.07B; total about 5.3B.
FROM vllm/vllm-openai:v0.32.0

ENV HF_HOME=/opt/hf \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

# sherpa-onnx runs the gipformer transducer; sentence-transformers is installed with --no-deps to keep the image's
# transformers/torch versions, so its other requirements (scipy, scikit-learn) are listed here
RUN pip install --no-cache-dir soundfile sherpa-onnx==2.0.1 scipy==1.18.1 scikit-learn==1.9.1 \
 && pip install --no-cache-dir --no-deps sentence-transformers==6.2.0

# model weights baked in at build time (the container runs offline)
COPY docker/fetch_models.py /app/docker/fetch_models.py
RUN HF_HUB_DISABLE_XET=1 python3 /app/docker/fetch_models.py
ENV HF_HUB_OFFLINE=1

WORKDIR /app
COPY lifemate/ /app/lifemate/
COPY scripts/predict.py /app/scripts/predict.py
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
# labelled train rows for few-shot demonstrations and usage notes, and their precomputed bge-m3 embeddings
COPY data/train+public_test/train/train.jsonl /app/assets/train.jsonl
COPY outputs/index/train_bge-m3_1545.npy /app/assets/index/train_bge-m3_1545.npy

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
