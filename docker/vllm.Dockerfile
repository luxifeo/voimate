# vLLM server for Gemma-4-E2B-it with audio input (adds the vllm[audio] extras the base image does not ship).
FROM vllm/vllm-openai:v0.32.0
RUN pip install --no-cache-dir av scipy soundfile soxr "mistral_common[audio]" librosa
