"""Download the models used by the baseline into the local HF cache."""
from huggingface_hub import snapshot_download

print(snapshot_download("google/gemma-4-E2B-it"), flush=True)
print(snapshot_download("BAAI/bge-m3", allow_patterns=["*.json", "pytorch_model.bin", "sentencepiece.bpe.model", "1_Pooling/*"]), flush=True)
