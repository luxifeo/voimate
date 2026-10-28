"""Download the models used by the pipeline into the local HF cache (same revisions as docker/fetch_models.py)."""
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"), flush=True)
print(snapshot_download("BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181",
                        allow_patterns=["*.json", "pytorch_model.bin", "sentencepiece.bpe.model", "1_Pooling/*"]), flush=True)
print(snapshot_download("g-group-ai-lab/gipformer1.5-68M-rnnt", revision="dd9227dcd8705c13f33bdbe59728d546ab94480f",
                        allow_patterns=["*.onnx", "tokens.txt", "config.json"]), flush=True)
