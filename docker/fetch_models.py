"""Bake the model weights into the image at build time, pinned to the revisions used in development.
refs/main is pointed at the pinned commit so offline lookups by repo id alone (vLLM, sentence-transformers,
lifemate.asr.TransducerASR) resolve to it."""
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = [
    ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", None),
    ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181",
     ["*.json", "pytorch_model.bin", "sentencepiece.bpe.model", "1_Pooling/*"]),
    ("g-group-ai-lab/gipformer1.5-68M-rnnt", "dd9227dcd8705c13f33bdbe59728d546ab94480f",
     ["*.onnx", "tokens.txt", "config.json"]),
]

for repo, rev, patterns in MODELS:
    snapshot = Path(snapshot_download(repo, revision=rev, allow_patterns=patterns))
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(rev)
    print(repo, rev, "->", snapshot)
