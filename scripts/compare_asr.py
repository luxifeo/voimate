"""Compare cached Gemma ASR transcripts with the ground-truth audio_text of a labelled split.

    uv run python scripts/compare_asr.py                      # train, cache outputs/asr/train_ctx.json
    uv run python scripts/compare_asr.py --cache train.json   # base prompt cache

Prints corpus WER/CER (final turn vs history turns), the per-clip WER distribution, the most frequent word
errors and examples; writes every clip to outputs/asr/<cache stem>_compare.tsv.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.asr import _words, error_rates  # noqa: E402
from lifemate.data import OUTPUT_ROOT, load_split  # noqa: E402


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    """Levenshtein alignment as (ref word, hyp word) pairs; "" marks an insertion or deletion."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((ref[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", hyp[j - 1]))
            j -= 1
    return pairs[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="train_ctx.json", help="file name under outputs/asr/")
    ap.add_argument("--examples", type=int, default=4, help="examples per WER bucket")
    ap.add_argument("--top", type=int, default=25, help="most frequent word errors to print")
    args = ap.parse_args()

    cache_path = OUTPUT_ROOT / "asr" / args.cache
    asr = json.loads(cache_path.read_text(encoding="utf-8"))
    clips = []  # (id, turn, gold, hyp)
    for r in load_split(args.split):
        turns = [t for t in r.get("history", []) if t.get("role") == "user" and t.get("audio")]
        for k, t in enumerate(turns, 1):
            clips.append((r["id"], f"u{k}", t["audio_text"], asr.get(t["audio"])))
        clips.append((r["id"], "final", r["audio_text"], asr.get(r["audio"])))
    missing = sum(c[3] is None for c in clips)
    clips = [c for c in clips if c[3] is not None]

    rows = []
    for sid, turn, gold, hyp in clips:
        e = error_rates([(gold, hyp)])
        rows.append({"id": sid, "turn": turn, "wer": e["wer"], "cer": e["cer"], "gold": gold, "asr": hyp})

    print(f"cache {cache_path.name}: {len(rows)} clips ({missing} without transcript)\n")
    print(f"{'turn':<10}{'clips':>7}{'WER':>9}{'CER':>9}")
    for name, sel in (("final", [r for r in rows if r["turn"] == "final"]),
                      ("history", [r for r in rows if r["turn"] != "final"]), ("all", rows)):
        e = error_rates([(r["gold"], r["asr"]) for r in sel])
        print(f"{name:<10}{len(sel):>7}{e['wer']:>8.1%}{e['cer']:>9.1%}")

    buckets = [(0, 0, "WER = 0"), (0, 0.2, "0-20%"), (0.2, 0.4, "20-40%"), (0.4, 0.6, "40-60%"),
               (0.6, 1.0, "60-100%"), (1.0, 99, ">100%")]

    def bucket(w: float) -> str:
        if w == 0:
            return buckets[0][2]
        return next(name for lo, hi, name in buckets[1:] if lo < w <= hi)

    final = [r for r in rows if r["turn"] == "final"]
    dist = Counter(bucket(r["wer"]) for r in final)
    print("\nPer-clip WER, final turn:")
    for _, _, name in buckets:
        print(f"  {name:<9}{dist[name]:>6}  {dist[name] / len(final):6.1%}")

    sub, dele, ins = Counter(), Counter(), Counter()
    for r in rows:
        for g, h in align(_words(r["gold"]), _words(r["asr"])):
            if g and h and g != h:
                sub[(g, h)] += 1
            elif g and not h:
                dele[g] += 1
            elif h and not g:
                ins[h] += 1
    print(f"\nWord errors (all clips): {sum(sub.values())} substitutions, {sum(dele.values())} deletions, "
          f"{sum(ins.values())} insertions")
    print(f"Top {args.top} substitutions (gold -> asr):")
    for (g, h), c in sub.most_common(args.top):
        print(f"  {c:>4}  {g} -> {h}")
    print("Top deletions:  " + ", ".join(f"{w} ({c})" for w, c in dele.most_common(12)))
    print("Top insertions: " + ", ".join(f"{w} ({c})" for w, c in ins.most_common(12)))

    rng = random.Random(0)
    print("\nExamples, final turn (random per bucket):")
    for _, _, name in buckets:
        pool = [r for r in final if bucket(r["wer"]) == name]
        for r in rng.sample(pool, min(args.examples, len(pool))):
            print(f"\n  [{r['id']}] WER {r['wer']:.0%}  CER {r['cer']:.0%}")
            print(f"    gold: {r['gold']}")
            print(f"    asr : {r['asr']}")

    out = cache_path.with_name(cache_path.stem + "_compare.tsv")
    with out.open("w", encoding="utf-8") as f:
        f.write("id\tturn\twer\tcer\tgold\tasr\n")
        for r in rows:
            f.write(f"{r['id']}\t{r['turn']}\t{r['wer']:.3f}\t{r['cer']:.3f}\t"
                    f"{r['gold'].replace(chr(9), ' ')}\t{r['asr'].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
