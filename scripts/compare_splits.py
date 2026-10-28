"""Compare the input distribution of train and public_test (no labels, no <split>_environment/ files).

    uv run python scripts/compare_splits.py                        # report + outputs/analysis/split_shift.json
    uv run python scripts/compare_splits.py --no-embed             # skip the bge-m3 transcript analysis
    uv run python scripts/compare_splits.py --pred-train v3_full_asrctx --pred-test v3

Sections: tool catalog, installed apps, history, context, final-turn audio, ASR transcripts (cached Gemma ASR,
same prompt on both splits), adversarial validation (train vs test classifier, AUC ~0.5 means no shift) and the
baseline's own predictions on both splits. Train gold labels are reported only as a reference column; they are
never compared with public_test predictions as if they were the same quantity.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.asr import _words  # noqa: E402
from lifemate.data import DATA_ROOT, OUTPUT_ROOT, load_split, user_audio_paths  # noqa: E402

SPLITS = ("train", "public_test")


def jsd(a: Counter, b: Counter) -> float:
    """Jensen-Shannon divergence (base 2, 0 = identical, 1 = disjoint) of two count distributions."""
    keys = sorted(set(a) | set(b))
    p = np.array([a.get(k, 0) for k in keys], float)
    q = np.array([b.get(k, 0) for k in keys], float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))
    return (kl(p, m) + kl(q, m)) / 2


def chi2_p(a: Counter, b: Counter, min_count: int = 5) -> float:
    """Chi-square test of homogeneity; categories rarer than min_count in total are pooled into one bucket."""
    keys = sorted(set(a) | set(b))
    rare = [k for k in keys if a.get(k, 0) + b.get(k, 0) < min_count]
    keep = [k for k in keys if k not in rare]
    rows = [[a.get(k, 0) for k in keep], [b.get(k, 0) for k in keep]]
    if rare:
        rows[0].append(sum(a.get(k, 0) for k in rare))
        rows[1].append(sum(b.get(k, 0) for k in rare))
    table = np.array(rows)
    table = table[:, table.sum(0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(table)[1])


def numeric(name: str, x: list[float], y: list[float]) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ks = stats.ks_2samp(x, y)
    return {
        "feature": name,
        "train_mean": round(float(x.mean()), 3), "test_mean": round(float(y.mean()), 3),
        "train_q": [round(float(v), 2) for v in np.quantile(x, [0.1, 0.5, 0.9])],
        "test_q": [round(float(v), 2) for v in np.quantile(y, [0.1, 0.5, 0.9])],
        "ks": round(float(ks.statistic), 3), "ks_p": float(ks.pvalue),
    }


def categorical(name: str, a: Counter, b: Counter, top: int = 12) -> dict:
    na, nb = sum(a.values()), sum(b.values())
    keys = sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0) / na + b.get(k, 0) / nb))
    return {
        "feature": name, "jsd": round(jsd(a, b), 4), "chi2_p": chi2_p(a, b),
        "rows": [(str(k), round(a.get(k, 0) / na, 3), round(b.get(k, 0) / nb, 3)) for k in keys[:top]],
    }


# ---------------------------------------------------------------- sections

def catalog_section(root: Path) -> dict:
    cats = {}
    for split in SPLITS:
        with open(root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
            cats[split] = json.load(f)
    tools = {s: {t["name"]: t for a in c["apps"] for t in a["tools"]} for s, c in cats.items()}
    apps = {s: {a["id"] for a in c["apps"]} for s, c in cats.items()}
    shared = sorted(set(tools["train"]) & set(tools["public_test"]))
    changed = [n for n in shared if tools["train"][n]["parameters"] != tools["public_test"][n]["parameters"]
               or tools["train"][n]["description"] != tools["public_test"][n]["description"]]
    test_only_apps = sorted(apps["public_test"] - apps["train"])
    return {
        "schema_version": {s: c.get("schema_version") for s, c in cats.items()},
        "n_apps": {s: len(v) for s, v in apps.items()},
        "n_tools": {s: len(v) for s, v in tools.items()},
        "shared_tools": len(shared),
        "shared_tools_changed": changed,
        "train_only_apps": sorted(apps["train"] - apps["public_test"]),
        "test_only_apps": test_only_apps,
        "test_only_tools": sorted(n for n in tools["public_test"] if n.split(".")[0] in test_only_apps),
        "app_group": {a["id"]: a.get("group") for c in cats.values() for a in c["apps"]},
    }


def installed_section(data: dict, cat: dict) -> tuple[list[dict], dict]:
    test_only = set(cat["test_only_apps"])
    train_only = set(cat["train_only_apps"])
    apps = {s: Counter(a for d in rows for a in d["context"]["installed_apps"]) for s, rows in data.items()}
    groups = {s: Counter(cat["app_group"].get(a) for d in rows for a in d["context"]["installed_apps"])
              for s, rows in data.items()}
    n = {s: len(rows) for s, rows in data.items()}
    out = [
        numeric("n_installed_apps", *[[len(d["context"]["installed_apps"]) for d in data[s]] for s in SPLITS]),
        numeric("n_installed_tools", *[[sum(len(a["tools"]) for a in d["context"]["installed_apps"].values())
                                        for d in data[s]] for s in SPLITS]),
        categorical("installed_app_group", groups["train"], groups["public_test"]),
    ]
    per_app = sorted(
        ((a, apps["train"].get(a, 0) / n["train"], apps["public_test"].get(a, 0) / n["public_test"])
         for a in set(apps["train"]) | set(apps["public_test"])),
        key=lambda r: -abs(r[1] - r[2]))
    extra = {
        "share_with_test_only_app": round(np.mean([bool(test_only & set(d["context"]["installed_apps"]))
                                                   for d in data["public_test"]]), 3),
        "share_with_train_only_app": round(np.mean([bool(train_only & set(d["context"]["installed_apps"]))
                                                    for d in data["train"]]), 3),
        "per_app_rate_largest_gap": [(a, round(x, 3), round(y, 3)) for a, x, y in per_app[:20]],
        "per_app_rate_shared_jsd": round(jsd(
            Counter({a: c for a, c in apps["train"].items() if a not in train_only}),
            Counter({a: c for a, c in apps["public_test"].items() if a not in test_only})), 4),
    }
    return out, extra


def history_section(data: dict, cat: dict) -> tuple[list[dict], dict]:
    def hist_tools(d):
        return [c["name"] for t in d["history"] if t.get("role") == "assistant" for c in (t.get("tool_calls") or [])]

    out = [
        numeric("n_history_turns", *[[len(d["history"]) for d in data[s]] for s in SPLITS]),
        categorical("has_history", *[Counter(bool(d["history"]) for d in data[s]) for s in SPLITS]),
        categorical("n_user_turns_total", *[Counter(len(user_audio_paths(d)) for d in data[s]) for s in SPLITS]),
        categorical("history_tool_app", *[Counter(n.split(".")[0] for d in data[s] for n in hist_tools(d))
                                          for s in SPLITS]),
        categorical("history_ask_missing_info",
                    *[Counter(any(n == "sys.ask_missing_info" for n in hist_tools(d)) for d in data[s] if d["history"])
                      for s in SPLITS]),
    ]
    return out, {}


def context_section(data: dict) -> tuple[list[dict], dict]:
    def times(s):
        return [datetime.fromisoformat(d["context"]["current_time"]) for d in data[s]]

    t = {s: times(s) for s in SPLITS}
    out = [
        categorical("city", *[Counter(d["context"]["user_profile"].get("city") for d in data[s]) for s in SPLITS]),
        categorical("timezone", *[Counter(d["context"]["timezone"] for d in data[s]) for s in SPLITS]),
        categorical("profile_keys", *[Counter(",".join(sorted(d["context"]["user_profile"])) for d in data[s])
                                      for s in SPLITS]),
        categorical("hour_of_day", *[Counter(x.hour for x in t[s]) for s in SPLITS], top=24),
        categorical("weekday", *[Counter(x.strftime("%a") for x in t[s]) for s in SPLITS], top=7),
        categorical("month", *[Counter(x.strftime("%Y-%m") for x in t[s]) for s in SPLITS]),
    ]
    extra = {"date_range": {s: [min(t[s]).isoformat(), max(t[s]).isoformat()] for s in SPLITS},
             "n_unique_user_names": {s: len({d["context"]["user_profile"].get("name") for d in data[s]})
                                     for s in SPLITS}}
    return out, extra


def audio_section(data: dict) -> tuple[list[dict], dict]:
    with open(OUTPUT_ROOT / "asr" / "audio_stats.json", encoding="utf-8") as f:
        audio = json.load(f)
    feats = {s: [audio.get(f"{s}/{d['audio']}") for d in data[s]] for s in SPLITS}
    missing = {s: sum(v is None for v in feats[s]) for s in SPLITS}
    out = [numeric(f"final_audio_{k}", *[[v[k] for v in feats[s] if v] for s in SPLITS])
           for k in ("dur", "snr", "noise_db", "peak", "clip_frac")]
    return out, {"missing_stats": missing}


def transcripts(data: dict, cache: str) -> dict[str, list[str]]:
    out = {}
    for s in SPLITS:
        with open(OUTPUT_ROOT / "asr" / f"{s}{cache}.json", encoding="utf-8") as f:
            asr = json.load(f)
        out[s] = [asr.get(d["audio"], "") for d in data[s]]
    return out


def transcript_section(data: dict, texts: dict) -> tuple[list[dict], dict]:
    words = {s: [_words(t) for t in texts[s]] for s in SPLITS}
    vocab = {s: Counter(w for ws in words[s] for w in ws) for s in SPLITS}
    oov = [w for ws in words["public_test"] for w in ws if w not in vocab["train"]]
    tokens_test = sum(vocab["public_test"].values())
    train_tokens = sum(vocab["train"].values())
    # words over-represented in test relative to train (add-one smoothed log ratio, min 5 occurrences in test)
    lift = sorted(((w, c, vocab["train"].get(w, 0),
                    np.log((c + 1) / tokens_test) - np.log((vocab["train"].get(w, 0) + 1) / train_tokens))
                   for w, c in vocab["public_test"].items() if c >= 5), key=lambda r: -r[3])
    out = [numeric("final_asr_words", *[[len(ws) for ws in words[s]] for s in SPLITS])]
    extra = {
        "empty_transcripts": {s: sum(not t.strip() for t in texts[s]) for s in SPLITS},
        "test_oov_token_rate": round(len(oov) / max(tokens_test, 1), 4),
        "test_oov_type_rate": round(sum(w not in vocab["train"] for w in vocab["public_test"])
                                    / max(len(vocab["public_test"]), 1), 4),
        "test_overrepresented_words": [(w, c, ct) for w, c, ct, _ in lift[:30]],
    }
    return out, extra


def tabular_features(d: dict, audio: dict, split: str, all_apps: list[str]) -> list[float]:
    t = datetime.fromisoformat(d["context"]["current_time"])
    a = audio.get(f"{split}/{d['audio']}") or {}
    installed = set(d["context"]["installed_apps"])
    return [len(installed), len(d["history"]), t.hour, t.weekday(),
            a.get("dur", 0), a.get("snr", 0), a.get("noise_db", 0), a.get("peak", 0)] + \
        [float(x in installed) for x in all_apps]


def adversarial_auc(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float, np.ndarray]:
    """5-fold CV ROC-AUC of a train-vs-test classifier; also returns out-of-fold P(test) per row."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"))
        clf.fit(x[tr], y[tr])
        oof[te] = clf.predict_proba(x[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))
    return float(np.mean(aucs)), float(np.std(aucs)), oof


def embedding_section(data: dict, texts: dict, cat: dict, device: str) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3", device=device)
    model.max_seq_length = 256
    emb = {s: model.encode(texts[s], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
           for s in SPLITS}
    # nearest train neighbour: test -> train vs train -> other train (leave-one-out)
    sim_tt = emb["train"] @ emb["train"].T
    np.fill_diagonal(sim_tt, -1)
    nn_train = sim_tt.max(1)
    nn_test = (emb["public_test"] @ emb["train"].T).max(1)
    with open(OUTPUT_ROOT / "asr" / "audio_stats.json", encoding="utf-8") as f:
        audio = json.load(f)
    all_apps = sorted(set(cat["app_group"]) - {"sys"})
    tab = np.array([tabular_features(d, audio, s, all_apps) for s in SPLITS for d in data[s]])
    shared_apps = [i for i, a in enumerate(all_apps)
                   if a not in set(cat["test_only_apps"]) | set(cat["train_only_apps"])]
    tab_shared = tab[:, list(range(8)) + [8 + i for i in shared_apps]]
    x_text = np.vstack([emb["train"], emb["public_test"]])
    y = np.array([0] * len(data["train"]) + [1] * len(data["public_test"]))
    auc = {}
    for name, x in (("text_bge_m3", x_text), ("tabular_all_apps", tab), ("tabular_shared_apps_only", tab_shared),
                    ("context_and_audio_no_apps", tab[:, :8])):
        m, sd, oof = adversarial_auc(x, y)
        auc[name] = {"auc": round(m, 3), "std": round(sd, 3)}
        if name == "text_bge_m3":
            oof_text = oof
    test_ids = [d["id"] for d in data["public_test"]]
    p_test = oof_text[len(data["train"]):]
    order = np.argsort(-p_test)
    return {
        "nn_cosine_train_loo_vs_test": numeric("nn_cos", nn_train, nn_test),
        "nn_share_below": {f"<{th}": {"train": round(float((nn_train < th).mean()), 3),
                                      "public_test": round(float((nn_test < th).mean()), 3)}
                           for th in (0.6, 0.7, 0.8)},
        "adversarial_auc": auc,
        "most_test_like_transcripts": [(test_ids[i], round(float(p_test[i]), 3), texts["public_test"][i])
                                       for i in order[:25]],
        "least_covered_test_transcripts": [(test_ids[i], round(float(nn_test[i]), 3), texts["public_test"][i])
                                           for i in np.argsort(nn_test)[:15]],
    }


def predictions_section(data: dict, runs: dict, cat: dict) -> tuple[list[dict], dict]:
    """Baseline predictions on both splits (same agent, ASR input). Proxy for request mix, not ground truth."""
    preds = {}
    for s in SPLITS:
        path = OUTPUT_ROOT / "runs" / s / runs[s] / "submission.jsonl"
        with open(path, encoding="utf-8") as f:
            preds[s] = {r["id"]: r["tool_calls"] for r in map(json.loads, f)}

    def kind(calls):
        names = [c["name"] for c in calls]
        if not names:
            return "empty"
        if names[-1] == "sys.ask_missing_info":
            return "ask_missing_info"
        return "action"

    def write_apps(calls):
        return {c["name"].split(".")[0] for c in calls} - {"sys"}

    p = {s: [preds[s].get(d["id"], []) for d in data[s]] for s in SPLITS}
    test_only = set(cat["test_only_apps"])
    out = [
        categorical("pred_kind", *[Counter(kind(c) for c in p[s]) for s in SPLITS]),
        numeric("pred_n_calls", *[[len(c) for c in p[s]] for s in SPLITS]),
        categorical("pred_app", *[Counter(a for c in p[s] for a in write_apps(c)) for s in SPLITS], top=25),
    ]
    gold = Counter(kind(d["label"]["tool_calls"]) for d in data["train"])
    extra = {
        "runs": runs,
        "train_gold_kind": {k: round(v / len(data["train"]), 3) for k, v in gold.items()},
        "test_pred_uses_test_only_app": round(np.mean([bool(write_apps(c) & test_only) for c in p["public_test"]]), 3),
        "test_pred_test_only_app_counts": Counter(a for c in p["public_test"] for a in write_apps(c) & test_only),
    }
    return out, extra


# ---------------------------------------------------------------- report

def fmt_p(p: float) -> str:
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def print_rows(rows: list[dict]) -> None:
    for r in rows:
        if "ks" in r:
            print(f"  {r['feature']:28s} mean {r['train_mean']:>8} vs {r['test_mean']:>8}  "
                  f"p10/50/90 {r['train_q']} vs {r['test_q']}  KS {r['ks']} (p {fmt_p(r['ks_p'])})")
        else:
            print(f"  {r['feature']:28s} JSD {r['jsd']}  chi2 p {fmt_p(r['chi2_p'])}")
            for k, a, b in r["rows"]:
                print(f"      {k:30s} {a:6.3f}  {b:6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-cache", default="_ctx", help="suffix of outputs/asr/<split><suffix>.json")
    ap.add_argument("--pred-train", default="v3_full_asrctx")
    ap.add_argument("--pred-test", default="v3")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUTPUT_ROOT / "analysis" / "split_shift.json"))
    args = ap.parse_args()

    data = {s: load_split(s) for s in SPLITS}
    cat = catalog_section(DATA_ROOT)
    report = {"n": {s: len(v) for s, v in data.items()}, "catalog": {k: v for k, v in cat.items() if k != "app_group"}}
    print(f"n: {report['n']}")
    print("\n[catalog]")
    for k, v in report["catalog"].items():
        print(f"  {k}: {v}")

    texts = transcripts(data, args.asr_cache)
    sections = [("installed_apps", lambda: installed_section(data, cat)),
                ("history", lambda: history_section(data, cat)),
                ("context", lambda: context_section(data)),
                ("audio", lambda: audio_section(data)),
                ("transcript", lambda: transcript_section(data, texts)),
                ("predictions", lambda: predictions_section(data, {"train": args.pred_train,
                                                                   "public_test": args.pred_test}, cat))]
    for name, fn in sections:
        rows, extra = fn()
        report[name] = {"tests": rows, **extra}
        print(f"\n[{name}]  (train vs public_test)")
        print_rows(rows)
        for k, v in extra.items():
            print(f"  {k}: {v}")

    if not args.no_embed:
        report["embedding"] = embedding_section(data, texts, cat, args.device)
        print("\n[embedding / adversarial validation]")
        for k, v in report["embedding"].items():
            if isinstance(v, list):
                print(f"  {k}:")
                for row in v:
                    print(f"      {row}")
            else:
                print(f"  {k}: {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
