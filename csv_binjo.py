"""
================================================================
SmishGuard v2 — CSV Benchmark Runner
================================================================
Supports the Mishra & Soni dataset format:
  Columns: LABEL | TEXT | URL | EMAIL | PHONE
  Labels : ham, Smishing (mapped to spam)

Usage:
  1. Place your CSV file in this folder
  2. Make sure server is running:
       uvicorn main1:app --reload --port 8001
  3. Run:
       python csv_benchmark.py

  Optional — specify a different CSV file:
       python csv_benchmark.py myfile.csv
================================================================
"""

import sys
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ================================================================
#  CONFIGURATION
# ================================================================
API_URL    = "http://127.0.0.1:8000/predict"
CSV_PATH   = sys.argv[1] if len(sys.argv) > 1 else "binjo_test.csv"
TIMEOUT    = 10
SAMPLE_SIZE = None   # Set to e.g. 200 to test a subset, None = all rows

# ================================================================
#  HELPERS
# ================================================================
def call_api(text: str) -> dict | None:
    try:
        r = requests.post(API_URL, json={"text": text}, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def check_server() -> bool:
    try:
        r = requests.get(API_URL.replace("/predict", "/health"), timeout=5)
        return r.status_code == 200
    except Exception:
        return False

# ================================================================
#  LOAD CSV — handles the Mishra & Soni format automatically
# ================================================================
def load_csv(path: str) -> pd.DataFrame:
    print(f"\nLoading: {path}")

    # Try common encodings
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"  Encoding     : {enc}")
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not read {path} with any encoding")

    print(f"  Raw shape    : {df.shape}")
    print(f"  Columns      : {list(df.columns)}")

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Find label column
    label_col = None
    for candidate in ["LABEL", "V1", "CLASS", "CATEGORY", "TYPE"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if not label_col:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")

    # Find text column
    text_col = None
    for candidate in ["TEXT", "V2", "MESSAGE", "SMS", "BODY", "CONTENT"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if not text_col:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")

    print(f"  Label column : {label_col}")
    print(f"  Text column  : {text_col}")

    # Extract and clean
    df = df[[label_col, text_col]].copy()
    df.columns = ["label", "text"]
    df.dropna(inplace=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"]  = df["text"].astype(str).str.strip()

    # Remove rows where text is empty
    df = df[df["text"].str.len() > 0]

    # Show raw label distribution
    print(f"\n  Raw label distribution:")
    for lbl, cnt in df["label"].value_counts().items():
        print(f"    '{lbl}': {cnt}")

    # Normalize labels → ham / spam
    # Handles: smishing, spam, phishing, scam → spam
    #          ham, legitimate, safe, normal  → ham
    spam_labels = {"spam", "smishing", "phishing", "scam", "fraud", "1"}
    ham_labels  = {"ham", "legitimate", "safe", "normal", "benign", "0"}

    df["label"] = df["label"].apply(
        lambda x: "spam" if x in spam_labels
        else ("ham" if x in ham_labels else None)
    )
    df.dropna(subset=["label"], inplace=True)

    print(f"\n  After normalization:")
    vc = df["label"].value_counts()
    print(f"    ham   : {vc.get('ham',  0)}")
    print(f"    spam  : {vc.get('spam', 0)}")
    print(f"  Total   : {len(df)}")

    # Sample if needed
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"  Sampled to  : {SAMPLE_SIZE} rows")

    return df


# ================================================================
#  RUN BENCHMARK
# ================================================================
def run_benchmark(df: pd.DataFrame):
    print(f"\n{'='*65}")
    print("📊 RUNNING PREDICTIONS AGAINST API")
    print(f"{'='*65}")
    print(f"Testing {len(df)} messages...\n")

    y_true    = []
    y_pred    = []
    y_scores  = []
    latencies = []
    errors    = 0

    # Store per-row details for analysis
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), ncols=70):
        t0     = time.time()
        result = call_api(str(row["text"]))
        lat    = (time.time() - t0) * 1000

        true_label = 1 if row["label"] == "spam" else 0

        if result and "is_phishing" in result:
            pred = 1 if result["is_phishing"] else 0
            try:
                score = float(result["final_risk_score"].replace("%", "")) / 100
            except Exception:
                score = 0.5

            y_true.append(true_label)
            y_pred.append(pred)
            y_scores.append(score)
            latencies.append(lat)

            rows.append({
                "text"        : str(row["text"])[:80],
                "true_label"  : row["label"],
                "predicted"   : "spam" if pred == 1 else "ham",
                "correct"     : pred == true_label,
                "score"       : f"{score*100:.1f}%",
                "ai_score"    : result.get("ai_score", "?"),
                "logic_mode"  : result.get("logic_mode", "?"),
                "warnings"    : result.get("link_warnings", ""),
            })
        else:
            errors += 1
            rows.append({
                "text"      : str(row["text"])[:80],
                "true_label": row["label"],
                "predicted" : "ERROR",
                "correct"   : False,
                "score"     : "?",
                "ai_score"  : "?",
                "logic_mode": "API Error",
                "warnings"  : "",
            })

    return y_true, y_pred, y_scores, latencies, errors, rows


# ================================================================
#  COMPUTE AND DISPLAY METRICS
# ================================================================
def compute_metrics(y_true, y_pred, y_scores, latencies, errors, rows):
    print(f"\n{'='*65}")
    print("📈 RESULTS")
    print(f"{'='*65}")

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_scores)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  Accuracy            : {acc*100:>7.2f}%          │")
    print(f"  │  Precision           : {prec*100:>7.2f}%          │")
    print(f"  │  Recall              : {rec*100:>7.2f}%          │")
    print(f"  │  F1 Score            : {f1*100:>7.2f}%          │")
    print(f"  │  ROC-AUC             : {auc:>8.4f}          │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │  True Positives      : {tp:>7}            │")
    print(f"  │  True Negatives      : {tn:>7}            │")
    print(f"  │  False Positives     : {fp:>7}  ← minimize │")
    print(f"  │  False Negatives     : {fn:>7}  ← minimize │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │  False Positive Rate : {fpr*100:>7.2f}%          │")
    print(f"  │  False Negative Rate : {fnr*100:>7.2f}%          │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │  Avg Latency         : {avg_lat:>7.1f} ms        │")
    print(f"  │  P95 Latency         : {p95_lat:>7.1f} ms        │")
    print(f"  │  API Errors          : {errors:>7}            │")
    print(f"  └──────────────────────────────────────────┘")

    # Show failures — most useful for debugging
    failures = [r for r in rows if not r["correct"] and r["predicted"] != "ERROR"]

    if failures:
        print(f"\n{'─'*65}")
        print(f"❌ FAILURES ({len(failures)} total)")
        print(f"{'─'*65}")

        fn_rows = [r for r in failures if r["true_label"] == "spam"]
        fp_rows = [r for r in failures if r["true_label"] == "ham"]

        if fn_rows:
            print(f"\n  FALSE NEGATIVES — Spam missed ({len(fn_rows)}):")
            for r in fn_rows[:10]:
                print(f"    Score:{r['score']:>7} | Mode: {r['logic_mode'][:30]}")
                print(f"    Text: {r['text'][:70]}")
                print()

        if fp_rows:
            print(f"\n  FALSE POSITIVES — Ham wrongly flagged ({len(fp_rows)}):")
            for r in fp_rows[:10]:
                print(f"    Score:{r['score']:>7} | Mode: {r['logic_mode'][:30]}")
                print(f"    Text: {r['text'][:70]}")
                print()

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["HAM (Safe)", "SPAM (Phishing)"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("SmishGuard v2 — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: confusion_matrix.png")

    # ROC curve
    fpr_c, tpr_c, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_c, tpr_c, color="royalblue", lw=2,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("SmishGuard v2 — ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: roc_curve.png")

    # Save report
    report = f"""
================================================================
SmishGuard v2 — CSV Benchmark Report
Dataset : {CSV_PATH}
Tested  : {len(y_true)} messages ({errors} errors)
================================================================

PERFORMANCE METRICS
─────────────────────────────────────────────────────────────
Accuracy            : {acc*100:.2f}%
Precision           : {prec*100:.2f}%
Recall              : {rec*100:.2f}%
F1 Score            : {f1*100:.2f}%
ROC-AUC             : {auc:.4f}
False Positive Rate : {fpr*100:.2f}%
False Negative Rate : {fnr*100:.2f}%

CONFUSION MATRIX
─────────────────────────────────────────────────────────────
True Positives  (Spam correctly blocked) : {tp}
True Negatives  (Safe correctly passed)  : {tn}
False Positives (Safe wrongly flagged)   : {fp}
False Negatives (Spam missed)            : {fn}

LATENCY
─────────────────────────────────────────────────────────────
Average  : {avg_lat:.1f} ms
P95      : {p95_lat:.1f} ms

FIGURES
─────────────────────────────────────────────────────────────
confusion_matrix.png
roc_curve.png
================================================================
"""
    with open("csv_benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ Saved: csv_benchmark_report.txt")
    print(report)

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
        "fpr": fpr, "fnr": fnr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    print("="*65)
    print("🛡️  SmishGuard v2 — CSV Benchmark Runner")
    print("="*65)

    # Check server
    if not check_server():
        print("\n❌ Server not running!")
        print("   Start it: uvicorn main1:app --reload --port 8001")
        sys.exit(1)
    print("✅ Server online")

    # Load CSV
    try:
        df = load_csv(CSV_PATH)
    except Exception as e:
        print(f"\n❌ Failed to load CSV: {e}")
        sys.exit(1)

    if len(df) == 0:
        print("❌ No valid rows found in CSV")
        sys.exit(1)

    # Run benchmark
    y_true, y_pred, y_scores, latencies, errors, rows = run_benchmark(df)

    if not y_true:
        print("❌ No predictions received. Is the API running on port 8001?")
        sys.exit(1)

    # Show metrics
    compute_metrics(y_true, y_pred, y_scores, latencies, errors, rows)