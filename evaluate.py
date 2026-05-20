"""
================================================================
SmishGuard v2 — Journal-Grade Evaluation Suite
================================================================
Generates all metrics required for academic publication:

  Level 1 — Standard benchmark  (accuracy, F1, AUC, confusion matrix)
  Level 2 — Adversarial test    (robustness against evasion attacks)
  Level 3 — Latency benchmark   (real-world speed measurement)

Usage:
  1. Make sure uvicorn main1:app --port 8001 is running
  2. Upload your test CSV (combined_smishing.csv or spam.csv)
  3. Run: python evaluate.py

Output:
  - Full metrics printed to console
  - journal_report.txt saved for copy-paste into paper
  - confusion_matrix.png saved for paper figures
  - roc_curve.png saved for paper figures
================================================================
"""

import requests
import pandas as pd
import numpy as np
import time
import json
import re
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ================================================================
#  CONFIGURATION — change port if needed
# ================================================================
API_URL    = "http://127.0.0.1:8000/predict"
CSV_PATH   = "test/Dataset_5971.csv"      # your test dataset
SAMPLE_SIZE = None                      # rows to test (set None for all)
TIMEOUT     = 10                        # seconds per request

# ================================================================
#  ADVERSARIAL TEST CASES
#  These are held-out attacks NOT in training data.
#  Tests robustness against real-world evasion techniques.
# ================================================================
ADVERSARIAL_CASES = [

    # ── Category 1: Friendly text + malicious URL (primary blind spot)
    (1, "Hi how are you could you send me some money http://pay-now.xyz"),
    (1, "Hey bro check this out http://bit.ly/free-cash99"),
    (1, "Good morning your reward is ready http://spin-win.club"),
    (1, "Hello dear I need your help urgently http://cash-verify.top"),
    (1, "Hi love please transfer here http://secure-pay.ru"),

    # ── Category 2: Homoglyph attacks
    (1, "URGENT your PαyPal account is limited http://secure-pp.com/verify"),
    (1, "Security alert Am@zon login from new device http://amzn-check.xyz"),
    (1, "Your G00gle account has been compromised http://google-verify.top"),

    # ── Category 3: Segmentation attacks
    (1, "N.e.t.f.l.i.x account on hold pay now http://netflix-fix.cam"),
    (1, "W-A-R-N-I-N-G your CitiBank card is blocked immediately"),

    # ── Category 4: Modern brand smishing
    (1, "Netflix payment failed update now http://netflix-billing.cam"),
    (1, "Binance withdrawal of 0.5 BTC requested http://bnb-secure.xyz"),
    (1, "Your Disney Plus subscription expired http://disney-renew.top"),
    (1, "Coinbase account suspended verify now http://coinbase-verify.live"),
    (1, "FedEx pay customs to release your parcel http://fedex-pay.top"),

    # ── Category 5: Positive sentiment attacks
    (1, "Congratulations you have been selected http://winner-claim.xyz"),
    (1, "Great news your refund is approved http://refund-portal.top"),
    (1, "Amazing you are our lucky winner today http://lucky-prize.cam"),

    # ── Category 6: OTP / Bank fraud (no URL)
    (1, "Your OTP is 847291 do NOT share with anyone call 1800123456"),
    (1, "SBI Alert Rs 9999 debited not you call 18001234567 immediately"),
    (1, "HDFC your card is blocked call immediately 18001234567"),

    # ── Category 7: Delivery scams
    (1, "Your parcel could not be delivered pay fee http://redeliver.xyz"),
    (1, "DHL one delivery attempt failed reschedule http://dhl-rebook.top"),

    # ── Category 8: Legitimate messages that MUST NOT be flagged
    (0, "Hey mom I will be home for dinner tonight"),
    (0, "Your appointment with Dr Smith is confirmed for Tuesday 10 AM"),
    (0, "The package was delivered to your front porch this afternoon"),
    (0, "Meeting rescheduled to 3pm please confirm attendance"),
    (0, "Happy birthday hope you have a wonderful day today"),
    (0, "Can you send me the notes from todays lecture please"),
    (0, "Redeem your Walmart credits via https://www.walmart.com"),
    (0, "Your Amazon order has shipped track at https://www.amazon.com"),
    (0, "Your Netflix account is active visit https://www.netflix.com"),
    (0, "Check your bank balance at https://www.chase.com"),
]

CATEGORY_NAMES = {
    0: "Friendly+Malicious URL",
    5: "Homoglyph Attack",
    8: "Segmentation Attack",
    10: "Modern Brand Smishing",
    15: "Positive Sentiment",
    18: "OTP/Bank Fraud",
    21: "Delivery Scam",
    23: "Legitimate HAM",
}


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
#  LEVEL 1 — STANDARD BENCHMARK
# ================================================================
def run_standard_benchmark() -> dict:
    print("\n" + "="*65)
    print("📊 LEVEL 1 — STANDARD BENCHMARK")
    print("="*65)

    # Load dataset
    try:
        df = pd.read_csv(CSV_PATH, encoding="latin-1")

        # Handle different column names
        df.columns = [str(c).lower() for c in df.columns]
        if "v1" in df.columns and "v2" in df.columns:
            df = df[["v1", "v2"]]
            df.columns = ["label", "text"]
        elif "label" in df.columns and "text" in df.columns:
            df = df[["label", "text"]]
        else:
            raise ValueError(f"Unknown columns: {list(df.columns)}")

        df["label"] = df["label"].str.lower().str.strip()
        df = df[df["label"].isin(["spam", "ham", "smishing"])].dropna()
        df["label_num"] = df["label"].map({"spam": 1, "smishing": 1, "ham": 0})

        # Sample if needed
        if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42)

        print(f"Dataset loaded  : {len(df)} rows")
        print(f"  HAM           : {(df['label']=='ham').sum()}")
        print(f"  SPAM          : {(df['label']=='spam').sum()}")

    except Exception as e:
        print(f"❌ Could not load {CSV_PATH}: {e}")
        print("   Place combined_smishing.csv or spam.csv in this folder")
        return {}

    # Run predictions
    y_true    = []
    y_pred    = []
    y_scores  = []
    errors    = 0
    latencies = []

    print(f"\nRunning predictions...")
    def fetch(row):
        t0 = time.time()
        result = call_api(str(row["text"]))
        latency = (time.time() - t0) * 1000
        return row, result, latency
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        for row, result, latency in tqdm(executor.map(fetch, [row for _, row in df.iterrows()]), total=len(df), ncols=70):
            if result and "is_phishing" in result:
                y_true.append(int(row["label_num"]))
                y_pred.append(1 if result["is_phishing"] else 0)

                # Extract numeric score from "52.14%"
                try:
                    score = float(result["final_risk_score"].replace("%", "")) / 100
                except Exception:
                    score = 0.5
                y_scores.append(score)
                latencies.append(latency)
            else:
                errors += 1

    if not y_true:
        print("❌ No predictions received. Is the server running?")
        return {}

    # Compute metrics
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = cm.ravel()
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0
    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)

    results = {
        "accuracy"       : acc,
        "precision"      : prec,
        "recall"         : rec,
        "f1"             : f1,
        "auc"            : auc,
        "fpr"            : fpr,
        "fnr"            : fnr,
        "tp"             : int(tp),
        "tn"             : int(tn),
        "fp"             : int(fp),
        "fn"             : int(fn),
        "avg_latency_ms" : avg_lat,
        "p95_latency_ms" : p95_lat,
        "total_tested"   : len(y_true),
        "errors"         : errors,
    }

    # Print results
    print(f"\n{'─'*65}")
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Precision       : {prec*100:.2f}%")
    print(f"  Recall          : {rec*100:.2f}%")
    print(f"  F1 Score        : {f1*100:.2f}%")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"{'─'*65}")
    print(f"  True Positives  : {tp}  (spam correctly blocked)")
    print(f"  True Negatives  : {tn}  (safe correctly passed)")
    print(f"  False Positives : {fp}  (safe wrongly flagged)    ← minimize")
    print(f"  False Negatives : {fn}  (spam missed)             ← minimize")
    print(f"{'─'*65}")
    print(f"  False Positive Rate : {fpr*100:.2f}%")
    print(f"  False Negative Rate : {fnr*100:.2f}%")
    print(f"{'─'*65}")
    print(f"  Avg Latency     : {avg_lat:.1f} ms")
    print(f"  P95 Latency     : {p95_lat:.1f} ms")
    print(f"  Errors          : {errors}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["HAM (Safe)", "SPAM (Phishing)"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("SmishGuard v2 — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✅ Saved: confusion_matrix.png")

    # Plot ROC curve
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_curve, tpr_curve, color="royalblue", lw=2,
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

    return results


# ================================================================
#  LEVEL 2 — ADVERSARIAL STRESS TEST
# ================================================================
def run_adversarial_test() -> dict:
    print("\n" + "="*65)
    print("🛡️  LEVEL 2 — ADVERSARIAL ROBUSTNESS TEST")
    print("="*65)

    categories = {
        "Friendly+Malicious URL" : [],
        "Homoglyph Attack"       : [],
        "Segmentation Attack"    : [],
        "Modern Brand Smishing"  : [],
        "Positive Sentiment"     : [],
        "OTP/Bank Fraud"         : [],
        "Delivery Scam"          : [],
        "Legitimate HAM"         : [],
    }

    cat_map = [
        ("Friendly+Malicious URL", 5),
        ("Homoglyph Attack",       3),
        ("Segmentation Attack",    2),
        ("Modern Brand Smishing",  5),
        ("Positive Sentiment",     3),
        ("OTP/Bank Fraud",         3),
        ("Delivery Scam",          2),
        ("Legitimate HAM",         8),
    ]

    passed = 0
    total  = len(ADVERSARIAL_CASES)
    idx    = 0

    for cat_name, cat_count in cat_map:
        cat_pass = 0
        cat_total = 0
        for i in range(cat_count):
            if idx >= total:
                break
            true_label, text = ADVERSARIAL_CASES[idx]
            idx += 1

            result = call_api(text)
            if not result:
                cat_total += 1
                continue

            predicted  = 1 if result["is_phishing"] else 0
            correct    = predicted == true_label
            score      = result.get("final_risk_score", "?")
            mode       = result.get("logic_mode", "?")

            if correct:
                passed   += 1
                cat_pass += 1
            cat_total += 1

            icon     = "✅" if correct else "❌"
            expected = "SPAM" if true_label == 1 else "HAM"
            got      = "SPAM" if predicted  == 1 else "HAM"
            print(f"  {icon} [{cat_name:<24}] "
                  f"Expected:{expected} Got:{got} "
                  f"Score:{score} | {text[:45]}...")

        cat_pct = (cat_pass / cat_total * 100) if cat_total > 0 else 0
        categories[cat_name] = {"pass": cat_pass, "total": cat_total, "pct": cat_pct}

    adv_score = passed / total * 100

    print(f"\n{'─'*65}")
    print("CATEGORY BREAKDOWN:")
    for cat, data in categories.items():
        bar = "█" * int(data["pct"] / 10) + "░" * (10 - int(data["pct"] / 10))
        print(f"  {cat:<26} [{bar}] {data['pass']}/{data['total']} ({data['pct']:.0f}%)")

    print(f"{'─'*65}")
    print(f"  Overall Adversarial Score: {passed}/{total} ({adv_score:.1f}%)")

    if adv_score >= 90:
        verdict = "🏆 Excellent — publication-grade robustness"
    elif adv_score >= 80:
        verdict = "✅ Good — minor improvements possible"
    elif adv_score >= 70:
        verdict = "⚠️  Acceptable — recommend more adversarial training data"
    else:
        verdict = "❌ Needs improvement — retrain with more adversarial examples"

    print(f"  Verdict: {verdict}")

    return {
        "adversarial_score"   : adv_score,
        "adversarial_passed"  : passed,
        "adversarial_total"   : total,
        "category_breakdown"  : categories,
    }


# ================================================================
#  LEVEL 3 — LATENCY BENCHMARK
# ================================================================
def run_latency_benchmark() -> dict:
    print("\n" + "="*65)
    print("⚡ LEVEL 3 — LATENCY BENCHMARK")
    print("="*65)

    test_messages = [
        "Hey mom I will be home for dinner tonight",
        "URGENT your account is blocked verify now http://secure-verify.xyz",
        "Your OTP is 847291 do not share call 18001234567 immediately",
        "Redeem your Walmart points at https://www.walmart.com",
        "Congratulations you won a prize click http://winner-now.cam",
    ]

    latencies = []
    runs      = 20   # Run each message multiple times for stable measurement

    print(f"Running {runs} iterations per message ({len(test_messages)} messages)...")

    for msg in test_messages:
        msg_lats = []
        for _ in range(runs):
            t0 = time.time()
            call_api(msg)
            msg_lats.append((time.time() - t0) * 1000)
        latencies.extend(msg_lats)

    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mn  = np.min(latencies)
    mx  = np.max(latencies)

    print(f"\n  Average latency  : {avg:.1f} ms")
    print(f"  Median  (P50)    : {p50:.1f} ms")
    print(f"  P95 latency      : {p95:.1f} ms")
    print(f"  P99 latency      : {p99:.1f} ms")
    print(f"  Min latency      : {mn:.1f} ms")
    print(f"  Max latency      : {mx:.1f} ms")

    if avg < 100:
        verdict = "🏆 Excellent — real-time capable"
    elif avg < 300:
        verdict = "✅ Good — suitable for production"
    elif avg < 1000:
        verdict = "⚠️  Acceptable — may need optimization"
    else:
        verdict = "❌ Slow — consider model optimization"

    print(f"  Verdict: {verdict}")

    return {
        "avg_ms": avg, "p50_ms": p50,
        "p95_ms": p95, "p99_ms": p99,
    }


# ================================================================
#  SAVE JOURNAL REPORT
# ================================================================
def save_report(std: dict, adv: dict, lat: dict):
    report = f"""
================================================================
SMISHGUARD v2 — JOURNAL EVALUATION REPORT
================================================================

STANDARD BENCHMARK (n={std.get('total_tested', 'N/A')})
─────────────────────────────────────────────────────────────
Accuracy            : {std.get('accuracy', 0)*100:.2f}%
Precision           : {std.get('precision', 0)*100:.2f}%
Recall              : {std.get('recall', 0)*100:.2f}%
F1 Score            : {std.get('f1', 0)*100:.2f}%
ROC-AUC             : {std.get('auc', 0):.4f}
False Positive Rate : {std.get('fpr', 0)*100:.2f}%
False Negative Rate : {std.get('fnr', 0)*100:.2f}%

CONFUSION MATRIX
─────────────────────────────────────────────────────────────
True Positives  (Spam correctly blocked) : {std.get('tp', 'N/A')}
True Negatives  (Safe correctly passed)  : {std.get('tn', 'N/A')}
False Positives (Safe wrongly flagged)   : {std.get('fp', 'N/A')}
False Negatives (Spam missed)            : {std.get('fn', 'N/A')}

ADVERSARIAL ROBUSTNESS
─────────────────────────────────────────────────────────────
Overall Score       : {adv.get('adversarial_score', 0):.1f}%
Passed              : {adv.get('adversarial_passed', 'N/A')}/{adv.get('adversarial_total', 'N/A')}

Category Breakdown:
"""
    for cat, data in adv.get("category_breakdown", {}).items():
        report += f"  {cat:<28}: {data['pass']}/{data['total']} ({data['pct']:.0f}%)\n"

    report += f"""
LATENCY BENCHMARK
─────────────────────────────────────────────────────────────
Average Latency     : {lat.get('avg_ms', 0):.1f} ms
Median  (P50)       : {lat.get('p50_ms', 0):.1f} ms
P95 Latency         : {lat.get('p95_ms', 0):.1f} ms
P99 Latency         : {lat.get('p99_ms', 0):.1f} ms

FIGURES GENERATED
─────────────────────────────────────────────────────────────
confusion_matrix.png  — for paper Figure X
roc_curve.png         — for paper Figure Y
================================================================
"""

    with open("journal_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n✅ Saved: journal_report.txt")
    print(report)


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    print("="*65)
    print("🛡️  SmishGuard v2 — Journal Evaluation Suite")
    print("="*65)

    # Check server
    print("\nChecking server...")
    if not check_server():
        print("❌ Server not running!")
        print("   Start it with: uvicorn main1:app --reload --port 8001")
        exit(1)
    print("✅ Server is online")

    # Run all three levels
    std_results = run_standard_benchmark()
    adv_results = run_adversarial_test()
    lat_results = run_latency_benchmark()

    # Save complete report
    if std_results:
        save_report(std_results, adv_results, lat_results)
    else:
        print("\n⚠️  Standard benchmark failed — report not saved")
        print("   Check that combined_smishing.csv is in this folder")