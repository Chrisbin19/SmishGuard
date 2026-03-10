"""
SmishGuard Full Evaluation Suite
================================
Tests the live API against spam.csv for accuracy, robustness, latency,
edge cases, and false positives. Produces a unified report card.

Usage:
    1. Make sure server is running: uvicorn main:app --reload
    2. Run: python smishguard_full_evaluation.py
"""

import requests
import pandas as pd
import time
import sys
import statistics
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = "spam.csv"
SAMPLE_SIZE = 500  # Total messages to test from spam.csv

# ==============================================================================
#   HELPER
# ==============================================================================
def call_api(text):
    """Send a single prediction request and return the result."""
    try:
        resp = requests.post(API_URL, json={"text": text}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
    return None

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ==============================================================================
#   MODULE 1: CORE ACCURACY (spam.csv)
# ==============================================================================
def run_accuracy_benchmark():
    print_header("MODULE 1: CORE ACCURACY BENCHMARK")
    print(f"  Dataset: {CSV_PATH} | Sample: {SAMPLE_SIZE} messages\n")

    try:
        df = pd.read_csv(CSV_PATH, encoding='latin-1', usecols=[0, 1])
        df.columns = ['label', 'text']
        df.dropna(inplace=True)
        df['text'] = df['text'].astype(str)
        df = df[df['text'].str.strip().astype(bool)]
    except Exception as e:
        print(f"  ERROR loading CSV: {e}")
        return None

    # Balance the sample: equal spam and ham
    spam_df = df[df['label'] == 'spam']
    ham_df = df[df['label'] == 'ham']
    half = min(SAMPLE_SIZE // 2, len(spam_df), len(ham_df))
    sample = pd.concat([spam_df.sample(n=half, random_state=42),
                        ham_df.sample(n=half, random_state=42)])
    sample = sample.sample(frac=1, random_state=42)  # shuffle

    print(f"  Testing {len(sample)} messages ({half} spam + {half} ham)...\n")

    tp = fp = tn = fn = errors = 0

    for i, (_, row) in enumerate(sample.iterrows()):
        result = call_api(row['text'])
        if result is None:
            errors += 1
            continue

        predicted_spam = result.get('is_phishing', False)
        actual_spam = (row['label'] == 'spam')

        if actual_spam and predicted_spam:
            tp += 1
        elif actual_spam and not predicted_spam:
            fn += 1
        elif not actual_spam and predicted_spam:
            fp += 1
        else:
            tn += 1

        # Progress indicator every 50 messages
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sample)}...")

    total = tp + fp + tn + fn
    accuracy  = ((tp + tn) / total * 100) if total else 0
    precision = (tp / (tp + fp) * 100) if (tp + fp) else 0
    recall    = (tp / (tp + fn) * 100) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f"\n  Results:")
    print(f"  {'Accuracy:':<20} {accuracy:.2f}%")
    print(f"  {'Precision:':<20} {precision:.2f}%")
    print(f"  {'Recall:':<20} {recall:.2f}%")
    print(f"  {'F1 Score:':<20} {f1:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives  (Spam caught):       {tp}")
    print(f"    False Negatives (Spam missed):        {fn}")
    print(f"    True Negatives  (Ham correct):        {tn}")
    print(f"    False Positives (Ham wrongly flagged): {fp}")
    print(f"    Errors:                               {errors}")

    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

# ==============================================================================
#   MODULE 2: ADVERSARIAL ROBUSTNESS
# ==============================================================================
def run_adversarial_test():
    print_header("MODULE 2: ADVERSARIAL ROBUSTNESS")

    attacks = [
        # Homoglyph Attacks
        {"text": "URGENT: Your PayPal account is limited. Restore here: http://secure-pp-verify.com", "type": "Homoglyph"},
        {"text": "Security Alert: Amazon login detected from Russia. Verify: http://amzn-security-check.xyz", "type": "Homoglyph"},

        # Segmentation / Noise
        {"text": "D.e.a.r C.u.s.t.o.m.e.r, your N.e.t.f.l.i.x is on hold. Pay: http://netflix-billing-update.com", "type": "Segmentation"},
        {"text": "W-A-R-N-I-N-G: Your CitiBank card is blocked. Action required: http://citi-confirm.net", "type": "Segmentation"},

        # Zero-Day Brands
        {"text": "Your Disney+ subscription has expired. Renew membership: http://disney-renewals.com", "type": "New Brand"},
        {"text": "Binance Withdrawal Alert: 0.5 BTC requested. Cancel transaction: http://binance-fraud-control.com", "type": "New Brand"},
        {"text": "Your Zelle payment of $200 failed. Retry: http://zelle-payment-fix.top", "type": "New Brand"},

        # Sentiment Inversion
        {"text": "We are happy to inform you that your refund has been approved! Click to accept: http://irs-refund-claim.xyz", "type": "Sentiment Inversion"},
        {"text": "Thank you for being a loyal customer. Free gift card: http://amazon-gifts-free.top", "type": "Sentiment Inversion"},

        # Semantic Mismatch
        {"text": "Login to your Microsoft Office account to view the document: http://google-drive-share.com", "type": "Semantic Mismatch"},
        {"text": "Your Apple ID was used in California. Verify: http://samsung-verify.club", "type": "Semantic Mismatch"},

        # URL Shortener Abuse
        {"text": "Your bank account is compromised! Secure it now: http://bit.ly/3xF9kQ2", "type": "URL Shortener"},
        {"text": "You have won a prize! Claim here: http://tinyurl.com/free-prize", "type": "URL Shortener"},

        # No-Link Pure Social Engineering
        {"text": "URGENT! Call 1-800-555-0199 immediately or your account will be closed in 24 hours!", "type": "No-Link Scam"},
        {"text": "Congratulations! You won $5000. Reply YES to claim your prize now!", "type": "No-Link Scam"},
    ]

    passed = 0
    failed_cases = []

    for i, case in enumerate(attacks):
        result = call_api(case['text'])
        if result is None:
            failed_cases.append((case['type'], "API Error"))
            continue

        if result.get('is_phishing', False):
            passed += 1
            status = "BLOCKED"
            icon = "+"
        else:
            failed_cases.append((case['type'], case['text'][:60]))
            status = "MISSED"
            icon = "-"

        ai = result.get('ai_score', '?')
        forensic = result.get('forensic_score', '?')
        mode = result.get('logic_mode', '?')
        print(f"  [{icon}] {case['type']:<20} | {status} | AI: {ai} | Forensic: {forensic} | Mode: {mode}")

    total = len(attacks)
    score = (passed / total * 100) if total else 0

    print(f"\n  Score: {passed}/{total} attacks blocked ({score:.1f}%)")
    if failed_cases:
        print(f"\n  Missed attacks:")
        for typ, txt in failed_cases:
            print(f"    - [{typ}] {txt}")

    return {"passed": passed, "total": total, "score": score}

# ==============================================================================
#   MODULE 3: LATENCY & THROUGHPUT
# ==============================================================================
def run_latency_test():
    print_header("MODULE 3: LATENCY & THROUGHPUT")

    test_messages = [
        "Hey, are you free tonight?",
        "URGENT: Your account is locked. Click http://verify-now.xyz",
        "Congratulations! You won $1000! Call now!",
        "Meeting moved to 3pm. See you there.",
        "Your package has been shipped. Track: http://fedex.com/track/123",
    ]

    latencies = []
    num_requests = 100

    print(f"  Sending {num_requests} requests...\n")
    overall_start = time.time()

    for i in range(num_requests):
        msg = test_messages[i % len(test_messages)]
        start = time.time()
        result = call_api(msg)
        elapsed = (time.time() - start) * 1000  # ms

        if result is not None:
            latencies.append(elapsed)

        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{num_requests}...")

    overall_elapsed = time.time() - overall_start

    if not latencies:
        print("  ERROR: All requests failed.")
        return None

    latencies.sort()
    avg = statistics.mean(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    throughput = len(latencies) / overall_elapsed

    print(f"\n  Results ({len(latencies)} successful requests):")
    print(f"  {'Min Latency:':<20} {min(latencies):.2f} ms")
    print(f"  {'Avg Latency:':<20} {avg:.2f} ms")
    print(f"  {'P50 (Median):':<20} {p50:.2f} ms")
    print(f"  {'P95 Latency:':<20} {p95:.2f} ms")
    print(f"  {'P99 Latency:':<20} {p99:.2f} ms")
    print(f"  {'Max Latency:':<20} {max(latencies):.2f} ms")
    print(f"  {'Throughput:':<20} {throughput:.2f} req/s")

    return {"avg": avg, "p50": p50, "p95": p95, "p99": p99,
            "min": min(latencies), "max": max(latencies),
            "throughput": throughput}

# ==============================================================================
#   MODULE 4: EDGE CASE RESILIENCE
# ==============================================================================
def run_edge_case_test():
    print_header("MODULE 4: EDGE CASE RESILIENCE")

    edge_cases = [
        {"text": "", "name": "Empty string"},
        {"text": " ", "name": "Single space"},
        {"text": "a", "name": "Single character"},
        {"text": "Hello " * 500, "name": "Very long text (2500 words)"},
        {"text": "http://example.com http://test.com http://foo.bar", "name": "Only URLs"},
        {"text": "12345678901234567890", "name": "Only numbers"},
        {"text": "!@#$%^&*()_+", "name": "Only special characters"},
        {"text": "😀🎉🔥💰🚨⚠️", "name": "Only emojis"},
        {"text": "Hola amigo, tu cuenta esta bloqueada", "name": "Spanish text"},
        {"text": "   \n\t  \r\n  ", "name": "Whitespace only"},
    ]

    passed = 0
    failed_cases = []

    for case in edge_cases:
        try:
            result = call_api(case['text'])
            if result is not None and 'error' not in result:
                passed += 1
                print(f"  [+] {case['name']:<35} -> {'Phishing' if result['is_phishing'] else 'Safe'}")
            elif result and 'error' in result:
                failed_cases.append((case['name'], f"API error: {result['error'][:50]}"))
                print(f"  [-] {case['name']:<35} -> ERROR: {result['error'][:50]}")
            else:
                failed_cases.append((case['name'], "No response"))
                print(f"  [-] {case['name']:<35} -> No response")
        except Exception as e:
            failed_cases.append((case['name'], str(e)[:50]))
            print(f"  [-] {case['name']:<35} -> CRASH: {str(e)[:50]}")

    total = len(edge_cases)
    score = (passed / total * 100) if total else 0
    print(f"\n  Score: {passed}/{total} handled gracefully ({score:.1f}%)")

    return {"passed": passed, "total": total, "score": score}

# ==============================================================================
#   MODULE 5: FALSE POSITIVE AUDIT
# ==============================================================================
def run_false_positive_audit():
    print_header("MODULE 5: FALSE POSITIVE AUDIT")
    print("  Testing legitimate messages that should NOT be flagged...\n")

    safe_messages = [
        {"text": "Your Amazon order #12345 has shipped. Track at http://amazon.com/track/12345", "name": "Amazon shipping"},
        {"text": "Click here to view the meeting agenda: http://docs.google.com/doc123", "name": "Google Docs link"},
        {"text": "Your Uber ride is arriving in 3 minutes.", "name": "Uber notification"},
        {"text": "Netflix: New episode of Stranger Things is now available!", "name": "Netflix update"},
        {"text": "Your appointment with Dr. Smith is confirmed for Tuesday 10 AM.", "name": "Doctor appointment"},
        {"text": "Hey, can you pick up milk on the way home?", "name": "Casual text"},
        {"text": "Reminder: Team standup at 9:30 AM tomorrow.", "name": "Work reminder"},
        {"text": "Your password was changed successfully. If this wasn't you, contact support.", "name": "Password change"},
        {"text": "Flight AA123 to New York is on time. Gate B12.", "name": "Flight update"},
        {"text": "Happy birthday! Hope you have a great day!", "name": "Birthday wish"},
    ]

    false_positives = 0
    details = []

    for case in safe_messages:
        result = call_api(case['text'])
        if result is None:
            print(f"  [?] {case['name']:<25} -> API Error")
            continue

        flagged = result.get('is_phishing', False)
        if flagged:
            false_positives += 1
            icon = "X"
            details.append(case['name'])
        else:
            icon = "+"

        ai = result.get('ai_score', '?')
        print(f"  [{icon}] {case['name']:<25} -> {'WRONGLY FLAGGED' if flagged else 'Correct (Safe)'} | AI: {ai}")

    total = len(safe_messages)
    fp_rate = (false_positives / total * 100) if total else 0
    clean = total - false_positives

    print(f"\n  Result: {false_positives}/{total} wrongly flagged (FP Rate: {fp_rate:.1f}%)")
    if details:
        print(f"  False positives: {', '.join(details)}")

    return {"false_positives": false_positives, "total": total,
            "fp_rate": fp_rate, "clean": clean}

# ==============================================================================
#   FINAL REPORT CARD
# ==============================================================================
def generate_report(accuracy_r, adversarial_r, latency_r, edge_r, fp_r):
    report_lines = []

    def add(line=""):
        report_lines.append(line)
        print(line)

    add("\n")
    add("=" * 62)
    add("   SMISHGUARD - COMPREHENSIVE EVALUATION REPORT CARD")
    add(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("=" * 62)

    add("\n  1. CORE ACCURACY (spam.csv)")
    add("  " + "-" * 40)
    if accuracy_r:
        add(f"    Accuracy:       {accuracy_r['accuracy']:.2f}%")
        add(f"    Precision:      {accuracy_r['precision']:.2f}%")
        add(f"    Recall:         {accuracy_r['recall']:.2f}%")
        add(f"    F1 Score:       {accuracy_r['f1']:.2f}%")
        add(f"    TP: {accuracy_r['tp']}  FN: {accuracy_r['fn']}  TN: {accuracy_r['tn']}  FP: {accuracy_r['fp']}")
    else:
        add("    FAILED - Could not load data")

    add("\n  2. ADVERSARIAL ROBUSTNESS")
    add("  " + "-" * 40)
    if adversarial_r:
        add(f"    Attacks Blocked: {adversarial_r['passed']}/{adversarial_r['total']} ({adversarial_r['score']:.1f}%)")
    else:
        add("    FAILED")

    add("\n  3. LATENCY & THROUGHPUT")
    add("  " + "-" * 40)
    if latency_r:
        add(f"    Avg Latency:    {latency_r['avg']:.2f} ms")
        add(f"    P95 Latency:    {latency_r['p95']:.2f} ms")
        add(f"    Throughput:     {latency_r['throughput']:.2f} req/s")
    else:
        add("    FAILED")

    add("\n  4. EDGE CASE RESILIENCE")
    add("  " + "-" * 40)
    if edge_r:
        add(f"    Handled:        {edge_r['passed']}/{edge_r['total']} ({edge_r['score']:.1f}%)")
    else:
        add("    FAILED")

    add("\n  5. FALSE POSITIVE AUDIT")
    add("  " + "-" * 40)
    if fp_r:
        add(f"    Clean Rate:     {fp_r['clean']}/{fp_r['total']} ({100 - fp_r['fp_rate']:.1f}% correct)")
        add(f"    False Positives: {fp_r['false_positives']}")
    else:
        add("    FAILED")

    add("\n" + "=" * 62)

    # Overall grade
    scores = []
    if accuracy_r: scores.append(accuracy_r['f1'])
    if adversarial_r: scores.append(adversarial_r['score'])
    if edge_r: scores.append(edge_r['score'])
    if fp_r: scores.append(100 - fp_r['fp_rate'])

    if scores:
        overall = sum(scores) / len(scores)
        if overall >= 90:
            grade = "A - Excellent"
        elif overall >= 75:
            grade = "B - Good"
        elif overall >= 60:
            grade = "C - Needs Improvement"
        elif overall >= 40:
            grade = "D - Poor"
        else:
            grade = "F - Critical Issues"

        add(f"   OVERALL GRADE: {grade} ({overall:.1f}%)")
    add("=" * 62)

    # Save to file
    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved to: evaluation_report.txt")

# ==============================================================================
#   MAIN
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("   SMISHGUARD FULL EVALUATION SUITE")
    print("=" * 62)

    # Quick health check
    print("\n  Checking API connection...")
    test = call_api("test")
    if test is None:
        print("  ERROR: Cannot reach API at", API_URL)
        print("  Make sure server is running: uvicorn main:app --reload")
        sys.exit(1)
    print("  API is online!\n")

    # Run all modules
    accuracy_result    = run_accuracy_benchmark()
    adversarial_result = run_adversarial_test()
    latency_result     = run_latency_test()
    edge_result        = run_edge_case_test()
    fp_result          = run_false_positive_audit()

    # Final Report
    generate_report(accuracy_result, adversarial_result, latency_result,
                    edge_result, fp_result)
