import numpy as np
import tensorflow as tf
import pickle
import os
import csv
import re
import requests
import tldextract
import pandas as pd
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

# ==========================================
# --- 1. SETUP & CORE CONFIGURATION ---
# ==========================================
app = Flask(__name__, 
            template_folder='../frontend/templates', 
            static_folder='../frontend/static')

# NLP Constraints
MAX_LEN = 200 
MAX_WORDS = 25000

# File Paths
MODEL_PATH = 'smishing_model.keras'
TOKENIZER_PATH = 'tokenizer.pickle'
LOG_FILE = '../dataset/review_me.csv'
BATCH_TEST_FILE = 'test_data.txt'
ACCURACY_FILE = 'testdata_new.csv'
CUSTOM_CSV_FILE = 'custom.csv'

# ==========================================
# --- 2. RESOURCE LOADER ---
# ==========================================
print("\n" + "="*50)
print("🛡️  SMISHGUARD SYSTEM INITIALIZATION")
print("="*50)

def load_ai_resources():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading LSTM Model...")
    try:
        # compile=False avoids errors if custom metrics were used during Colab training
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model: {e}")
        return None, None

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading Pickled Tokenizer...")
    try:
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("✅ Tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load tokenizer: {e}")
        return None, None
    
    return model, tokenizer

model, tokenizer = load_ai_resources()

if model is None:
    print("System halted due to missing resources.")
    exit()

# ==========================================
# --- 3. FORENSIC & NLP UTILITIES ---
# ==========================================

def expand_url(short_url):
    """ 
    Follows redirects (bit.ly, t.co) to find the true destination.
    Uses a strict timeout to maintain API responsiveness.
    """
    try:
        response = requests.head(short_url, allow_redirects=True, timeout=1.8)
        return response.url
    except Exception as e:
        # If site is down or times out, we analyze the string offline
        return short_url

def inspect_url_and_context(text):
    """
    Performs forensic inspection of URLs found within text.
    Returns: (Risk Score 0.0-1.0, Reason String)
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = url_pattern.findall(text)
    
    if not urls:
        return 0.0, "No URL found"
    
    raw_url = urls[0]
    resolved_url = expand_url(raw_url)
    
    # Precise Domain Parsing
    ext = tldextract.extract(resolved_url)
    registered_domain = f"{ext.domain}.{ext.suffix}".lower()
    
    risk_score = 0.0
    flags = []
    text_lower = text.lower()

    # --- FORENSIC RULE 1: Brand Consistency ---
    # This acts as a deterministic "Smoking Gun"
    consistency_rules = {
        'whatsapp': 'whatsapp.com', 
        'facebook': 'facebook.com',
        'paypal': 'paypal.com', 
        'amazon': 'amazon.com',
        'apple': 'apple.com', 
        'netflix': 'netflix.com',
        'hdfc': 'hdfcbank.com',
        'sbi': 'sbi.co.in',
        'icici': 'icicibank.com',
        'axis': 'axisbank.com',
        'chase': 'chase.com',
        'bank of america': 'bankofamerica.com'
    }

    for brand, legit_domain in consistency_rules.items():
        if brand in text_lower:
            if legit_domain != registered_domain:
                print(f"🚩 ALERT: Forensic Mismatch | Claims {brand}, leads to {registered_domain}")
                return 1.0, f"⚠️ Brand Mismatch: Claimed {brand.title()}, but link is '{registered_domain}'"

    # --- FORENSIC RULE 2: Anomalies ---
    # Check for raw IP address as domain
    if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ext.domain):
        flags.append("URL is a raw IP address")
        risk_score += 0.9

    # Suspicious TLDs
    suspicious_tlds = ['xyz', 'top', 'vip', 'info', 'site', 'win', 'online', 'club', 'co']
    if ext.suffix in suspicious_tlds:
        flags.append(f"Suspicious TLD (.{ext.suffix})")
        risk_score += 0.4

    # High Subdomain Density
    if resolved_url.count('.') > 4:
        flags.append("Excessive subdomains")
        risk_score += 0.3

    # Redirection check
    if raw_url != resolved_url:
        flags.append("Hidden redirect expanded")
        risk_score += 0.1

    final_score = min(risk_score, 1.0)
    reason = ", ".join(flags) if flags else "Standard URL Pattern"
    return final_score, reason

# ==========================================
# --- 4. CORE ENGINE: HYBRID FUSION ---
# ==========================================

def get_smishing_verdict(text):
    # 1. Forensic Pass
    url_score, url_reason = inspect_url_and_context(text)
    
    # 2. AI Inference
    # We use the augmented text for the LSTM as planned for your hybrid model
    augmented_text = text
    if url_score >= 1.0: augmented_text += " [TOKEN_URL_SPOOF]"
    elif url_score > 0.6: augmented_text += " [TOKEN_URL_SUSPICIOUS]"
    
    seq = tokenizer.texts_to_sequences([augmented_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    lstm_score = float(model.predict(padded, verbose=0)[0][0])

    # --- MODIFIED FUSION LOGIC ---
    if url_reason == "No URL found":
        # Case A: No link. We rely on AI but use a higher threshold (0.9) 
        # to avoid flagging "I'm in office" or "Call me" messages.
        final_score = lstm_score
        status = "DANGER" if final_score > 0.9 else "SAFE"
        final_reason = "Language Pattern Analysis"
    
    elif url_score >= 0.99:
        # Case B: Forensic "Smoking Gun" (Brand spoof)
        # We override the AI completely for technical certainty.
        final_score = 1.0
        status = "DANGER"
        final_reason = url_reason
        
    else:
        # Case C: Message HAS a link. Use your hybrid 60/40 fusion.
        # $$Final Score = (LSTM_{score} \times 0.6) + (URL_{score} \times 0.4)$$
        final_score = (lstm_score * 0.6) + (url_score * 0.4)
        status = "DANGER" if final_score > 0.5 else "SAFE"
        final_reason = url_reason if url_score > 0.4 else "Suspicious Language Patterns"

    # Terminal Trace for your debugging
    print(f"| Final Fusion: {final_score:.4f} -> {status} ({final_reason})")

    return {
        "status": status,
        "probability": f"{final_score*100:.2f}%",
        "reason": final_reason,
        "raw_score": final_score
    }

# ==========================================
# --- 5. LOGGING & SYSTEM AUDIT ---
# ==========================================

def log_unknown_message(text, score, label='?'):
    """
    Captures edge cases and errors to review_me.csv for Active Learning.
    """
    # Target "Unsure" cases (between 30% and 70%) or manual corrections
    is_unsure = 0.3 < score < 0.7
    is_report = label != '?'
    
    if is_unsure or is_report:
        try:
            file_path = LOG_FILE
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file_exists = os.path.isfile(file_path)
            
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['label', 'text'])
                writer.writerow([label, text])
            
            log_type = "ERROR_REPORT" if is_report else "SYSTEM_LOG"
            print(f"📝 [{log_type}] Logged potential sample to review_me.csv")
        except Exception as e:
            print(f"⚠️ Logger Warning: Failed to write to CSV: {e}")

# ==========================================
# --- 6. ROUTES: USER INTERFACE ---
# ==========================================

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_message():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text field provided in JSON'}), 400
        
        msg_text = data['text'].strip()
        if not msg_text:
            return jsonify({'error': 'Input text is empty'}), 400
        
        result = get_smishing_verdict(msg_text)
        log_unknown_message(msg_text, result['raw_score'])
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

@app.route('/report', methods=['POST'])
def report_error():
    """ Allows frontend users to correct the model's judgment. """
    try:
        data = request.get_json()
        text = data.get('text')
        label = data.get('label') # 0 for Safe, 1 for Scam
        
        if text and label is not None:
            log_unknown_message(text, 1.0, label)
            return jsonify({'status': 'success', 'msg': 'Reported for retraining.'})
        return jsonify({'error': 'Missing data'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# --- 7. ROUTES: DATASET MANAGEMENT ---
# ==========================================

@app.route('/batch_scan', methods=['GET'])
def batch_scan():
    """
    Rapidly scans the server-side test_data.txt file.
    Useful for quick developer sanity checks.
    """
    try:
        if not os.path.exists(BATCH_TEST_FILE):
            return jsonify({'error': f'Batch file {BATCH_TEST_FILE} not found on server.'}), 404
        
        results = []
        with open(BATCH_TEST_FILE, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            results.append(get_smishing_verdict(line))

        summary = {
            "total": len(results),
            "scam": len([r for r in results if r['status'] == 'DANGER']),
            "safe": len([r for r in results if r['status'] == 'SAFE']),
            "processed_at": datetime.now().isoformat()
        }
        return jsonify({"summary": summary, "details": results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_accuracy', methods=['GET'])
def test_accuracy():
    """
    Automated Benchmark Route.
    Iterates through testdata_new.csv and calculates accuracy metrics.
    """
    try:
        if not os.path.exists(ACCURACY_FILE):
            return jsonify({'error': f'Standard dataset {ACCURACY_FILE} missing.'}), 404
        
        correct_count = 0
        wrong_count = 0
        fp = 0 # False Positives
        fn = 0 # False Negatives
        failed_cases = []
        
        with open(ACCURACY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Skip Header
            
            for row in reader:
                if len(row) < 2: continue
                
                # Dynamic Label Normalization
                actual_label_raw = row[0].lower().strip()
                actual_label = 1 if actual_label_raw in ['spam', '1', 'smish', 'danger', 'malicious'] else 0
                message_text = row[1].strip()

                prediction = get_smishing_verdict(message_text)
                predicted_label = 1 if prediction['status'] == "DANGER" else 0
                
                if predicted_label == actual_label:
                    correct_count += 1
                else:
                    wrong_count += 1
                    err_type = "False Positive" if predicted_label == 1 else "False Negative"
                    if predicted_label == 1: fp += 1
                    else: fn += 1
                    
                    failed_cases.append({
                        'text': message_text,
                        'actual': "SCAM" if actual_label == 1 else "SAFE",
                        'pred': prediction['status'],
                        'reason': prediction['reason'],
                        'error_type': err_type
                    })
                    # Auto-log failures to the active learning dataset
                    log_unknown_message(message_text, 1.0, actual_label)

        total = correct_count + wrong_count
        acc_pct = (correct_count / total * 100) if total > 0 else 0
        
        print(f"📊 Accuracy Benchmark: {acc_pct:.2f}% across {total} samples.")
        
        return jsonify({
            'summary': {
                'accuracy': f"{acc_pct:.2f}%",
                'correct': correct_count,
                'wrong': wrong_count,
                'false_positives': fp,
                'false_negatives': fn,
                'total': total
            },
            'failures': failed_cases
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_custom_csv', methods=['GET'])
def test_custom_csv():
    """
    Pandas-powered route to handle large, complex custom datasets.
    Includes automated column identification and encoding fallbacks.
    """
    try:
        if not os.path.exists(CUSTOM_CSV_FILE):
            return jsonify({'error': 'custom.csv not found. Please place it in the backend folder.'}), 404

        # Robust Data Loading
        try:
            df = pd.read_csv(CUSTOM_CSV_FILE, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            print("⚠️ UTF-8 decoding failed. Switching to Latin-1 fallback.")
            df = pd.read_csv(CUSTOM_CSV_FILE, encoding='latin-1', on_bad_lines='skip')
        
        # Heuristic Column Finding
        # Looking for common names like 'Fulltext', 'message', 'url'
        target_col = None
        for col in df.columns:
            if col.lower() in ['fulltext', 'message', 'text', 'url', 'body']:
                target_col = col
                break
        
        if not target_col:
            return jsonify({'error': f'Could not find a text column in CSV headers: {df.columns.tolist()}'}), 400

        print(f"Processing custom dataset using column: '{target_col}'")
        
        results = []
        # We only process the first 1000 rows to prevent server timeout
        limit = min(len(df), 1000)
        for i in range(limit):
            msg = str(df.iloc[i][target_col]).strip()
            if msg:
                results.append(get_smishing_verdict(msg))

        return jsonify({
            "status": "success",
            "total_processed": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({'error': f"Pandas Processing Error: {str(e)}"}), 500

@app.route('/accuracy')
def accuracy_page():
    """ Renders the advanced system benchmark dashboard. """
    return render_template('accuracy.html')

if __name__ == '__main__':
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] SmishGuard API v2 is now LIVE.")
    print("Point your Android app or Frontend to:")
    print(" > http://127.0.0.1:5000")
    print(" > http://your-internal-ip:5000 (for testing on same WiFi)")
    
    # Debug=True ensures terminal logs every HTTP request automatically
    app.run(debug=True, host='0.0.0.0', port=5000)