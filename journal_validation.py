import pandas as pd
import requests
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = "testdata_new.csv" 

def run_benchmark():
    print("1. Loading Unseen Test Data (Smart Mode)...")
    
    # --- SMART LOADER FIX ---
    # Instead of pd.read_csv (which crashes on extra commas), 
    # we manually read the file and split ONLY on the first comma.
    data = []
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue # Skip empty lines
                
                # Split the line into exactly 2 parts: Label, and The Rest
                parts = line.split(',', 1) 
                
                if len(parts) == 2:
                    label = parts[0].strip()
                    text = parts[1].strip()
                    
                    # Remove surrounding quotes if they exist
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                        
                    data.append({'label': label, 'text': text})
                    
        df = pd.DataFrame(data)
        
        # Convert labels to numbers (spam=1, ham=0)
        df['actual_numeric'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Drop rows where mapping failed (e.g. headers)
        df.dropna(subset=['actual_numeric'], inplace=True)
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    # ------------------------

    print(f"🚀 Starting Benchmark on {len(df)} unseen messages...")
    
    predictions = []
    true_labels = []
    errors = 0
    start_time = time.time()

    # Loop through the dataset
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Hit the API
            response = requests.post(API_URL, json={"text": row['text']})
            
            if response.status_code == 200:
                result = response.json()
                
                # Store Real Answer
                true_labels.append(row['actual_numeric'])
                
                # Store Model Answer (Convert True/False to 1/0)
                pred = 1 if result['is_phishing'] else 0
                predictions.append(pred)
            else:
                errors += 1
                
        except Exception:
            errors += 1

    total_time = time.time() - start_time
    
    # --- GENERATE METRICS ---
    if len(predictions) == 0:
        print("Test Failed. Is server running?")
        return

    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)

    print("\n" + "="*60)
    print(f"📄 JOURNAL PUBLICATION METRICS")
    print("="*60)
    print(f"✅ Accuracy:      {acc * 100:.2f}%")
    print(f"🎯 Precision:     {prec * 100:.2f}%")
    print(f"🔍 Recall:        {rec * 100:.2f}%")
    print(f"⚖️  F1-Score:      {f1 * 100:.2f}%")
    print("-" * 60)
    print("CONFUSION MATRIX:")
    try:
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negatives (Safe correctly identified): {tn}")
        print(f"False Positives (Safe marked as Danger):    {fp}")
        print(f"False Negatives (Spam missed):              {fn}")
        print(f"True Positives (Spam correctly identified): {tp}")
    except:
        print("Matrix shape mismatch (not enough data classes)")
    print("="*60)
    print(f"⏱️ Total Time: {total_time:.2f}s | Avg Latency: {(total_time/len(df))*1000:.1f}ms")

if __name__ == "__main__":
    run_benchmark() 