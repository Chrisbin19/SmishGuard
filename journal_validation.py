import pandas as pd
import requests
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = "journal_test_set.csv" # We use the file we just created

def run_benchmark():
    print("1. Loading Unseen Test Data...")
    try:
        df = pd.read_csv(CSV_PATH)
        # Convert labels to numbers (spam=1, ham=0) for math
        df['actual_numeric'] = df['label'].map({'spam': 1, 'ham': 0})
    except Exception as e:
        print(f"Error: {e}")
        return

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
    prec = precision_score(true_labels, predictions)
    rec = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)

    print("\n" + "="*60)
    print(f"📄 JOURNAL PUBLICATION METRICS")
    print("="*60)
    print(f"✅ Accuracy:      {acc * 100:.2f}%  (Overall correctness)")
    print(f"🎯 Precision:     {prec * 100:.2f}%  (Avoids false alarms)")
    print(f"🔍 Recall:        {rec * 100:.2f}%  (Catches hidden spam)")
    print(f"⚖️  F1-Score:      {f1 * 100:.2f}%  (The 'Real' Efficiency Score)")
    print("-" * 60)
    print("CONFUSION MATRIX:")
    print(f"True Negatives (Safe correctly identified): {cm[0][0]}")
    print(f"False Positives (Safe marked as Danger):    {cm[0][1]}")
    print(f"False Negatives (Spam missed):              {cm[1][0]}")
    print(f"True Positives (Spam correctly identified): {cm[1][1]}")
    print("="*60)
    print(f"⏱️ Total Time: {total_time:.2f}s | Avg Latency: {(total_time/len(df))*1000:.1f}ms")

if __name__ == "__main__":
    run_benchmark()