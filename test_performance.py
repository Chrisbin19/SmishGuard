import pandas as pd
import requests
import time
from tqdm import tqdm

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = "real_world_test.csv"
SAMPLE_SIZE = 200 

def run_stress_test():
    print("1. Loading and Cleaning Data...")
    try:
        # Load specific columns
        df = pd.read_csv(CSV_PATH, encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'text']
        
        # CLEAN DATA: Remove empty rows and force text format
        df.dropna(inplace=True)
        df['text'] = df['text'].astype(str) 
        df = df[df['text'].str.strip().astype(bool)]
        
        # Take sample
        real_n = min(SAMPLE_SIZE, len(df))
        test_data = df.sample(n=real_n)
        
    except Exception as e:
        print(f"CRITICAL ERROR loading CSV: {e}")
        return

    correct_predictions = 0
    errors = 0
    total_time = 0

    print(f"\n🚀 Testing {SAMPLE_SIZE} messages against Localhost API...")
    
    # Run the loop
    for index, row in tqdm(test_data.iterrows(), total=SAMPLE_SIZE):
        actual_label = row['label'] # 'ham' or 'spam'
        text_content = row['text']
        
        start_time = time.time()
        try:
            # Send Request
            response = requests.post(API_URL, json={"text": text_content})
            
            # Check if API accepted it
            if response.status_code != 200:
                errors += 1
                continue

            result = response.json()
            
            # Measure time
            total_time += (time.time() - start_time)
            
            # --- THE FIX IS HERE ---
            # Your server returns "is_phishing": True/False
            # So we use that directly.
            predicted_is_spam = result['is_phishing']
            
            # The CSV says "spam", so we check if they match
            actually_is_spam = (actual_label == "spam")
            
            if predicted_is_spam == actually_is_spam:
                correct_predictions += 1
                
        except Exception as e:
            print(f"Connection Failed: {e}")
            errors += 1

    # Report
    valid_attempts = SAMPLE_SIZE - errors
    if valid_attempts == 0:
        print("\n❌ All requests failed. Is the server running?")
        return

    accuracy = (correct_predictions / valid_attempts) * 100
    avg_latency = (total_time / valid_attempts) * 1000

    print("\n" + "="*50)
    print(f"📊 REAL PERFORMANCE REPORT")
    print("="*50)
    print(f"✅ Accuracy:      {accuracy:.2f}%")
    print(f"⚡ Avg Latency:   {avg_latency:.2f} ms")
    print(f"❌ Errors:        {errors}")
    print("="*50)

if __name__ == "__main__":
    run_stress_test()