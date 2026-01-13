import numpy as np
import tensorflow as tf
import pickle
import os
import csv  # <--- NEW: Needed for logging
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- SETUP FLASK TO FIND THE HTML ---
app = Flask(__name__, 
            template_folder='../frontend/templates', 
            static_folder='../frontend/static')

# --- CONFIGURATION ---
MAX_LEN = 100 
MODEL_PATH = 'smishing_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'
LOG_FILE = '../dataset/review_me.csv' # <--- NEW: Where we save tricky messages

# --- LOAD RESOURCES ---
print("Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading Tokenizer...")
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# --- HELPER FUNCTION: LOGGING ---
def log_unknown_message(text, score):
    """
    Saves the message if the model is unsure (0.3 < score < 0.7)
    or if it's a very strong scam (score > 0.9) to build our dataset.
    """
    # 1. Define conditions for saving
    is_unsure = 0.3 < score < 0.7
    is_definite_scam = score > 0.9
    
    if is_unsure or is_definite_scam:
        try:
            # Check if file exists to write header
            file_exists = os.path.isfile(LOG_FILE)
            
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # If file is new, add the header columns
                if not file_exists:
                    writer.writerow(['label', 'text']) 
                
                # We log it with '?' as the label so you can manually check it later
                # Or 'spam' if it's super obvious, but '?' is safer for review
                label = '?' 
                writer.writerow([label, text])
                
            print(f"📝 Logged message for review (Score: {score:.4f})")
        except Exception as e:
            print(f"⚠️ Failed to log message: {e}")

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_message():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        sms_text = data['text']
        
        # --- DEBUGGING START ---
        print(f"Original Text: {sms_text}")
        seq = tokenizer.texts_to_sequences([sms_text])
        # print(f"Converted Numbers: {seq}") # Uncomment if needed
        # --- DEBUGGING END ---

        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Predict
        prediction = model.predict(padded)
        score = float(prediction[0][0])

        # --- NEW STEP: Log for Continuous Learning ---
        log_unknown_message(sms_text, score)

        # Verdict Logic
        verdict = "DANGER" if score > 0.5 else "SAFE"
        
        return jsonify({
            "status": verdict,
            "probability": f"{score*100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)