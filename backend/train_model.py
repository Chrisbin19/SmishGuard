import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import glob
import os
import re
import tldextract
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- CONFIGURATION ---
DATASET_PATH = '../dataset/'
MAX_WORDS = 25000       # Increased to accommodate new forensic tokens
MAX_LEN = 200 
EMBEDDING_DIM = 100 

# --- NEW: FORENSIC AUGMENTATION HELPER ---
def augment_training_text(text):
    """
    Analyzes URLs in training data and appends special tokens.
    This teaches the LSTM to recognize technical red flags.
    """
    text_str = str(text).lower()
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = url_pattern.findall(text_str)
    
    if not urls:
        return text_str
    
    url = urls[0]
    ext = tldextract.extract(url)
    reg_domain = f"{ext.domain}.{ext.suffix}"
    
    tokens = []
    # Brand Spoof Check (The "Offline Detective")
    brands = {
        'whatsapp': 'whatsapp.com', 'facebook': 'facebook.com', 
        'paypal': 'paypal.com', 'amazon': 'amazon.com',
        'apple': 'apple.com', 'netflix': 'netflix.com'
    }
    for brand, legit in brands.items():
        if brand in text_str and legit != reg_domain:
            tokens.append("[TOKEN_URL_SPOOF]")
            break
            
    # Technical Red Flags
    if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ext.domain):
        tokens.append("[TOKEN_URL_IP]")
    if ext.suffix in ['xyz', 'top', 'vip', 'info']:
        tokens.append("[TOKEN_URL_SUSPICIOUS]")
        
    return f"{text_str} {' '.join(tokens)}"

def load_specific_file(filepath):
    filename = os.path.basename(filepath)
    df = None
    print(f"Processing: {filename} ...")

    try:
        # CASE 1: Standard Spam (Latin-1)
        if 'spam.csv' in filename:
            df = pd.read_csv(filepath, encoding='latin-1')
            if 'v1' in df.columns:
                df = df.rename(columns={'v1': 'label', 'v2': 'text'})

        # CASE 2: Collection/Txt Files
        elif 'Collection' in filename or filename.endswith('.txt'):
            try:
                df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'text'])
            except:
                df = pd.read_csv(filepath, on_bad_lines='skip')

        # CASE 3: Malicious Phish
        elif 'malicious_phish' in filename:
            df = pd.read_csv(filepath)
            df = df.rename(columns={'url': 'text', 'type': 'label'})

        # CASE 4: PhiUSIIL
        elif 'PhiUSIIL' in filename:
            df = pd.read_csv(filepath)
            df = df.rename(columns={'URL': 'text'})

        # CASE 5: Top 1M Benign URLs (Force Label 0)
        elif 'top-1m' in filename:
            df = pd.read_csv(filepath, header=None, names=['rank', 'text'])
            df['label'] = 0

        # CASE 6: Review Me (Your Manual Corrections)
        elif 'review_me' in filename:
            df = pd.read_csv(filepath)
            # OVER-SAMPLING: Multiply rows to ensure the model prioritizes these fixes
            df = pd.concat([df] * 10, ignore_index=True) 

        else:
            df = pd.read_csv(filepath)
            if 'URL' in df.columns: df.rename(columns={'URL': 'text'}, inplace=True)
            if 'message' in df.columns: df.rename(columns={'message': 'text'}, inplace=True)

        if df is not None:
            return df[['text', 'label']]
            
    except Exception as e:
        print(f"  -> Error loading {filename}: {e}")
        return None

# --- 1. LOAD & NORMALIZE ---
all_files = glob.glob(os.path.join(DATASET_PATH, "*"))
df_list = []

for filepath in all_files:
    df = load_specific_file(filepath)
    if df is not None:
        df_list.append(df)

data = pd.concat(df_list, ignore_index=True)

# Normalize Labels (Handling ham, spam, benign, etc.)
label_map = {
    'ham': 0, 'safe': 0, 'benign': 0, '0': 0, 0: 0,
    'spam': 1, 'smish': 1, 'phishing': 1, 'malicious': 1, '1': 1, 1: 1
}
data['label'] = data['label'].map(label_map)
data.dropna(subset=['label', 'text'], inplace=True)
data['label'] = data['label'].astype(int)

# --- 2. DYNAMIC FORENSIC AUGMENTATION ---
print("Augmenting text with forensic tokens (Dynamic NLP Step)...")
data['text'] = data['text'].apply(augment_training_text)

# --- 3. BALANCE DATA ---
spam_df = data[data['label'] == 1]
ham_df = data[data['label'] == 0]
min_len = min(len(ham_df), len(spam_df))
balanced_data = pd.concat([ham_df.sample(n=min_len, random_state=42), 
                           spam_df.sample(n=min_len, random_state=42)])

# --- 4. TOKENIZATION ---
texts = balanced_data['text'].tolist()
labels = balanced_data['label'].values

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# --- 5. BUILD MODEL ---
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- 6. TRAIN ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('smishing_model.keras', save_best_only=True, monitor='val_accuracy')
]

model.fit(X_train, y_train, epochs=10, batch_size=64, 
          validation_data=(X_test, y_test), callbacks=callbacks)

print("✅ Hybrid Model Training Complete! Best model saved as 'smishing_model.keras'")