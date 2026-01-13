import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import glob
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- CONFIGURATION ---
DATASET_PATH = '../dataset/'
MAX_WORDS = 20000       # Large vocabulary to cover both SMS slang and URL parts
MAX_LEN = 200           # URLs can be long, so we increase this
EMBEDDING_DIM = 100     # Dimension of the word vectors

def load_specific_file(filepath):
    filename = os.path.basename(filepath)
    df = None

    print(f"Processing: {filename} ...")

    try:
        # CASE 1: The standard Spam Dataset (needs Latin-1)
        if 'spam.csv' in filename:
            df = pd.read_csv(filepath, encoding='latin-1')
            # Usually v1=label, v2=text
            if 'v1' in df.columns:
                df = df.rename(columns={'v1': 'label', 'v2': 'text'})

        # CASE 2: The Collection Files (Txt/Tab-separated, No Header)
        elif 'Collection' in filename or filename.endswith('.txt'):
            try:
                df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'text'])
            except:
                # Fallback if it's not actually tab separated
                df = pd.read_csv(filepath, on_bad_lines='skip')

        # CASE 3: Malicious Phish (url, type)
        elif 'malicious_phish' in filename:
            df = pd.read_csv(filepath)
            df = df.rename(columns={'url': 'text', 'type': 'label'})

        # CASE 4: PhiUSIIL (Complex dataset)
        elif 'PhiUSIIL' in filename:
            df = pd.read_csv(filepath)
            df = df.rename(columns={'URL': 'text'}) # Label column already exists
        
        # CASE 5: Generic / Review CSVs
        else:
            df = pd.read_csv(filepath)
            # Generic cleanup
            if 'URL' in df.columns: df.rename(columns={'URL': 'text'}, inplace=True)
            if 'message' in df.columns: df.rename(columns={'message': 'text'}, inplace=True)

        # FINAL CHECK: Ensure we have text and label
        if df is not None:
            # Ensure columns exist
            if 'text' not in df.columns or 'label' not in df.columns:
                print(f"  -> Skipping {filename}: Missing 'text' or 'label' columns.")
                return None
            
            # Keep only what we need
            return df[['text', 'label']]
            
    except Exception as e:
        print(f"  -> Error loading {filename}: {e}")
        return None

    return None

# --- 1. LOAD DATA ---
all_files = glob.glob(os.path.join(DATASET_PATH, "*"))
df_list = []

for filepath in all_files:
    df = load_specific_file(filepath)
    if df is not None:
        # Clean 'text' column just in case
        df['text'] = df['text'].astype(str)
        df_list.append(df)
        print(f"  -> Successfully loaded {len(df)} rows.")

if not df_list:
    print("CRITICAL ERROR: No data loaded.")
    exit()

data = pd.concat(df_list, ignore_index=True)
print(f"\nTotal Raw Rows: {len(data)}")

# --- 2. CLEAN LABELS ---
# Map everything to integers. Drop unknown labels like '?'
label_map = {
    'ham': 0, 'safe': 0, 'benign': 0, '0': 0, 0: 0,
    'spam': 1, 'smish': 1, 'phishing': 1, 'malicious': 1, '1': 1, 1: 1
}

data['label'] = data['label'].map(label_map)
data.dropna(subset=['label'], inplace=True) # Drop '?' or errors
data['label'] = data['label'].astype(int)

# --- 3. BALANCE DATA ---
# This ensures the model doesn't just guess "Safe" because most data is Safe
spam_df = data[data['label'] == 1]
ham_df = data[data['label'] == 0]

print(f"Safe Count: {len(ham_df)}")
print(f"Scam Count: {len(spam_df)}")

# Balance them 50/50
min_len = min(len(ham_df), len(spam_df))
ham_downsampled = ham_df.sample(n=min_len, random_state=42)
spam_downsampled = spam_df.sample(n=min_len, random_state=42)

balanced_data = pd.concat([ham_downsampled, spam_downsampled])
print(f"Training on {len(balanced_data)} balanced rows.")

# --- 4. TOKENIZATION ---
texts = balanced_data['text'].tolist()
labels = balanced_data['label'].values

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Save Tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# --- 5. BUILD MODEL ---
# Using Bidirectional LSTM for better context understanding
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- 6. TRAIN WITH CALLBACKS ---
# EarlyStopping: Stop if not improving
# ModelCheckpoint: Save the BEST model, not just the last one
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('smishing_model.keras', save_best_only=True, monitor='val_accuracy')
]

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

print("Training Complete. Best model saved as 'smishing_model.keras'")