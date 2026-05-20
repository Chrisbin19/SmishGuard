import pandas as pd

SEED = 42

print("Loading combined_smishing.csv...")
df = pd.read_csv('combined_smishing.csv', encoding='latin-1')
df = df[['label', 'text']]
df.dropna(inplace=True)
df['text']  = df['text'].astype(str)
df['label'] = df['label'].str.lower().str.strip()
df = df[df['text'].str.strip().astype(bool)]
df = df[df['label'].isin(['spam', 'ham'])]

# Step 1: Shuffle exactly as done in training
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Extract the 80% training set (to find its indices)
train_df = df.sample(frac=0.8, random_state=SEED)

# Step 3: Extract the 20% test set
test_df = df.drop(train_df.index)

print(f"Total valid samples: {len(df)}")
print(f"Training samples (used to train model): {len(train_df)}")
print(f"Test samples (completely unseen): {len(test_df)}")

# Save to CSV
test_df.to_csv('true_holdout_test.csv', index=False)
print("Saved completely unseen test data to 'true_holdout_test.csv'")
