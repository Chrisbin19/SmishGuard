import pandas as pd

# 1. Load the original full dataset
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except:
    print("❌ Error: Could not find 'spam.csv'. Make sure it is in this folder.")
    exit()

# 2. Clean it exactly like we did before
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df.dropna(inplace=True)
df['text'] = df['text'].astype(str)
df = df[df['text'].str.strip().astype(bool)]

# 3. THE MAGIC TRICK: Replicate the split
# We use random_state=42, just like your training script.
# This isolates the Training Data...
train_df = df.sample(frac=0.8, random_state=42)

# ...so we can DROP it and keep only the Test Data
test_df = df.drop(train_df.index)

# 4. Save this "Unseen" data to a new file
print(f"Original Dataset: {len(df)} rows")
print(f"Training Data (Ignored): {len(train_df)} rows")
print(f"Unseen Test Data (Saved): {len(test_df)} rows")

test_df.to_csv('journal_test_set.csv', index=False)
print("✅ Success! Created 'journal_test_set.csv'. Use this for validation.")