import pandas as pd
import os
import glob

# Path to your dataset folder
DATASET_PATH = '../dataset/'

print(f"--- INSPECTING DATASETS IN: {DATASET_PATH} ---\n")

# Get all files
files = glob.glob(os.path.join(DATASET_PATH, "*"))

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"FILE: {filename}")
    
    try:
        # Try reading as standard CSV first
        df = pd.read_csv(filepath, nrows=3)
        print(f"  [CSV Read] Columns found: {list(df.columns)}")
        print(f"  [Sample Row]: {df.values[0] if len(df) > 0 else 'Empty'}\n")
        
    except:
        try:
            # If that fails, try as TSV (Tab Separated) - common for SMS collections
            df = pd.read_csv(filepath, sep='\t', nrows=3)
            print(f"  [TSV Read] Columns found: {list(df.columns)}")
            print(f"  [Sample Row]: {df.values[0] if len(df) > 0 else 'Empty'}\n")
        except Exception as e:
            print(f"  [!] Could not read file. Error: {e}\n")

print("--- END OF INSPECTION ---")