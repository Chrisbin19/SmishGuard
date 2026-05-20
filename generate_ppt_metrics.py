import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import asyncio
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import main_o2

CSV_PATH = "true_holdout_test.csv"

async def evaluate_text(text):
    req = main_o2.SMSRequest(text=text)
    return await main_o2.predict(req)

def run():
    print(f"Loading Test Data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        df['actual_numeric'] = df['label'].map({'spam': 1, 'ham': 0})
        df.dropna(subset=['actual_numeric'], inplace=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Evaluating {len(df)} messages...")
    
    true_labels = []
    pred_labels = []
    pred_probs = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        true_labels.append(row['actual_numeric'])
        
        # Async call
        res = asyncio.run(evaluate_text(row['text']))
        
        if 'error' in res:
            # Fallback
            pred_labels.append(0)
            pred_probs.append(0.0)
            continue
            
        pred = 1 if res['is_phishing'] else 0
        pred_labels.append(pred)
        
        # Extract float probability from final_risk_score like "45.00%"
        prob_str = res.get('final_risk_score', '0%').replace('%', '')
        try:
            prob = float(prob_str) / 100.0
        except:
            prob = 0.0
        pred_probs.append(prob)

    # Calculate metrics
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    print("\n" + "="*50)
    print("📊 MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50)
    
    # --- 1. Plot Confusion Matrix ---
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham (Safe)', 'Spam (Phishing)'], 
                yticklabels=['Ham (Safe)', 'Spam (Phishing)'],
                annot_kws={"size": 14})
    plt.title('SmishGuard Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_ppt.png', dpi=300)
    print("✅ Saved Confusion Matrix to 'confusion_matrix_ppt.png'")
    plt.close()

    # --- 2. Plot ROC Curve ---
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_ppt.png', dpi=300)
    print("✅ Saved ROC Curve to 'roc_curve_ppt.png'")
    plt.close()

if __name__ == "__main__":
    run()
