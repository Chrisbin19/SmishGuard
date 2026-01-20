import requests

API_URL = "http://127.0.0.1:8000/predict"

test_cases = [
    # 1. Obvious SAFE message
    {"text": "Hey mom, I will be home for dinner.", "expected": "SAFE"},
    
    # 2. Obvious DANGEROUS message (Link)
    {"text": "URGENT: Your account is blocked. Click http://bit.ly/scam now.", "expected": "DANGEROUS"},
    
    # 3. Obvious DANGEROUS message (Keywords)
    {"text": "Congratulations! You won $1000. Call now to claim.", "expected": "DANGEROUS"}
]

print("--- 🧠 BRAIN DIAGNOSTIC ---")
for case in test_cases:
    response = requests.post(API_URL, json={"text": case["text"]})
    result = response.json()
    
    # Get the status
    if result['is_phishing']:
        status = "DANGEROUS"
    else:
        status = "SAFE"
        
    # Compare
    is_correct = (status == case["expected"])
    icon = "✅" if is_correct else "❌"
    
    print(f"\nInput:    {case['text']}")
    print(f"Expected: {case['expected']}")
    print(f"AI Says:  {status} (Risk: {result['final_risk_score']})")
    print(f"Result:   {icon}")

print("\n---------------------------")