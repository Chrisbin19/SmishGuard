import requests
import pandas as pd
import time
from colorama import Fore, Style, init

# Initialize colors for pretty printing
init(autoreset=True)

API_URL = "http://127.0.0.1:8000/predict"

# --- THE MODERN THREAT DATABASE ---
# These are attacks designed to BREAK standard AI models.
test_cases = [
    # CATEGORY 1: HOMOGLYPH ATTACKS (Visual Spoofing)
    # The 'a' in PαyPal is Greek alpha. The 'o' in Am@zon is weird.
    {"text": "URGENT: Your PαyPal account is limited. Restore here: http://secure-pp-verify.com", "type": "Homoglyph"},
    {"text": "Security Alert: Am@zon login detected from Russia. Verify: http://amzn-security-check.xyz", "type": "Homoglyph"},
    
    # CATEGORY 2: SEGMENTATION & NOISE (Evasion)
    # Breaking words to bypass keyword filters.
    {"text": "D.e.a.r C.u.s.t.o.m.e.r, your N.e.t.f.l.i.x is on hold. Pay: http://netflix-billing-update.com", "type": "Segmentation"},
    {"text": "W-A-R-N-I-N-G: Your C.i.t.i.B.a.n.k card is blocked. Action required: http://citi-confirm.net", "type": "Segmentation"},

    # CATEGORY 3: ZERO-DAY BRANDS (New companies not in 2015 dataset)
    # Your 2015 AI has NEVER seen "Disney+" or "Binance".
    {"text": "Your Disney+ subscription has expired. Renew membership: http://disney-renewals.com", "type": "New Gen Brand"},
    {"text": "Binance Withdrawal Alert: 0.5 BTC requested. Cancel transaction: http://binance-fraud-control.com", "type": "New Gen Brand"},
    
    # CATEGORY 4: SENTIMENT INVERSION (The "Good Guy" Trap)
    # AI models equate "Polite/Happy" with "Safe".
    {"text": "We are happy to inform you that your refund has been approved! Great news! Click to accept: http://irs-refund-claim.xyz", "type": "Positive Sentiment"},
    {"text": "Thank you for being a loyal customer. Here is a free gift card for you: http://amazon-gifts-free.top", "type": "Positive Sentiment"},
    
    # CATEGORY 5: THE ULTIMATE DECEPTION (Semantic Mismatch)
    # Text says one thing, link says another.
    {"text": "Please login to your Microsoft Office account to view the document: http://google-drive-share.com", "type": "Semantic Mismatch"},
]

def run_stress_test():
    print(Fore.CYAN + "="*60)
    print(Fore.CYAN + "🛡️  STARTING ADVERSARIAL ROBUSTNESS STRESS TEST")
    print(Fore.CYAN + "="*60 + "\n")
    
    score_card = {"Pass": 0, "Fail": 0}
    results = []

    for i, case in enumerate(test_cases):
        print(f"Testing Case {i+1} [{case['type']}]:")
        print(Fore.YELLOW + f"📝 Input: {case['text']}")
        
        start_time = time.time()
        try:
            response = requests.post(API_URL, json={"text": case['text']})
            data = response.json()
            latency = (time.time() - start_time) * 1000
            
            # We EXPECT all these to be flagged as PHISHING (True)
            prediction = data['is_phishing']
            reason = data.get('logic_mode', 'Unknown')
            entities = data.get('entities_detected', [])
            
            if prediction == True:
                print(Fore.GREEN + f"✅ BLOCKED | Reason: {reason} | Latency: {latency:.1f}ms")
                print(Fore.GREEN + f"   -> Entities Found: {entities}")
                score_card["Pass"] += 1
                result = "Pass"
            else:
                print(Fore.RED + f"❌ FAILED | The system thought this was Safe.")
                print(Fore.RED + f"   -> AI Score: {data.get('ai_score')} | Forensic: {data.get('forensic_score')}")
                score_card["Fail"] += 1
                result = "Fail"
                
        except Exception as e:
            print(Fore.RED + f"❌ ERROR: {e}")
            result = "Error"

        results.append({"type": case['type'], "result": result})
        print("-" * 50)

    # FINAL REPORT
    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.CYAN + "📊 FINAL ROBUSTNESS REPORT")
    print(Fore.CYAN + "="*60)
    total = len(test_cases)
    accuracy = (score_card["Pass"] / total) * 100
    
    print(f"Total Attacks: {total}")
    print(Fore.GREEN + f"Attacks Blocked: {score_card['Pass']}")
    print(Fore.RED + f"Attacks Missed:  {score_card['Fail']}")
    print(Fore.YELLOW + f"🛡️  ADVERSARIAL ROBUSTNESS SCORE: {accuracy:.2f}%")
    
    if accuracy > 85:
        print(Fore.GREEN + "\n🏆 CONCLUSION: Journal-Grade Robustness Achieved.")
        print("Your Neuro-Symbolic Logic successfully covers the gaps of the old dataset.")
    else:
        print(Fore.RED + "\n⚠️ CONCLUSION: System needs tuning.")

if __name__ == "__main__":
    run_stress_test()