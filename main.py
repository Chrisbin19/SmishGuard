from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import re # <-- New Tool: Regular Expressions (The Pattern Finder)

app = FastAPI()

# 1. Allow Frontend Connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Brain
print("Loading AI Model...")
model = tf.keras.models.load_model('smishguard_model.keras', compile=False)
print("Model Loaded!")

class SMSRequest(BaseModel):
    text: str

# --- NEW FEATURE: The URL Detective ---
def analyze_links(text):
    # Step 1: Find links using Regex (looks for http:// or https://)
    urls = re.findall(r'https?://\S+', text)
    
    if not urls:
        return {"has_links": False, "risk_score": 0, "details": "No links found"}

    risk_score = 0
    details = []

    # Step 2: Check each link for danger signs
    suspicious_tlds = ['.xyz', '.top', '.club', '.info', '.ru', '.cn']
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd']

    for url in urls:
        # Check A: Is it an IP address? (e.g., http://192.168.1.1) -> VERY DANGEROUS
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            risk_score += 50
            details.append(f"IP Address URL detected: {url}")

        # Check B: Is it a URL Shortener? (Hides the real destination) -> SUSPICIOUS
        if any(short in url for short in shorteners):
            risk_score += 30
            details.append(f"URL Shortener detected: {url}")

        # Check C: Is it a cheap/spammy domain? (.xyz, .top) -> SUSPICIOUS
        if any(tld in url for tld in suspicious_tlds):
            risk_score += 20
            details.append(f"Suspicious Domain detected: {url}")

    return {
        "has_links": True, 
        "risk_score": risk_score, 
        "details": "; ".join(details) if details else "Links appear standard"
    }
# --------------------------------------

@app.post("/predict")
async def predict_sms(request: SMSRequest):
    try:
        # 1. AI PREDICTION (The Brain)
        tensor_text = tf.constant([request.text])
        ai_score = float(model.predict(tensor_text)[0][0]) * 100 # Convert to 0-100 scale
        
        # 2. LINK ANALYSIS (The Detective)
        link_analysis = analyze_links(request.text)
        
        # 3. COMBINE SCORES
        # We start with the AI score, but if the links are bad, we add to the risk.
        final_risk = ai_score + link_analysis['risk_score']
        
        # Cap the score at 100%
        if final_risk > 100:
            final_risk = 100
            
        is_phishing = final_risk > 50

        return {
            "is_phishing": is_phishing,
            "final_risk_score": f"{final_risk:.2f}%",
            "ai_score": f"{ai_score:.2f}%",
            "link_warnings": link_analysis['details'],
            "message": "CAUTION: Phishing Detected!" if is_phishing else "Message seems Safe."
        }
        
    except Exception as e:
        return {"error": str(e)}