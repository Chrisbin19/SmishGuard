import os
# 1. FORCE UTF-8 (Prevents Windows Crash)
os.environ["PYTHONUTF8"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import re
import math
from urllib.parse import urlparse
import difflib 
import spacy # <-- The New Dynamic Brain

app = FastAPI()

# --- CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD PIPELINE A: The Deep Learning Brain (Bi-LSTM)
print("🧠 Loading Neuro-Symbolic Core...")
try:
    model = tf.keras.models.load_model('smishguard_model.keras', compile=False)
    print("✅ Pipeline A (Bi-LSTM) Active.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Model not found. {e}")
    model = None

# 3. LOAD PIPELINE B: The Linguistic Brain (spaCy NER)
print("🗣️ Loading Dynamic Forensic Core...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ Pipeline B (Dynamic NER) Active.")
except:
    print("❌ ERROR: spaCy model not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

class SMSRequest(BaseModel):
    text: str

# ==============================================================================
#   COMPONENT 1: THE DYNAMIC FORENSIC AGENT (Pipeline B)
# ==============================================================================
class DynamicForensicAgent:
    def __init__(self):
        # We only keep TLDs (infrastructure), NO hardcoded brands.
        self.suspicious_tlds = ['.xyz', '.top', '.club', '.info', '.ru', '.cn', '.tk', '.cam', '.work', '.net']

    def calculate_entropy(self, text):
        """Math to detect random gibberish like 'a8z9-q2.com'"""
        if not text: return 0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy

    def extract_organizations(self, text):
        """
        Dynamically finds companies/brands in text using NLP.
        Example: "Your Disney+ account is locked" -> Extracts "Disney+"
        """
        if not nlp: return []
        doc = nlp(text)
        # Extract entities labeled as ORG (Organization)
        orgs = [ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"]
        
        # Fallback: Regex for capitalized words (Robustness for new brands)
        if not orgs:
            # Finds capitalized words that aren't at the start of a sentence
            orgs = re.findall(r'(?<!^)\b[A-Z][a-zA-Z0-9]+\b', text)
            orgs = [o.lower() for o in orgs]
            
        return list(set(orgs))

    def analyze(self, text, url):
        risk_score = 0
        logs = []
        is_critical = False 
        
        # 1. Parse Domain
        try:
            domain = urlparse(url).netloc.lower()
            if not domain: domain = url
            clean_domain = domain.replace("www.", "")
            domain_body = clean_domain.split('.')[0]
        except:
            domain = url
            domain_body = url

        # --- A. INFRASTRUCTURE CHECKS (Universal) ---
        
        # Check 1: IP Address (Instant Block)
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
            return 100, [f"🚨 CRITICAL: Host is an IP Address ({domain})"], True

        # Check 2: High Entropy (Gibberish)
        entropy = self.calculate_entropy(domain)
        if entropy > 3.9:
            risk_score += 40
            logs.append(f"⚠️ High Entropy URL (Randomness {entropy:.2f})")

        # --- B. DYNAMIC SEMANTIC CONSISTENCY (The Innovation) ---
        
        # Step 1: Who does the text CLAIM to be?
        claimed_orgs = self.extract_organizations(text)
        
        # Step 2: Does the Link Match the Claim?
        if claimed_orgs:
            match_found = False
            for org in claimed_orgs:
                # Fuzzy Logic: Is the Org name inside the domain?
                # Check 1: Substring match (e.g. "netflix" in "netflix-verify.com")
                if org in clean_domain:
                    match_found = True
                
                # Check 2: Fuzzy Logic (Levenshtein Distance)
                # Catches "Pay-Pal" vs "PayPal"
                elif difflib.SequenceMatcher(None, org, domain_body).ratio() > 0.70:
                    match_found = True
            
            # THE VERDICT
            if not match_found:
                # If text claims a Brand, but URL has NO relation to it -> Deception
                risk_score += 80 # Massive penalty
                is_critical = True
                logs.append(f"🎣 DYNAMIC MISMATCH: Text claims to be '{claimed_orgs[0].title()}' but URL is '{domain}'")
            else:
                # If they DO match, we check the TLD just in case
                if any(tld in domain for tld in self.suspicious_tlds):
                    risk_score += 50
                    logs.append(f"⚠️ Brand detected but TLD is suspicious ({domain})")

        # Check 3: Suspicious TLD (Fallback)
        if any(tld in domain for tld in self.suspicious_tlds) and risk_score < 50:
            risk_score += 40
            logs.append(f"🚩 Suspicious Top-Level Domain")

        return min(risk_score, 100), logs, is_critical

# Initialize Dynamic Agent
forensic_agent = DynamicForensicAgent()

# ==============================================================================
#   COMPONENT 2: SANITIZATION
# ==============================================================================
def sanitize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\$\!\.\?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
#   COMPONENT 3: FUSION ENDPOINT
# ==============================================================================
# ==============================================================================
#   COMPONENT 3: THE FIXED "SMART TRUST" ENDPOINT
# ==============================================================================
@app.post("/predict")
async def predict_sms(request: SMSRequest):
    try:
        # 1. AI PREDICTION (Pipeline A)
        clean_text = sanitize_text(request.text)
        try:
            if model:
                tensor_text = tf.constant([clean_text])
                ai_probability = float(model.predict(tensor_text)[0][0]) * 100
            else:
                ai_probability = 0
        except:
            ai_probability = 0
            
        # 2. DYNAMIC LOGIC (Pipeline B)
        forensic_risk = 0
        forensic_logs = []
        is_critical = False
        detected_entities = [] # Capture entities for the response
        
        urls = re.findall(r'https?://\S+', request.text)
        if urls:
            risk, logs, critical = forensic_agent.analyze(request.text, urls[0])
            forensic_risk = risk
            forensic_logs = logs
            is_critical = critical
            # Re-extract entities just for the frontend display
            detected_entities = forensic_agent.extract_organizations(request.text)
        
        # --- 3. THE FIXED FUSION LOGIC (Smart Trust) ---
        
        # Scenario A: CRITICAL THREAT (Agent found hard evidence) -> BLOCK
        if is_critical:
            final_score = 100.0
            reason = "Forensic Override (Critical)"
            
        # Scenario B: NO LINKS FOUND (Agent is blind) -> TRUST AI 100%
        # THIS IS THE FIX. We do NOT average with 0.
        elif not urls:
            final_score = ai_probability
            reason = "AI Intuition (No Links)"
            
        # Scenario C: LINKS FOUND (Both brains have an opinion)
        else:
            # If Agent sees distinct danger (Risk > 50), trust the Agent.
            if forensic_risk >= 50:
                 final_score = max(ai_probability, forensic_risk)
                 reason = "High Forensic Risk"
            
            # If Agent is unsure (Risk < 50), check the AI.
            else:
                # If AI is very confident (>80), trust the AI.
                if ai_probability > 80:
                    final_score = ai_probability
                    reason = "AI Dominance"
                # Only average them if BOTH are unsure.
                else:
                    final_score = (ai_probability * 0.5) + (forensic_risk * 0.5)
                    reason = "Hybrid Consensus"

        # Cap score at 100 just in case
        final_score = min(final_score, 100)
        is_phishing = final_score > 50

        return {
            "is_phishing": is_phishing,
            "final_risk_score": f"{final_score:.2f}%",
            "ai_score": f"{ai_probability:.2f}%",
            "forensic_score": f"{forensic_risk:.2f}%",
            "link_warnings": "; ".join(forensic_logs) if forensic_logs else "No anomalies.",
            "logic_mode": reason,
            "entities_detected": detected_entities
        }
        
    except Exception as e:
        return {"error": str(e)}