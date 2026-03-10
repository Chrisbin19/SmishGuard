import os
os.environ["PYTHONUTF8"] = "1"

# ================================================================
#  SmishGuard v2 — Production API
#  Architecture : Neuro-Symbolic Dual Pipeline
#  Pipeline A   : Bi-LSTM Deep Learning Brain
#  Pipeline B   : Lightweight Forensic Agent (URL signals only)
#  Fusion       : Confidence-weighted, threshold-calibrated
# ================================================================

import re
import json
import math
import difflib
import logging
from urllib.parse import urlparse

import spacy
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smishguard")

app = FastAPI(
    title="SmishGuard v2",
    description="Neuro-Symbolic SMS Phishing Detection API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
#  CONFIGURATION
#  Loads optimal threshold from Colab-trained config
#  Falls back to safe default if file missing
# ================================================================
try:
    with open("smishguard_config.json") as f:
        _cfg = json.load(f)
    THRESHOLD     = _cfg.get("threshold_percent", 55.0)
    MODEL_VERSION = _cfg.get("model_version", "v2")
    log.info(f"✅ Config loaded — threshold: {THRESHOLD:.4f}% | version: {MODEL_VERSION}")
except Exception:
    THRESHOLD     = 55.0
    MODEL_VERSION = "v2_default"
    log.warning(f"⚠️  Config not found — using default threshold: {THRESHOLD}%")

# ================================================================
#  PIPELINE A — Bi-LSTM Deep Learning Brain
#  Trained in Colab on ~12,000 real + synthetic smishing samples
#  Understands: context, sentiment, urgency, brand patterns
# ================================================================
log.info("🧠 Loading Bi-LSTM brain...")
try:
    model = tf.keras.models.load_model(
        "smishguard_model (2).keras", compile=False
    )
    log.info("✅ Pipeline A (Bi-LSTM) active")
except Exception as e:
    log.error(f"❌ Model load failed: {e}")
    model = None

# ================================================================
#  PIPELINE B — spaCy NER
#  Used ONLY for brand entity extraction (dynamic, not hardcoded)
#  Identifies organization names to check against URL domain
# ================================================================
log.info("🗣️  Loading NER engine...")
try:
    nlp = spacy.load("en_core_web_sm")
    log.info("✅ Pipeline B (spaCy NER) active")
except Exception:
    nlp = None
    log.warning("⚠️  spaCy not found — NER disabled. Run: python -m spacy download en_core_web_sm")


# ================================================================
#  REQUEST SCHEMA
# ================================================================
class SMSRequest(BaseModel):
    text: str


# ================================================================
#  PREPROCESSOR
#  MUST be identical to the one used during Colab training
#  Any difference = model receives unfamiliar input format
# ================================================================
HOMOGLYPHS = str.maketrans({
    "\u03b1": "a",   # Greek alpha  → PαyPal
    "\u0435": "e",   # Cyrillic е
    "\u043e": "o",   # Cyrillic о
    "\u0440": "r",   # Cyrillic р
    "\u0455": "s",   # Cyrillic ѕ
    "@"     : "a",   # Am@zon
    "3"     : "e",   # fr33
    "0"     : "o",   # G00gle
    "1"     : "l",   # paypa1
    "5"     : "s",   # 5ecure
})

def preprocess(text: str) -> str:
    """
    Normalizes raw SMS text into model-ready format.
    Converts attack patterns into consistent tokens so the
    Bi-LSTM recognizes them regardless of obfuscation technique.
    """
    text = str(text).lower().strip()

    # 1. Normalize look-alike characters (homoglyph attacks)
    text = text.translate(HOMOGLYPHS)

    # 2. Replace URLs with semantic token
    #    Keeps the signal (URL present) without memorizing domains
    text = re.sub(r"https?://\S+|www\.\S+", " url_token ", text)

    # 3. Normalize phone numbers
    text = re.sub(r"\b[\d\-\.\(\)\s]{9,}\b", " phone_token ", text)

    # 4. Normalize monetary values
    text = re.sub(r"[\$\£\€\u20b9][\s\d,\.]+", " money_token ", text)

    # 5. Decode dot-segmentation: N.e.t.f.l.i.x → netflix
    text = re.sub(
        r"(?:\b\w\.){2,}\w\b",
        lambda m: m.group().replace(".", ""),
        text
    )

    # 6. Decode hyphen-segmentation: W-A-R-N-I-N-G → warning
    text = re.sub(
        r"(?:\b\w\-){2,}\w\b",
        lambda m: m.group().replace("-", ""),
        text
    )

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ================================================================
#  FORENSIC URL AGENT
#  Minimal rule-based component — only activated when URL present
#  Focuses on mathematical signals (entropy, mismatch) rather
#  than static keyword lists, making it robust to new attacks
# ================================================================
# Only infrastructure-level TLDs kept — no brand keywords
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".ru", ".cn",
    ".tk", ".cam", ".work", ".live", ".click",
    ".link", ".gq", ".ml", ".ga", ".cf"
}

def _entropy(text: str) -> float:
    """Shannon entropy — detects gibberish/random domains."""
    if not text:
        return 0.0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)

def _extract_orgs(text: str) -> list[str]:
    """
    Dynamically extracts brand/org names via NLP.
    No hardcoded brand list — works on any brand including new ones.
    """
    if not nlp:
        return []
    doc  = nlp(text)
    orgs = [e.text.lower() for e in doc.ents if e.label_ == "ORG"]
    if not orgs:
        # Fallback: capitalized tokens (catches new brands NER misses)
        orgs = [w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)]
    return list(set(orgs))

def analyze_url(raw_text: str, url: str) -> tuple[float, list[str], bool]:
    """
    Analyzes URL for objective threat signals.
    Returns: (risk_score 0-100, log_messages, is_critical)

    Signal priority (high → low):
    1. IP address host        → instant critical block
    2. Brand-domain mismatch  → very high risk (semantic deception)
    3. High entropy domain    → high risk (gibberish = generated)
    4. Suspicious TLD         → medium risk (infrastructure signal)
    """
    risk       = 0.0
    logs       = []
    is_critical = False

    # Parse domain
    try:
        parsed      = urlparse(url if url.startswith("http") else "http://" + url)
        domain      = parsed.netloc.lower().replace("www.", "") or url
        domain_body = domain.split(".")[0]
    except Exception:
        domain = domain_body = url

    # ── Signal 1: IP address host (mathematical certainty of malice) ──
    if re.match(r"\d{1,3}(\.\d{1,3}){3}", domain):
        return 100.0, [f"🚨 IP address host detected: {domain}"], True

    # ── Signal 2: Dynamic brand-domain semantic mismatch ─────────────
    # This is the core innovation — no hardcoded brands needed
    # NLP extracts claimed brand from text, checks if URL matches
    claimed_orgs = _extract_orgs(raw_text)
    if claimed_orgs:
        match_found = any(
            org in domain or
            difflib.SequenceMatcher(None, org, domain_body).ratio() > 0.72
            for org in claimed_orgs
        )
        if not match_found:
            risk       += 75.0
            is_critical = True
            logs.append(
                f"🎣 Semantic mismatch: claims '{claimed_orgs[0].title()}' "
                f"but links to '{domain}'"
            )
        elif any(tld in domain for tld in SUSPICIOUS_TLDS):
            risk += 45.0
            logs.append(f"⚠️  Brand present but TLD is suspicious: {domain}")

    # ── Signal 3: High entropy domain (randomness = generated domain) ──
    entropy = _entropy(domain)
    if entropy > 3.9:
        risk += 35.0
        logs.append(f"⚠️  High entropy domain (score: {entropy:.2f})")

    # ── Signal 4: Suspicious TLD (infrastructure signal) ─────────────
    if any(tld in domain for tld in SUSPICIOUS_TLDS) and risk < 45.0:
        risk += 35.0
        logs.append(f"🚩 Suspicious top-level domain: {domain}")

    return min(risk, 100.0), logs, is_critical


def analyze_phone(text: str) -> tuple[float, list[str]]:
    """
    Detects OTP/bank fraud via phone number + urgency co-occurrence.
    Handles smishing with no URL (Pipeline B blind spot without this).
    """
    phones = re.findall(
        r"\b(?:\+?1[\-\.\s]?)?\(?\d{3}\)?[\-\.\s]?\d{3}[\-\.\s]?\d{4}\b",
        text
    )
    if not phones:
        return 0.0, []

    urgency_terms = [
        "urgent", "immediately", "suspend", "block", "otp",
        "verify", "not you", "debit", "credit", "call now",
        "action required", "unauthorized", "compromised"
    ]
    text_lower = text.lower()
    urgency_hit = any(term in text_lower for term in urgency_terms)

    if urgency_hit:
        return 65.0, [f"📞 Suspicious phone number + urgency: {phones[0]}"]
    return 0.0, []


# ================================================================
#  FUSION ENGINE
#  Confidence-weighted combination of both pipelines
#  Design principle: minimize hardcoded rules, maximize signal use
#
#  Scenarios (in priority order):
#  A. Critical forensic evidence     → override with 100%
#  B. No URL, no phone signal        → trust AI entirely
#  C. Phone fraud detected (no URL)  → take max of AI + phone risk
#  D. High forensic risk (≥ 50)      → forensic-dominant fusion
#  E. AI highly confident (> 80%)    → AI-dominant fusion
#  F. Both uncertain                 → weighted average
# ================================================================
def fuse(
    ai_prob     : float,
    forensic    : float,
    is_critical : bool,
    has_url     : bool,
    phone_risk  : float,
) -> tuple[float, str]:

    has_phone = phone_risk > 0

    # A — Critical hard evidence (IP or brand mismatch)
    if is_critical:
        return 100.0, "Forensic Override — Critical Signal"

    # B — No links, no phone → AI is the only brain active
    if not has_url and not has_phone:
        return ai_prob, "AI Only — No External Signals"

    # C — Phone fraud pattern detected
    if not has_url and has_phone:
        score = max(ai_prob, phone_risk)
        return score, "Phone Fraud Pattern Detected"

    # D — Forensic sees strong evidence
    if forensic >= 50.0:
        # Weighted: forensic 65%, AI 35%
        score = (forensic * 0.65) + (ai_prob * 0.35)
        return min(score, 100.0), "Forensic Dominant Fusion"

    # E — AI is highly confident, forensic is uncertain
    if ai_prob > 80.0:
        # Weighted: AI 70%, forensic 30%
        score = (ai_prob * 0.70) + (forensic * 0.30)
        return min(score, 100.0), "AI Dominant Fusion"

    # F — Both uncertain → conservative weighted average
    score = (ai_prob * 0.55) + (forensic * 0.45)
    return min(score, 100.0), "Hybrid Consensus"


# ================================================================
#  PREDICT ENDPOINT
# ================================================================
@app.post("/predict")
async def predict(request: SMSRequest):
    try:
        raw_text = request.text

        # ── Pipeline A: Bi-LSTM ──────────────────────────────────
        clean = preprocess(raw_text)
        try:
            if model:
                ai_prob = float(
                    model.predict(
                        tf.constant([clean]), verbose=0
                    )[0][0]
                ) * 100.0
            else:
                ai_prob = 0.0
        except Exception as e:
            log.warning(f"AI prediction failed: {e}")
            ai_prob = 0.0

        # ── Pipeline B: Forensic Agent ───────────────────────────
        forensic_risk = 0.0
        forensic_logs = []
        is_critical   = False
        entities      = []
        phone_risk    = 0.0
        phone_logs    = []

        urls = re.findall(r"https?://\S+", raw_text)

        if urls:
            forensic_risk, forensic_logs, is_critical = analyze_url(raw_text, urls[0])
            entities = _extract_orgs(raw_text)

        phone_risk, phone_logs = analyze_phone(raw_text)

        all_logs  = forensic_logs + phone_logs
        has_url   = bool(urls)

        # ── Fusion ───────────────────────────────────────────────
        final_score, logic_mode = fuse(
            ai_prob, forensic_risk, is_critical, has_url, phone_risk
        )

        is_phishing = final_score >= THRESHOLD

        return {
            "is_phishing"      : is_phishing,
            "final_risk_score" : f"{final_score:.2f}%",
            "ai_score"         : f"{ai_prob:.2f}%",
            "forensic_score"   : f"{forensic_risk:.2f}%",
            "phone_risk_score" : f"{phone_risk:.2f}%",
            "logic_mode"       : logic_mode,
            "link_warnings"    : "; ".join(all_logs) if all_logs else "No anomalies detected.",
            "entities_detected": entities,
            "threshold_used"   : f"{THRESHOLD:.4f}%",
            "model_version"    : MODEL_VERSION,
        }

    except Exception as e:
        log.error(f"Prediction error: {e}")
        return {"error": str(e)}


# ================================================================
#  HEALTH CHECK ENDPOINT
# ================================================================
@app.get("/health")
async def health():
    return {
        "status"        : "online",
        "model_loaded"  : model is not None,
        "nlp_loaded"    : nlp is not None,
        "threshold"     : f"{THRESHOLD:.4f}%",
        "model_version" : MODEL_VERSION,
    }