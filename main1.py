import os
os.environ["PYTHONUTF8"]       = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ================================================================
#  SmishGuard v2 — Production API (Robust Edition v3)
#
#  Architecture : Neuro-Symbolic Dual Pipeline
#  Pipeline A   : Bi-LSTM Deep Learning Brain
#  Pipeline B   : Lightweight Forensic Agent
#  Fusion       : Evidence-gated, confidence-weighted
#
#  Fixes in this version:
#  [1] Brand + suspicious TLD → risk raised to 55 (was 40)
#      Triggers Forensic Dominant fusion at ≥50 threshold
#      Fixes: "Disney Plus...disney-renew.top"
#
#  [2] No-URL urgency boost via keyword co-occurrence scoring
#      Fixes: "W-A-R-N-I-N-G CitiBank blocked" (no URL, no phone)
#
#  [3] Delivery/service brand fallback for spaCy blind spots
#      FedEx, UPS, DHL etc. that NER commonly misclassifies
#      Fixes: "FedEx pay customs...fedex-pay.top"
#
#  [4] Suspicious TLD with no brand → raises risk to 45
#      Fixes: "Hi love...http://secure-pay.ru" (friendly + .ru)
#
#  [5] URL shortener exact match (not substring)
#      Fixes: "t.co" matching inside "walmart.com"
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

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("smishguard")

# ================================================================
#  APP
# ================================================================
app = FastAPI(
    title       = "SmishGuard v2",
    description = "Neuro-Symbolic SMS Phishing Detection API",
    version     = "2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ================================================================
#  CONFIGURATION
# ================================================================
try:
    with open("smishguard_config.json", encoding="utf-8") as f:
        _cfg = json.load(f)
    THRESHOLD     = max(float(_cfg.get("threshold_percent", 45.0)), 40.0)
    MODEL_VERSION = _cfg.get("model_version", "v2")
    log.info(f"✅ Config loaded — threshold: {THRESHOLD:.2f}% | version: {MODEL_VERSION}")
except Exception:
    THRESHOLD     = 45.0
    MODEL_VERSION = "v2_default"
    log.warning(f"⚠️  Config not found — using default: {THRESHOLD}%")

# ================================================================
#  PIPELINE A — Bi-LSTM
# ================================================================
log.info("🧠 Loading Bi-LSTM brain...")
model = None
try:
    model = tf.keras.models.load_model(
        "smishguard_model (2).keras",
        compile   = False,
        safe_mode = False
    )
    log.info("✅ Pipeline A (Bi-LSTM) active")
except Exception as e:
    log.error(f"❌ Model load failed: {e}")

# ================================================================
#  PIPELINE B — spaCy NER
# ================================================================
log.info("🗣️  Loading NER engine...")
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
    log.info("✅ Pipeline B (spaCy NER) active")
except Exception:
    log.warning("⚠️  Run: python -m spacy download en_core_web_sm")

# ================================================================
#  SCHEMA
# ================================================================
class SMSRequest(BaseModel):
    text: str

# ================================================================
#  PREPROCESSOR — identical to Colab training preprocessor
# ================================================================
HOMOGLYPHS = str.maketrans({
    "\u03b1": "a",  "\u0435": "e",  "\u043e": "o",
    "\u0440": "r",  "\u0455": "s",
    "@": "a", "3": "e", "0": "o", "1": "l", "5": "s",
})

def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(HOMOGLYPHS)
    text = re.sub(r"https?://\S+|www\.\S+",    " url_token ",   text)
    text = re.sub(r"\b[\d\-\.\(\)\s]{9,}\b",   " phone_token ", text)
    text = re.sub(r"[\$\£\€\u20b9][\s\d,\.]+", " money_token ", text)
    text = re.sub(r"(?:\b\w\.){2,}\w\b",  lambda m: m.group().replace(".", ""), text)
    text = re.sub(r"(?:\b\w\-){2,}\w\b",  lambda m: m.group().replace("-", ""), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================================================================
#  FORENSIC CONSTANTS
# ================================================================
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".ru", ".cn", ".tk",
    ".cam", ".work", ".live", ".click", ".link",
    ".gq",  ".ml",  ".ga",  ".cf",  ".pw"
}

CLEAN_TLDS = {
    ".com", ".org", ".net", ".edu", ".gov",
    ".co.uk", ".co.in", ".io", ".app"
}

# EXACT domain match only — NOT substring
# "t.co" in "walmart.com" would be True with substring — this prevents that
URL_SHORTENERS = {
    "bit.ly",    "tinyurl.com", "t.co",      "goo.gl",
    "ow.ly",     "is.gd",       "buff.ly",   "cutt.ly",
    "rb.gy",     "short.io",    "tiny.cc",   "bl.ink",
    "shorte.st", "rebrand.ly",
}

# ── Fix [3]: Delivery/service brands spaCy commonly misses ───────
# Small, justified list — only brands where NER is known to fail
# These are not "hardcoded smishing brands" — they are NER corrections
KNOWN_SERVICE_BRANDS = {
    "fedex", "ups", "usps", "dhl", "hermes", "evri",
    "royal mail", "australia post", "canada post",
    "irs", "hmrc", "medicare", "medicaid",
}

# ── Fix [2]: Urgency keywords for no-URL messages ─────────────────
# Used to boost AI score when model underestimates urgency-only attacks
URGENCY_KEYWORDS = [
    "warning", "urgent", "blocked", "suspended", "compromised",
    "unauthorized", "immediately", "verify now", "action required",
    "account locked", "unusual activity", "suspicious activity",
    "call now", "limited", "restricted", "frozen",
]

# ================================================================
#  FORENSIC HELPERS
# ================================================================
def _entropy(text: str) -> float:
    if not text:
        return 0.0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)


def _extract_orgs(text: str) -> list[str]:
    """
    Dynamically extracts brand names via NLP.
    Falls back to capitalized tokens, then service brand list.
    """
    orgs = []

    if nlp:
        doc  = nlp(text)
        orgs = [e.text.lower() for e in doc.ents if e.label_ == "ORG"]

    # Fallback 1: capitalized tokens (catches brands NER misses)
    if not orgs:
        orgs = [w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)]

    # Fallback 2 [Fix 3]: known service brands — catches FedEx, UPS etc.
    text_lower = text.lower()
    for brand in KNOWN_SERVICE_BRANDS:
        if brand in text_lower and brand not in orgs:
            orgs.append(brand)

    return list(set(orgs))


def _domain_is_clean(domain: str) -> bool:
    has_clean_tld     = any(domain.endswith(t) for t in CLEAN_TLDS)
    reasonable_length = len(domain) < 30
    no_bad_keywords   = not any(
        kw in domain for kw in [
            "verify-", "-verify", "secure-", "-secure",
            "login-",  "-login",  "update-", "-update",
            "confirm", "billing", "suspend", "-alert",
            "account-", "-account",
        ]
    )
    return has_clean_tld and reasonable_length and no_bad_keywords


def _urgency_score(text: str) -> float:
    """
    [Fix 2] Counts urgency keyword hits in text.
    Used to boost AI score for no-URL high-urgency messages.
    Returns a boost score 0-40 based on hit count.
    """
    text_lower = text.lower()
    hits = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
    if hits == 0:
        return 0.0
    elif hits == 1:
        return 15.0
    elif hits == 2:
        return 28.0
    else:
        return 40.0   # 3+ urgency hits → strong signal


# ================================================================
#  FORENSIC URL ANALYSIS
# ================================================================
def analyze_url(raw_text: str, url: str) -> tuple[float, list[str], bool, bool]:
    """
    Signals (priority order):
      1. IP address host       → instant critical block (100%)
      2. URL shortener         → exact match only (+40%)
      3. Brand-domain mismatch → NLP + fallback (+75%, critical)
      4. Brand + suspicious TLD→ +55% [Fix 1] (was 40, now triggers forensic dominant)
      5. High entropy domain   → +35%
      6. Suspicious TLD alone  → +45% [Fix 4] (was 35, catches friendly+.ru)
    """
    risk        = 0.0
    logs        = []
    is_critical = False

    try:
        parsed      = urlparse(url if url.startswith("http") else "http://" + url)
        domain      = parsed.netloc.lower().replace("www.", "") or url
        domain_body = domain.split(".")[0]
    except Exception:
        domain = domain_body = url

    # Signal 1: IP address host
    if re.match(r"\d{1,3}(\.\d{1,3}){3}", domain):
        return 100.0, [f"🚨 IP address host: {domain}"], True, False

    # Signal 2: URL shortener — EXACT set membership only
    if domain in URL_SHORTENERS:
        risk += 40.0
        logs.append(f"⚠️  URL shortener detected: {domain}")

    # Signal 3 & 4: Brand mismatch or brand + bad TLD
    claimed_orgs = _extract_orgs(raw_text)
    if claimed_orgs:
        match_found = any(
            org in domain or
            difflib.SequenceMatcher(None, org, domain_body).ratio() > 0.72
            for org in claimed_orgs
        )
        if not match_found:
            # Brand claimed but URL has no relation → deception
            risk       += 75.0
            is_critical = True
            logs.append(
                f"🎣 Semantic mismatch: claims "
                f"'{claimed_orgs[0].title()}' but links to '{domain}'"
            )
        else:
            # Brand matches domain name but TLD is suspicious
            # Fix [1]: raised from 40 → 55 to cross forensic dominant threshold
            if any(tld in domain for tld in SUSPICIOUS_TLDS):
                risk += 55.0
                logs.append(
                    f"⚠️  Brand present but TLD is suspicious: {domain}"
                )

    # Signal 5: High entropy
    entropy = _entropy(domain)
    if entropy > 3.9:
        risk += 35.0
        logs.append(f"⚠️  High entropy domain (score: {entropy:.2f})")

    # Signal 6: Suspicious TLD alone (no brand context)
    # Fix [4]: raised from 35 → 45 to catch friendly+malicious-TLD messages
    if any(tld in domain for tld in SUSPICIOUS_TLDS) and risk < 40.0:
        risk += 45.0
        logs.append(f"🚩 Suspicious TLD: {domain}")

    # Clean URL check
    url_is_clean = (
        risk == 0.0     and
        not is_critical and
        _domain_is_clean(domain)
    )

    return min(risk, 100.0), logs, is_critical, url_is_clean


# ================================================================
#  PHONE FRAUD DETECTION
# ================================================================
def analyze_phone(text: str) -> tuple[float, list[str]]:
    phones = re.findall(
        r"\b(?:\+?1[\-\.\s]?)?\(?\d{3}\)?[\-\.\s]?\d{3}[\-\.\s]?\d{4}\b",
        text
    )
    if not phones:
        return 0.0, []

    urgency_terms = [
        "urgent", "immediately", "suspend", "block", "otp",
        "verify", "not you", "debit", "credit", "call now",
        "action required", "unauthorized", "compromised",
        "unusual activity", "suspicious",
    ]
    hits = [t for t in urgency_terms if t in text.lower()]

    if hits:
        return 65.0, [
            f"📞 Suspicious phone + urgency "
            f"({', '.join(hits[:2])}): {phones[0]}"
        ]
    return 0.0, []


# ================================================================
#  FUSION ENGINE
#
#  A. Critical forensic evidence       → 100% block
#  B. No URL, no phone, low urgency    → AI only
#  B2. No URL, no phone, high urgency  → AI + urgency boost [Fix 2]
#  C. Phone fraud (no URL)             → max(AI, phone)
#  D. Clean legitimate URL             → dampen AI by 70%
#  E. Strong forensic (≥ 50)           → forensic dominant (65/35)
#  F. AI confident (> 80%) + forensic  → AI dominant (65/35)
#  G. Both uncertain                   → conservative 50/50
# ================================================================
def fuse(
    ai_prob      : float,
    forensic     : float,
    is_critical  : bool,
    has_url      : bool,
    phone_risk   : float,
    url_is_clean : bool,
    urgency_boost: float,
) -> tuple[float, str]:

    has_phone = phone_risk > 0.0

    # A — Critical forensic
    if is_critical:
        return 100.0, "Forensic Override — Critical Signal"

    # B — No external signals
    if not has_url and not has_phone:
        # Fix [2]: boost AI with urgency signal for no-URL attacks
        if urgency_boost > 0:
            boosted = min(ai_prob + urgency_boost, 100.0)
            return boosted, f"AI + Urgency Boost (+{urgency_boost:.0f}%)"
        return ai_prob, "AI Only — No External Signals"

    # C — Phone fraud
    if not has_url and has_phone:
        return max(ai_prob, phone_risk), "Phone Fraud Pattern Detected"

    # D — Clean URL
    if url_is_clean and forensic == 0.0:
        dampened = ai_prob * 0.30
        return dampened, "Clean URL — AI Score Dampened"

    # E — Strong forensic
    if forensic >= 50.0:
        score = (forensic * 0.65) + (ai_prob * 0.35)
        return min(score, 100.0), "Forensic Dominant Fusion"

    # F — AI dominant
    if ai_prob > 80.0 and forensic > 0.0:
        score = (ai_prob * 0.65) + (forensic * 0.35)
        return min(score, 100.0), "AI Dominant Fusion"

    # G — Hybrid
    score = (ai_prob * 0.50) + (forensic * 0.50)
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
                    model.predict(tf.constant([clean]), verbose=0)[0][0]
                ) * 100.0
            else:
                ai_prob = 0.0
        except Exception as e:
            log.warning(f"AI prediction error: {e}")
            ai_prob = 0.0

        # ── Pipeline B: URL Forensics ────────────────────────────
        forensic_risk = 0.0
        forensic_logs = []
        is_critical   = False
        url_is_clean  = False
        entities      = []
        urls          = re.findall(r"https?://\S+", raw_text)

        if urls:
            forensic_risk, forensic_logs, is_critical, url_is_clean = \
                analyze_url(raw_text, urls[0])
            entities = _extract_orgs(raw_text)

        # ── Pipeline B: Phone Fraud ──────────────────────────────
        phone_risk, phone_logs = analyze_phone(raw_text)

        # ── Urgency boost for no-URL messages ────────────────────
        urgency_boost = _urgency_score(raw_text) if not urls else 0.0

        # ── Fusion ───────────────────────────────────────────────
        final_score, logic_mode = fuse(
            ai_prob       = ai_prob,
            forensic      = forensic_risk,
            is_critical   = is_critical,
            has_url       = bool(urls),
            phone_risk    = phone_risk,
            url_is_clean  = url_is_clean,
            urgency_boost = urgency_boost,
        )

        is_phishing = final_score >= THRESHOLD

        return {
            "is_phishing"      : is_phishing,
            "final_risk_score" : f"{final_score:.2f}%",
            "ai_score"         : f"{ai_prob:.2f}%",
            "forensic_score"   : f"{forensic_risk:.2f}%",
            "phone_risk_score" : f"{phone_risk:.2f}%",
            "urgency_boost"    : f"{urgency_boost:.2f}%",
            "logic_mode"       : logic_mode,
            "link_warnings"    : "; ".join(forensic_logs + phone_logs)
                                 if (forensic_logs or phone_logs)
                                 else "No anomalies detected.",
            "entities_detected": entities,
            "url_is_clean"     : url_is_clean,
            "threshold_used"   : f"{THRESHOLD:.2f}%",
            "model_version"    : MODEL_VERSION,
        }

    except Exception as e:
        log.error(f"Prediction error: {e}")
        return {"error": str(e)}


# ================================================================
#  HEALTH CHECK
# ================================================================
@app.get("/health")
async def health():
    return {
        "status"       : "online",
        "model_loaded" : model is not None,
        "nlp_loaded"   : nlp   is not None,
        "threshold"    : f"{THRESHOLD:.2f}%",
        "model_version": MODEL_VERSION,
    }