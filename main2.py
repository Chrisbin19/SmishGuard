import os
os.environ["PYTHONUTF8"]       = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ================================================================
#  SmishGuard v2 — Production API (Robust Edition)
#
#  Architecture : Neuro-Symbolic Dual Pipeline
#  Pipeline A   : Bi-LSTM Deep Learning Brain
#  Pipeline B   : Lightweight Forensic Agent
#  Fusion       : Evidence-gated, confidence-weighted
#
#  Bugs fixed in this version:
#  [1] URL shortener now uses EXACT domain match — not substring.
#      "t.co" was matching inside "walmart.com" via substring.
#      Fix: `domain in URL_SHORTENERS` (exact set membership)
#  [2] Clean URL dampening now works correctly because [1] is fixed.
#      walmart.com → forensic=0 → url_is_clean=True → AI dampened.
#  [3] safe_mode=False fixes Windows charmap codec error on load.
#  [4] Threshold floor 40% prevents over-flagging.
#  [5] Phone/OTP fraud detection covers no-URL blind spot.
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
        safe_mode = False          # Required for TextVectorization on Windows
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
#  PREPROCESSOR
#  Must be identical to Colab training preprocessor
# ================================================================
HOMOGLYPHS = str.maketrans({
    "\u03b1": "a",  # Greek alpha  PαyPal  → paypal
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "r",  # Cyrillic р
    "\u0455": "s",  # Cyrillic ѕ
    "@"     : "a",  # Am@zon      → amazon
    "3"     : "e",  # fr33        → free
    "0"     : "o",  # G00gle      → google
    "1"     : "l",  # paypa1      → paypal
    "5"     : "s",  # 5ecure      → secure
})

def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(HOMOGLYPHS)
    text = re.sub(r"https?://\S+|www\.\S+",   " url_token ",   text)
    text = re.sub(r"\b[\d\-\.\(\)\s]{9,}\b",  " phone_token ", text)
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

# ================================================================
#  CRITICAL FIX: Exact domain members only — NO substring matching
#  Old code: any(s in domain for s in URL_SHORTENERS)
#            → "t.co" IN "walmart.com" = TRUE (wrong!)
#  New code: domain in URL_SHORTENERS
#            → "walmart.com" IN set = FALSE (correct!)
# ================================================================
URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rb.gy",
    "short.io",
    "tiny.cc",
    "bl.ink",
    "shorte.st",
    "rebrand.ly",
}

# ================================================================
#  FORENSIC HELPERS
# ================================================================
def _entropy(text: str) -> float:
    """Shannon entropy — detects algorithmically-generated domains."""
    if not text:
        return 0.0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)


def _extract_orgs(text: str) -> list[str]:
    """
    Dynamically extracts brand/org names via NLP.
    No hardcoded brand list — works on any brand including new ones.
    Falls back to capitalized word detection if NER finds nothing.
    """
    if not nlp:
        return []
    doc  = nlp(text)
    orgs = [e.text.lower() for e in doc.ents if e.label_ == "ORG"]
    if not orgs:
        # Fallback: capitalized tokens (catches brands NER might miss)
        orgs = [w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)]
    return list(set(orgs))


def _domain_is_clean(domain: str) -> bool:
    """
    Returns True if domain appears to be a legitimate well-known domain.
    Three criteria must ALL be true:
      1. Has a trusted TLD (.com, .org, .gov etc.)
      2. Domain is short (< 30 chars) — legit domains are concise
      3. Domain body has no suspicious keyword patterns
    """
    has_clean_tld     = any(domain.endswith(t) for t in CLEAN_TLDS)
    reasonable_length = len(domain) < 30
    no_bad_keywords   = not any(
        kw in domain for kw in [
            "verify-", "-verify", "secure-", "-secure",
            "login-",  "-login",  "update-", "-update",
            "confirm", "billing", "suspend", "-alert",
            "account-", "-account"
        ]
    )
    return has_clean_tld and reasonable_length and no_bad_keywords


# ================================================================
#  FORENSIC URL ANALYSIS
# ================================================================
def analyze_url(raw_text: str, url: str) -> tuple[float, list[str], bool, bool]:
    """
    Analyzes URL against objective, mathematical threat signals.

    Signals (priority order):
      1. IP address host     → instant critical block
      2. URL shortener       → EXACT domain match only (bug fix)
      3. Brand-URL mismatch  → dynamic NLP, no hardcoded brands
      4. High entropy domain → Shannon entropy > 3.9
      5. Suspicious TLD      → infrastructure signal, fallback only

    Returns:
      risk        : 0.0–100.0
      logs        : human-readable findings
      is_critical : True = hard evidence → override fusion
      url_is_clean: True = legitimate domain → dampen AI score
    """
    risk        = 0.0
    logs        = []
    is_critical = False

    # Parse domain
    try:
        parsed      = urlparse(url if url.startswith("http") else "http://" + url)
        domain      = parsed.netloc.lower().replace("www.", "") or url
        domain_body = domain.split(".")[0]
    except Exception:
        domain = domain_body = url

    # ── Signal 1: IP address host ─────────────────────────────────
    if re.match(r"\d{1,3}(\.\d{1,3}){3}", domain):
        return 100.0, [f"🚨 IP address host: {domain}"], True, False

    # ── Signal 2: URL shortener ───────────────────────────────────
    # EXACT SET MEMBERSHIP — fixes the walmart.com false positive
    # "walmart.com" not in URL_SHORTENERS → False ✅
    # "t.co"        in URL_SHORTENERS     → True  ✅
    if domain in URL_SHORTENERS:
        risk += 40.0
        logs.append(f"⚠️  URL shortener detected: {domain}")

    # ── Signal 3: Dynamic brand-domain semantic mismatch ──────────
    # NLP extracts claimed brand from SMS text.
    # Checks whether the URL domain is related to that brand.
    # No match → deception detected (e.g. "Netflix" → scam-site.xyz)
    claimed_orgs = _extract_orgs(raw_text)
    if claimed_orgs:
        match_found = any(
            org in domain or
            difflib.SequenceMatcher(None, org, domain_body).ratio() > 0.72
            for org in claimed_orgs
        )
        if not match_found:
            # Text claims a brand but URL has no relation to it
            risk       += 75.0
            is_critical = True
            logs.append(
                f"🎣 Semantic mismatch: text claims "
                f"'{claimed_orgs[0].title()}' but links to '{domain}'"
            )
        else:
            # Brand matches domain — still check TLD
            if any(tld in domain for tld in SUSPICIOUS_TLDS):
                risk += 40.0
                logs.append(
                    f"⚠️  Brand present but TLD is suspicious: {domain}"
                )

    # ── Signal 4: High entropy ────────────────────────────────────
    entropy = _entropy(domain)
    if entropy > 3.9:
        risk += 35.0
        logs.append(f"⚠️  High entropy domain (score: {entropy:.2f})")

    # ── Signal 5: Suspicious TLD (fallback) ──────────────────────
    if any(tld in domain for tld in SUSPICIOUS_TLDS) and risk < 40.0:
        risk += 35.0
        logs.append(f"🚩 Suspicious TLD: {domain}")

    # ── Clean URL determination ───────────────────────────────────
    # Only clean if ALL are true:
    # - No risk signals fired (risk == 0)
    # - Not a critical threat
    # - Domain passes the clean domain check
    url_is_clean = (
        risk == 0.0         and
        not is_critical     and
        _domain_is_clean(domain)
    )

    return min(risk, 100.0), logs, is_critical, url_is_clean


# ================================================================
#  PHONE FRAUD DETECTION
# ================================================================
def analyze_phone(text: str) -> tuple[float, list[str]]:
    """
    Detects OTP/bank fraud via phone number + urgency co-occurrence.
    Covers smishing that has no URL — Pipeline B blind spot otherwise.
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
        "action required", "unauthorized", "compromised",
        "unusual activity", "suspicious"
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
#  Scenario priority (A → G):
#
#  A. Critical forensic evidence (IP / brand mismatch)
#     → Hard block at 100%
#
#  B. No URL and no phone signal
#     → Trust AI fully (pure text classification)
#
#  C. Phone fraud pattern, no URL
#     → max(AI, phone_risk)
#
#  D. Clean legitimate URL (forensic = 0, domain verified clean)
#     → Dampen AI by 70% — protects walmart.com style messages
#     → "Redeem Walmart credits via walmart.com" = SAFE
#
#  E. Strong forensic evidence (≥ 50%)
#     → Forensic dominant: 65% forensic + 35% AI
#
#  F. AI highly confident (> 80%) + forensic present but weak
#     → AI dominant: 65% AI + 35% forensic
#
#  G. Both uncertain
#     → Conservative 50/50 average
# ================================================================
def fuse(
    ai_prob     : float,
    forensic    : float,
    is_critical : bool,
    has_url     : bool,
    phone_risk  : float,
    url_is_clean: bool,
) -> tuple[float, str]:

    has_phone = phone_risk > 0.0

    # A — Critical forensic evidence
    if is_critical:
        return 100.0, "Forensic Override — Critical Signal"

    # B — No external signals
    if not has_url and not has_phone:
        return ai_prob, "AI Only — No External Signals"

    # C — Phone fraud (no URL)
    if not has_url and has_phone:
        return max(ai_prob, phone_risk), "Phone Fraud Pattern Detected"

    # D — Clean legitimate URL
    #     forensic == 0.0 means NO threat signals fired at all
    #     url_is_clean means domain passed all legitimacy checks
    #     Together: URL is genuinely safe → heavily dampen AI
    if url_is_clean and forensic == 0.0:
        dampened = ai_prob * 0.30
        return dampened, "Clean URL — AI Score Dampened"

    # E — Strong forensic evidence
    if forensic >= 50.0:
        score = (forensic * 0.65) + (ai_prob * 0.35)
        return min(score, 100.0), "Forensic Dominant Fusion"

    # F — AI confident, forensic weak but not zero
    if ai_prob > 80.0 and forensic > 0.0:
        score = (ai_prob * 0.65) + (forensic * 0.35)
        return min(score, 100.0), "AI Dominant Fusion"

    # G — Both uncertain
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
                    model.predict(
                        tf.constant([clean]), verbose=0
                    )[0][0]
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

        # ── Fusion ───────────────────────────────────────────────
        final_score, logic_mode = fuse(
            ai_prob      = ai_prob,
            forensic     = forensic_risk,
            is_critical  = is_critical,
            has_url      = bool(urls),
            phone_risk   = phone_risk,
            url_is_clean = url_is_clean,
        )

        is_phishing = final_score >= THRESHOLD

        return {
            "is_phishing"      : is_phishing,
            "final_risk_score" : f"{final_score:.2f}%",
            "ai_score"         : f"{ai_prob:.2f}%",
            "forensic_score"   : f"{forensic_risk:.2f}%",
            "phone_risk_score" : f"{phone_risk:.2f}%",
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
#  GET http://127.0.0.1:8001/health
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