
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "phishing_model.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.joblib"

SUSPICIOUS_WORDS = [
    "login", "verify", "password", "account",
    "secure", "update", "banking", "signin",
    "confirm", "wallet", "recover", "payment",
]


def extract_features(url: str) -> dict:
    """Extract the same 13 features used during model training."""
    url = str(url).strip()
    parsed = urlsplit(url if "://" in url else "http://" + url)

    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    full_url = url.lower()

    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0

    host_parts = [part for part in host.split(".") if part]
    subdomains = max(
        0,
        len(host_parts) - 2
        - (1 if host.startswith("www.") else 0),
    )

    # Preserve the feature calculation used in train_model.py.
    return {
        "url_length": len(url),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_symbol": int("@" in url),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip": has_ip,
        "subdomains": subdomains,
        "digit_count": sum(ch.isdigit() for ch in url),
        "slash_count": url.count("/"),
        "question_count": url.count("?"),
        "equals_count": url.count("="),
        "suspicious_words": sum(
            len(
                re.findall(
                    r"\b" + re.escape(word) + r"\b",
                    full_url,
                )
            )
            for word in SUSPICIOUS_WORDS
        ),
        "path_length": len(path),
    }


if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not FEATURE_NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"Feature names not found: {FEATURE_NAMES_PATH}"
    )

# Load only artifacts created by your trusted training script.
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

if len(feature_names) != model.n_features_in_:
    raise RuntimeError(
        "Model and feature-name counts do not match."
    )

if len(set(feature_names)) != len(feature_names):
    raise RuntimeError("Duplicate feature names were found.")

if not hasattr(model, "predict_proba"):
    raise RuntimeError(
        "The loaded model does not support predict_proba."
    )

# Check that model inference works with the saved feature schema.
probe = pd.DataFrame(
    [[0] * len(feature_names)],
    columns=feature_names,
)
model.predict(probe)


def predict_url(url: str) -> dict:
    """Return the prediction and extracted features for one URL."""
    features = extract_features(url)

    # Explicit column ordering is essential for compatibility.
    input_data = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability
        in zip(model.classes_, probabilities)
    }

    phishing_probability = class_probabilities.get(0, 0.0)
    legitimate_probability = class_probabilities.get(1, 0.0)

    return {
        "prediction": (
            "Phishing" if prediction == 0 else "Legitimate"
        ),
        "prediction_code": prediction,
        "phishing_probability": round(
            phishing_probability * 100, 2
        ),
        "legitimate_probability": round(
            legitimate_probability * 100, 2
        ),
        "features": features,
    }
