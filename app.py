
import ipaddress
import re
from urllib.parse import urlsplit

import joblib
import pandas as pd
import streamlit as st


SUSPICIOUS_WORDS = [
    "login", "verify", "password", "account",
    "secure", "update", "banking", "signin",
    "confirm", "wallet", "recover", "payment",
]


def extract_features(url):
    url = str(url).strip()
    parsed = urlsplit(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    full_url = url.lower()

    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0

    parts = [p for p in host.split(".") if p]
    subdomains = max(
        0, len(parts) - 2 - (1 if host.startswith("www.") else 0)
    )

    return {
        "url_length": len(url),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_symbol": int("@" in url),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip": has_ip,
        "subdomains": subdomains,
        "digit_count": sum(c.isdigit() for c in url),
        "slash_count": url.count("/"),
        "question_count": url.count("?"),
        "equals_count": url.count("="),
        "suspicious_words": sum(
            len(re.findall(r"\b" + re.escape(word) + r"\b", full_url))
            for word in SUSPICIOUS_WORDS
        ),
        "path_length": len(parsed.path),
    }


st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔐",
    layout="centered",
)

st.title("🔐 Phishing URL Detection")
st.write(
    "Analyze URL patterns using a machine learning model."
)

st.info(
    "Safety notice: This tool analyzes the URL text only. "
    "It does not verify the live website or guarantee safety."
)

try:
    model = joblib.load("artifacts/phishing_model.joblib")
    feature_names = joblib.load("artifacts/feature_names.joblib")
except FileNotFoundError:
    st.error(
        "Model files are missing. Run `python train_model.py` first."
    )
    st.stop()

url = st.text_input(
    "Enter a website URL",
    placeholder="https://example.com",
)

if st.button("Analyze URL", type="primary"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        candidate = url.strip()
        parsed = urlsplit(
            candidate if "://" in candidate else "http://" + candidate
        )

        try:
            valid_host = bool(parsed.hostname) and "." in parsed.hostname
            if parsed.hostname:
                parsed.hostname.encode("idna")
        except (ValueError, UnicodeError):
            valid_host = False

        if not valid_host or any(c.isspace() for c in candidate):
            st.error("Please enter a valid-looking URL.")
        else:
            features = extract_features(candidate)
            input_data = pd.DataFrame(
                [features], columns=feature_names
            )

            prediction = int(model.predict(input_data)[0])

            st.subheader("Analysis result")

            if prediction == 0:
                st.error("⚠️ Potentially Phishing")
                st.write(
                    "The model detected patterns associated with "
                    "phishing URLs. Avoid entering personal details."
                )
            else:
                st.success("Likely Legitimate")
                st.write(
                    "The model classified this URL as legitimate. "
                    "This does not mean the website is safe."
                )

            st.subheader("Extracted URL features")
            st.dataframe(
                pd.DataFrame(
                    features.items(), columns=["Feature", "Value"]
                ),
                hide_index=True,
                use_container_width=True,
            )

st.caption(
    "Educational cybersecurity project. Never automatically open "
    "submitted URLs. Model predictions may be incorrect."
)
