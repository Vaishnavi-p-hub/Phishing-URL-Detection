
import os
import ipaddress
import re
from urllib.parse import urlsplit

import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ID = 967

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(
    ARTIFACTS_DIR, "phishing_model.joblib"
)
FEATURE_NAMES_PATH = os.path.join(
    ARTIFACTS_DIR, "feature_names.joblib"
)

# Dataset label convention used by this project:
# 0 = Phishing
# 1 = Legitimate
PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1

TEST_SIZE = 0.20
RANDOM_STATE = 42

SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "password",
    "account",
    "secure",
    "update",
    "banking",
    "signin",
    "confirm",
    "wallet",
    "recover",
    "payment",
]


# ============================================================
# FEATURE EXTRACTION
# Keep these features identical to backend/model_service.py.
# ============================================================

def extract_features(url):
    """Convert a URL string into the model's 13 numeric features."""

    url = str(url).strip()

    parsed = urlsplit(
        url if "://" in url else "http://" + url
    )

    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    full_url = url.lower()

    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0

    host_parts = [
        part for part in host.split(".") if part
    ]

    subdomains = max(
        0,
        len(host_parts)
        - 2
        - (1 if host.startswith("www.") else 0),
    )

    features = {
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

    return features


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_dataset():
    print("Downloading the public UCI dataset...")

    dataset = fetch_ucirepo(id=DATASET_ID)

    if "URL" not in dataset.data.features.columns:
        raise ValueError(
            "The expected URL column was not found. "
            f"Available columns: "
            f"{list(dataset.data.features.columns)}"
        )

    if dataset.data.targets is None:
        raise ValueError(
            "The dataset does not contain target labels."
        )

    raw_urls = dataset.data.features["URL"]
    raw_labels = dataset.data.targets.iloc[:, 0]

    # Keep missing URLs as missing instead of converting them
    # into the literal string "nan".
    urls = raw_urls.astype("string").str.strip()
    labels = pd.to_numeric(
        raw_labels, errors="coerce"
    )

    valid = (
        urls.notna()
        & urls.ne("")
        & labels.isin(
            [PHISHING_LABEL, LEGITIMATE_LABEL]
        )
    )

    data = pd.DataFrame({
        "url": urls[valid].astype(str),
        "label": labels[valid].astype(int),
    })

    if data.empty:
        raise ValueError(
            "No valid URLs and labels were found."
        )

    print(f"Valid records loaded: {len(data)}")
    print("\nOriginal label counts:")
    print(data["label"].value_counts().sort_index())

    print(
        "\nLabel convention: "
        "0 = Phishing, 1 = Legitimate"
    )

    print(
        "Important: verify this convention against the "
        "dataset's official label definition."
    )

    # Diagnose exact URL duplicates before splitting.
    duplicate_mask = data["url"].duplicated(keep=False)

    duplicate_rows = int(duplicate_mask.sum())
    conflicting_groups = (
        data.groupby("url")["label"].nunique().gt(1).sum()
    )

    print(f"\nRows belonging to duplicate URLs: {duplicate_rows}")
    print(
        "URLs associated with conflicting labels: "
        f"{conflicting_groups}"
    )

    # Exact URL duplicates must not appear across the train
    # and test sets. Retain one record per exact URL.
    # If duplicate labels conflict, the first retained label
    # is used; the conflict count above makes this visible.
    data = data.drop_duplicates(
        subset=["url"], keep="first"
    ).reset_index(drop=True)

    print(
        "Records after exact URL deduplication: "
        f"{len(data)}"
    )

    if data["label"].nunique() != 2:
        raise ValueError(
            "Both phishing and legitimate classes are "
            "required for training."
        )

    return data


# ============================================================
# FEATURE MATRIX
# ============================================================

def build_feature_matrix(urls):
    print("\nExtracting URL features...")

    feature_rows = [
        extract_features(url) for url in urls
    ]

    X = pd.DataFrame(feature_rows)

    expected_features = [
        "url_length",
        "dot_count",
        "hyphen_count",
        "at_symbol",
        "https",
        "has_ip",
        "subdomains",
        "digit_count",
        "slash_count",
        "question_count",
        "equals_count",
        "suspicious_words",
        "path_length",
    ]

    # Explicitly preserve the schema expected by FastAPI.
    X = X[expected_features]

    if X.isnull().any().any():
        raise ValueError(
            "Missing values were found in the feature matrix."
        )

    print(f"Feature count: {X.shape[1]}")
    print(f"Feature names: {list(X.columns)}")

    return X


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def main():
    data = load_dataset()

    X = build_feature_matrix(data["url"])
    y = data["label"]

    # Split only after exact URL deduplication.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDataset split:")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("\nTraining the Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test, predictions
    )

    phishing_precision = precision_score(
        y_test,
        predictions,
        pos_label=PHISHING_LABEL,
        zero_division=0,
    )

    phishing_recall = recall_score(
        y_test,
        predictions,
        pos_label=PHISHING_LABEL,
        zero_division=0,
    )

    phishing_f1 = f1_score(
        y_test,
        predictions,
        pos_label=PHISHING_LABEL,
        zero_division=0,
    )

    legitimate_precision = precision_score(
        y_test,
        predictions,
        pos_label=LEGITIMATE_LABEL,
        zero_division=0,
    )

    legitimate_recall = recall_score(
        y_test,
        predictions,
        pos_label=LEGITIMATE_LABEL,
        zero_division=0,
    )

    legitimate_f1 = f1_score(
        y_test,
        predictions,
        pos_label=LEGITIMATE_LABEL,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nPhishing metrics (positive class = 0):")
    print(f"Precision: {phishing_precision:.4f}")
    print(f"Recall:    {phishing_recall:.4f}")
    print(f"F1-score:  {phishing_f1:.4f}")

    print("\nLegitimate metrics (class = 1):")
    print(f"Precision: {legitimate_precision:.4f}")
    print(f"Recall:    {legitimate_recall:.4f}")
    print(f"F1-score:  {legitimate_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[
                PHISHING_LABEL,
                LEGITIMATE_LABEL,
            ],
            target_names=[
                "Phishing",
                "Legitimate",
            ],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            PHISHING_LABEL,
            LEGITIMATE_LABEL,
        ],
    )

    print(
        "Confusion matrix "
        "(rows = actual, columns = predicted):"
    )
    print(matrix)

    print("\nInterpretation:")
    print("Top-left: phishing correctly detected.")
    print("Top-right: phishing missed.")
    print("Bottom-left: legitimate URL falsely flagged.")
    print("Bottom-right: legitimate URL correctly classified.")

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\nTop feature importances:")

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance", ascending=False
    )

    print(importance_df.to_string(index=False))

    # --------------------------------------------------------
    # SAVE MODEL ARTIFACTS
    # --------------------------------------------------------

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(
        list(X.columns),
        FEATURE_NAMES_PATH,
    )

    print("\nModel saved successfully:")
    print(f"Model: {MODEL_PATH}")
    print(f"Features: {FEATURE_NAMES_PATH}")

    # --------------------------------------------------------
    # CHECK KNOWN LEGITIMATE URLS
    # This is a diagnostic only, not a benchmark.
    # --------------------------------------------------------

    print("\nDiagnostic predictions for known websites:")

    sample_urls = [
        "https://www.microsoft.com/",
        "https://www.google.com/",
        "https://www.wikipedia.org/",
        "https://github.com/",
        "https://www.python.org/",
    ]

    sample_features = pd.DataFrame(
        [
            extract_features(url)
            for url in sample_urls
        ],
        columns=list(X.columns),
    )

    sample_predictions = model.predict(
        sample_features
    )

    for url, prediction in zip(
        sample_urls, sample_predictions
    ):
        label = (
            "Phishing"
            if int(prediction) == PHISHING_LABEL
            else "Legitimate"
        )
        print(f"{url} -> {label}")

    print(
        "\nNote: these example URLs are diagnostic checks, "
        "not a substitute for an independent labeled test set."
    )


if __name__ == "__main__":
    main()
