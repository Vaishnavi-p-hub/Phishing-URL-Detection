
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.model_service import predict_url
from backend.schemas import ScanRequest, ScanResponse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Phishing URL Detection API",
    description=(
        "Classifies URL patterns using a trained Random Forest model. "
        "Submitted URLs are never opened by this API."
    ),
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)


def validate_url(candidate: str) -> str:
    """Validate URL syntax without making any network request."""
    candidate = candidate.strip()

    if not candidate or any(ch.isspace() for ch in candidate):
        raise HTTPException(
            status_code=422,
            detail="Enter a URL without whitespace.",
        )

    if any(ord(ch) < 32 or ord(ch) == 127 for ch in candidate):
        raise HTTPException(
            status_code=422,
            detail="The URL contains invalid control characters.",
        )

    # Support URLs pasted without a scheme.
    normalized = (
        candidate if "://" in candidate
        else "http://" + candidate
    )

    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="The URL format is invalid.",
        )

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or "." not in hostname and not _is_ip(hostname)
        or len(hostname) > 253
        or any(char.isspace() for char in hostname)
    ):
        raise HTTPException(
            status_code=422,
            detail="Enter a valid-looking HTTP or HTTPS URL.",
        )

    return normalized


def _is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "RandomForestClassifier",
        "features": 13,
    }


@app.post("/api/scan", response_model=ScanResponse)
def scan_url(request: ScanRequest):
    normalized_url = validate_url(request.url)

    try:
        result = predict_url(normalized_url)
    except Exception:
        # Log details on the server in a production deployment.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs.",
        )

    return ScanResponse(
        url=normalized_url,
        **result,
        disclaimer=(
            "Educational prediction only. The model may be wrong; "
            "a legitimate prediction does not guarantee safety."
        ),
    )
