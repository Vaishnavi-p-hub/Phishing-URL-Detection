
from typing import Literal

from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="URL to classify; it will not be opened.",
    )


class ScanResponse(BaseModel):
    url: str
    prediction: Literal["Phishing", "Legitimate"]
    prediction_code: Literal[0, 1]
    phishing_probability: float
    legitimate_probability: float
    features: dict[str, int]
    disclaimer: str
