
const form = document.getElementById("scan-form");
const urlInput = document.getElementById("url-input");
const scanButton = document.getElementById("scan-button");
const buttonText = document.getElementById("button-text");
const message = document.getElementById("message");
const results = document.getElementById("results");

function showMessage(text, type = "error") {
  message.textContent = text;
  message.className = `message ${type}`;
  message.hidden = false;
}

function clearMessage() {
  message.textContent = "";
  message.hidden = true;
  message.className = "message";
}

function setLoading(loading) {
  scanButton.disabled = loading;
  buttonText.textContent = loading ? "Scanning..." : "Scan URL";
}

function renderResults(data) {
  const phishing = data.prediction === "Phishing";
  const category = phishing ? "phishing" : "legitimate";

  document.getElementById("result-badge").textContent =
    phishing ? "Potentially Phishing" : "Likely Legitimate";
  document.getElementById("result-badge").className =
    `badge ${category}`;

  document.getElementById("result-symbol").textContent =
    phishing ? "!" : "✓";
  document.getElementById("result-symbol").className =
    `result-symbol ${category}`;

  document.getElementById("result-title").textContent =
    phishing ? "Suspicious URL detected" : "Likely legitimate URL";

  document.getElementById("result-description").textContent =
    phishing
      ? "The model detected patterns associated with phishing. Avoid entering personal or financial information."
      : "The model classified this URL as legitimate. This result does not guarantee that the website is safe.";

  document.getElementById("scanned-url").textContent = data.url;

  document.getElementById("phishing-score").textContent =
    `${data.phishing_probability.toFixed(2)}%`;
  document.getElementById("legitimate-score").textContent =
    `${data.legitimate_probability.toFixed(2)}%`;

  document.getElementById("phishing-meter").style.width =
    `${data.phishing_probability}%`;
  document.getElementById("legitimate-meter").style.width =
    `${data.legitimate_probability}%`;

  const featuresBody = document.getElementById("features-body");
  featuresBody.replaceChildren();

  for (const [name, value] of Object.entries(data.features)) {
    const row = document.createElement("tr");
    const featureCell = document.createElement("td");
    const valueCell = document.createElement("td");

    featureCell.textContent = name;
    valueCell.textContent = String(value);

    row.append(featureCell, valueCell);
    featuresBody.appendChild(row);
  }

  document.getElementById("feature-count").textContent =
    `${Object.keys(data.features).length} features`;

  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessage();
  results.hidden = true;

  const url = urlInput.value.trim();

  if (!url) {
    showMessage("Please enter a URL to scan.");
    urlInput.focus();
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/api/scan", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url }),
    });

    const data = await response.json();

    if (!response.ok) {
      const detail = typeof data.detail === "string"
        ? data.detail
        : "The URL could not be analyzed.";

      throw new Error(detail);
    }

    renderResults(data);
    showMessage("Scan completed successfully.", "success");
  } catch (error) {
    showMessage(
      error instanceof TypeError
        ? "Cannot reach the API. Check that the backend is running."
        : error.message || "An unexpected error occurred."
    );
  } finally {
    setLoading(false);
  }
});
