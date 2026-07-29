import { useState } from "react";
import { Newspaper, Loader2, TriangleAlert, ArrowRight } from "lucide-react";

// ---------------------------------------------------------------
// Real backend, confirmed working. Contract: POST /api/v1/predict,
// body { headline } -> response { sentiment }. No confidence score.
// If John deploys this somewhere other than localhost, just update
// API_URL below.
// ---------------------------------------------------------------
const API_URL = "http://127.0.0.1:8000/api/v1/predict";

async function predictSentiment(headline) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ headline }),
  });

  if (!res.ok) {
    if (res.status === 422) throw new Error("Headline can't be empty.");
    throw new Error(`API error: ${res.status}`);
  }

  return res.json(); // { sentiment }
}

const SENTIMENT_STYLES = {
  positive: { color: "#1a7f4e", bg: "#e7f6ee", label: "Positive" },
  neutral: { color: "#3d5a80", bg: "#e8eef5", label: "Neutral" },
  negative: { color: "#b3402b", bg: "#fbeae6", label: "Negative" },
};

export default function SentimentDemo() {
  const [headline, setHeadline] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = headline.trim();

    if (!trimmed) {
      setStatus("error");
      setErrorMsg("Type a headline first.");
      return;
    }

    setStatus("loading");
    setErrorMsg("");

    try {
      const data = await predictSentiment(trimmed);
      setResult(data);
      setStatus("success");
    } catch (err) {
      setStatus("error");
      setErrorMsg(
        err.message === "Failed to fetch"
          ? "Couldn't reach the model. Is the API running on http://127.0.0.1:8000?"
          : err.message
      );
    }
  }

  const sentimentStyle = result ? SENTIMENT_STYLES[result.sentiment] : null;

  return (
    <div
      style={{
        minHeight: "100%",
        width: "100%",
        display: "flex",
        justifyContent: "center",
        padding: "48px 20px",
        fontFamily:
          "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color: "#1c2733",
      }}
    >
      <div style={{ width: "100%", maxWidth: 560 }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <Newspaper size={22} strokeWidth={2} color="#3d5a80" />
          <span style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.06em", color: "#6b7785", textTransform: "uppercase" }}>
            Kenyan News Sentiment
          </span>
        </div>
        <h1 style={{ fontSize: 26, fontWeight: 700, margin: "0 0 6px", lineHeight: 1.25 }}>
          What does this headline sound like?
        </h1>
        <p style={{ fontSize: 14.5, color: "#5b6675", margin: "0 0 28px", lineHeight: 1.5 }}>
          Paste any news headline below and the model will classify it as positive, neutral, or negative.
        </p>

        {/* Input form */}
        <form onSubmit={handleSubmit} style={{ marginBottom: 20 }}>
          <div
            style={{
              display: "flex",
              gap: 10,
              alignItems: "stretch",
            }}
          >
            <input
              type="text"
              value={headline}
              onChange={(e) => setHeadline(e.target.value)}
              placeholder="e.g. Ruto launches new SME funding programme"
              style={{
                flex: 1,
                padding: "14px 16px",
                fontSize: 15,
                border: "1.5px solid #d6dce3",
                borderRadius: 10,
                outline: "none",
                fontFamily: "inherit",
                transition: "border-color 0.15s ease",
              }}
              onFocus={(e) => (e.target.style.borderColor = "#3d5a80")}
              onBlur={(e) => (e.target.style.borderColor = "#d6dce3")}
              disabled={status === "loading"}
            />
            <button
              type="submit"
              disabled={status === "loading"}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "0 20px",
                fontSize: 15,
                fontWeight: 600,
                color: "#fff",
                background: status === "loading" ? "#8fa0b3" : "#3d5a80",
                border: "none",
                borderRadius: 10,
                cursor: status === "loading" ? "default" : "pointer",
                transition: "background 0.15s ease",
                whiteSpace: "nowrap",
              }}
            >
              {status === "loading" ? (
                <Loader2 size={17} className="spin" />
              ) : (
                <>
                  Predict <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>
        </form>

        {/* Error state */}
        {status === "error" && (
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 10,
              padding: "14px 16px",
              background: "#fbeae6",
              border: "1px solid #f0c9be",
              borderRadius: 10,
              color: "#8a3421",
              fontSize: 14,
              marginBottom: 20,
            }}
          >
            <TriangleAlert size={18} style={{ flexShrink: 0, marginTop: 1 }} />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Result card */}
        {status === "success" && result && sentimentStyle && (
          <div
            style={{
              padding: "22px 24px",
              background: sentimentStyle.bg,
              borderRadius: 14,
              border: `1px solid ${sentimentStyle.color}22`,
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", color: "#6b7785", textTransform: "uppercase", marginBottom: 8 }}>
              Predicted sentiment
            </div>
            <div style={{ marginBottom: 4 }}>
              <span style={{ fontSize: 28, fontWeight: 700, color: sentimentStyle.color }}>
                {sentimentStyle.label}
              </span>
            </div>
            <p style={{ fontSize: 13.5, color: "#5b6675", margin: "10px 0 0", lineHeight: 1.5 }}>
              "{headline}"
            </p>
          </div>
        )}

        {/* Empty / idle state hint */}
        {status === "idle" && (
          <p style={{ fontSize: 13.5, color: "#93a0af", textAlign: "center", marginTop: 8 }}>
            Try a real headline from the dataset, or make one up.
          </p>
        )}

        <style>{`
          .spin { animation: spin 0.8s linear infinite; }
          @keyframes spin { to { transform: rotate(360deg); } }
        `}</style>
      </div>
    </div>
  );
}
