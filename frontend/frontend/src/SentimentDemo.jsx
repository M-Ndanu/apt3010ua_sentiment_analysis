import { useState } from "react";
import { Newspaper, Loader2, TriangleAlert, ArrowRight } from "lucide-react";
import { predictSentiment } from "./api";
import "./SentimentDemo.css";

const SENTIMENT_STYLES = {
    positive: { color: "var(--positive-color)", bg: "var(--positive-bg)", label: "Positive" },
    neutral: { color: "var(--neutral-color)", bg: "var(--neutral-bg)", label: "Neutral" },
    negative: { color: "var(--negative-color)", bg: "var(--negative-bg)", label: "Negative" },
};

export default function SentimentDemo() {
    const [headline, setHeadline] = useState("");
    const [status, setStatus] = useState("idle"); // idle | loading | success | error
    const [result, setResult] = useState(null);
    const [errorMsg, setErrorMsg] = useState("");

    const handleSubmit = async (e) => {
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
        } catch (error) {
            console.error("Error:", error);
            setStatus("error");
            setErrorMsg(
                error.message === "Failed to fetch"
                    ? "Couldn't reach the model. Is the API running?"
                    : error.message
            );
        }
    };

    const sentimentStyle = result ? SENTIMENT_STYLES[result.sentiment] : null;

    return (
        <div className="sentimentPage">
            <div className="sentimentContainer">
                {/* Header */}
                <div className="sentimentHeader">
                    <Newspaper size={22} strokeWidth={2} color="var(--sentiment-blue)" />
                    <span className="sentimentHeaderLabel">Kenyan News Sentiment</span>
                </div>
                <h1 className="sentimentTitle">What does this headline sound like?</h1>
                <p className="sentimentSubtitle">
                    Paste any news headline below and the model will classify it as positive, neutral, or negative.
                </p>

                {/* Input form */}
                <form className="sentimentForm" onSubmit={handleSubmit}>
                    <div className="sentimentInputRow">
                        <input
                            type="text"
                            className="sentimentInput"
                            value={headline}
                            onChange={(e) => setHeadline(e.target.value)}
                            placeholder="e.g. Ruto launches new SME funding programme"
                            disabled={status === "loading"}
                        />
                        <button
                            type="submit"
                            className="sentimentSubmitButton"
                            disabled={status === "loading"}
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
                    <div className="sentimentError">
                        <TriangleAlert size={18} style={{ flexShrink: 0, marginTop: 1 }} />
                        <span>{errorMsg}</span>
                    </div>
                )}

                {/* Result card */}
                {status === "success" && result && sentimentStyle && (
                    <div
                        className="sentimentResultCard"
                        style={{ background: sentimentStyle.bg, borderColor: sentimentStyle.color }}
                    >
                        <div className="sentimentResultLabel">Predicted sentiment</div>
                        <div>
                            <span className="sentimentResultValue" style={{ color: sentimentStyle.color }}>
                                {sentimentStyle.label}
                            </span>
                        </div>
                        <p className="sentimentResultHeadline">"{headline}"</p>
                    </div>
                )}

                {/* Empty / idle state hint */}
                {status === "idle" && (
                    <p className="sentimentIdleHint">Try a real headline from the dataset, or make one up.</p>
                )}
            </div>
        </div>
    );
}
