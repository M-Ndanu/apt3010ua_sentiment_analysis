
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function predictSentiment(headline) {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ headline })
    });

    if (!response.ok) {
        if (response.status === 422) {
            throw new Error("Headline can't be empty.");
        }
        throw new Error(`API error: ${response.status}`);
    }

    return response.json(); // { sentiment }
}
