import { useState } from 'react'

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1'

const SENTIMENT_STYLES = {
    positive: { emoji: '🙂', className: 'sentiment-positive' },
    negative: { emoji: '🙁', className: 'sentiment-negative' },
    neutral: { emoji: '😐', className: 'sentiment-neutral' },
}

function App() {
    const [headline, setHeadline] = useState('')
    const [status, setStatus] = useState('idle')
    const [result, setResult] = useState(null)
    const [errorMessage, setErrorMessage] = useState('')

const handleSubmit = async (e) => {
    e.preventDefault()

    const trimmed = headline.trim()
    if (!trimmed) {
        setStatus('error')
        setErrorMessage('Please enter a headline before submitting.')
        setResult(null)
        return
    }

    setStatus('loading')
    setErrorMessage('')
    setResult(null)

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ headline: trimmed }),
        })

        if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail || `Request failed (status ${response.status})`)
        }

        const data = await response.json()
        setResult(data.sentiment)
        setStatus('success')
    } catch (err) {
        setStatus('error')
        if (err instanceof TypeError) {
        setErrorMessage(
        "Couldn't reach the API. Make sure the backend is running at " + API_BASE_URL
        )
        } else {
        setErrorMessage(err.message)
        }
    }
    }

    const sentimentInfo = result ? SENTIMENT_STYLES[result.toLowerCase()] : null

    return (
    <div className="page">
        <div className="card">
        <h1>Kenyan News Sentiment Analysis</h1>
        <p className="subtitle">Enter a news headline to predict its sentiment.</p>

        <form onSubmit={handleSubmit} className="form">
            <input
            type="text"
            value={headline}
            onChange={(e) => setHeadline(e.target.value)}
            placeholder="e.g. Government unveils affordable housing programme"
            className="input"
            disabled={status === 'loading'}
            />
            <button type="submit" className="button" disabled={status === 'loading'}>
            {status === 'loading' ? 'Analyzing...' : 'Predict Sentiment'}
            </button>
        </form>

        <div className="result-area">
            {status === 'loading' && <p className="loading">Analyzing headline…</p>}

            {status === 'error' && <p className="error">⚠️ {errorMessage}</p>}

            {status === 'success' && result && (
            <div className={`result ${sentimentInfo?.className || ''}`}>
                <span className="result-emoji">{sentimentInfo?.emoji}</span>
                <span className="result-label">{result.toUpperCase()}</span>
            </div>
            )}
        </div>
        </div>
    </div>
    )
}

export default App