# Sentiment Frontend

Milestone 5 demo interface for the Kenyan News Sentiment Analysis project.
Talks to John's FastAPI backend (`POST /api/v1/predict`).

## Run it

1. Make sure the backend is running first (in the `apt3010ua_sentiment_analysis` repo):
   ```
   uvicorn app.main:app --reload
   ```
   It should be live at `http://127.0.0.1:8000`.

2. In this folder, install dependencies:
   ```
   npm install
   ```

3. Start the dev server:
   ```
   npm run dev
   ```

4. Open the URL it prints (usually `http://localhost:5173`).

## If the API URL changes

Open `src/SentimentDemo.jsx` and update the `API_URL` constant near the top
of the file.

## Notes

- Backend needs to allow CORS from whatever port this runs on. `http://localhost:5173`
  is already whitelisted in `app/config.py` on the backend, so this should just work.
- No confidence score is shown since the current API only returns `{ "sentiment": "..." }`.
