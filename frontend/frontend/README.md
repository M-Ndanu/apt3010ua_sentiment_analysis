#Frontend

Milestone 5 demo interface for the Kenyan News Sentiment Analysis project.
Talks to the FastAPI backend (`POST /api/v1/predict`).

## Run it

1. Make sure the backend is running first (in the `apt3010ua_sentiment_analysis` repo):
   uvicorn app.main:app --reload
   It should be live at http://127.0.0.1:8000'.

2. In this folder, install dependencies:
   npm install

3. Start the dev server:
   npm run dev

4. Open the URL it prints (usually 'http://localhost:5173').

## Project structure

- src/api.js - the only file that talks to the backend. All fetch calls
  live here, so if we add more API calls later (or reuse this in another
  part of the app), we just add another function to this file and import it.
- src/SentimentDemo.jsx - the UI. It imports predictSentiment from
  api.js instead of calling fetch itself.
- src/SentimentDemo.css — styles for the component, using plain CSS
  classes instead of inline styles.

## If the API URL changes

The base URL now comes from an environment variable instead of being
hardcoded. Copy .env.example` to .env and edit it:

VITE_API_BASE_URL=http://127.0.0.1:8000


## Notes

- Backend needs to allow CORS from whatever port this runs on. 'http://localhost:5173'
  is already whitelisted in app/config.py on the backend, so this should just work.

