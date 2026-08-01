# Frontend

Small React + Vite demonstration UI for the Kenyan News Sentiment Analysis project. It's a single form: type a headline, hit Predict, see whether the model calls it positive, neutral, or negative.

Talks to the FastAPI backend at `POST /api/v1/predict` (see [../../app/README.md](../../app/README.md)).

## Run it

1. Start the backend first, from the project root (two levels up):

   ```bash
   uvicorn app.main:app --reload
   ```

   It should be live at `http://127.0.0.1:8000`.

2. From this folder (`frontend/frontend`), install dependencies:

   ```bash
   npm install
   ```

3. Start the dev server:

   ```bash
   npm run dev
   ```

4. Open the URL it prints — `http://localhost:5173` by default (fixed in `vite.config.js`).

Other scripts: `npm run build` (production build), `npm run preview` (serve the build locally).

## Project structure

* `src/api.js` — the only file that talks to the backend. All fetch calls live here, so if we add more API calls later (or reuse this in another part of the app), we just add another function to this file and import it.
* `src/SentimentDemo.jsx` — the UI. It imports `predictSentiment` from `api.js` instead of calling `fetch` itself.
* `src/SentimentDemo.css` — styles for the component, using plain CSS classes instead of inline styles.
* `src/main.jsx` — React entry point.

## Configuring the API URL

The base URL comes from the `VITE_API_BASE_URL` environment variable, read in `src/api.js`, falling back to `http://127.0.0.1:8000` if unset.

There is no `.env.example` in this folder — create a `.env` file yourself if you need to point at a different backend URL:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

`.env` is git-ignored, so this stays local to your machine.

## Notes

* The backend needs to allow CORS from whatever port this runs on. `http://localhost:5173` is already whitelisted in `app/config.py` on the backend, so this should just work with the default Vite port.
* This folder is nested as `frontend/frontend/` (not `frontend/`) — that's the actual layout in this repo, not a typo.
