# AstroHoops — Quick Start Guide

## Prerequisites
- **Python 3.8+** with a virtual environment (`.venv/`)
- **Node.js 18+** and npm

## First-Time Setup

### 1. Backend (Python / FastAPI)
```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend (Next.js)
```bash
cd frontend
npm install
```

### 3. Train the ML Model (if not already trained)
```bash
cd models
python train_model.py
```
This fetches historical NBA data, engineers features, trains a model, and saves `.pkl` artifacts.

---

## Running AstroHoops

### Option A: One-Click Launch (Windows)
Double-click **`start.bat`** in the project root.

This opens two terminal windows:
- **Backend** → `http://127.0.0.1:8000`
- **Frontend** → `http://localhost:3000`

### Option B: Manual Launch

**Terminal 1 — Backend:**
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Using AstroHoops

| Page | URL | What It Does |
|------|-----|--------------|
| **Dashboard** | `/` | Today's games with AI predictions, market odds, value edge |
| **Predictions** | `/predictions` | Pick any two teams for a custom AI prediction |
| **Live Games** | `/live` | Real-time win probability tracker |
| **Model Insights** | `/insights` | Feature importance & model explanation |
| **History** | `/history` | ROI tracking and accuracy metrics |

### Market Data (Optional)
To overlay live Polymarket/Kalshi odds, run the market stream in a third terminal:
```bash
cd market_agents
python unify_live_feed.py
```
This writes to `data/live_probs.jsonl`, which the backend reads automatically.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/teams` | List all 30 NBA teams |
| GET | `/api/games/today` | Today's games with ML predictions |
| POST | `/api/predict` | Custom matchup prediction (`{ "home_team": "LAL", "away_team": "BOS" }`) |

---

## Troubleshooting

- **Backend won't start?** Make sure you're in the `.venv` and have `pip install -r backend/requirements.txt`
- **Frontend errors?** Run `npm install` inside `frontend/`
- **No predictions?** Train the model first: `python models/train_model.py`
- **Market data missing?** That's optional — the app works fine without it

Good luck! 🏀
