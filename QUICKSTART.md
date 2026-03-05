# Quick Start Guide - NBA Game Predictor

## Installation & Setup

### 1. Prerequisites
- Python 3.8+
- macOS/Linux/Windows

### 2. Clone & Setup
```bash
cd /Users/spencerlebiedzinski/nba-game-predicter
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install nba_api flask scikit-learn joblib requests websockets aiohttp python-dateutil
```

### 4. Train the Model (First Run)
```bash
python train_model.py
```

This will:
- Fetch 8 years of NBA game data (2015-2023)
- Engineer 12 features from game statistics
- Train a logistic regression model
- Save model artifacts (model.pkl, team_features.pkl, etc.)
- Display accuracy: ~60.24%

### 5. Start the Web App
```bash
python app.py
# App will be available at http://127.0.0.1:5001
```

## Using the Predictor

### Manual Predictions
1. Go to http://127.0.0.1:5001/
2. Select home and away teams
3. Click "🔮 Predict Winner"
4. See prediction with confidence scores

### View Upcoming Games
1. Click "📅 View Upcoming Games" on main page
2. See:
   - Your AI model's prediction
   - Probability for each team
   - Confidence level
   - [Optional] Live market odds comparison

## Market Data Integration (Optional)

### If you have API access to Polymarket/Kalshi:

**Terminal 1: Web App**
```bash
python app.py
```

**Terminal 2: Live Market Data**
```bash
python unify_live_feed.py
```

This will:
- Discover active NBA markets on both exchanges
- Stream live prices via WebSocket
- Write updates to `data/live_probs.jsonl`
- Dashboard will show market odds vs your predictions

## Dashboard Interpretation

### Prediction Confidence
- 🟢 **High** (>10% difference): Strong conviction
- 🟡 **Medium** (5-10% difference): Moderate confidence
- 🔴 **Low** (<5% difference): Close matchup

### Market Odds Comparison
Shows your edge: **Your Probability - Market Probability**

- **Positive edge**: Market undervalues your prediction
- **Negative edge**: Market overvalues your prediction

Example:
```
Your prediction: Celtics 65%
Market odds:    Celtics 60%
Edge:           +5% (undervalued, betting opportunity)
```

## Model Features

The AI model uses 12 features:
1. **Home Point Differential (L10)** - Average margin in last 10 games
2. **Away Point Differential (L10)**
3. **Home Win Rate (L10)** - % of wins in last 10 games
4. **Away Win Rate (L10)**
5. **Home Points Per Game (L10)**
6. **Away Points Per Game (L10)**
7. **Home Defense (PAPG L10)** - Points allowed per game
8. **Away Defense (PAPG L10)**
9. **Home Consistency** - Std dev of point differentials
10. **Away Consistency**
11. **Home Rest Days** - Days since last game
12. **Away Rest Days**

## Example Workflow

```
1. Open http://127.0.0.1:5001/
   ↓
2. View upcoming games dashboard
   ↓
3. AI shows: Celtics 62% vs Warriors 38%
   ↓
4. Market odds: Celtics -110 (52% implied)
   ↓
5. Your edge: +10% (Celtics undervalued)
   ↓
6. Decision: Bet Celtics if you agree with model
```

## Monitoring Model Performance

After each season, retrain:
```bash
python train_model.py
```

Check accuracy metrics:
- Look for consistent >60% accuracy
- Monitor against baseline (57% home win %)
- Track by team strength

## Troubleshooting

### Model fails to train
- Check internet connection for NBA API
- Ensure you have 4GB+ free disk space
- Try running: `python train_model.py 2>&1 | head -50`

### Web app won't start
- Port 5001 in use? Change in app.py: `app.run(port=5002)`
- Missing dependencies? Run pip install again
- Check Flask is installed: `pip list | grep Flask`

### Market data not showing
- Optional feature - app works without it
- Need API access to Polymarket/Kalshi
- Check `data/live_probs.jsonl` file size

## API Documentation

See `MARKET_INTEGRATION.md` for detailed API specs.

## Support

For questions or issues:
1. Check the logs in terminal
2. Verify all dependencies installed
3. Ensure Python 3.8+ with `python --version`
4. Review MARKET_INTEGRATION.md for advanced features

## Next Steps

1. **Validate predictions**: Track against actual game results
2. **Optimize bets**: Focus on high-edge opportunities
3. **Expand data**: Consider adding more seasons or features
4. **Live monitoring**: Set up alerts for positive edges
5. **Risk management**: Size bets based on confidence

Good luck! 🏀
