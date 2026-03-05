from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

# Fetch data for seasons 2015-16 through 2022-23
all_games = []
for season in range(2015, 2023):
    season_str = f'{season}-{str(season+1)[-2:]}'
    print(f"Fetching {season_str}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season_str)
    games = gamefinder.get_data_frames()[0]
    all_games.append(games)

# Combine all seasons
df = pd.concat(all_games, ignore_index=True)

# --- Make sure types are clean ---
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

# Separate home/away rows using MATCHUP text
home = df[df["MATCHUP"].str.contains("vs\\.", na=False)].copy()
away = df[df["MATCHUP"].str.contains("@", na=False)].copy()

# Keep only columns we need for merging
keep = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]
home = home[keep].rename(columns={
    "TEAM_ID": "HOME_TEAM_ID",
    "TEAM_ABBREVIATION": "HOME_TEAM",
    "PTS": "HOME_PTS"
})
away = away[keep].rename(columns={
    "TEAM_ID": "AWAY_TEAM_ID",
    "TEAM_ABBREVIATION": "AWAY_TEAM",
    "PTS": "AWAY_PTS"
})

games = home.merge(away, on=["GAME_ID", "GAME_DATE"], how="inner")



# Create a per-team log from the merged games table
home_log = games[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "HOME_PTS", "AWAY_PTS"]].copy()
home_log.rename(columns={"HOME_TEAM_ID":"TEAM_ID", "HOME_PTS":"PTS", "AWAY_PTS":"OPP_PTS"}, inplace=True)

away_log = games[["GAME_ID", "GAME_DATE", "AWAY_TEAM_ID", "AWAY_PTS", "HOME_PTS"]].copy()
away_log.rename(columns={"AWAY_TEAM_ID":"TEAM_ID", "AWAY_PTS":"PTS", "HOME_PTS":"OPP_PTS"}, inplace=True)

team_games = pd.concat([home_log, away_log], ignore_index=True)
team_games["PD"] = team_games["PTS"] - team_games["OPP_PTS"]
team_games = team_games.sort_values(["TEAM_ID", "GAME_DATE"])

# --- ENHANCED FEATURES ---
# 1. Rolling Point Differential (last 10 games)
team_games["PD_L10"] = (
    team_games.groupby("TEAM_ID")["PD"]
    .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
)

# 2. Win Rate (last 10 games)
team_games["WIN"] = (team_games["PD"] > 0).astype(int)
team_games["WIN_RATE_L10"] = (
    team_games.groupby("TEAM_ID")["WIN"]
    .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
)

# 3. Points Per Game (last 10 games)
team_games["PPG_L10"] = (
    team_games.groupby("TEAM_ID")["PTS"]
    .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
)

# 4. Points Against Per Game (last 10 games)
team_games["PAPG_L10"] = (
    team_games.groupby("TEAM_ID")["OPP_PTS"]
    .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
)

# 5. Consistency/Standard Deviation of PD (last 10 games)
team_games["PD_STD_L10"] = (
    team_games.groupby("TEAM_ID")["PD"]
    .transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
)

# 6. Rest Days (approximate using date diff)
team_games = team_games.sort_values(["TEAM_ID", "GAME_DATE"])
team_games["DAYS_REST"] = team_games.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(1)

# Attach features back to games (home + away)
games = games.merge(
    team_games[["GAME_ID","TEAM_ID","PD_L10","WIN_RATE_L10","PPG_L10","PAPG_L10","PD_STD_L10","DAYS_REST"]],
    left_on=["GAME_ID","HOME_TEAM_ID"],
    right_on=["GAME_ID","TEAM_ID"],
    how="left"
).rename(columns={
    "PD_L10":"HOME_PD_L10",
    "WIN_RATE_L10":"HOME_WIN_RATE_L10",
    "PPG_L10":"HOME_PPG_L10",
    "PAPG_L10":"HOME_PAPG_L10",
    "PD_STD_L10":"HOME_PD_STD_L10",
    "DAYS_REST":"HOME_DAYS_REST"
}).drop(columns=["TEAM_ID"])

games = games.merge(
    team_games[["GAME_ID","TEAM_ID","PD_L10","WIN_RATE_L10","PPG_L10","PAPG_L10","PD_STD_L10","DAYS_REST"]],
    left_on=["GAME_ID","AWAY_TEAM_ID"],
    right_on=["GAME_ID","TEAM_ID"],
    how="left"
).rename(columns={
    "PD_L10":"AWAY_PD_L10",
    "WIN_RATE_L10":"AWAY_WIN_RATE_L10",
    "PPG_L10":"AWAY_PPG_L10",
    "PAPG_L10":"AWAY_PAPG_L10",
    "PD_STD_L10":"AWAY_PD_STD_L10",
    "DAYS_REST":"AWAY_DAYS_REST"
}).drop(columns=["TEAM_ID"])
# Label (target) - determine winner before trying to display it
games["HOME_WIN"] = (games["HOME_PTS"] > games["AWAY_PTS"]).astype(int)

# show a few rows including the engineered features and the label
feature_cols = ["GAME_DATE","HOME_TEAM","AWAY_TEAM","HOME_PD_L10","HOME_WIN_RATE_L10","HOME_PPG_L10","AWAY_PD_L10","AWAY_WIN_RATE_L10","AWAY_PPG_L10","HOME_WIN"]
print(games[feature_cols].head(15))

print(games.head())
print("Unique games:", games["GAME_ID"].nunique())

# Calculate home team win percentage
home_wins = games["HOME_WIN"].sum()
total_games = len(games)
home_win_percentage = (home_wins / total_games) * 100

print(f"\n--- HOME TEAM STATISTICS (2015-2023) ---")
print(f"Total games: {total_games}")
print(f"Home wins: {home_wins}")
print(f"Home team win percentage: {home_win_percentage:.2f}%")






#------------------------------------------------------
# Train a simple logistic regression model to predict home 
# wins based on the rolling average point differential features.
#-------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Drop early season games without enough history
model_df = games.dropna(subset=["HOME_PD_L10","AWAY_PD_L10","HOME_WIN_RATE_L10","AWAY_WIN_RATE_L10","HOME_DAYS_REST","AWAY_DAYS_REST"]).copy()

# Time split (very important)
cutoff = model_df["GAME_DATE"].quantile(0.8)

train = model_df[model_df["GAME_DATE"] <= cutoff]
test  = model_df[model_df["GAME_DATE"] > cutoff]

# Enhanced feature set
feature_cols = ["HOME_PD_L10","AWAY_PD_L10","HOME_WIN_RATE_L10","AWAY_WIN_RATE_L10",
                "HOME_PPG_L10","AWAY_PPG_L10","HOME_PAPG_L10","AWAY_PAPG_L10",
                "HOME_PD_STD_L10","AWAY_PD_STD_L10","HOME_DAYS_REST","AWAY_DAYS_REST"]

X_train = train[feature_cols].fillna(0)
y_train = train["HOME_WIN"]

X_test = test[feature_cols].fillna(0)
y_test = test["HOME_WIN"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:,1]
predictions = (probabilities >= 0.5).astype(int)

print("Test Accuracy:", accuracy_score(y_test, predictions))
print("Test Log Loss:", log_loss(y_test, probabilities))
print("Baseline (home win %):", y_test.mean())

# Save the model
import joblib
joblib.dump(model, 'nba_predictor_model.pkl')
print("Model saved as nba_predictor_model.pkl")

# Get list of teams for the web app
teams = sorted(df["TEAM_ABBREVIATION"].unique())
joblib.dump(teams, 'teams.pkl')
print("Teams saved as teams.pkl")

# Compute average features per team for prediction
team_features = team_games.groupby("TEAM_ID").agg({
    "PD_L10": "mean",
    "WIN_RATE_L10": "mean",
    "PPG_L10": "mean",
    "PAPG_L10": "mean",
    "PD_STD_L10": "mean",
    "DAYS_REST": "mean"
}).to_dict()

# Map team IDs to abbreviations
team_id_to_abbrev = df.set_index("TEAM_ID")["TEAM_ABBREVIATION"].to_dict()

# Convert to abbreviation-based dicts
team_features_abbrev = {}
for team_id, abbrev in team_id_to_abbrev.items():
    team_features_abbrev[abbrev] = {}
    for feature, values_dict in team_features.items():
        team_features_abbrev[abbrev][feature] = values_dict.get(team_id, 0)

joblib.dump(team_features_abbrev, 'team_features.pkl')
print("Team features saved as team_features.pkl")

# Also save feature column names for the web app
joblib.dump(feature_cols, 'feature_cols.pkl')
print("Feature columns saved as feature_cols.pkl")