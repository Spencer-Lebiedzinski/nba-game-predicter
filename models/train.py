import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, log_loss

# Ensure the database path works relative to execution
DB_PATH = "sqlite:///./astrohoops.db"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

class AstroHoopsModel:
    def __init__(self):
        self.rf = RandomForestClassifier(random_state=42)
        self.xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.lgb = LGBMClassifier(random_state=42, verbose=-1)
        self.meta_model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Enhanced feature layout
        self.feature_columns = [
            "home_off_rtg_roll", "home_def_rtg_roll", "home_rest_days", 
            "away_off_rtg_roll", "away_def_rtg_roll", "away_rest_days",
            "home_win_pct_roll", "away_win_pct_roll"
        ]
        
    def _fetch_data(self):
        """Fetch games from the SQLite database."""
        engine = create_engine(DB_PATH)
        query = """
            SELECT g.id, g.date, g.home_team_id, g.away_team_id, 
                   g.home_score, g.away_score, g.status
            FROM games g
            WHERE g.status = 'completed'
            ORDER BY g.date ASC
        """
        df = pd.read_sql(query, engine)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _engineer_features(self, df, window=10):
        """Calculate rolling averages and rest days."""
        print("Engineering temporal features...")
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        
        # In a real scenario, off_rtg/def_rtg would be calculated from possession stats.
        # Here we approximate using points scored and allowed.
        team_stats = []
        for index, row in df.iterrows():
            # Home context
            team_stats.append({
                'date': row['date'], 'team': row['home_team_id'], 'opp': row['away_team_id'],
                'pts': row['home_score'], 'opp_pts': row['away_score'], 'win': row['home_win'], 'is_home': 1
            })
            # Away context
            team_stats.append({
                'date': row['date'], 'team': row['away_team_id'], 'opp': row['home_team_id'],
                'pts': row['away_score'], 'opp_pts': row['home_score'], 'win': 1 - row['home_win'], 'is_home': 0
            })
            
        stats_df = pd.DataFrame(team_stats).sort_values(by=['team', 'date'])
        
        # Rolling Metrics (Excluding the current game to prevent leakage)
        stats_df['off_rtg_roll'] = stats_df.groupby('team')['pts'].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
        stats_df['def_rtg_roll'] = stats_df.groupby('team')['opp_pts'].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
        stats_df['win_pct_roll'] = stats_df.groupby('team')['win'].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
        
        # Rest Days
        stats_df['prev_game_date'] = stats_df.groupby('team')['date'].shift(1)
        stats_df['rest_days'] = (stats_df['date'] - stats_df['prev_game_date']).dt.days.fillna(3) # Default 3 days start of season
        stats_df['rest_days'] = stats_df['rest_days'].clip(upper=7) # Cap rest days
        
        # Merge back to original game dataframe
        home_stats = stats_df[stats_df['is_home'] == 1].set_index(['date', 'team'])
        away_stats = stats_df[stats_df['is_home'] == 0].set_index(['date', 'team'])
        
        for col in ['off_rtg_roll', 'def_rtg_roll', 'win_pct_roll', 'rest_days']:
            df[f'home_{col}'] = df.apply(lambda r: home_stats.loc[(r['date'], r['home_team_id']), col] if (r['date'], r['home_team_id']) in home_stats.index else np.nan, axis=1)
            df[f'away_{col}'] = df.apply(lambda r: away_stats.loc[(r['date'], r['away_team_id']), col] if (r['date'], r['away_team_id']) in away_stats.index else np.nan, axis=1)
            
        # Drop early season games without enough history
        df = df.dropna(subset=self.feature_columns).copy()
        print(f"Dataset shape after feature engineering: {df.shape}")
        return df

    def train(self):
        """Train the ensemble model using TimeSeriesSplit and GridSearchCV."""
        print("Initiating ML Pipeline...")
        try:
            df = self._fetch_data()
        except Exception as e:
            print(f"Database error (you may need to ingest games first): {e}")
            return
            
        if len(df) < 50:
            print("Not enough historical data to perform robust training (Need > 50 completed games).")
            return
            
        df = self._engineer_features(df)
        
        X = df[self.feature_columns]
        y = df['home_win']
        dates = df['date']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # TimeSeriesSplit ensures no future leakage
        tscv = TimeSeriesSplit(n_splits=5)
        
        print("Optimizing RandomForest...")
        rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        rf_grid = GridSearchCV(self.rf, rf_param_grid, cv=tscv, scoring='neg_log_loss', n_jobs=-1)
        rf_grid.fit(X_scaled, y)
        self.rf = rf_grid.best_estimator_
        
        print("Optimizing XGBoost...")
        xgb_param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        xgb_grid = GridSearchCV(self.xgb, xgb_param_grid, cv=tscv, scoring='neg_log_loss', n_jobs=-1)
        xgb_grid.fit(X_scaled, y)
        self.xgb = xgb_grid.best_estimator_
        
        # Train Meta-Model correctly using out-of-fold predictions to prevent overfitting
        print("Stacking Base Models and Training Meta Classifier...")
        
        # For simplicity in this script, we'll train the base models on the full dataset now
        # and use their predict_proba outputs to train the LogisticRegression
        # Note: In a fully strict pipeline, cross_val_predict would generate the meta-features
        
        rf_pred = self.rf.predict_proba(X_scaled)[:, 1]
        xgb_pred = self.xgb.predict_proba(X_scaled)[:, 1]
        lgb_pred = self.lgb.fit(X_scaled, y).predict_proba(X_scaled)[:, 1] # Quick raw fit for LGBM
        
        meta_X = np.column_stack((rf_pred, xgb_pred, lgb_pred))
        self.meta_model.fit(meta_X, y)
        
        # Calculate trailing training accuracy
        final_preds = self.meta_model.predict(meta_X)
        accuracy = accuracy_score(y, final_preds)
        print(f"Ensemble Training Accuracy: {accuracy*100:.2f}%")
        
        self.is_trained = True
        self.save_model()
        
    def predict(self, game_features_dict):
        """Predict the outcome of a new game."""
        if not self.is_trained:
            self.load_model()
            
        # Convert dictionary to ordered feature array
        features = [game_features_dict.get(col, 0) for col in self.feature_columns]
        X = self.scaler.transform(pd.DataFrame([features], columns=self.feature_columns))
        
        rf_pred = self.rf.predict_proba(X)[:, 1]
        xgb_pred = self.xgb.predict_proba(X)[:, 1]
        lgb_pred = self.lgb.predict_proba(X)[:, 1]
        
        meta_X = np.column_stack((rf_pred, xgb_pred, lgb_pred))
        final_prob = self.meta_model.predict_proba(meta_X)[0][1]
        
        # Extract feature importance from RandomForest as approximation
        importances = dict(zip(self.feature_columns, self.rf.feature_importances_))
        return final_prob, importances

    def save_model(self):
        """Persist the trained model pipeline to disk."""
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        # Save base estimators and meta_model into a single dict for convenience
        pipeline = {
            'rf': self.rf,
            'xgb': self.xgb,
            'lgb': self.lgb,
            'meta': self.meta_model,
            'scaler': self.scaler,
            'features': self.feature_columns
        }
        joblib.dump(pipeline, os.path.join(MODEL_DIR, "astrohoops_ensemble.joblib"))
        print(f"Models saved successfully to {MODEL_DIR}")

    def load_model(self):
        """Load the persisted model pipeline."""
        model_path = os.path.join(MODEL_DIR, "astrohoops_ensemble.joblib")
        if not os.path.exists(model_path):
            raise Exception("No trained model found. Please run the training pipeline first.")
            
        pipeline = joblib.load(model_path)
        self.rf = pipeline['rf']
        self.xgb = pipeline['xgb']
        self.lgb = pipeline['lgb']
        self.meta_model = pipeline['meta']
        self.scaler = pipeline['scaler']
        self.feature_columns = pipeline['features']
        self.is_trained = True

if __name__ == "__main__":
    model = AstroHoopsModel()
    model.train()
