from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database.db import Base
from datetime import datetime

class Team(Base):
    __tablename__ = "teams"
    id = Column(String, primary_key=True, index=True) # e.g., 'LAL'
    name = Column(String, index=True)
    city = Column(String)
    logo_url = Column(String)
    offensive_rating = Column(Float, default=0.0)
    defensive_rating = Column(Float, default=0.0)

class Player(Base):
    __tablename__ = "players"
    id = Column(String, primary_key=True, index=True)
    team_id = Column(String, ForeignKey("teams.id"))
    name = Column(String)
    efficiency_rating = Column(Float, default=0.0)
    is_injured = Column(Boolean, default=False)
    
    team = relationship("Team")

class Game(Base):
    __tablename__ = "games"
    id = Column(String, primary_key=True, index=True) # e.g., '2026-LAL-BOS'
    date = Column(DateTime)
    home_team_id = Column(String, ForeignKey("teams.id"))
    away_team_id = Column(String, ForeignKey("teams.id"))
    status = Column(String) # 'scheduled', 'live', 'completed'
    home_score = Column(Integer, default=0)
    away_score = Column(Integer, default=0)
    
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

class Prediction(Base):
    __tablename__ = "predictions"
    game_id = Column(String, ForeignKey("games.id"), primary_key=True)
    prob_home_win = Column(Float)
    prob_away_win = Column(Float)
    predicted_score_home = Column(Integer)
    predicted_score_away = Column(Integer)
    confidence = Column(String)
    features_json = Column(String) # JSON string of feature importance/explanation
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketOdd(Base):
    __tablename__ = "market_odds"
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey("games.id"))
    source = Column(String) # 'polymarket', 'kalshi', 'vegas'
    home_implied_prob = Column(Float)
    away_implied_prob = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)
