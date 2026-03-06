"use client";

import { useEffect, useState } from 'react';

export default function Dashboard() {
  const [games, setGames] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch from FastAPI backend
    fetch('http://127.0.0.1:8000/api/games/today')
      .then(res => res.json())
      .then(data => {
        setGames(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load games", err);
        // Fallback for visual testing if api is down
        setGames([
          {
            id: "LAL_BOS",
            home_team: { name: "Los Angeles Lakers", logo_url: "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg" },
            away_team: { name: "Boston Celtics", logo_url: "https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg" },
            prob_home_win: 0.61,
            prob_away_win: 0.39,
            predicted_score_home: 118,
            predicted_score_away: 112,
            home_market_implied: 0.54,
            away_market_implied: 0.46,
            confidence: "High",
            value_edge_home: 0.07,
            value_edge_away: -0.07,
            features: {
              "lakers_off_rtg_adv": "+4.2",
              "celtics_b2b": "Yes",
              "home_court": "Yes"
            }
          }
        ]);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[60vh]">
        <div className="animate-pulse flex flex-col items-center">
          <div className="w-16 h-16 border-4 border-[#45f3ff] border-t-transparent rounded-full animate-spin"></div>
          <p className="mt-4 text-[#b026ff] font-bold tracking-widest uppercase">Initializing Models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in zoom-in duration-700">
      <header className="mb-10 text-center md:text-left">
        <h2 className="text-3xl font-extrabold mb-2 text-white">Today's NBA Predictions</h2>
        <p className="text-gray-400 text-lg">AI-powered analytics and predictive market value edge.</p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {games.map((game, idx) => (
          <div key={idx} className="glass-panel rounded-3xl p-6 lg:p-8 flex flex-col space-y-6 hover:-translate-y-1 transition-transform duration-300">
            {/* Header / Teams */}
            <div className="flex justify-between items-center bg-black/30 rounded-2xl p-4 border border-white/5">
              <div className="flex items-center space-x-4">
                <img src={game.away_team.logo_url} alt={game.away_team.name} className="w-14 h-14" />
                <div className="font-bold text-xl">{game.away_team.name}</div>
              </div>
              <div className="text-sm font-black text-[#45f3ff] mx-4 uppercase tracking-widest text-shadow-neon">VS</div>
              <div className="flex items-center space-x-4">
                <div className="font-bold text-xl text-right">{game.home_team.name}</div>
                <img src={game.home_team.logo_url} alt={game.home_team.name} className="w-14 h-14" />
              </div>
            </div>

            {/* Core Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* AI Prediction */}
              <div className="flex flex-col items-center justify-center p-4 bg-white/5 rounded-2xl border border-white/10">
                <span className="text-xs uppercase tracking-wider text-gray-400 mb-2">AI Prediction</span>
                <div className="text-3xl font-black text-white mb-2">
                  {game.prob_home_win > game.prob_away_win ? game.home_team.name.split(' ').pop() : game.away_team.name.split(' ').pop()}
                </div>
                <div className="w-full bg-white/10 rounded-full h-3 mt-2 overflow-hidden flex">
                  <div className="bg-[#b026ff] h-full" style={{ width: `${game.prob_away_win * 100}%` }}></div>
                  <div className="bg-[#45f3ff] h-full" style={{ width: `${game.prob_home_win * 100}%` }}></div>
                </div>
                <div className="w-full flex justify-between mt-2 text-xs font-bold">
                  <span className="text-[#b026ff]">{(game.prob_away_win * 100).toFixed(1)}%</span>
                  <span className="text-[#45f3ff]">{(game.prob_home_win * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Market Odds */}
              <div className="flex flex-col items-center justify-center p-4 bg-white/5 rounded-2xl border border-white/10">
                <span className="text-xs uppercase tracking-wider text-gray-400 mb-2">Market Odds</span>
                <div className="flex justify-between w-full mt-4 text-sm font-semibold">
                  <span>{game.away_team.name.split(' ').pop()}</span>
                  <span>{(game.away_market_implied * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between w-full mt-2 text-sm font-semibold">
                  <span>{game.home_team.name.split(' ').pop()}</span>
                  <span>{(game.home_market_implied * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Value Edge & Confidence */}
              <div className="flex flex-col space-y-3">
                <div className="flex-1 bg-white/5 rounded-2xl p-4 flex flex-col items-center justify-center border border-white/10">
                  <span className="text-xs uppercase tracking-wider text-gray-400">Value Edge</span>
                  <span className={`text-2xl font-black mt-1 ${game.value_edge_home > 0 ? 'text-green-400 drop-shadow-[0_0_10px_rgba(74,222,128,0.5)]' : 'text-red-400'}`}>
                    {game.value_edge_home > 0 ? '+' : ''}{(game.value_edge_home * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex-1 bg-white/5 rounded-2xl p-4 flex flex-col items-center justify-center border border-white/10">
                  <span className="text-xs uppercase tracking-wider text-gray-400">Confidence</span>
                  <span className="text-lg font-bold mt-1 text-[#45f3ff]">{game.confidence}</span>
                </div>
              </div>
            </div>

            {/* Score Prediction & Why */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gradient-to-br from-[#0f172a] to-[#040816] rounded-2xl p-5 border border-[#45f3ff]/30 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-[#b026ff]/10 blur-2xl rounded-full"></div>
                <span className="text-xs uppercase tracking-wider text-gray-400 mb-3 block">Predicted Score</span>
                <div className="flex flex-col space-y-2 relative z-10">
                  <div className="flex justify-between items-center text-lg font-bold">
                    <span>{game.away_team.name}</span>
                    <span className="text-2xl">{game.predicted_score_away}</span>
                  </div>
                  <div className="flex justify-between items-center text-lg font-bold">
                    <span>{game.home_team.name}</span>
                    <span className="text-2xl text-[#45f3ff]">{game.predicted_score_home}</span>
                  </div>
                </div>
              </div>

              <div className="bg-white/5 rounded-2xl p-5 border border-white/10">
                <span className="text-xs uppercase tracking-wider text-[#b026ff] font-bold mb-3 block">Model Explanation</span>
                <ul className="space-y-2 text-sm text-gray-300">
                  {Object.entries(game.features).map(([key, val], i) => (
                    <li key={i} className="flex items-start">
                      <span className="text-green-400 mr-2">+</span>
                      <span>{key.replace(/_/g, ' ')}: <strong className="text-white">{String(val)}</strong></span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

          </div>
        ))}
      </div>
    </div>
  );
}
