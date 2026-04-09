"use client";

import { useEffect, useState } from "react";

interface Team {
    id: string;
    name: string;
    logo_url: string;
}

interface PredictResult {
    home_team: string;
    away_team: string;
    home_name: string;
    away_name: string;
    home_logo: string;
    away_logo: string;
    prob_home_win: number;
    prob_away_win: number;
    winner: string;
    confidence: number;
}

export default function Predictions() {
    const [teams, setTeams] = useState<Team[]>([]);
    const [home, setHome] = useState("");
    const [away, setAway] = useState("");
    const [result, setResult] = useState<PredictResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        fetch("http://127.0.0.1:8000/api/teams")
            .then((r) => r.json())
            .then(setTeams)
            .catch(() => setError("Could not load teams. Is the backend running?"));
    }, []);

    const handlePredict = async () => {
        if (!home || !away) return;
        if (home === away) {
            setError("Please select two different teams.");
            return;
        }
        setError("");
        setLoading(true);
        setResult(null);
        try {
            const res = await fetch("http://127.0.0.1:8000/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ home_team: home, away_team: away }),
            });
            if (!res.ok) throw new Error("Prediction failed");
            const data: PredictResult = await res.json();
            setResult(data);
        } catch {
            setError("Prediction request failed.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-700">
            <header className="mb-10 text-center md:text-left">
                <h2 className="text-3xl font-extrabold mb-2 text-white">
                    Matchup Predictor
                </h2>
                <p className="text-gray-400 text-lg">
                    Select any two teams and let the AI make its call.
                </p>
            </header>

            {/* Team Picker */}
            <div className="glass-panel rounded-3xl p-8">
                <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-6 items-end">
                    {/* Home */}
                    <div className="flex flex-col">
                        <label className="text-xs uppercase tracking-widest text-gray-400 mb-2 font-bold">
                            Home Team
                        </label>
                        <select
                            value={home}
                            onChange={(e) => setHome(e.target.value)}
                            className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-[#45f3ff] transition-colors appearance-none cursor-pointer"
                        >
                            <option value="">Select home team...</option>
                            {teams.map((t) => (
                                <option key={t.id} value={t.id}>
                                    {t.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* VS Badge */}
                    <div className="hidden md:flex items-center justify-center">
                        <span className="text-sm font-black text-[#45f3ff] uppercase tracking-widest px-4 py-2 bg-white/5 rounded-full border border-white/10">
                            VS
                        </span>
                    </div>

                    {/* Away */}
                    <div className="flex flex-col">
                        <label className="text-xs uppercase tracking-widest text-gray-400 mb-2 font-bold">
                            Away Team
                        </label>
                        <select
                            value={away}
                            onChange={(e) => setAway(e.target.value)}
                            className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-[#45f3ff] transition-colors appearance-none cursor-pointer"
                        >
                            <option value="">Select away team...</option>
                            {teams.map((t) => (
                                <option key={t.id} value={t.id}>
                                    {t.name}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {error && (
                    <p className="mt-4 text-red-400 text-sm font-semibold text-center">
                        {error}
                    </p>
                )}

                <div className="flex justify-center mt-8">
                    <button
                        onClick={handlePredict}
                        disabled={loading || !home || !away}
                        className="px-10 py-3 rounded-full font-bold text-sm uppercase tracking-widest transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed bg-gradient-to-r from-[#b026ff] to-[#45f3ff] text-white hover:shadow-[0_0_30px_rgba(69,243,255,0.4)] hover:-translate-y-0.5 active:translate-y-0"
                    >
                        {loading ? (
                            <span className="flex items-center gap-2">
                                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                                Analyzing...
                            </span>
                        ) : (
                            "Run Prediction"
                        )}
                    </button>
                </div>
            </div>

            {/* Result Card */}
            {result && (
                <div className="glass-panel rounded-3xl p-8 animate-in fade-in zoom-in duration-500">
                    {/* Teams header */}
                    <div className="flex justify-between items-center bg-black/30 rounded-2xl p-5 border border-white/5 mb-6">
                        <div className="flex items-center space-x-4">
                            <img
                                src={result.away_logo}
                                alt={result.away_name}
                                className="w-14 h-14"
                            />
                            <div className="font-bold text-xl">{result.away_name}</div>
                        </div>
                        <div className="text-sm font-black text-[#45f3ff] mx-4 uppercase tracking-widest">
                            VS
                        </div>
                        <div className="flex items-center space-x-4">
                            <div className="font-bold text-xl text-right">
                                {result.home_name}
                            </div>
                            <img
                                src={result.home_logo}
                                alt={result.home_name}
                                className="w-14 h-14"
                            />
                        </div>
                    </div>

                    {/* Winner + Probabilities */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="flex flex-col items-center justify-center p-6 bg-gradient-to-br from-[#0f172a] to-[#040816] rounded-2xl border border-[#45f3ff]/30 relative overflow-hidden">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-[#b026ff]/10 blur-2xl rounded-full"></div>
                            <span className="text-xs uppercase tracking-wider text-gray-400 mb-2">
                                Predicted Winner
                            </span>
                            <span className="text-4xl font-black text-white z-10">
                                {result.winner === result.home_team
                                    ? result.home_name
                                    : result.away_name}
                            </span>
                            <span className="mt-2 text-[#45f3ff] font-bold text-lg">
                                {result.confidence}% Confidence
                            </span>
                        </div>

                        <div className="flex flex-col items-center justify-center p-6 bg-white/5 rounded-2xl border border-white/10">
                            <span className="text-xs uppercase tracking-wider text-gray-400 mb-4">
                                Win Probability
                            </span>
                            {/* Prob bars */}
                            <div className="w-full space-y-3">
                                <div>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span>{result.home_name}</span>
                                        <span className="text-[#45f3ff] font-bold">
                                            {(result.prob_home_win * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                                        <div
                                            className="bg-[#45f3ff] h-full transition-all duration-700"
                                            style={{ width: `${result.prob_home_win * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                                <div>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span>{result.away_name}</span>
                                        <span className="text-[#b026ff] font-bold">
                                            {(result.prob_away_win * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                                        <div
                                            className="bg-[#b026ff] h-full transition-all duration-700"
                                            style={{ width: `${result.prob_away_win * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
