"use client";

import { useEffect, useState } from 'react';

export default function LiveGames() {
    const [winProb, setWinProb] = useState(54);

    useEffect(() => {
        const interval = setInterval(() => {
            setWinProb(prev => Math.min(Math.max(prev + (Math.random() * 4 - 2), 10), 90));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-700">
            <header className="mb-10 text-center md:text-left">
                <h2 className="text-3xl font-extrabold mb-2 text-white"><span className="text-red-500 animate-pulse mr-2">●</span>Live In-Game Analytics</h2>
                <p className="text-gray-400 text-lg">Real-time win probability shifts and play-by-play model updates.</p>
            </header>

            <div className="glass-panel p-8 rounded-3xl">
                <div className="flex justify-between items-center mb-8">
                    <div className="text-2xl font-bold">LAL <span className="text-sm font-normal text-gray-400 block">102</span></div>
                    <div className="text-sm text-[#45f3ff] font-bold px-4 py-1 bg-white/5 rounded-full border border-white/10">Q4 05:12</div>
                    <div className="text-2xl font-bold text-right">BOS <span className="text-sm font-normal text-gray-400 block">98</span></div>
                </div>

                <div className="w-full bg-white/5 rounded-full h-8 overflow-hidden relative border border-white/10">
                    <div
                        className="bg-gradient-to-r from-[#b026ff] to-[#45f3ff] h-full absolute left-0 transition-all duration-1000 ease-out"
                        style={{ width: `${winProb}%` }}
                    ></div>
                </div>
                <div className="flex justify-between mt-3 text-sm font-black tracking-wider uppercase">
                    <span className="text-white">LAL Win Prob: {winProb.toFixed(1)}%</span>
                    <span className="text-white">BOS Win Prob: {(100 - winProb).toFixed(1)}%</span>
                </div>
            </div>
        </div>
    );
}
