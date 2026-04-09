export default function History() {
    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-700">
            <header className="mb-10 text-center md:text-left">
                <h2 className="text-3xl font-extrabold mb-2 text-white">Historical Accuracy</h2>
                <p className="text-gray-400 text-lg">Track ROI vs Markets and algorithmic performance calibration.</p>
            </header>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="glass-panel rounded-2xl p-6 text-center border-t-2 border-t-[#00ff88]">
                    <span className="block text-gray-400 uppercase text-xs tracking-wider mb-2">Season ROI</span>
                    <span className="text-3xl font-black text-[#00ff88]">+14.2%</span>
                </div>
                <div className="glass-panel rounded-2xl p-6 text-center border-t-2 border-t-[#45f3ff]">
                    <span className="block text-gray-400 uppercase text-xs tracking-wider mb-2">Accuracy (ML Preds)</span>
                    <span className="text-3xl font-black text-white">68.4%</span>
                </div>
                <div className="glass-panel rounded-2xl p-6 text-center border-t-2 border-t-[#b026ff]">
                    <span className="block text-gray-400 uppercase text-xs tracking-wider mb-2">Total Games Modeled</span>
                    <span className="text-3xl font-black text-white">412</span>
                </div>
            </div>
            <div className="glass-panel p-8 rounded-3xl h-64 flex items-center justify-center bg-black/40 relative overflow-hidden">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-[1px] bg-[#45f3ff]/20"></div>
                <div className="absolute inset-x-8 bottom-8 top-8 opacity-30">
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full stroke-[#b026ff]" fill="none" strokeWidth="2">
                        <polyline points="0,80 20,70 40,40 60,60 80,30 100,10" />
                    </svg>
                </div>
                <p className="text-gray-300 font-bold z-10 relative bg-black/50 px-6 py-2 rounded-xl backdrop-blur-sm shadow-xl">Bankroll Growth Trajectory</p>
            </div>
        </div>
    );
}
