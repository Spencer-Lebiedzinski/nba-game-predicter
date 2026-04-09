export default function Insights() {
    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-700">
            <header className="mb-10 text-center md:text-left">
                <h2 className="text-3xl font-extrabold mb-2 text-white">Model Insights</h2>
                <p className="text-gray-400 text-lg">Feature importance, tree breakdowns, and calibration visualizations.</p>
            </header>
            <div className="glass-panel p-8 rounded-3xl grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                    <h3 className="text-[#45f3ff] text-xl font-bold mb-4">Feature Importance</h3>
                    <ul className="space-y-3">
                        <li className="flex flex-col">
                            <span className="text-gray-300 text-sm mb-1 uppercase tracking-wide">Offensive Rating Diff</span>
                            <div className="w-full bg-white/10 rounded-full h-2">
                                <div className="bg-[#b026ff] h-full" style={{ width: '85%' }}></div>
                            </div>
                        </li>
                        <li className="flex flex-col">
                            <span className="text-gray-300 text-sm mb-1 uppercase tracking-wide">Rest Days</span>
                            <div className="w-full bg-white/10 rounded-full h-2">
                                <div className="bg-[#b026ff] h-full" style={{ width: '60%' }}></div>
                            </div>
                        </li>
                        <li className="flex flex-col">
                            <span className="text-gray-300 text-sm mb-1 uppercase tracking-wide">Defensive Rating Diff</span>
                            <div className="w-full bg-white/10 rounded-full h-2">
                                <div className="bg-[#b026ff] h-full" style={{ width: '55%' }}></div>
                            </div>
                        </li>
                    </ul>
                </div>
                <div className="flex items-center justify-center p-6 border border-white/10 rounded-xl bg-black/40">
                    <p className="text-gray-400 italic font-light text-center">Interactive D3/Plotly charts to be injected here</p>
                </div>
            </div>
        </div>
    );
}
