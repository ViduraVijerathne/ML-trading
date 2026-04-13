import { useState } from "react";
import ModelSection from "./components/ModelSection";
import BacktestSection from "./components/BacktestSection";
import TradingSection from "./components/TradingSection";
import AviatorSection from "./components/AviatorSection";

type Tab = "model" | "backtest" | "trading" | "aviator";

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("aviator");

  const tabs: { key: Tab; label: string }[] = [
    { key: "aviator", label: "Aviator Bot" },
    { key: "model", label: "ML Model" },
    { key: "backtest", label: "Backtesting" },
    { key: "trading", label: "Live Trading" },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-sm font-bold">
                S
              </div>
              <div>
                <h1 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  SOL Futures Trader
                </h1>
                <p className="text-xs text-gray-500">SOLUSDT Perpetual &middot; Binance Testnet</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 bg-gray-800 px-3 py-1 rounded-full">
                10x Leverage
              </span>
              <span className="text-xs text-gray-500 bg-gray-800 px-3 py-1 rounded-full">
                $1 / Trade
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Tab navigation */}
      <div className="border-b border-gray-800 bg-gray-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex gap-1 -mb-px">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-5 py-3 text-sm font-medium transition-colors border-b-2 ${
                  activeTab === tab.key
                    ? "text-blue-400 border-blue-400"
                    : "text-gray-400 border-transparent hover:text-gray-300 hover:border-gray-600"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === "aviator" && <AviatorSection />}
        {activeTab === "model" && <ModelSection />}
        {activeTab === "backtest" && <BacktestSection />}
        {activeTab === "trading" && <TradingSection />}
      </main>
    </div>
  );
}

export default App;
