import { useState, useEffect, useRef } from "react";
import {
  aviatorTrain,
  aviatorPredict,
  aviatorModelStatus,
  aviatorSimulate,
  aviatorBotStart,
  aviatorBotStop,
  aviatorBotStatus,
  aviatorBotSettings,
  aviatorBotReset,
  aviatorProcessRound,
  aviatorGenerateData,
  type AviatorTrainResult,
  type AviatorPrediction,
  type AviatorBotStatus as BotStatusType,
  type AviatorSimulationResult,
} from "../lib/api";
import StatsCard from "./StatsCard";

type SubTab = "model" | "bot" | "simulate";

export default function AviatorSection() {
  const [subTab, setSubTab] = useState<SubTab>("model");
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");

  // Model state
  const [modelTrained, setModelTrained] = useState(false);
  const [trainResult, setTrainResult] = useState<AviatorTrainResult | null>(null);
  const [prediction, setPrediction] = useState<AviatorPrediction | null>(null);
  const [useSynthetic, setUseSynthetic] = useState(true);
  const [nRounds, setNRounds] = useState(10000);

  // Bot state
  const [botStatus, setBotStatus] = useState<BotStatusType | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [manualCrash, setManualCrash] = useState("");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Bot settings form
  const [betAmount, setBetAmount] = useState(1);
  const [cashoutTarget, setCashoutTarget] = useState(2);
  const [maxLosses, setMaxLosses] = useState(10);
  const [useMartingale, setUseMartingale] = useState(false);
  const [initialBalance, setInitialBalance] = useState(100);

  // Simulation state
  const [simResult, setSimResult] = useState<AviatorSimulationResult | null>(null);
  const [simBetAmount, setSimBetAmount] = useState(1);
  const [simBalance, setSimBalance] = useState(100);
  const [simMartingale, setSimMartingale] = useState(false);

  useEffect(() => {
    aviatorModelStatus().then((s) => setModelTrained(s.trained)).catch(() => {});
    aviatorBotStatus().then(setBotStatus).catch(() => {});
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        aviatorBotStatus().then(setBotStatus).catch(() => {});
      }, 3000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh]);

  const handle = async (action: string, fn: () => Promise<unknown>) => {
    setLoading(action);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    }
    setLoading("");
  };

  const subTabs: { key: SubTab; label: string }[] = [
    { key: "model", label: "ML Model" },
    { key: "bot", label: "Auto-Bet Bot" },
    { key: "simulate", label: "Simulation" },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="text-2xl">&#9992;</span> Aviator Prediction Bot
          </h2>
          <p className="text-sm text-gray-400 mt-1">
            ML-powered crash prediction &middot; Auto-betting when multiplier predicted &ge; 2x
          </p>
        </div>
        <span
          className={`text-xs px-3 py-1 rounded-full ${
            modelTrained
              ? "bg-green-500/20 text-green-400 border border-green-500/30"
              : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
          }`}
        >
          {modelTrained ? "Model Trained" : "Model Not Trained"}
        </span>
      </div>

      {/* Sub-tabs */}
      <div className="flex gap-1 border-b border-gray-800">
        {subTabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setSubTab(tab.key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              subTab === tab.key
                ? "text-orange-400 border-orange-400"
                : "text-gray-400 border-transparent hover:text-gray-300"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* ML Model Tab */}
      {subTab === "model" && (
        <div className="space-y-6">
          {/* Training Controls */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Train Prediction Model</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Data Source</label>
                <select
                  value={useSynthetic ? "synthetic" : "history"}
                  onChange={(e) => setUseSynthetic(e.target.value === "synthetic")}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none"
                >
                  <option value="synthetic">Synthetic Data</option>
                  <option value="history">Imported History</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Rounds</label>
                <input
                  type="number"
                  min={1000}
                  max={50000}
                  step={1000}
                  value={nRounds}
                  onChange={(e) => setNRounds(Number(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={() =>
                    handle("train", async () => {
                      const result = await aviatorTrain({
                        use_synthetic: useSynthetic,
                        n_rounds: nRounds,
                      });
                      setTrainResult(result);
                      setModelTrained(true);
                    })
                  }
                  disabled={loading === "train"}
                  className="w-full bg-orange-600 hover:bg-orange-700 disabled:bg-orange-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
                >
                  {loading === "train" ? "Training..." : "Train Model"}
                </button>
              </div>
              <div className="flex items-end">
                <button
                  onClick={() =>
                    handle("predict", async () => {
                      const p = await aviatorPredict();
                      setPrediction(p);
                    })
                  }
                  disabled={!modelTrained || loading === "predict"}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
                >
                  {loading === "predict" ? "Predicting..." : "Get Prediction"}
                </button>
              </div>
            </div>
            <button
              onClick={() =>
                handle("generate", async () => {
                  await aviatorGenerateData({ n_rounds: nRounds });
                })
              }
              disabled={loading === "generate"}
              className="text-xs text-gray-400 hover:text-gray-300 underline"
            >
              {loading === "generate" ? "Generating..." : "Generate synthetic data only (without training)"}
            </button>
          </div>

          {/* Training Results */}
          {trainResult && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatsCard
                  title="Test Accuracy"
                  value={`${trainResult.test_accuracy}%`}
                  color={trainResult.test_accuracy >= 75 ? "green" : trainResult.test_accuracy >= 60 ? "yellow" : "red"}
                />
                <StatsCard
                  title="High-Conf Accuracy"
                  value={`${trainResult.high_confidence_accuracy}%`}
                  color={trainResult.high_confidence_accuracy >= 75 ? "green" : "yellow"}
                  subtitle={`${trainResult.high_confidence_trades} trades`}
                />
                <StatsCard
                  title="CV Mean Accuracy"
                  value={`${trainResult.cv_mean_accuracy}%`}
                  color="blue"
                  subtitle={`±${trainResult.cv_std}%`}
                />
                <StatsCard
                  title="Confidence Threshold"
                  value={`${trainResult.confidence_threshold}%`}
                  color="default"
                  subtitle={`${trainResult.total_samples} samples`}
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-3">Class Distribution</h3>
                  <div className="flex gap-4">
                    <div className="flex-1 bg-green-500/10 border border-green-500/30 rounded-lg p-3 text-center">
                      <p className="text-2xl font-bold text-green-400">{trainResult.class_distribution.above_2x}</p>
                      <p className="text-xs text-gray-400">&ge; 2x rounds</p>
                    </div>
                    <div className="flex-1 bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-center">
                      <p className="text-2xl font-bold text-red-400">{trainResult.class_distribution.below_2x}</p>
                      <p className="text-xs text-gray-400">&lt; 2x rounds</p>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-3">Top Features</h3>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {Object.entries(trainResult.feature_importance)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 8)
                      .map(([name, imp]) => (
                        <div key={name} className="flex items-center gap-2">
                          <div className="flex-1 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-orange-500 h-2 rounded-full"
                              style={{ width: `${Math.min(imp * 500, 100)}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-400 w-32 truncate">{name}</span>
                          <span className="text-xs text-gray-500 w-12 text-right">{(imp * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Prediction Result */}
          {prediction && (
            <div
              className={`border rounded-xl p-5 ${
                prediction.should_bet
                  ? "bg-green-500/10 border-green-500/30"
                  : "bg-gray-800/50 border-gray-700"
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-400">Latest Prediction</h3>
                <span
                  className={`text-sm font-bold px-3 py-1 rounded-full ${
                    prediction.should_bet
                      ? "bg-green-500/20 text-green-400"
                      : "bg-gray-700 text-gray-400"
                  }`}
                >
                  {prediction.should_bet ? "BET" : "SKIP"}
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Prediction:</span>{" "}
                  <span className={`font-medium ${prediction.predicted_class === 1 ? "text-green-400" : "text-red-400"}`}>
                    {prediction.prediction}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">P(&ge;2x):</span>{" "}
                  <span className="text-white">{prediction.probability_above_2x}%</span>
                </div>
                <div>
                  <span className="text-gray-500">Confidence:</span>{" "}
                  <span className="text-white">{prediction.confidence}%</span>
                </div>
                <div>
                  <span className="text-gray-500">Last Crash:</span>{" "}
                  <span className="text-white">{prediction.last_crash_point}x</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Auto-Bet Bot Tab */}
      {subTab === "bot" && (
        <div className="space-y-6">
          {/* Bot Controls */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => handle("botStart", async () => { await aviatorBotStart(); setBotStatus(await aviatorBotStatus()); })}
              disabled={loading === "botStart" || !modelTrained}
              className="bg-green-600 hover:bg-green-700 disabled:bg-green-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
            >
              {loading === "botStart" ? "Starting..." : "Start Bot"}
            </button>
            <button
              onClick={() => handle("botStop", async () => { await aviatorBotStop(); setBotStatus(await aviatorBotStatus()); })}
              disabled={loading === "botStop"}
              className="bg-red-600 hover:bg-red-700 disabled:bg-red-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
            >
              {loading === "botStop" ? "Stopping..." : "Stop Bot"}
            </button>
            <button
              onClick={() => handle("botReset", async () => { await aviatorBotReset(initialBalance); setBotStatus(await aviatorBotStatus()); })}
              disabled={!!loading}
              className="bg-gray-600 hover:bg-gray-500 disabled:bg-gray-600/50 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors"
            >
              Reset
            </button>
            <button
              onClick={() => aviatorBotStatus().then(setBotStatus)}
              className="bg-gray-700 hover:bg-gray-600 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors"
            >
              Refresh
            </button>
            <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer ml-auto">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded bg-gray-700 border-gray-600"
              />
              Auto-refresh (3s)
            </label>
          </div>

          {/* Bot Settings */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Bot Settings</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Bet Amount ($)</label>
                <input type="number" min={0.1} step={0.1} value={betAmount} onChange={(e) => setBetAmount(Number(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Cashout Target (x)</label>
                <input type="number" min={1.1} step={0.1} value={cashoutTarget} onChange={(e) => setCashoutTarget(Number(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Max Losses</label>
                <input type="number" min={1} step={1} value={maxLosses} onChange={(e) => setMaxLosses(Number(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Initial Balance ($)</label>
                <input type="number" min={1} step={1} value={initialBalance} onChange={(e) => setInitialBalance(Number(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none" />
              </div>
              <div className="flex items-end gap-2">
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                  <input type="checkbox" checked={useMartingale} onChange={(e) => setUseMartingale(e.target.checked)}
                    className="rounded bg-gray-700 border-gray-600" />
                  Martingale
                </label>
              </div>
            </div>
            <div className="mt-4">
              <button
                onClick={() =>
                  handle("saveSettings", async () => {
                    await aviatorBotSettings({
                      bet_amount: betAmount,
                      cashout_target: cashoutTarget,
                      max_consecutive_losses: maxLosses,
                      martingale: useMartingale,
                      initial_balance: initialBalance,
                    });
                    setBotStatus(await aviatorBotStatus());
                  })
                }
                disabled={loading === "saveSettings"}
                className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
              >
                {loading === "saveSettings" ? "Saving..." : "Save Settings"}
              </button>
            </div>
          </div>

          {/* Manual Round Input */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Feed Round Result</h3>
            <p className="text-xs text-gray-500 mb-3">Enter the crash multiplier from 1xBet Aviator to feed it to the bot</p>
            <div className="flex gap-3">
              <input
                type="number"
                min={1}
                step={0.01}
                placeholder="Crash multiplier (e.g. 2.35)"
                value={manualCrash}
                onChange={(e) => setManualCrash(e.target.value)}
                className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none"
              />
              <button
                onClick={() =>
                  handle("processRound", async () => {
                    const val = parseFloat(manualCrash);
                    if (isNaN(val) || val < 1) throw new Error("Enter a valid multiplier >= 1");
                    await aviatorProcessRound(val);
                    setBotStatus(await aviatorBotStatus());
                    setManualCrash("");
                  })
                }
                disabled={loading === "processRound" || !manualCrash}
                className="bg-orange-600 hover:bg-orange-700 disabled:bg-orange-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
              >
                {loading === "processRound" ? "Processing..." : "Submit Round"}
              </button>
            </div>
          </div>

          {/* Bot Status */}
          {botStatus && (
            <>
              <div className="flex items-center gap-3">
                <span className={`inline-block w-3 h-3 rounded-full ${botStatus.is_running ? "bg-green-500 animate-pulse" : "bg-gray-500"}`} />
                <span className="text-sm text-gray-400">
                  {botStatus.is_running ? "Bot is ACTIVE" : "Bot is STOPPED"}
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatsCard title="Balance" value={`$${botStatus.balance.toFixed(2)}`} color="blue" />
                <StatsCard
                  title="Total P&L"
                  value={`${botStatus.total_pnl >= 0 ? "+" : ""}$${botStatus.total_pnl.toFixed(2)}`}
                  color={botStatus.total_pnl >= 0 ? "green" : "red"}
                />
                <StatsCard
                  title="Win Rate"
                  value={`${botStatus.win_rate}%`}
                  color={botStatus.win_rate >= 60 ? "green" : botStatus.win_rate >= 50 ? "yellow" : "red"}
                  subtitle={`${botStatus.total_wins}W / ${botStatus.total_losses}L`}
                />
                <StatsCard title="Bets / Skipped" value={`${botStatus.total_bets} / ${botStatus.total_skipped}`} color="default" />
              </div>

              {/* Last Prediction */}
              {botStatus.last_prediction && (
                <div className={`border rounded-xl p-4 ${botStatus.last_prediction.should_bet ? "bg-green-500/10 border-green-500/30" : "bg-gray-800/50 border-gray-700"}`}>
                  <h3 className="text-sm font-medium text-gray-400 mb-2">Last Prediction</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div>
                      <span className="text-gray-500">Prediction:</span>{" "}
                      <span className={botStatus.last_prediction.predicted_class === 1 ? "text-green-400" : "text-red-400"}>
                        {botStatus.last_prediction.prediction}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Confidence:</span>{" "}
                      <span className="text-white">{botStatus.last_prediction.confidence}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Action:</span>{" "}
                      <span className={botStatus.last_prediction.should_bet ? "text-green-400 font-bold" : "text-gray-400"}>
                        {botStatus.last_prediction.should_bet ? `BET $${botStatus.current_bet_amount}` : "SKIP"}
                      </span>
                    </div>
                    {("reason" in botStatus.last_prediction) && (botStatus.last_prediction as Record<string, unknown>).reason ? (
                      <div>
                        <span className="text-gray-500">Reason:</span>{" "}
                        <span className="text-gray-300">{String((botStatus.last_prediction as Record<string, unknown>).reason)}</span>
                      </div>
                    ) : null}
                  </div>
                </div>
              )}

              {/* Bet History */}
              {botStatus.bet_history && botStatus.bet_history.length > 0 && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
                  <div className="px-4 py-3 border-b border-gray-700">
                    <h3 className="text-sm font-medium text-gray-400">
                      Bet History ({botStatus.bet_history.length} bets)
                    </h3>
                  </div>
                  <div className="overflow-x-auto max-h-72 overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-800 sticky top-0">
                        <tr className="text-gray-400 text-xs uppercase">
                          <th className="px-3 py-2 text-left">#</th>
                          <th className="px-3 py-2 text-left">Outcome</th>
                          <th className="px-3 py-2 text-right">Crash</th>
                          <th className="px-3 py-2 text-right">Bet</th>
                          <th className="px-3 py-2 text-right">Profit</th>
                          <th className="px-3 py-2 text-right">Balance</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...botStatus.bet_history].reverse().map((b, idx) => (
                          <tr key={idx} className="border-t border-gray-700/50 hover:bg-gray-700/30">
                            <td className="px-3 py-2 text-gray-400">{b.round_id}</td>
                            <td className="px-3 py-2">
                              <span className={`font-medium ${b.outcome === "WIN" ? "text-green-400" : "text-red-400"}`}>
                                {b.outcome}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right text-gray-300">{b.crash_point}x</td>
                            <td className="px-3 py-2 text-right text-gray-300">${b.bet_amount}</td>
                            <td className={`px-3 py-2 text-right font-medium ${b.profit >= 0 ? "text-green-400" : "text-red-400"}`}>
                              {b.profit >= 0 ? "+" : ""}${b.profit.toFixed(2)}
                            </td>
                            <td className="px-3 py-2 text-right text-gray-300">${b.balance_after}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Simulation Tab */}
      {subTab === "simulate" && (
        <div className="space-y-6">
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Run Simulation</h3>
            <p className="text-xs text-gray-500 mb-4">
              Train the model and simulate the bot on historical data to evaluate performance
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Initial Balance ($)</label>
                <input type="number" min={10} step={10} value={simBalance} onChange={(e) => setSimBalance(Number(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Bet Amount ($)</label>
                <input type="number" min={0.1} step={0.1} value={simBetAmount} onChange={(e) => setSimBetAmount(Number(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-orange-500 focus:outline-none" />
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                  <input type="checkbox" checked={simMartingale} onChange={(e) => setSimMartingale(e.target.checked)}
                    className="rounded bg-gray-700 border-gray-600" />
                  Martingale
                </label>
              </div>
              <div className="flex items-end">
                <button
                  onClick={() =>
                    handle("simulate", async () => {
                      const result = await aviatorSimulate({
                        use_synthetic: true,
                        n_rounds: 10000,
                        initial_balance: simBalance,
                        bet_amount: simBetAmount,
                        cashout_target: 2.0,
                        martingale: simMartingale,
                      });
                      setSimResult(result);
                      setModelTrained(true);
                    })
                  }
                  disabled={loading === "simulate"}
                  className="w-full bg-orange-600 hover:bg-orange-700 disabled:bg-orange-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
                >
                  {loading === "simulate" ? "Simulating..." : "Run Simulation"}
                </button>
              </div>
            </div>
          </div>

          {simResult && (
            <>
              {/* Training Metrics */}
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-3">Model Training Results</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <StatsCard title="Test Accuracy" value={`${simResult.training.test_accuracy}%`}
                    color={simResult.training.test_accuracy >= 75 ? "green" : "yellow"} />
                  <StatsCard title="High-Conf Accuracy" value={`${simResult.training.high_confidence_accuracy}%`}
                    color={simResult.training.high_confidence_accuracy >= 75 ? "green" : "yellow"}
                    subtitle={`${simResult.training.high_confidence_trades} trades`} />
                  <StatsCard title="Conf. Threshold" value={`${simResult.training.confidence_threshold}%`} color="default" />
                  <StatsCard title="CV Accuracy" value={`${simResult.training.cv_mean_accuracy}%`} color="blue"
                    subtitle={`±${simResult.training.cv_std}%`} />
                </div>
              </div>

              {/* Simulation Results */}
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-3">Bot Simulation Results</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <StatsCard title="Final Balance" value={`$${simResult.simulation.final_balance.toFixed(2)}`}
                    color={simResult.simulation.final_balance > simResult.simulation.initial_balance ? "green" : "red"} />
                  <StatsCard title="Total P&L"
                    value={`${simResult.simulation.total_pnl >= 0 ? "+" : ""}$${simResult.simulation.total_pnl.toFixed(2)}`}
                    color={simResult.simulation.total_pnl >= 0 ? "green" : "red"} />
                  <StatsCard title="Win Rate" value={`${simResult.simulation.win_rate}%`}
                    color={simResult.simulation.win_rate >= 60 ? "green" : simResult.simulation.win_rate >= 50 ? "yellow" : "red"}
                    subtitle={`${simResult.simulation.wins}W / ${simResult.simulation.losses}L`} />
                  <StatsCard title="Bets / Skipped" value={`${simResult.simulation.bets_placed} / ${simResult.simulation.skipped}`}
                    color="default" subtitle={`of ${simResult.simulation.total_rounds} rounds`} />
                </div>
              </div>

              {/* Equity Curve */}
              {simResult.simulation.equity_curve.length > 0 && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-3">Equity Curve</h3>
                  <div className="h-48 flex items-end gap-px">
                    {(() => {
                      const curve = simResult.simulation.equity_curve;
                      const maxBal = Math.max(...curve.map((c) => c.balance));
                      const minBal = Math.min(...curve.map((c) => c.balance));
                      const range = maxBal - minBal || 1;
                      const step = Math.max(1, Math.floor(curve.length / 150));
                      const sampled = curve.filter((_, i) => i % step === 0);
                      return sampled.map((point, i) => {
                        const height = ((point.balance - minBal) / range) * 100;
                        const isProfit = point.balance >= simResult!.simulation.initial_balance;
                        return (
                          <div key={i} className="flex-1 min-w-[1px]" title={`Round ${point.round}: $${point.balance}`}>
                            <div
                              className={`w-full rounded-t-sm ${isProfit ? "bg-green-500/70" : "bg-red-500/70"}`}
                              style={{ height: `${Math.max(height, 2)}%` }}
                            />
                          </div>
                        );
                      });
                    })()}
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-2">
                    <span>Start: ${simResult.simulation.initial_balance}</span>
                    <span>End: ${simResult.simulation.final_balance.toFixed(2)}</span>
                  </div>
                </div>
              )}

              {/* Bet Log */}
              {simResult.simulation.bet_log.length > 0 && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
                  <div className="px-4 py-3 border-b border-gray-700">
                    <h3 className="text-sm font-medium text-gray-400">
                      Simulated Bet Log (last {simResult.simulation.bet_log.length})
                    </h3>
                  </div>
                  <div className="overflow-x-auto max-h-72 overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-800 sticky top-0">
                        <tr className="text-gray-400 text-xs uppercase">
                          <th className="px-3 py-2 text-left">Round</th>
                          <th className="px-3 py-2 text-left">Outcome</th>
                          <th className="px-3 py-2 text-right">Crash</th>
                          <th className="px-3 py-2 text-right">Bet</th>
                          <th className="px-3 py-2 text-right">Profit</th>
                          <th className="px-3 py-2 text-right">Balance</th>
                          <th className="px-3 py-2 text-right">Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...simResult.simulation.bet_log].reverse().map((b, idx) => (
                          <tr key={idx} className="border-t border-gray-700/50 hover:bg-gray-700/30">
                            <td className="px-3 py-2 text-gray-400">{b.round}</td>
                            <td className="px-3 py-2">
                              <span className={`font-medium ${b.outcome === "WIN" ? "text-green-400" : "text-red-400"}`}>
                                {b.outcome}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right text-gray-300">{b.crash_point}x</td>
                            <td className="px-3 py-2 text-right text-gray-300">${b.bet_amount}</td>
                            <td className={`px-3 py-2 text-right font-medium ${b.profit >= 0 ? "text-green-400" : "text-red-400"}`}>
                              {b.profit >= 0 ? "+" : ""}${b.profit.toFixed(2)}
                            </td>
                            <td className="px-3 py-2 text-right text-gray-300">${b.balance.toFixed(2)}</td>
                            <td className="px-3 py-2 text-right text-gray-400">{b.confidence}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
