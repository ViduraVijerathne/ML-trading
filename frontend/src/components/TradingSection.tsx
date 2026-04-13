import { useState, useEffect, useRef } from "react";
import {
  startTrading,
  stopTrading,
  getTradingStatus,
  executeOnce,
  closePosition,
  updateTradingSettings,
  type TradingStatus,
} from "../lib/api";
import StatsCard from "./StatsCard";

export default function TradingSection() {
  const [status, setStatus] = useState<TradingStatus | null>(null);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Settings form state
  const [leverage, setLeverage] = useState(10);
  const [tradeAmount, setTradeAmount] = useState(1);
  const [takeProfitPct, setTakeProfitPct] = useState(1);
  const [stopLossPct, setStopLossPct] = useState(0.5);
  const [commissionFee, setCommissionFee] = useState(0.04);
  const [settingsSaved, setSettingsSaved] = useState(false);

  const fetchStatus = async () => {
    try {
      const s = await getTradingStatus();
      setStatus(s);
    } catch (e) {
      setError(String(e));
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  // Sync settings from server status
  useEffect(() => {
    if (status?.settings) {
      setLeverage(status.settings.leverage);
      setTradeAmount(status.settings.trade_amount);
      setTakeProfitPct(status.settings.take_profit_pct * 100);
      setStopLossPct(status.settings.stop_loss_pct * 100);
      setCommissionFee(status.settings.commission_fee * 100);
    }
  }, [status?.settings?.leverage, status?.settings?.trade_amount, status?.settings?.take_profit_pct, status?.settings?.stop_loss_pct, status?.settings?.commission_fee]);

  const saveSettings = async () => {
    setLoading("settings");
    setError("");
    try {
      await updateTradingSettings({
        leverage,
        trade_amount: tradeAmount,
        take_profit_pct: takeProfitPct / 100,
        stop_loss_pct: stopLossPct / 100,
        commission_fee: commissionFee / 100,
      });
      setSettingsSaved(true);
      setTimeout(() => setSettingsSaved(false), 2000);
      await fetchStatus();
    } catch (e) {
      setError(String(e));
    }
    setLoading("");
  };

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchStatus, 5000);
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
      await fetchStatus();
    } catch (e) {
      setError(String(e));
    }
    setLoading("");
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-bold text-white">Live Trading</h2>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded bg-gray-700 border-gray-600"
            />
            Auto-refresh (5s)
          </label>
          <button
            onClick={() => fetchStatus()}
            className="bg-gray-700 hover:bg-gray-600 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={() => handle("start", startTrading)}
            disabled={loading === "start" || status?.is_running === true}
            className="bg-green-600 hover:bg-green-700 disabled:bg-green-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
          >
            {loading === "start" ? "Starting..." : "Start Trading"}
          </button>
          <button
            onClick={() => handle("stop", stopTrading)}
            disabled={loading === "stop" || status?.is_running === false}
            className="bg-red-600 hover:bg-red-700 disabled:bg-red-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
          >
            {loading === "stop" ? "Stopping..." : "Stop Trading"}
          </button>
          <button
            onClick={() => handle("execute", executeOnce)}
            disabled={!!loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors"
          >
            {loading === "execute" ? "Executing..." : "Execute Once"}
          </button>
          <button
            onClick={() => handle("close", closePosition)}
            disabled={!!loading}
            className="bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-600/50 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors"
          >
            {loading === "close" ? "Closing..." : "Close Position"}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Settings panel */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Trading Settings</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Leverage</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={1}
                max={125}
                step={1}
                value={leverage}
                onChange={(e) => setLeverage(Number(e.target.value))}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-500 focus:outline-none"
              />
              <span className="text-gray-400 text-sm">x</span>
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Trade Amount ($)</label>
            <input
              type="number"
              min={0.1}
              step={0.1}
              value={tradeAmount}
              onChange={(e) => setTradeAmount(Number(e.target.value))}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Take Profit (%)</label>
            <input
              type="number"
              min={0.1}
              step={0.1}
              value={takeProfitPct}
              onChange={(e) => setTakeProfitPct(Number(e.target.value))}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Stop Loss (%)</label>
            <input
              type="number"
              min={0.1}
              step={0.1}
              value={stopLossPct}
              onChange={(e) => setStopLossPct(Number(e.target.value))}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Commission (%)</label>
            <input
              type="number"
              min={0}
              step={0.01}
              value={commissionFee}
              onChange={(e) => setCommissionFee(Number(e.target.value))}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={saveSettings}
            disabled={loading === "settings"}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
          >
            {loading === "settings" ? "Saving..." : "Save Settings"}
          </button>
          {settingsSaved && (
            <span className="text-green-400 text-sm">Settings saved!</span>
          )}
          <span className="text-xs text-gray-500 ml-auto">
            Position size: ${tradeAmount} x {leverage}x = ${(tradeAmount * leverage).toFixed(2)}
          </span>
        </div>
      </div>

      {status && (
        <>
          {/* Status indicator */}
          <div className="flex items-center gap-3">
            <span
              className={`inline-block w-3 h-3 rounded-full ${
                status.is_running ? "bg-green-500 animate-pulse" : "bg-gray-500"
              }`}
            />
            <span className="text-sm text-gray-400">
              {status.is_running ? "Trading is ACTIVE" : "Trading is STOPPED"}
            </span>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatsCard
              title="Balance"
              value={`$${status.balance.toFixed(2)}`}
              color="blue"
            />
            <StatsCard
              title="Total P&L"
              value={`${status.total_pnl >= 0 ? "+" : ""}$${status.total_pnl.toFixed(2)}`}
              color={status.total_pnl >= 0 ? "green" : "red"}
            />
            <StatsCard
              title="Win Rate"
              value={`${status.win_rate.toFixed(1)}%`}
              color={status.win_rate >= 60 ? "green" : status.win_rate >= 50 ? "yellow" : "red"}
              subtitle={`${status.winning_trades}W / ${status.losing_trades}L`}
            />
            <StatsCard
              title="Total Trades"
              value={status.total_trades}
              color="default"
            />
          </div>

          {/* Current position */}
          {status.current_position && (
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Current Position</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Side:</span>{" "}
                  <span
                    className={`font-medium ${
                      status.current_position.side === "LONG" ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {status.current_position.side}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Entry:</span>{" "}
                  <span className="text-white">${status.current_position.entry_price.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Quantity:</span>{" "}
                  <span className="text-white">{status.current_position.quantity.toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Unrealized P&L:</span>{" "}
                  <span
                    className={
                      status.current_position.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"
                    }
                  >
                    ${status.current_position.unrealized_pnl.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Trade history */}
          {status.trade_history && status.trade_history.length > 0 && (
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-700">
                <h3 className="text-sm font-medium text-gray-400">
                  Live Trade History ({status.trade_history.length} trades)
                </h3>
              </div>
              <div className="overflow-x-auto max-h-72 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-800 sticky top-0">
                    <tr className="text-gray-400 text-xs uppercase">
                      <th className="px-3 py-2 text-left">#</th>
                      <th className="px-3 py-2 text-left">Signal</th>
                      <th className="px-3 py-2 text-right">Entry</th>
                      <th className="px-3 py-2 text-right">Exit</th>
                      <th className="px-3 py-2 text-right">P&L</th>
                      <th className="px-3 py-2 text-left">Reason</th>
                      <th className="px-3 py-2 text-left">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {status.trade_history.map((t) => (
                      <tr
                        key={t.trade_id}
                        className="border-t border-gray-700/50 hover:bg-gray-700/30"
                      >
                        <td className="px-3 py-2 text-gray-400">{t.trade_id}</td>
                        <td className="px-3 py-2">
                          <span
                            className={`font-medium ${
                              t.signal === "LONG" ? "text-green-400" : "text-red-400"
                            }`}
                          >
                            {t.signal}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right text-gray-300">
                          ${t.entry_price.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right text-gray-300">
                          ${t.exit_price.toFixed(2)}
                        </td>
                        <td
                          className={`px-3 py-2 text-right font-medium ${
                            t.pnl >= 0 ? "text-green-400" : "text-red-400"
                          }`}
                        >
                          {t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(4)}
                        </td>
                        <td className="px-3 py-2">
                          <span
                            className={`text-xs px-2 py-0.5 rounded-full ${
                              t.exit_reason === "take_profit"
                                ? "bg-green-500/20 text-green-400"
                                : t.exit_reason === "stop_loss"
                                ? "bg-red-500/20 text-red-400"
                                : "bg-yellow-500/20 text-yellow-400"
                            }`}
                          >
                            {t.exit_reason.replace("_", " ")}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-gray-400 text-xs">{t.timestamp}</td>
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
  );
}
