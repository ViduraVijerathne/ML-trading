import { useState } from "react";
import { runBacktest, type BacktestResult } from "../lib/api";
import StatsCard from "./StatsCard";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

export default function BacktestSection() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState("");

  const [interval, setInterval_] = useState("15m");
  const [limit, setLimit] = useState(1500);
  const [balance, setBalance] = useState(10);
  const [tradeAmt, setTradeAmt] = useState(1);
  const [leverage, setLeverage] = useState(10);
  const [tp, setTp] = useState(0.01);
  const [sl, setSl] = useState(0.005);
  const [feeRate, setFeeRate] = useState(0.04);

  const handleRun = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await runBacktest({
        interval,
        limit,
        initial_balance: balance,
        trade_amount: tradeAmt,
        leverage,
        take_profit_pct: tp,
        stop_loss_pct: sl,
        fee_rate: feeRate / 100,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    }
    setLoading(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-bold text-white">Backtesting</h2>
        <button
          onClick={handleRun}
          disabled={loading}
          className="bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
        >
          {loading ? "Running..." : "Run Backtest"}
        </button>
      </div>

      {/* Parameters */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
        {[
          { label: "Interval", value: interval, type: "select", options: ["5m", "15m", "1h", "4h"], onChange: setInterval_ },
          { label: "Candles", value: limit, type: "number", onChange: (v: string) => setLimit(Number(v)) },
          { label: "Balance ($)", value: balance, type: "number", onChange: (v: string) => setBalance(Number(v)) },
          { label: "Trade ($)", value: tradeAmt, type: "number", onChange: (v: string) => setTradeAmt(Number(v)) },
          { label: "Leverage", value: leverage, type: "number", onChange: (v: string) => setLeverage(Number(v)) },
          { label: "TP (%)", value: tp * 100, type: "number", onChange: (v: string) => setTp(Number(v) / 100) },
          { label: "SL (%)", value: sl * 100, type: "number", onChange: (v: string) => setSl(Number(v) / 100) },
          { label: "Fee (%)", value: feeRate, type: "number", onChange: (v: string) => setFeeRate(Number(v)) },
        ].map((p) => (
          <div key={p.label}>
            <label className="text-xs text-gray-500 block mb-1">{p.label}</label>
            {p.type === "select" ? (
              <select
                value={String(p.value)}
                onChange={(e) => (p.onChange as (v: string) => void)(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
              >
                {p.options?.map((o) => (
                  <option key={o} value={o}>{o}</option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                value={p.value}
                onChange={(e) => (p.onChange as (v: string) => void)(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                step="any"
              />
            )}
          </div>
        ))}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {result && (
        <>
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatsCard
              title="Final Balance"
              value={`$${result.final_balance.toFixed(2)}`}
              color={result.total_pnl >= 0 ? "green" : "red"}
              subtitle={`Started at $${result.initial_balance}`}
            />
            <StatsCard
              title="Total P&L"
              value={`${result.total_pnl >= 0 ? "+" : ""}$${result.total_pnl.toFixed(2)}`}
              color={result.total_pnl >= 0 ? "green" : "red"}
            />
            <StatsCard
              title="Win Rate"
              value={`${result.win_rate}%`}
              color={result.win_rate >= 60 ? "green" : result.win_rate >= 50 ? "yellow" : "red"}
              subtitle={`${result.winning_trades}W / ${result.losing_trades}L`}
            />
            <StatsCard
              title="Total Trades"
              value={result.total_trades}
              color="blue"
              subtitle={`${result.leverage}x leverage`}
            />
          </div>

          {/* Equity curve */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Equity Curve</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={result.equity_curve}>
                <XAxis dataKey="trade" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                <YAxis
                  tick={{ fill: "#9ca3af", fontSize: 11 }}
                  domain={["dataMin - 0.5", "dataMax + 0.5"]}
                  tickFormatter={(v) => `$${Number(v).toFixed(1)}`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8 }}
                  labelStyle={{ color: "#fff" }}
                  formatter={(v: number) => [`$${v.toFixed(4)}`, "Balance"]}
                  labelFormatter={(l) => `Trade #${l}`}
                />
                <ReferenceLine y={result.initial_balance} stroke="#6b7280" strokeDasharray="3 3" />
                <Line
                  type="monotone"
                  dataKey="balance"
                  stroke={result.total_pnl >= 0 ? "#10b981" : "#ef4444"}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Trade history table */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-700">
              <h3 className="text-sm font-medium text-gray-400">Trade History ({result.trades.length} trades)</h3>
            </div>
            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-800 sticky top-0">
                  <tr className="text-gray-400 text-xs uppercase">
                    <th className="px-3 py-2 text-left">#</th>
                    <th className="px-3 py-2 text-left">Signal</th>
                    <th className="px-3 py-2 text-right">Entry</th>
                    <th className="px-3 py-2 text-right">Exit</th>
                    <th className="px-3 py-2 text-right">P&L</th>
                    <th className="px-3 py-2 text-left">Exit Reason</th>
                    <th className="px-3 py-2 text-right">Conf.</th>
                    <th className="px-3 py-2 text-right">Balance</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades.map((t) => (
                    <tr key={t.trade_id} className="border-t border-gray-700/50 hover:bg-gray-700/30">
                      <td className="px-3 py-2 text-gray-400">{t.trade_id}</td>
                      <td className="px-3 py-2">
                        <span className={`font-medium ${t.signal === "LONG" ? "text-green-400" : "text-red-400"}`}>
                          {t.signal}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right text-gray-300">${t.entry_price.toFixed(2)}</td>
                      <td className="px-3 py-2 text-right text-gray-300">${t.exit_price.toFixed(2)}</td>
                      <td className={`px-3 py-2 text-right font-medium ${t.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
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
                      <td className="px-3 py-2 text-right text-gray-400">{t.confidence.toFixed(0)}%</td>
                      <td className="px-3 py-2 text-right text-gray-300">${t.balance_after.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
