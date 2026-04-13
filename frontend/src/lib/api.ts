const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

// ---- Model ----
export interface TrainResult {
  train_accuracy: number;
  test_accuracy: number;
  high_confidence_accuracy: number;
  high_confidence_trades: number;
  train_samples: number;
  test_samples: number;
  total_samples: number;
  classification_report: Record<string, unknown>;
  feature_importance: Record<string, number>;
}

export interface PredictResult {
  signal: string;
  confidence: number;
  price: number;
  timestamp: string;
  reason?: string;
  features?: Record<string, number>;
}

export interface ModelStatus {
  trained: boolean;
  model_path?: string;
}

export function trainModel(interval = "15m", limit = 1500) {
  return request<TrainResult>("/api/model/train", {
    method: "POST",
    body: JSON.stringify({ interval, limit }),
  });
}

export function predictSignal() {
  return request<PredictResult>("/api/model/predict");
}

export function getModelStatus() {
  return request<ModelStatus>("/api/model/status");
}

// ---- Backtest ----
export interface BacktestTrade {
  trade_id: number;
  signal: string;
  entry_price: number;
  exit_price: number;
  entry_time: string;
  quantity: number;
  pnl: number;
  fees: number;
  exit_reason: string;
  confidence: number;
  balance_after: number;
}

export interface BacktestResult {
  initial_balance: number;
  final_balance: number;
  total_pnl: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  leverage: number;
  trade_amount: number;
  trades: BacktestTrade[];
  equity_curve: { trade: number; balance: number }[];
}

export interface BacktestParams {
  interval?: string;
  limit?: number;
  initial_balance?: number;
  trade_amount?: number;
  leverage?: number;
  take_profit_pct?: number;
  stop_loss_pct?: number;
  fee_rate?: number;
}

export function runBacktest(params: BacktestParams = {}) {
  return request<BacktestResult>("/api/backtest/run", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

// ---- Trading ----
export interface TradingSettings {
  leverage: number;
  trade_amount: number;
  take_profit_pct: number;
  stop_loss_pct: number;
  commission_fee: number;
}

export interface TradingStatus {
  is_running: boolean;
  balance: number;
  total_pnl: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  current_position: {
    side: string;
    entry_price: number;
    quantity: number;
    unrealized_pnl: number;
  } | null;
  trade_history: {
    trade_id: number;
    signal: string;
    entry_price: number;
    exit_price: number;
    pnl: number;
    timestamp: string;
    exit_reason: string;
  }[];
  settings?: TradingSettings;
}

export function startTrading() {
  return request<{ message: string }>("/api/trading/start", { method: "POST" });
}

export function stopTrading() {
  return request<{ message: string }>("/api/trading/stop", { method: "POST" });
}

export function getTradingStatus() {
  return request<TradingStatus>("/api/trading/status");
}

export function executeOnce() {
  return request<{ message: string; signal?: PredictResult }>("/api/trading/execute-once", { method: "POST" });
}

export function closePosition() {
  return request<{ message: string }>("/api/trading/close-position", { method: "POST" });
}

export function updateTradingSettings(settings: Partial<TradingSettings>) {
  return request<{ message: string; settings: TradingSettings }>("/api/trading/settings", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}
