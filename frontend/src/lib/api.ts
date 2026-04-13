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

// ---- Aviator Prediction Bot ----
export interface AviatorTrainResult {
  train_accuracy: number;
  test_accuracy: number;
  high_confidence_accuracy: number;
  high_confidence_trades: number;
  confidence_threshold: number;
  cv_mean_accuracy: number;
  cv_std: number;
  train_samples: number;
  test_samples: number;
  total_samples: number;
  class_distribution: { above_2x: number; below_2x: number };
  classification_report: Record<string, unknown>;
  feature_importance: Record<string, number>;
}

export interface AviatorPrediction {
  prediction: string;
  predicted_class: number;
  probability_above_2x: number;
  probability_below_2x: number;
  confidence: number;
  confidence_threshold: number;
  should_bet: boolean;
  last_crash_point: number;
  features: Record<string, number>;
}

export interface AviatorModelStatus {
  trained: boolean;
  confidence_threshold: number | null;
  n_features: number;
}

export interface AviatorBotSettings {
  bet_amount: number;
  cashout_target: number;
  max_consecutive_losses: number;
  stop_loss_balance: number;
  take_profit_balance: number;
  martingale: boolean;
  martingale_multiplier: number;
  max_martingale_bet: number;
}

export interface AviatorBetRecord {
  round_id: number;
  timestamp: number;
  outcome: string;
  bet_amount: number;
  payout: number;
  profit: number;
  crash_point: number;
  target: number;
  balance_after: number;
}

export interface AviatorBotStatus {
  is_running: boolean;
  balance: number;
  initial_balance: number;
  total_pnl: number;
  total_bets: number;
  total_wins: number;
  total_losses: number;
  total_skipped: number;
  win_rate: number;
  consecutive_losses: number;
  current_bet_amount: number;
  last_prediction: AviatorPrediction | null;
  settings: AviatorBotSettings;
  bet_history: AviatorBetRecord[];
}

export interface AviatorSimulationResult {
  training: AviatorTrainResult;
  simulation: {
    initial_balance: number;
    final_balance: number;
    total_pnl: number;
    total_rounds: number;
    bets_placed: number;
    wins: number;
    losses: number;
    skipped: number;
    win_rate: number;
    bet_amount: number;
    cashout_target: number;
    martingale: boolean;
    equity_curve: { round: number; balance: number }[];
    bet_log: {
      round: number;
      crash_point: number;
      outcome: string;
      bet_amount: number;
      profit: number;
      balance: number;
      confidence: number;
    }[];
  };
}

export interface AviatorDataInfo {
  message: string;
  total_rounds: number;
  above_2x: number;
  below_2x: number;
  ratio_above_2x?: number;
  mean_crash?: number;
  median_crash?: number;
}

export function aviatorGenerateData(params: {
  n_rounds?: number;
  seed?: number;
  pattern_strength?: number;
} = {}) {
  return request<AviatorDataInfo>("/api/aviator/generate-data", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export function aviatorImportCsv(csvContent: string) {
  return request<AviatorDataInfo>("/api/aviator/import-csv", {
    method: "POST",
    body: JSON.stringify({ csv_content: csvContent }),
  });
}

export function aviatorTrain(params: {
  use_synthetic?: boolean;
  n_rounds?: number;
  seed?: number;
  pattern_strength?: number;
} = {}) {
  return request<AviatorTrainResult>("/api/aviator/train", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export function aviatorPredict() {
  return request<AviatorPrediction>("/api/aviator/predict");
}

export function aviatorModelStatus() {
  return request<AviatorModelStatus>("/api/aviator/model-status");
}

export function aviatorBotStart() {
  return request<{ message: string; status: AviatorBotStatus }>("/api/aviator/bot/start", {
    method: "POST",
  });
}

export function aviatorBotStop() {
  return request<{ message: string; status: AviatorBotStatus }>("/api/aviator/bot/stop", {
    method: "POST",
  });
}

export function aviatorBotStatus() {
  return request<AviatorBotStatus>("/api/aviator/bot/status");
}

export function aviatorBotSettings(settings: Partial<AviatorBotSettings & { initial_balance: number }>) {
  return request<AviatorBotSettings>("/api/aviator/bot/settings", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}

export function aviatorBotReset(initialBalance = 100) {
  return request<{ message: string; status: AviatorBotStatus }>("/api/aviator/bot/reset", {
    method: "POST",
    body: JSON.stringify({ initial_balance: initialBalance }),
  });
}

export function aviatorProcessRound(crashPoint: number, roundId?: number) {
  return request<{
    round: { round_id: number; crash_point: number; timestamp: number };
    crash_point: number;
    bet_placed: boolean;
    bet_result: { outcome: string; bet_amount: number; payout: number; profit: number } | null;
    next_prediction: AviatorPrediction | null;
  }>("/api/aviator/bot/process-round", {
    method: "POST",
    body: JSON.stringify({ crash_point: crashPoint, round_id: roundId }),
  });
}

export function aviatorSimulate(params: {
  use_synthetic?: boolean;
  n_rounds?: number;
  seed?: number;
  pattern_strength?: number;
  initial_balance?: number;
  bet_amount?: number;
  cashout_target?: number;
  martingale?: boolean;
} = {}) {
  return request<AviatorSimulationResult>("/api/aviator/simulate", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export function aviatorHistory(limit = 100) {
  return request<{ total_rounds: number; rounds: { round_id: number; crash_point: number; timestamp: number }[] }>(
    `/api/aviator/history?limit=${limit}`
  );
}

export function aviatorAddRound(crashPoint: number, roundId?: number) {
  return request<{ round_id: number; crash_point: number; timestamp: number }>("/api/aviator/add-round", {
    method: "POST",
    body: JSON.stringify({ crash_point: crashPoint, round_id: roundId }),
  });
}
