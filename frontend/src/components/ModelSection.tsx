import { useState } from "react";
import { trainModel, getModelStatus, predictSignal, type TrainResult, type PredictResult, type ModelStatus } from "../lib/api";
import StatsCard from "./StatsCard";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

export default function ModelSection() {
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [prediction, setPrediction] = useState<PredictResult | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [error, setError] = useState("");
  const [interval, setInterval_] = useState("15m");
  const [limit, setLimit] = useState(1500);

  const handleTrain = async () => {
    setTraining(true);
    setError("");
    try {
      const result = await trainModel(interval, limit);
      setTrainResult(result);
    } catch (e) {
      setError(String(e));
    }
    setTraining(false);
  };

  const handlePredict = async () => {
    setError("");
    try {
      const result = await predictSignal();
      setPrediction(result);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleCheckStatus = async () => {
    try {
      const status = await getModelStatus();
      setModelStatus(status);
    } catch (e) {
      setError(String(e));
    }
  };

  const featureData = trainResult
    ? Object.entries(trainResult.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 15)
        .map(([name, value]) => ({ name: name.replace(/_/g, " "), value: +(value * 100).toFixed(2) }))
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white">ML Model</h2>
        <div className="flex items-center gap-3">
          <select
            value={interval}
            onChange={(e) => setInterval_(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
          >
            <option value="5m">5 min</option>
            <option value="15m">15 min</option>
            <option value="1h">1 hour</option>
            <option value="4h">4 hour</option>
          </select>
          <input
            type="number"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white w-24"
            placeholder="Candles"
          />
          <button
            onClick={handleTrain}
            disabled={training}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
          >
            {training ? "Training..." : "Train Model"}
          </button>
          <button
            onClick={handlePredict}
            className="bg-purple-600 hover:bg-purple-700 text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
          >
            Predict Signal
          </button>
          <button
            onClick={handleCheckStatus}
            className="bg-gray-700 hover:bg-gray-600 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors"
          >
            Status
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {modelStatus && (
        <div className="bg-gray-800/50 border border-gray-700 px-4 py-3 rounded-lg text-sm text-gray-300">
          Model Status: {modelStatus.trained ? "Trained" : "Not Trained"}
          {modelStatus.model_path && <span className="text-gray-500 ml-2">({modelStatus.model_path})</span>}
        </div>
      )}

      {prediction && (
        <div
          className={`border px-4 py-3 rounded-lg ${
            prediction.signal === "LONG"
              ? "bg-green-500/10 border-green-500/30"
              : prediction.signal === "SHORT"
              ? "bg-red-500/10 border-red-500/30"
              : "bg-yellow-500/10 border-yellow-500/30"
          }`}
        >
          <div className="flex items-center gap-4">
            <span
              className={`text-lg font-bold ${
                prediction.signal === "LONG"
                  ? "text-green-400"
                  : prediction.signal === "SHORT"
                  ? "text-red-400"
                  : "text-yellow-400"
              }`}
            >
              {prediction.signal}
            </span>
            <span className="text-gray-400 text-sm">
              Confidence: {prediction.confidence}%
            </span>
            <span className="text-gray-400 text-sm">
              Price: ${prediction.price?.toFixed(2)}
            </span>
            {prediction.reason && (
              <span className="text-gray-500 text-sm">{prediction.reason}</span>
            )}
          </div>
        </div>
      )}

      {trainResult && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatsCard
              title="Train Accuracy"
              value={`${trainResult.train_accuracy}%`}
              color="blue"
            />
            <StatsCard
              title="Test Accuracy"
              value={`${trainResult.test_accuracy}%`}
              color={trainResult.test_accuracy >= 60 ? "green" : "yellow"}
            />
            <StatsCard
              title="High-Conf Accuracy"
              value={`${trainResult.high_confidence_accuracy}%`}
              color={trainResult.high_confidence_accuracy >= 70 ? "green" : "yellow"}
              subtitle={`${trainResult.high_confidence_trades} trades`}
            />
            <StatsCard
              title="Total Samples"
              value={trainResult.total_samples}
              subtitle={`${trainResult.train_samples} train / ${trainResult.test_samples} test`}
              color="default"
            />
          </div>

          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Top Feature Importance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureData} layout="vertical" margin={{ left: 80 }}>
                <XAxis type="number" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} width={80} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8 }}
                  labelStyle={{ color: "#fff" }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {featureData.map((_, idx) => (
                    <Cell key={idx} fill={idx < 5 ? "#3b82f6" : "#6b7280"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
