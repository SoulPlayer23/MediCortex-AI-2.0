import { useEffect, useState } from 'react';

const API = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8001';

interface Summary {
  total_responses: number;
  avg_request_ms: number;
  avg_judge_score: number;
  judged_count: number;
}

interface AgentStat {
  agent: string;
  usage_count: number;
}

interface NodeTiming {
  node: string;
  avg_ms: number;
  min_ms: number;
  max_ms: number;
  sample_count: number;
}

interface RequestRow {
  id: number;
  session_id: string;
  timestamp: string;
  request_elapsed_ms: number | null;
  judge_score: number | null;
  agents_used: string[];
  node_timings: Record<string, number>;
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-[#2a2a2a] rounded-xl p-5 flex flex-col gap-1">
      <span className="text-xs text-neutral-400 uppercase tracking-wide">{label}</span>
      <span className="text-2xl font-semibold text-white">{value}</span>
    </div>
  );
}

export default function Dashboard() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [agents, setAgents] = useState<AgentStat[]>([]);
  const [nodeTimings, setNodeTimings] = useState<NodeTiming[]>([]);
  const [requests, setRequests] = useState<RequestRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const headers = { 'Content-Type': 'application/json' };
    Promise.all([
      fetch(`${API}/api/dashboard/summary`, { headers }).then(r => r.json()),
      fetch(`${API}/api/dashboard/agents`, { headers }).then(r => r.json()),
      fetch(`${API}/api/dashboard/node-timings`, { headers }).then(r => r.json()),
      fetch(`${API}/api/dashboard/requests?limit=50`, { headers }).then(r => r.json()),
    ])
      .then(([s, a, n, r]) => {
        setSummary(s);
        setAgents(a);
        setNodeTimings(n);
        setRequests(r);
      })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="min-h-screen bg-[#212121] text-white p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Observability Dashboard</h1>
            <p className="text-sm text-neutral-400 mt-1">MediCortex AI — pipeline metrics</p>
          </div>
          <a
            href="/"
            className="text-sm text-neutral-400 hover:text-white transition-colors"
          >
            ← Back to chat
          </a>
        </div>

        {error && (
          <div className="bg-red-900/40 border border-red-700 rounded-lg p-4 text-sm text-red-300">
            Failed to load metrics: {error}
          </div>
        )}

        {/* Summary cards */}
        {summary && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Total responses" value={summary.total_responses} />
            <StatCard label="Avg latency" value={`${summary.avg_request_ms} ms`} />
            <StatCard label="Avg judge score" value={`${summary.avg_judge_score} / 5`} />
            <StatCard label="Judged requests" value={summary.judged_count} />
          </div>
        )}

        {/* Agent usage + Node timings side-by-side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Agent usage */}
          <div className="bg-[#2a2a2a] rounded-xl p-5">
            <h2 className="text-sm font-semibold text-neutral-300 mb-4 uppercase tracking-wide">
              Agent Usage
            </h2>
            {agents.length === 0 ? (
              <p className="text-sm text-neutral-500">No data yet.</p>
            ) : (
              <ul className="space-y-2">
                {agents.map(a => (
                  <li key={a.agent} className="flex items-center justify-between text-sm">
                    <span className="text-neutral-200">{a.agent}</span>
                    <span className="bg-neutral-700 rounded-full px-2 py-0.5 text-xs font-mono">
                      {a.usage_count}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Node timings */}
          <div className="bg-[#2a2a2a] rounded-xl p-5">
            <h2 className="text-sm font-semibold text-neutral-300 mb-4 uppercase tracking-wide">
              Node Latency (avg ms)
            </h2>
            {nodeTimings.length === 0 ? (
              <p className="text-sm text-neutral-500">No data yet.</p>
            ) : (
              <ul className="space-y-2">
                {nodeTimings.map(n => (
                  <li key={n.node} className="text-sm">
                    <div className="flex justify-between mb-0.5">
                      <span className="text-neutral-200">{n.node}</span>
                      <span className="text-neutral-400 font-mono">{n.avg_ms} ms</span>
                    </div>
                    <div className="w-full bg-neutral-700 rounded-full h-1.5">
                      <div
                        className="bg-blue-500 h-1.5 rounded-full"
                        style={{
                          width: `${Math.min(100, (n.avg_ms / (nodeTimings[0]?.avg_ms || 1)) * 100)}%`,
                        }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Recent requests table */}
        <div className="bg-[#2a2a2a] rounded-xl p-5">
          <h2 className="text-sm font-semibold text-neutral-300 mb-4 uppercase tracking-wide">
            Recent Requests
          </h2>
          {requests.length === 0 ? (
            <p className="text-sm text-neutral-500">No requests yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead>
                  <tr className="text-neutral-500 border-b border-neutral-700">
                    <th className="pb-2 pr-4 font-normal">Time</th>
                    <th className="pb-2 pr-4 font-normal">Elapsed</th>
                    <th className="pb-2 pr-4 font-normal">Score</th>
                    <th className="pb-2 font-normal">Agents</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800">
                  {requests.map(r => (
                    <tr key={r.id} className="hover:bg-neutral-800/40 transition-colors">
                      <td className="py-2 pr-4 text-neutral-400 font-mono text-xs whitespace-nowrap">
                        {new Date(r.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="py-2 pr-4 font-mono text-xs">
                        {r.request_elapsed_ms != null ? `${r.request_elapsed_ms} ms` : '—'}
                      </td>
                      <td className="py-2 pr-4">
                        {r.judge_score != null ? (
                          <span
                            className={`px-1.5 py-0.5 rounded text-xs font-mono ${
                              r.judge_score >= 4
                                ? 'bg-green-900/50 text-green-300'
                                : r.judge_score >= 3
                                ? 'bg-yellow-900/50 text-yellow-300'
                                : 'bg-red-900/50 text-red-300'
                            }`}
                          >
                            {r.judge_score}/5
                          </span>
                        ) : (
                          <span className="text-neutral-600 text-xs">—</span>
                        )}
                      </td>
                      <td className="py-2 text-xs text-neutral-400">
                        {(r.agents_used || []).join(', ') || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
