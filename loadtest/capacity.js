// loadtest/capacity.js - send a FIXED number of requests per second for a while.
// Used by harness/capacity_local.sh to find how many req/s one pod can take.
//   RATE=4 DURATION=30s URL=http://localhost:8001/work SUMMARY=/tmp/cap.json k6 run loadtest/capacity.js
import http from 'k6/http';

export const options = {
  scenarios: {
    fixed: {
      executor: 'constant-arrival-rate',
      rate: Number(__ENV.RATE || 2),
      timeUnit: '1s',
      duration: __ENV.DURATION || '30s',
      preAllocatedVUs: 20,
      maxVUs: 300,
    },
  },
  discardResponseBodies: true,
  summaryTrendStats: ['avg', 'med', 'p(95)', 'max'],
};

export default function () {
  http.get(__ENV.URL || 'http://localhost:8001/work', { timeout: '10s' });
}

export function handleSummary(data) {
  const out = { stdout: '' };
  if (__ENV.SUMMARY) out[__ENV.SUMMARY] = JSON.stringify(data);
  return out;
}
