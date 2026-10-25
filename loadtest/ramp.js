// loadtest/ramp.js - ramping arrival-rate load for the demo app.
// "Arrival rate" = we choose requests per second; k6 adds virtual users as needed, so the load
// does not slow down when the app slows down (like a real traffic surge).
//
// Everything is set with environment variables, so one script serves training AND eval runs:
//   URL     address of /work (default http://localhost:8080/work = the Ingress; never port-forward)
//   BASE    calm rate in req/s (default 2)
//   CALM    seconds of calm before the first ramp (default 180; B needs >= 3 min of warm-up)
//   PEAK    peak rate in req/s,  RAMP seconds to climb to it,  HOLD seconds at the peak
//   DOWN    seconds to go back down to BASE (default 60)
//   BURSTS  number of bursts, 1 or 2 (default 1);  GAP seconds of calm between bursts (default 150)
//   TAIL    seconds of calm at the end, to watch recovery and scale-down (default 120)
//   MAX_VUS most virtual users k6 may use (default 400)
//   SUMMARY optional file path: a JSON summary of the run is written there (the runner reads it)
import http from 'k6/http';

const num = (name, def) => Number(__ENV[name] || def);
const BASE = num('BASE', 2), PEAK = num('PEAK', 12), RAMP = num('RAMP', 90), HOLD = num('HOLD', 120);
const CALM = num('CALM', 180), DOWN = num('DOWN', 60), TAIL = num('TAIL', 120);
const BURSTS = num('BURSTS', 1), GAP = num('GAP', 150);
const URL = __ENV.URL || 'http://localhost:8080/work';

const stages = [{ target: BASE, duration: `${CALM}s` }];
for (let b = 0; b < BURSTS; b++) {
  if (b > 0) stages.push({ target: BASE, duration: `${GAP}s` });
  stages.push({ target: PEAK, duration: `${RAMP}s` });   // climb
  stages.push({ target: PEAK, duration: `${HOLD}s` });   // stay at the peak
  stages.push({ target: BASE, duration: `${DOWN}s` });   // come back down
}
stages.push({ target: BASE, duration: `${TAIL}s` });

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-arrival-rate',
      startRate: BASE,
      timeUnit: '1s',
      preAllocatedVUs: 50,
      maxVUs: num('MAX_VUS', 400),
      stages: stages,
    },
  },
  discardResponseBodies: true,
  summaryTrendStats: ['avg', 'med', 'p(95)', 'max'],
};

export default function () {
  http.get(URL, { timeout: '10s' });
}

// short summary on screen + full JSON to a file (used by harness/run_profiles.sh)
export function handleSummary(data) {
  const m = data.metrics;
  const v = (name, key) => (m[name] && m[name].values[key] !== undefined ? m[name].values[key] : 0);
  const line = `requests ${v('http_reqs', 'count')}, ` +
    `p95 ${Math.round(v('http_req_duration', 'p(95)'))} ms, ` +
    `failed ${(100 * v('http_req_failed', 'rate')).toFixed(1)}%, ` +
    `dropped ${v('dropped_iterations', 'count')} (should be 0)\n`;
  const out = { stdout: line };
  if (__ENV.SUMMARY) out[__ENV.SUMMARY] = JSON.stringify(data);
  return out;
}
