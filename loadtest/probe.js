// loadtest/probe.js - measure how many req/s ONE pod can take before p95 passes 250 ms.
//
// Run it ONLY while the HPA is pinned to a single replica (ask A, they own infra/):
//   kubectl -n aegis patch hpa demo -p '{"spec":{"minReplicas":1,"maxReplicas":1}}'
// and put it back afterwards:
//   kubectl -n aegis patch hpa demo -p '{"spec":{"minReplicas":1,"maxReplicas":6}}'
//
//   k6 run loadtest/probe.js                      # default staircase
//   STEPS=2,4,6,8,10,12,16,20 STEP_S=60 k6 run loadtest/probe.js
//
// Each step holds one arrival rate for STEP_S seconds. Afterwards read p95 and replicas per step
// in Grafana / Prometheus. Prometheus p95 uses a 30 s window, so read it near the END of each step.
import http from 'k6/http';

const STEPS = (__ENV.STEPS || '1,2,3,4,6,8,10,12,15,20').split(',').map(Number);
const STEP_S = Number(__ENV.STEP_S || 45);

export const options = {
  scenarios: {
    probe: {
      executor: 'ramping-arrival-rate',
      startRate: STEPS[0],
      timeUnit: '1s',
      preAllocatedVUs: 100,
      maxVUs: 500,
      stages: STEPS.flatMap((r) => [
        { target: r, duration: '2s' },        // jump to the new rate
        { target: r, duration: `${STEP_S}s` }, // hold it
      ]),
    },
  },
  discardResponseBodies: true,
};

export default function () {
  http.get(`${__ENV.BASE_URL || 'http://localhost:8080'}/work`, { timeout: '10s' });
}
