import os, time, hashlib
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

START = time.time()
DELAY = float(os.getenv("STARTUP_DELAY", "30"))
ITERS = int(os.getenv("WORK_ITERS", "20000"))
BUCKETS = [.01, .025, .05, .1, .15, .2, .25, .3, .4, .5, .75, 1, 2, 5]

REQS = Counter("http_requests", "requests", ["route", "status"])
LAT = Histogram("http_request_duration_seconds", "latency", ["route"], buckets=BUCKETS)
app = FastAPI()

@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    resp = await call_next(request)
    if request.url.path == "/work":
        LAT.labels("/work").observe(time.perf_counter() - t)
        REQS.labels("/work", str(resp.status_code)).inc()
    return resp

@app.get("/work")
def work():
    h = b"x"
    for _ in range(ITERS):
        h = hashlib.sha256(h).digest()
    return {"ok": True}

@app.get("/readyz")
async def readyz(response: Response):
    if time.time() - START < DELAY:
        response.status_code = 503
        return {"ready": False}
    return {"ready": True}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
