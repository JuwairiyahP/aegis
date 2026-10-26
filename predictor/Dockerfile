# Predictor image. Build context = this folder (repo: predictor/).
#   docker build --build-arg MODEL_DIR=models/v1 -t aegis-predictor:v1 .
#   k3d image import aegis-predictor:v1 -c aegis
FROM python:3.11-slim
# xgboost's Linux wheel needs the OpenMP runtime, which the slim image lacks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY predictor/ predictor/
ARG MODEL_DIR=models/v1
COPY ${MODEL_DIR}/model.json ${MODEL_DIR}/meta.json /app/model/
ENV MODEL_DIR=/app/model \
    PROM_URL=http://mon-kube-prometheus-stack-prometheus.monitoring.svc:9090 \
    AEGIS_NAMESPACE=aegis AEGIS_DEPLOYMENT=demo AEGIS_POD_REGEX="demo-.*" AEGIS_ROUTE=/work \
    RISK_POLL_SECONDS=5
EXPOSE 8000
CMD ["uvicorn", "predictor.service:app", "--host", "0.0.0.0", "--port", "8000"]
