#!/usr/bin/env bash
# Runs ON THE LOADGEN instance. Deploys monitoring + AEGIS onto the K3s node and
# prepares the harness. Usage:  NODE_PRIV=<node private ip> bash setup_loadgen.sh
set -euo pipefail
: "${NODE_PRIV:?set NODE_PRIV to the node's private IP}"
cd ~
[ -d aegis ] || git clone https://github.com/JuwairiyahP/aegis.git
cd aegis && git pull -q

# kubeconfig from the node (copied to ~/k3s.yaml by the laptop), pointed at the private IP
mkdir -p ~/.kube
sed "s#127.0.0.1#${NODE_PRIV}#" ~/k3s.yaml > ~/.kube/config
chmod 600 ~/.kube/config
kubectl get nodes

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo update >/dev/null
helm upgrade --install mon prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace -f infra/k8s/prom-values.yaml --wait --timeout 10m

kubectl apply -f infra/k8s/app.yaml || { sleep 5; kubectl apply -f infra/k8s/app.yaml; }
# 2-vCPU node: ~0.7 vCPU for system + 4 pods x 0.25 = fits; 6 would not
sed 's/maxReplicas: 6/maxReplicas: 4/' infra/k8s/hpa.yaml | kubectl apply -f -
kubectl apply -f infra/k8s/rbac.yaml
kubectl apply -f infra/k8s/aegis.yaml
kubectl -n aegis scale deploy policy --replicas=0
kubectl -n aegis rollout status deploy/demo --timeout=180s
kubectl -n aegis rollout status deploy/predictor --timeout=180s || true
curl -s "http://${NODE_PRIV}/work" && echo "  <- app reachable from loadgen"
kubectl -n aegis get pods,hpa
