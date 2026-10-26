#!/usr/bin/env bash
# Runs ON THE K3s NODE. Builds the 3 images from the repo and loads them into K3s.
set -euo pipefail
until [ -f /etc/rancher/k3s/k3s.yaml ]; do echo "waiting for k3s..."; sleep 5; done
sudo apt-get update -y -q && sudo apt-get install -y -q docker.io git
cd ~ && { [ -d aegis ] || git clone https://github.com/JuwairiyahP/aegis.git; } && cd aegis && git pull -q
sudo docker build -q -t aegis-demo:v2 app/demo
sudo docker build -q -t aegis-predictor:v3 --build-arg MODEL_DIR=models/v3 predictor
sudo docker build -q -t aegis-policy:v1 policy
sudo docker save aegis-demo:v2 aegis-predictor:v3 aegis-policy:v1 | sudo k3s ctr images import -
sudo systemctl stop docker docker.socket   # free the CPU for the experiment
sudo k3s ctr images ls -q | grep aegis
