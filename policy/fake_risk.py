# policy/fake_risk.py - a pretend predictor, to test the controller on your computer without a cluster.
#   python fake_risk.py                      # serves http://127.0.0.1:8100/risk
# The risk value is read from a small file, so you can change it while it runs:
#   Windows PowerShell:  Set-Content -Path "$env:TEMP\risk.txt" -Value 0.9        ("503" = predictor down)
#   Linux/WSL:           echo 0.9 > /tmp/risk.txt
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

RISK_FILE = os.path.join(tempfile.gettempdir(), "risk.txt")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            value = open(RISK_FILE).read().strip()
        except FileNotFoundError:
            value = "0.1"
        if value == "503":
            self.send_response(503)
            self.end_headers()
            return
        body = json.dumps({"risk": float(value), "threshold": 0.5, "in_breach": False, "model_version": "fake"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


print("fake predictor on http://127.0.0.1:8100/risk ; change the risk by writing a number into", RISK_FILE)
HTTPServer(("127.0.0.1", 8100), Handler).serve_forever()
