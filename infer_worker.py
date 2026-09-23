"""
Standalone inference worker. Deliberately does NOT import cv2.

Reads one base64-encoded JPEG per line from stdin, POSTs it to Roboflow,
writes the JSON response as one line to stdout. Runs until stdin closes.

Why this exists: on this machine, having an active cv2/AVFoundation
camera capture session in the same process as urllib3/ssl corrupts TLS
(SSL bad_record_mac on nearly every request), even though the network
code is otherwise correct -- proven by this exact request working
every time when run standalone. Isolating it in its own process (no
cv2 ever loaded here) works around it.
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["ROBOFLOW_API_KEY"]
MODEL_ID = os.environ["ROBOFLOW_MODEL_ID"]
INFER_URL = f"https://serverless.roboflow.com/{MODEL_ID}?api_key={API_KEY}"
MAX_ATTEMPTS = int(os.environ.get("INFER_MAX_ATTEMPTS", 3))

for line in sys.stdin:
    image_b64 = line.strip()
    if not image_b64:
        continue
    try:
        # Connection-level failures (incl. the intermittent SSL bad_record_mac) are
        # transient, so retry those; HTTP errors and timeouts fail straight away.
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = requests.post(
                    INFER_URL,
                    data=image_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=10,
                )
                break
            except requests.exceptions.Timeout:
                raise  # ConnectTimeout is also a ConnectionError; don't stall the video 3x10s
            except requests.exceptions.ConnectionError:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
        response.raise_for_status()
        print(response.text.replace("\n", ""), flush=True)
    except requests.exceptions.RequestException as e:
        # requests puts the full URL (including api_key) in its error text; never print the key.
        error_msg = str(e).replace(API_KEY, "***").replace('"', "'").replace("\n", " ")
        print(f'{{"error": "{error_msg}"}}', flush=True)
