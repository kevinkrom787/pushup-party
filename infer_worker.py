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

for line in sys.stdin:
    image_b64 = line.strip()
    if not image_b64:
        continue
    try:
        response = requests.post(
            INFER_URL,
            data=image_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        response.raise_for_status()
        print(response.text.replace("\n", ""), flush=True)
    except requests.exceptions.RequestException as e:
        error_msg = str(e).replace('"', "'").replace("\n", " ")
        print(f'{{"error": "{error_msg}"}}', flush=True)
