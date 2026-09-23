"""
Phase 1: Mac webcam -> Roboflow hosted inference API -> live overlay.

No rep counting yet. This just proves the pipeline: capture a frame,
send it to Roboflow, draw back whatever it predicts. Press 'q' to quit.

Inference runs in a separate worker process (infer_worker.py) that never
imports cv2. On this machine, an active cv2/AVFoundation camera capture
session in the same process as urllib3/ssl corrupts TLS (SSL
bad_record_mac). Isolating the network call in its own process works
around it.
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import supervision as sv
from dotenv import load_dotenv

load_dotenv()
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))
INFER_INTERVAL_SEC = 0.5

worker_path = Path(__file__).parent / "infer_worker.py"
worker = subprocess.Popen(
    [sys.executable, str(worker_path)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam (index {CAMERA_INDEX})")

detections = sv.Detections.empty()
labels = []
last_infer_time = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    now = time.monotonic()
    if now - last_infer_time >= INFER_INTERVAL_SEC:
        last_infer_time = now
        _, buffer = cv2.imencode(".jpg", frame)
        image_b64 = base64.b64encode(buffer).decode("ascii")

        worker.stdin.write(image_b64 + "\n")
        worker.stdin.flush()
        result = json.loads(worker.stdout.readline())

        if "error" in result:
            print(f"Inference request failed, keeping last result: {result['error']}")
        else:
            detections = sv.Detections.from_inference(result)
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(detections["class_name"], detections.confidence)
            ]

    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    cv2.imshow("PushPass - Phase 1", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
worker.stdin.close()
worker.wait(timeout=2)
