"""
Phase 1: Mac webcam -> Roboflow hosted inference API -> live overlay.

Capture a frame, send it to Roboflow, draw back whatever it predicts,
and count reps: one pushup = a "down" prediction followed by an "up"
prediction. hud.py draws the game-style overlay and milestone celebrations.
Press 'q' to quit, 'r' to reset the session, 'd' to toggle debug boxes,
's' to save a screenshot (without the key hints) to screenshots/.

Inference runs in a separate worker process (infer_worker.py) that never
imports cv2. On this machine, an active cv2/AVFoundation camera capture
session in the same process as urllib3/ssl corrupts TLS (SSL
bad_record_mac). Isolating the network call in its own process works
around it.
"""

import base64
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import supervision as sv
from dotenv import load_dotenv

from hud import MILESTONES, Hud

load_dotenv()
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))
INFER_INTERVAL_SEC = 0.5
# One-off inference failures are silently skipped; only a streak this long gets reported.
FAILURE_STREAK_TO_REPORT = 5
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
# One row per inference, overwritten each run, for diagnosing missed/extra reps.
LOG_PATH = Path(__file__).parent / "data" / "infer-log.csv"
# Class names the model emits for each position; override in .env if yours differ.
UP_CLASS = os.environ.get("UP_CLASS", "pushup")
DOWN_CLASS = os.environ.get("DOWN_CLASS", "pushdown")
MIN_CONFIDENCE = 0.5
# Anti-cheat geometry checks (assumes a side-on camera). A real pushup has a
# horizontal body (box wider than tall), and the top of the box must drop by
# at least MIN_DROP (fraction of frame height) between the up and down poses.
MIN_ASPECT = float(os.environ.get("MIN_ASPECT", 1.3))
MIN_DROP = float(os.environ.get("MIN_DROP", 0.05))

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
hud = Hud()
show_debug = False

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam (index {CAMERA_INDEX})")

detections = sv.Detections.empty()
labels = []
last_infer_time = 0.0
infer_requests = 0
infer_failures = 0
failure_streak = 0
pushup_count = 0
# Last confident position seen ("up"/"down"); a rep is counted on down -> up.
position = None
# Lowest box top (largest y, as a fraction of frame height) seen in the current down phase.
down_top = None
status = ""
unknown_classes = set()

LOG_PATH.parent.mkdir(exist_ok=True)
log_file = open(LOG_PATH, "w", newline="")
log = csv.writer(log_file)
log.writerow(["t_sec", "latency_ms", "ok", "n_dets", "class", "conf", "aspect", "top",
              "position", "count", "status"])
start_time = time.monotonic()

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
        latency_ms = (time.monotonic() - now) * 1000
        row = {"class": "", "conf": "", "aspect": "", "top": "", "n_dets": 0}

        infer_requests += 1
        if "error" in result:
            infer_failures += 1
            failure_streak += 1
            if failure_streak == FAILURE_STREAK_TO_REPORT:
                print(f"Inference has failed {failure_streak} times in a row: {result['error']}")
        else:
            if failure_streak >= FAILURE_STREAK_TO_REPORT:
                print("Inference recovered.")
            failure_streak = 0
            detections = sv.Detections.from_inference(result)
            widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
            heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            aspects = widths / heights
            labels = [
                f"{class_name} {confidence:.2f} w/h {aspect:.1f}"
                for class_name, confidence, aspect in zip(
                    detections["class_name"], detections.confidence, aspects
                )
            ]

            # Use the single most confident prediction as this frame's position.
            if len(detections) > 0:
                best = detections.confidence.argmax()
                best_class = detections["class_name"][best]
                top = detections.xyxy[best][1] / frame.shape[0]
                row = {
                    "class": best_class, "conf": f"{detections.confidence[best]:.2f}",
                    "aspect": f"{aspects[best]:.2f}", "top": f"{top:.3f}", "n_dets": len(detections),
                }
                if detections.confidence[best] < MIN_CONFIDENCE:
                    pass
                elif best_class in (UP_CLASS, DOWN_CLASS) and aspects[best] < MIN_ASPECT:
                    # Upright body (e.g. sitting with arms out): not a pushup, drop any rep in progress.
                    position, down_top = None, None
                    status = f"not horizontal (w/h {aspects[best]:.1f} < {MIN_ASPECT})"
                elif best_class == DOWN_CLASS:
                    down_top = top if down_top is None else max(down_top, top)
                    position = "down"
                elif best_class == UP_CLASS:
                    if position == "down":
                        drop = down_top - top
                        if drop >= MIN_DROP:
                            pushup_count += 1
                            status = f"rep! drop {drop:.0%}"
                            print(f"Pushups: {pushup_count}")
                            if pushup_count in MILESTONES:
                                print(f"*** {MILESTONES[pushup_count][0]} ***")
                            hud.on_rep(pushup_count)
                        else:
                            status = f"too shallow (drop {drop:.0%} < {MIN_DROP:.0%})"
                    position, down_top = "up", None
                elif best_class not in unknown_classes:
                        unknown_classes.add(best_class)
                        print(
                            f"Unrecognized class '{best_class}' -- set UP_CLASS/DOWN_CLASS "
                            f"in .env (currently '{UP_CLASS}'/'{DOWN_CLASS}')"
                        )

        log.writerow([
            f"{now - start_time:.2f}", f"{latency_ms:.0f}", "error" not in result, row["n_dets"],
            row["class"], row["conf"], row["aspect"], row["top"], position, pushup_count, status,
        ])
        log_file.flush()

    annotated = frame.copy()
    if show_debug:
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        cv2.putText(
            annotated, status, (20, 200),  # just below the counter panel, clear of the footer
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
    annotated = hud.draw(annotated, pushup_count)

    # Key hints only go on the on-screen copy so saved screenshots stay clean.
    display = annotated.copy()
    hints = "s screenshot   r reset   d debug   q quit"
    (hint_w, _), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    cv2.putText(
        display, hints, (display.shape[1] - hint_w - 16, 30),
        cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )

    cv2.imshow("PushPass", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        shot = SCREENSHOT_DIR / f"pushpass-{time.strftime('%Y%m%d-%H%M%S')}.png"
        cv2.imwrite(str(shot), annotated)
        print(f"Saved screenshot: {shot}")
    if key == ord("q"):
        break
    if key == ord("r"):
        pushup_count = 0
        position, down_top = None, None
        status = ""
        hud.reset()
    if key == ord("d"):
        show_debug = not show_debug

if infer_failures:
    print(f"Inference: {infer_failures}/{infer_requests} requests failed (skipped, counting continued).")

log_file.close()
cap.release()
cv2.destroyAllWindows()
worker.stdin.close()
worker.wait(timeout=2)
