"""
One-off helper: cycle through camera indices so you can see which one
is the MacBook's built-in camera vs. an external one.

Press any key to advance to the next index, 'q' to quit.
"""

import cv2

for index in range(5):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"index {index}: not available")
        cap.release()
        continue

    ok, frame = cap.read()
    if not ok:
        print(f"index {index}: opened but no frame")
        cap.release()
        continue

    cv2.putText(
        frame, f"index {index} - press any key for next, q to quit",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    cv2.imshow("Camera probe", frame)
    key = cv2.waitKey(0) & 0xFF
    cap.release()
    if key == ord("q"):
        break

cv2.destroyAllWindows()
