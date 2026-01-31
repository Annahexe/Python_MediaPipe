import cv2
import subprocess
import sys
import os

WINDOW = "MediaPipe scripts Launcher"

SCRIPTS = [
    ("01 Like Dislike System", "likeDislikeGestureSystem.py", []),
    ("02 Faces Detection", "facesDetectionCounter.py", []),
    ("02 Index up/down", "indexHandGesture.py", []),
    ("03 Arm raised", "armRaisedDetection.py", []),
    ("04 Pose Video Recorder", "poseVideoRecorder.py", []),
    ("05 Pose draw", "poseDrawing.py", []),
]

buttons = []
clicked_index = None


def abs_path(filename: str) -> str:

    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


def mouse_cb(event, x, y, flags, param):
    global clicked_index
    if event == cv2.EVENT_LBUTTONDOWN:
        for (x1, y1, x2, y2, idx) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_index = idx
                break


def draw_menu(frame):
    global buttons
    buttons = []

    h, w = frame.shape[:2]

    panel_w = int(w * 0.45)
    x1 = 20
    y = 20
    btn_h = 52
    gap = 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + (btn_h + gap) * (len(SCRIPTS) + 1) + 20), (0, 0, 0), -1)
    alpha = 0.45
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, "Select a script (click)", (x1, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += 40

    # Buttons
    for idx, (label, _, _) in enumerate(SCRIPTS):
        bx1, by1 = x1, y
        bx2, by2 = x1 + panel_w - 30, y + btn_h

        # Button rectangle
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        cv2.putText(frame, label, (bx1 + 12, by1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        buttons.append((bx1, by1, bx2, by2, idx))
        y += btn_h + gap

    cv2.putText(frame, "ESC: quit launcher", (x1, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def run_script(idx: int):
    label, script_name, script_args = SCRIPTS[idx]
    script_path = abs_path(script_name)

    if not os.path.exists(script_path):
        print(f"[Launcher] Script not found: {script_path}")
        return

    print(f"[Launcher] Running: {label} -> {script_name}")

    # Use the same Python interpreter you're using for the launcher
    cmd = [sys.executable, script_path, *script_args]

    # This blocks until the script finishes (user closes its window)
    subprocess.run(cmd, check=False)


def main():
    global clicked_index

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = draw_menu(frame)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        # If user clicked a button, launch corresponding script
        if clicked_index is not None:
            idx = clicked_index
            clicked_index = None

            # Close camera + window before launching another OpenCV script
            cap.release()
            cv2.destroyAllWindows()

            run_script(idx)

            # Re-open launcher after script ends
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not re-open camera 0")
            cv2.namedWindow(WINDOW)
            cv2.setMouseCallback(WINDOW, mouse_cb)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
