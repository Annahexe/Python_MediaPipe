import cv2
import mediapipe as mp

# This script processes a video file using MediaPipe Pose,
# draws body landmarks and simple geometric shapes based on key body points,
# and saves the processed result into a new MP4 video file.

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

output_path = "result.mp4"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def get_point(landmarks, index, width, height):
    lm = landmarks[index]
    return (int(lm.x * width), int(lm.y * height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
        )

        h, w, c = frame.shape
        landmarks = results.pose_landmarks.landmark

        left_shoulder = get_point(landmarks, 11, w, h)
        right_shoulder = get_point(landmarks, 12, w, h)
        nose = get_point(landmarks, 0, w, h)

        left_hip = get_point(landmarks, 23, w, h)
        right_hip = get_point(landmarks, 24, w, h)
        left_ankle = get_point(landmarks, 27, w, h)
        right_ankle = get_point(landmarks, 28, w, h)

        # Upper triangle
        cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)
        cv2.line(frame, right_shoulder, nose, (0, 255, 0), 2)
        cv2.line(frame, nose, left_shoulder, (0, 255, 0), 2)

        # Lower square / rectangle
        cv2.line(frame, left_hip, right_hip, (0, 255, 0), 2)
        cv2.line(frame, right_hip, right_ankle, (0, 255, 0), 2)
        cv2.line(frame, right_ankle, left_ankle, (0, 255, 0), 2)
        cv2.line(frame, left_ankle, left_hip, (0, 255, 0), 2)

    # Show result and write frames to output video
    cv2.putText(frame, "ESC: quit script", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Body Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
