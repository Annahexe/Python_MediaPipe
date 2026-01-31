import cv2
import mediapipe as mp

# Drawing a figure with the body (pose drawing)
# Draws a geometric shape by connecting points on the body

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video capture or Camera Capture
# cap = cv2.VideoCapture("VideoFileHere.mp4")
cap = cv2.VideoCapture(0)

def obtener_punto(landmarks, indice, ancho, alto):
    lm = landmarks[indice]
    return (int(lm.x * ancho), int(lm.y * alto))

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

        left_shoulder = obtener_punto(landmarks, 11, w, h)
        right_shoulder = obtener_punto(landmarks, 12, w, h)
        nose = obtener_punto(landmarks, 0, w, h)

        left_hip = obtener_punto(landmarks, 23, w, h)
        right_hip = obtener_punto(landmarks, 24, w, h)
        left_ankle = obtener_punto(landmarks, 27, w, h)
        right_ankle = obtener_punto(landmarks, 28, w, h)

        # Upper triangle
        cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)
        cv2.line(frame, right_shoulder, nose, (0, 255, 0), 2)
        cv2.line(frame, nose, left_shoulder, (0, 255, 0), 2)

        # Lower square / rectangle
        cv2.line(frame, left_hip, right_hip, (0, 255, 0), 2)
        cv2.line(frame, right_hip, right_ankle, (0, 255, 0), 2)
        cv2.line(frame, right_ankle, left_ankle, (0, 255, 0), 2)
        cv2.line(frame, left_ankle, left_hip, (0, 255, 0), 2)

    cv2.putText(frame, "ESC: quit script", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Body Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()