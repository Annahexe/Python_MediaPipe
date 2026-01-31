import cv2
import mediapipe as mp

# Right Arm Raised Detection - Determine whether the user’s right arm is in a correct position.
# It analyzes the relative positions of the right wrist and right shoulder and displays a message indicating a “correct position”
# when the wrist is detected above the shoulder in the image.

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
#cap = cv2.VideoCapture("VideoNameHere.mp4")
cap = cv2.VideoCapture(0)

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
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        h, w, c = frame.shape
        landmarks = results.pose_landmarks.landmark

        right_shoulder_y = int(landmarks[12].y * h)

        right_wrist_y = int(landmarks[16].y * h)
        right_wrist_x = int(landmarks[16].x * w)

        if right_wrist_y < right_shoulder_y:
            cv2.circle(frame, (right_wrist_x, right_wrist_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "ARM RAISED", (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.circle(frame, (right_wrist_x, right_wrist_y), 10, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "ARM DOWN", (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "ESC: quit script", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Body Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()