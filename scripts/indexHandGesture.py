import cv2
import mediapipe as mp

# Detecting simple hand gestures (index finger up/down)
# Detect whether the index finger is raised and display a message:
# • “Index finger UP”
# • “Index finger DOWN”

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draws points and connections
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape
            index_x = int(hand_landmarks.landmark[8].x * w)
            index_y = int(hand_landmarks.landmark[8].y * h)

            knuckle_y = int(hand_landmarks.landmark[5].y * h)

            if index_y > knuckle_y:
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, "Index finger: DOWN", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "Index finger: UP", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "ESC: quit script", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()