import cv2
import mediapipe as mp

#  Faces detected counter 😁🙂

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,       # 0: near faces, 1: far
    min_detection_confidence=0.6
)

#cap = cv2.VideoCapture("VideoNameHere.mp4") # Can use a video instead of the camera as an alternative
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    faces = 0
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)
            faces += 1

    cv2.putText(frame, "Faces detected: " + str(faces), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "ESC: quit script", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Face Detection", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
