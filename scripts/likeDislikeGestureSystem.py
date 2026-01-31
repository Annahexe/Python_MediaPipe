import cv2
import mediapipe as mp
import math
import time

# Like–Dislike Gesture System
# Detects thumbs up/down with a closed hand, shows Like/Dislike/Neutral text + colored thumb marker,
# and spawns small floating reaction icons near the thumb (wiggle up + fade out).

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

like_icon = cv2.imread("thumb-up.png", cv2.IMREAD_UNCHANGED)      # RGBA PNG
dislike_icon = cv2.imread("thumb-down.png", cv2.IMREAD_UNCHANGED) # RGBA PNG

def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def overlay_png(background, overlay, x, y, scale=1.0, alpha=1.0):
    if overlay is None or overlay.ndim != 3 or overlay.shape[2] < 4:
        return background

    if scale != 1.0:
        ow = max(1, int(overlay.shape[1] * scale))
        oh = max(1, int(overlay.shape[0] * scale))
        overlay = cv2.resize(overlay, (ow, oh), interpolation=cv2.INTER_AREA)

    h, w = overlay.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(background.shape[1], x + w)
    y2 = min(background.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return background

    overlay_crop = overlay[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]
    overlay_rgb = overlay_crop[:, :, :3].astype("float32")
    overlay_a = (overlay_crop[:, :, 3].astype("float32") / 255.0) * float(alpha)
    overlay_a = overlay_a[..., None]

    roi = background[y1:y2, x1:x2].astype("float32")
    blended = roi * (1.0 - overlay_a) + overlay_rgb * overlay_a
    background[y1:y2, x1:x2] = blended.astype("uint8")
    return background

# --- Floating reaction particles ---
particles = []

def spawn_particle(icon, x, y, scale=0.10):
    t = time.time()
    particles.append({
        "icon": icon,
        "x0": x,
        "y0": y,
        "age": 0,
        "lifetime": 40,  # frames
        "amp": 10 + 6 * (math.sin(t * 3.1) * 0.5 + 0.5),   # wiggle amplitude (px)
        "freq": 6 + 3 * (math.sin(t * 2.2) * 0.5 + 0.5),   # wiggle speed
        "speed": 1.6 + 0.9 * (math.sin(t * 1.7) * 0.5 + 0.5),  # upward speed
        "scale": scale
    })

def update_and_draw_particles(frame):
    alive = []
    for p in particles:
        p["age"] += 1
        t = p["age"]

        x = int(p["x0"] + p["amp"] * math.sin(t / p["freq"]))
        y = int(p["y0"] - p["speed"] * t)

        fade_start = int(p["lifetime"] * 0.55)
        if t < fade_start:
            alpha = 1.0
        else:
            alpha = max(0.0, 1.0 - (t - fade_start) / max(1, (p["lifetime"] - fade_start)))

        frame = overlay_png(frame, p["icon"], x, y, scale=p["scale"], alpha=alpha)

        if t < p["lifetime"]:
            alive.append(p)

    particles.clear()
    particles.extend(alive)
    return frame

# --- Camera ---
cap = cv2.VideoCapture(0)

last_spawn_time = 0.0
spawn_cooldown = 0.25  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    h, w, c = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb points
            thumb_tip = hand_landmarks.landmark[4]
            thumb_base = hand_landmarks.landmark[2]

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            thumb_up = thumb_tip.y < thumb_base.y
            thumb_down = thumb_tip.y > thumb_base.y

            # Hand open/closed
            wrist = hand_landmarks.landmark[0]
            finger_tips = [8, 12, 16, 20]
            avg_dist = sum(distance(hand_landmarks.landmark[i], wrist) for i in finger_tips) / len(finger_tips)
            hand_open = avg_dist > 0.18

            if thumb_up and not hand_open:
                text = "Like :)"
                color = (0, 255, 0)

                # Spawn floating icon
                now = time.time()
                if now - last_spawn_time >= spawn_cooldown:
                    spawn_particle(like_icon, thumb_x + 15, thumb_y - 80, scale=0.08)
                    last_spawn_time = now

            elif thumb_down and not hand_open:
                text = "Dislike :("
                color = (0, 0, 255)

                now = time.time()
                if now - last_spawn_time >= spawn_cooldown:
                    spawn_particle(dislike_icon, thumb_x + 15, thumb_y - 80, scale=0.08)
                    last_spawn_time = now

            else:
                text = "Neutral"
                color = (255, 255, 255)

            # Thumb marker + text
            cv2.circle(frame, (thumb_x, thumb_y), 10, color, cv2.FILLED)
            cv2.putText(frame, text, (thumb_x + 5, thumb_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Debug
            cv2.putText(frame, f"avg_dist: {avg_dist:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    frame = update_and_draw_particles(frame)
    cv2.putText(frame, "ESC: quit script", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
