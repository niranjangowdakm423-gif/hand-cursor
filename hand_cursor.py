import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # remove extra delay

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)

# ==================== CONFIG ====================
SMOOTHING        = 0.7
AMPLIFY          = 2.2
PREDICT_GAIN     = 0.9
SCROLL_HOLD_TIME = 0.6       # seconds → very reliable now
SCROLL_SPEED     = 4.0       # ↑ higher = faster scroll
# ===============================================

# State
prev_cx = prev_cy = None
click_down = False

scroll_mode = False
scroll_start_time = None
last_scroll_y = None          # pixel y (higher = lower on screen)

def is_index_middle_up(lm):
    # Finger is extended if tip is above (lower normalized y) the PIP joint
    indexextended  = lm[8].y < lm[6].y
    middle_extended = lm[12].y < lm[10].y
    ring_down       = lm[16].y > lm[14].y
    pinky_down      = lm[20].y > lm[18].y
    return index_extended and middle_extended and ring_down and pinky_down

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        # ─────── CURSOR (index finger tip) ───────
        x = lm[8].x
        y = lm[8].y
        x = np.clip((x - 0.5) * AMPLIFY + 0.5, 0, 1)
        y = np.clip((y - 0.5) * AMPLIFY + 0.5, 0, 1)
        target_x = int(x * screen_w)
        target_y = int(y * screen_h)

        # Super smooth + prediction
        if prev_cx is None:
            sx = target_x
            sy = target_y
        else:
            sx = prev_cx + (target_x - prev_cx) * (1 - SMOOTHING)
            sy = prev_cy + (target_y - prev_cy) * (1 - SMOOTHING)

        vx = sx - (prev_cx or sx)
        vy = sy - (prev_cy or sy)
        px = int(np.clip(sx + vx * PREDICT_GAIN, 0, screen_w-1))
        py = int(np.clip(sy + vy * PREDICT_GAIN, 0, screen_h-1))
        pyautogui.moveTo(px, py)

        prev_cx, prev_cy = sx, sy

        # ─────── PINCH CLICK ───────
        thumb_index_dist = abs(lm[4].x - lm[8].x) + abs(lm[4].y - lm[8].y)
        if thumb_index_dist < 0.05 and not click_down:
            pyautogui.mouseDown()
            click_down = True
        elif thumb_index_dist > 0.07 and click_down:
            pyautogui.mouseUp()
            click_down = False

def is_two_finger_scroll(lm):
    # Both index & middle extended
    index_extended  = lm[8].y < lm[6].y
    middle_extended = lm[12].y < lm[10].y
    # Ring & pinky folded
    ring_down       = lm[16].y > lm[14].y
    pinky_down      = lm[20].y > lm[18].y
    # Fingers close together (avoid false positives)
    fingers_close   = abs(lm[8].x - lm[12].x) < 0.05
    return index_extended and middle_extended and ring_down and pinky_down and fingers_close

# ─────── TWO-FINGER SCROLL DETECTION ───────
pose_active = is_two_finger_scroll(lm)

if pose_active:
    if scroll_start_time is None:
        scroll_start_time = time.time()
        last_scroll_y = (lm[8].y + lm[12].y) / 2 * h
else:
    scroll_mode = False
    scroll_start_time = None
    last_scroll_y = None

# Activate scroll mode after holding pose
if scroll_start_time and (time.time() - scroll_start_time) >= SCROLL_HOLD_TIME:
    scroll_mode = True

# ─────── ACTUAL SCROLLING ───────
if scroll_mode and last_scroll_y is not None:
    current_y = (lm[8].y + lm[12].y) / 2 * h
    delta_y = last_scroll_y - current_y
    scroll_amount = int(delta_y / 10)   # divide to keep values small

    if scroll_amount != 0:
        pyautogui.scroll(scroll_amount)
        last_scroll_y = current_y
# ─────── DRAWING & FEEDBACK ───────
mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

status = "SCROLL MODE - MOVE HAND UP/DOWN" if scroll_mode else "Cursor Mode"
color = (0, 255, 0) if scroll_mode else (0, 165, 255)
cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 3)

if click_down:
    cv2.putText(frame, "CLICK", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

else:
   prev_cx = prev_cy = None
   scroll_start_time = None
   scroll_mode = False

   cv2.imshow("Working Hand Cursor + Two-Finger Scroll", frame)

cap.release()
cv2.destroyAllWindows()
