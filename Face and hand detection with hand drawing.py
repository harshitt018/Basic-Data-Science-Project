import cv2
import mediapipe as mp
import numpy as np
import warnings
import time
import math
import os

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_cap = cv2.VideoCapture(0)

if not video_cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

canvas = None
drawing = False
prev_x, prev_y = None, None
brush_color = (0, 255, 0)
brush_radius = 10  
eraser_mode = False

# Color palette coordinates
palette = {
    'red': ((20, 20), (60, 60), (0, 0, 255)),
    'green': ((80, 20), (120, 60), (0, 255, 0)),
    'blue': ((140, 20), (180, 60), (255, 0, 0)),
    'white': ((200, 20), (240, 60), (255, 255, 255)),
    'yellow': ((260, 20), (300, 60), (0, 255, 255)),
}

def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1), fingers

# Auto-create screenshot folder
screenshot_folder = "screenshots"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw color palette
    for name, (start, end, color) in palette.items():
        cv2.rectangle(frame, start, end, color, -1)
        cv2.rectangle(frame, start, end, (255, 255, 255), 2)

    # Detect faces and draw rectangles
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            thumb = hand_landmarks.landmark[4]
            pinky = hand_landmarks.landmark[20]

            cx_index, cy_index = int(index.x * w), int(index.y * h)
            cx_thumb, cy_thumb = int(thumb.x * w), int(thumb.y * h)
            cx_middle, cy_middle = int(middle.x * w), int(middle.y * h)
            cx_pinky, cy_pinky = int(pinky.x * w), int(pinky.y * h)

            # Count fingers
            count, fingers = count_fingers(hand_landmarks)

            # 🤙 Gesture: Screenshot (thumb and pinky only)
            if fingers[0] and fingers[4] and not any(fingers[1:4]):
                filename = os.path.join(screenshot_folder, f"screenshot_{int(time.time())}.png")
                success = cv2.imwrite(filename, frame)
                if success:
                    print("✅ Screenshot saved at:", os.path.abspath(filename))
                else:
                    print("❌ Failed to save screenshot.")

            # 👆 Select color/tool
            elif count == 1:
                for name, (start, end, color) in palette.items():
                    if start[0] <= cx_index <= end[0] and start[1] <= cy_index <= end[1]:
                        brush_color = color
                        print(f"Color changed to {name}")
                        break

            # ✋ Eraser mode (all fingers up)
            elif count == 5:
                eraser_mode = True
                drawing = True

            # ✌️ Drawing mode (index and middle fingers only)
            elif fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
                eraser_mode = False
                drawing = True
            else:
                drawing = False

            # Drawing or erasing
            if drawing:
                if prev_x is not None and prev_y is not None:
                    color = (0, 0, 0) if eraser_mode else brush_color
                    thickness = brush_radius if not eraser_mode else 30  
                    cv2.line(canvas, (prev_x, prev_y), (cx_index, cy_index), color, thickness)
                prev_x, prev_y = cx_index, cy_index
            else:
                prev_x, prev_y = None, None

    # Merge canvas with live frame
    frame = cv2.add(frame, canvas)

    # Show info
    cv2.putText(frame, f'Mode: {"Eraser" if eraser_mode else "Draw"}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if eraser_mode else (0, 255, 0), 2)
    cv2.putText(frame, f'Brush Size: {brush_radius}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show final output
    cv2.imshow("Hand + Face Drawing App", frame)
    key = cv2.waitKey(2) & 0xFF
    if key == ord('a'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        print("Canvas cleared.")
    elif key == ord('s'):
        filename = os.path.join(screenshot_folder, f"screenshot_{int(time.time())}.png")
        success = cv2.imwrite(filename, frame)
        if success:
            print("✅ Screenshot saved at:", os.path.abspath(filename))
        else:
            print("❌ Failed to save screenshot.")

video_cap.release()
cv2.destroyAllWindows()
