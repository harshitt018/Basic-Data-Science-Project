import cv2
import mediapipe as mp
import numpy as np
import warnings
import time
import os

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Initialize MediaPipe with optimized settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,  
    min_tracking_confidence=0.7    
)
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_cap = cv2.VideoCapture(0)

if not video_cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Set camera properties for better performance
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_cap.set(cv2.CAP_PROP_FPS, 30)

canvas = None
drawing = False
prev_x, prev_y = None, None
brush_color = (0, 255, 0)
brush_radius = 10
eraser_mode = False
last_color_change_time = 0

# Color palette (removed black)
palette = {
    'red': ((50, 50), (100, 90), (0, 0, 255)),
    'green': ((120, 50), (170, 90), (0, 255, 0)),
    'blue': ((190, 50), (240, 90), (255, 0, 0)),
    'white': ((260, 50), (310, 90), (255, 255, 255)),
    'yellow': ((330, 50), (380, 90), (0, 255, 255)),
}

def count_fingers(hand_landmarks):
    """Optimized finger counting with better accuracy"""
    tips_ids = [4, 8, 12, 16, 20]
    fingers = [0] * 5

    # Thumb - check if tip is to the left of the previous joint
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers[0] = 1

    # Other 4 fingers - check if tip is above the joint 2 positions down
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers[i] = 1

    return sum(fingers), fingers

def is_point_in_palette(x, y):
    """Check if point is in any palette color area"""
    for name, (start, end, color) in palette.items():
        if start[0] <= x <= end[0] and start[1] <= y <= end[1]:
            return name, color
    return None, None

def draw_palette(frame):
    """Draw color palette with labels"""
    for name, (start, end, color) in palette.items():
        cv2.rectangle(frame, start, end, color, -1)
        cv2.rectangle(frame, start, end, (255, 255, 255), 2)
        cv2.putText(frame, name.capitalize(), (start[0] + 5, end[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Auto-create screenshot folder
screenshot_folder = "screenshots"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

print("Hand Gesture Drawing App Started!")
print("Gestures:")
print("👆 One finger (index): Select color from palette")
print("✌️  Two fingers (index + middle): Draw mode")
print("✋ Five fingers: Eraser mode")
print("🤙 Thumb + Pinky: Screenshot")
print("Keys: 'c' = clear canvas, 's' = save screenshot, 'q' = quit")

# Optimization variables
frame_skip = 2  # Process every 2nd frame for hand detection
frame_count = 0
gesture_buffer = []
buffer_size = 3  # Reduced buffer size

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Draw color palette
    draw_palette(frame)

    # Face detection (every 5th frame for efficiency)
    if frame_count % 5 == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    # Draw face rectangles from last detection
    try:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    except NameError:
        faces = []

    # Hand detection (optimized frequency)
    process_hands = frame_count % frame_skip == 0
    current_time = time.time()
    
    if process_hands:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
    else:
        try:
            results = last_results
        except NameError:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
    
    last_results = results
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Lighter hand landmarks for better performance
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1))

            h, w, _ = frame.shape
            
            # Get key landmark positions
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            thumb = hand_landmarks.landmark[4]
            pinky = hand_landmarks.landmark[20]

            cx_index, cy_index = int(index.x * w), int(index.y * h)
            cx_middle, cy_middle = int(middle.x * w), int(middle.y * h)

            # Optimized gesture recognition with smaller buffer
            count, fingers = count_fingers(hand_landmarks)
            gesture_buffer.append((count, fingers))
            
            if len(gesture_buffer) > buffer_size:
                gesture_buffer.pop(0)
            
            # Use most recent stable gesture
            if len(gesture_buffer) >= 2:
                recent_counts = [g[0] for g in gesture_buffer[-2:]]
                if recent_counts[0] == recent_counts[1]:
                    stable_count, stable_fingers = gesture_buffer[-1]
                else:
                    stable_count, stable_fingers = count, fingers
            else:
                stable_count, stable_fingers = count, fingers

            # Show finger count for debugging
            cv2.putText(frame, f"Fingers: {stable_count}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Screenshot gesture (thumb and pinky only)
            if stable_fingers[0] and stable_fingers[4] and not any(stable_fingers[1:4]):
                cv2.putText(frame, "SCREENSHOT MODE", (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if current_time - last_color_change_time > 2:
                    filename = os.path.join(screenshot_folder, f"screenshot_{int(time.time())}.png")
                    combined_frame = cv2.add(frame, canvas)
                    success = cv2.imwrite(filename, combined_frame)
                    if success:
                        print("✅ Screenshot saved at:", os.path.abspath(filename))
                        last_color_change_time = current_time

            # Color selection (index finger only)
            elif stable_count == 1 and stable_fingers[1] and sum(stable_fingers) == 1:
                cv2.putText(frame, "COLOR SELECT MODE", (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                color_name, selected_color = is_point_in_palette(cx_index, cy_index)
                if color_name and selected_color and current_time - last_color_change_time > 0.5:
                    brush_color = selected_color
                    eraser_mode = False
                    print(f"Color changed to {color_name}")
                    last_color_change_time = current_time
                    
                cv2.circle(frame, (cx_index, cy_index), 15, (255, 255, 255), 3)

            # Eraser mode (all five fingers up)
            elif stable_count == 5:
                cv2.putText(frame, "ERASER MODE", (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                eraser_mode = True
                drawing = True
                
                if prev_x is not None and prev_y is not None:
                    # Create eraser mask and apply it to canvas
                    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
                    cv2.line(mask, (prev_x, prev_y), (cx_index, cy_index), 255, 30)
                    cv2.circle(mask, (cx_index, cy_index), 15, 255, -1)
                    canvas[mask == 255] = [0, 0, 0]
                
                prev_x, prev_y = cx_index, cy_index

            # Drawing mode (index and middle fingers only)
            elif stable_count == 2 and stable_fingers[1] and stable_fingers[2] and sum(stable_fingers) == 2:
                cv2.putText(frame, "DRAW MODE", (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                eraser_mode = False
                drawing = True
                
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (cx_index, cy_index), brush_color, brush_radius)
                    cv2.circle(canvas, (cx_index, cy_index), brush_radius//2, brush_color, -1)
                
                prev_x, prev_y = cx_index, cy_index
            
            else:
                drawing = False
                prev_x, prev_y = None, None

    else:
        prev_x, prev_y = None, None
        drawing = False

    # Combine canvas with frame
    frame = cv2.add(frame, canvas)

    # Show status information
    status_color = (0, 0, 255) if eraser_mode else (0, 255, 0)
    mode_text = "Eraser" if eraser_mode else "Draw"
    cv2.putText(frame, f'Mode: {mode_text}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    cv2.putText(frame, f'Brush Size: {brush_radius}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Show current brush color (only if not in eraser mode)
    if not eraser_mode:
        cv2.rectangle(frame, (10, 80), (60, 120), brush_color, -1)
        cv2.rectangle(frame, (10, 80), (60, 120), (255, 255, 255), 2)
    else:
        # Show eraser indicator
        cv2.rectangle(frame, (10, 80), (60, 120), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 80), (60, 120), (255, 255, 255), 2)
        cv2.putText(frame, "E", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand + Face Drawing App", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        print("Canvas cleared.")
    elif key == ord('s'):
        filename = os.path.join(screenshot_folder, f"screenshot_{int(time.time())}.png")
        success = cv2.imwrite(filename, frame)
        if success:
            print("✅ Screenshot saved at:", os.path.abspath(filename))

print("Application closed.")
video_cap.release()
cv2.destroyAllWindows()
