# Import required libraries
import cv2
import mediapipe as mp
import numpy as np
import warnings
import time
import math

# Suppress specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Initializing MediaPipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Loading Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initializing the laptop's camera with index 0
video_cap = cv2.VideoCapture(0)

# Initializing a canvas for drawing
canvas = None

# Drawing state
drawing = False
prev_x, prev_y = None, None

# Start the time for FPS calculation
start_time = time.time()
frame_count = 0

# Setting the thresold for drawing in pixels (adjust this as needed)
distance_threshold = 40 

# Calculating the Euclidean distance (shortest distance between two points.)
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while True:
    # Capturing frame-by-frame
    ret, frame = video_cap.read()
    frame_count += 1

    # Checking the camera is accessible or not
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally(for a later selfie-view display)
    frame = cv2.flip(frame, 1)

    # Initialize canvas if not initialized
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Converting the video to RGB 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the video to detect hands
    results = hands.process(rgb_frame)

    # Converting the video to grayscale (required for face detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale video
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for faces

    # Draw hand nodes/landmarks 
    if results.multi_hand_landmarks:
        # If detected and use the index finger tip for drawing
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green color for connections
            )
            
            # Get coordinates of the index finger tip (node/landmark 8) and middle finger tip (node/landmark 12)
            h, w, _ = frame.shape
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            cx_index, cy_index = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cx_middle, cy_middle = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

            # Calculating the Euclidean distance between the index finger tip and middle finger tip
            distance = calculate_distance(cx_index, cy_index, cx_middle, cy_middle)

            # Toggle drawing based on the distance between the fingers
            if distance < distance_threshold:
                #If index fingure and middle finger arre inn contact then no drawing
                drawing = False
            else:
                drawing = True

            # Draw if drawing mode is enabled
            if drawing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (cx_index, cy_index), (0, 255, 0), 5)
                prev_x, prev_y = cx_index, cy_index
            else:
                prev_x, prev_y = None, None

    # Overlay the canvas on the frame
    frame = cv2.add(frame, canvas)

    # Display drawing mode status
    cv2.putText(frame, f'Drawing Mode: {"ON" if drawing else "OFF"}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if drawing else (0, 0, 255), 2)

    # Display the resulting frame with face rectangles, hand landmarks, and drawings
    cv2.imshow("Multi-Face and Multi-Hand Detector with Drawing", frame)

    # Check key presses to exit
    key = cv2.waitKey(5) & 0xFF
    if key == ord('a'):
        break

# Release the video capture object and close all OpenCV windows
video_cap.release()
cv2.destroyAllWindows()