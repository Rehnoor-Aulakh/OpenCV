import cv2
import pyautogui
import mediapipe as mp

#1 opening the video camera
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands model and drawing utilities
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get the screen resolution
screen_width, screen_height = pyautogui.size()
# Define parameters for cursor smoothing
smoothed_cursor_y=0
smooth_factor = 0.5  # Smoothing factor for exponential moving average filter
prev_cursor_x, prev_cursor_y = 0, 0  # Previous cursor position
prev_thumb_x, prev_thumb_y = 0, 0  # Previous thumb position

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Failed to capture frame")
        break

    #inverting the frame to prevent mirroring
    frame = cv2.flip(frame, 1)

    # Get the dimensions of the frame
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting hands in the frame
    outputs = hand_detector.process(rgb_frame)
    hands = outputs.multi_hand_landmarks

    # Process each detected hand
    if hands:
        for hand in hands:
            # Process landmarks
            for id, landmark in enumerate(hand.landmark):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # Smooth cursor movement using exponential moving average (EMA) filter
                if id == 8:  # Index finger

                    cursor_x = int(screen_width / frame_width * x)
                    cursor_y = int(screen_height / frame_height * y)
                    #marking the index finger by yellow circle
                    cv2.circle(frame, (x,y), 10, (0, 255, 255), 5)

                    # Update previous cursor position using smoothing
                    smoothed_cursor_x = smooth_factor * cursor_x + (1 - smooth_factor) * prev_cursor_x
                    smoothed_cursor_y = smooth_factor * cursor_y + (1 - smooth_factor) * prev_cursor_y

                    # Moving the cursor to the smoothed coordinates
                    pyautogui.moveTo(smoothed_cursor_x, smoothed_cursor_y)

                    # Updating previous cursor position
                    prev_cursor_x, prev_cursor_y = smoothed_cursor_x, smoothed_cursor_y


                if id == 4:  # Thumb
                    thumb_x = int(screen_width / frame_width * x)
                    thumb_y = int(screen_height / frame_height * y)
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), 5)

                    # Update previous thumb position using smoothing
                    smoothed_thumb_x = smooth_factor * thumb_x + (1 - smooth_factor) * prev_thumb_x
                    smoothed_thumb_y = smooth_factor * thumb_y + (1 - smooth_factor) * prev_thumb_y

                    # Checking for click, when thumb and index finger position are close enough
                    if abs(smoothed_cursor_y - smoothed_thumb_y) < 50:
                        pyautogui.click()

                    # Update previous thumb position
                    prev_thumb_x, prev_thumb_y = smoothed_thumb_x, smoothed_thumb_y

    # Show the frame
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
