import cv2
import time

cap = cv2.VideoCapture(14)  # or try 0
time.sleep(2)  # Wait for 2 seconds
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
if ret:
    cv2.imwrite('test_frame.jpg', frame)
    print("Frame captured and saved as test_frame.jpg")
else:
    print("Failed to capture frame")

cap.release()