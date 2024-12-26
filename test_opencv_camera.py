import cv2

# Open the camera with Video4Linux backend
cap = cv2.VideoCapture('/dev/video14', cv2.CAP_V4L2)
if not cap.isOpened():
    print("Cannot open camera at /dev/video14. Please check the device path or permissions.")
    exit()

# Set a lower resolution to reduce bandwidth
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Start capturing frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the frame in a window
    cv2.imshow('Camera Test', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting the camera feed.")
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
