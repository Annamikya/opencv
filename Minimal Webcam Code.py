import cv2  # OpenCV library

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()  # Read frame from webcam
    
    if not ret:
        break  # Stop if camera fails

    cv2.imshow("Webcam Feed", frame)  # Show frame

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()  # Release camera
cv2.destroyAllWindows()  # Close window
