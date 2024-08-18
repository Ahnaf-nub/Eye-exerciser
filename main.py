import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

prev_left_pupil = None
prev_right_pupil = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    # Iterate over the detected eyes
    for (ex, ey, ew, eh) in eyes:
        # Draw a rectangle around each eye
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        # Get the region of interest (ROI) in the grayscale image
        roi_gray = gray[ey:ey+eh, ex:ex+ew]
        roi_color = frame[ey:ey+eh, ex:ex+ew]

        # Use a threshold to binarize the image for pupil detection
        _, threshold = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour which should be the pupil
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the center and radius of the pupil
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw a circle around the pupil
            cv2.circle(roi_color, center, radius, (0, 5, 0), 2)

            # Determine which eye is being processed (left or right)
            if ex < frame.shape[1] // 2:
                current_pupil = "left"
                prev_pupil = prev_left_pupil
                prev_left_pupil = center
            else:
                current_pupil = "right"
                prev_pupil = prev_right_pupil
                prev_right_pupil = center

            # Check the movement direction if previous pupil position exists
            if prev_pupil is not None:
                dx = center[0] - prev_pupil[0]
                dy = center[1] - prev_pupil[1]
                
                if abs(dx) > abs(dy):  # Horizontal movement
                    if dx > 0:
                        direction = f"{current_pupil} pupil moving right"
                    else:
                        direction = f"{current_pupil} pupil moving left"
                else:  # Vertical movement
                    if dy > 0:
                        direction = f"{current_pupil} pupil moving down"
                    else:
                        direction = f"{current_pupil} pupil moving up"
                
                cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Pupil Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
