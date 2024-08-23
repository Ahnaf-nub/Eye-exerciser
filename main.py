import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Eye Exerciser")
root.geometry("800x600")

# Label to display the eye direction
direction_label = Label(root, text="Direction: Straight", font=("Arial", 25))
direction_label.pack()

# Label to display the video feed
video_label = Label(root)
video_label.pack()

# OpenCV eye cascade for detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
camera = cv2.VideoCapture(0)

prev_left_pupil = None
prev_right_pupil = None
current_direction = "Straight"

def detect_movement(dx, dy):
    if dx > abs(dy):  # More horizontal movement to the right
        return "Right"
    elif -dx > abs(dy):  # More horizontal movement to the left
        return "Left"
    elif dy > abs(dx):  # More vertical movement downwards
        return "Down"
    elif -dy > abs(dx):  # More vertical movement upwards
        return "Up"
    else:
        return "Straight"  # Little or no significant movement
    
def check_eye_movement():
    global current_direction
    if current_direction == "Straight":
        direction_label.config(text="Move your eye to the Right")
        current_direction = "Right"
    elif current_direction == "Right":
        direction_label.config(text="Move your eye to the Left")
        current_direction = "Left"
    elif current_direction == "Left":
        direction_label.config(text="Move your eye Down")
        current_direction = "Down"
    elif current_direction == "Down":
        direction_label.config(text="Move your eye Up")
        current_direction = "Up"
    elif current_direction == "Up":
        direction_label.config(text="Move your eye to the Right")
        current_direction = "Right"

def detect_eyes(frame):
    global prev_left_pupil, prev_right_pupil, current_direction

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    for (ex, ey, ew, eh) in eyes:
        roi_gray = gray[ey:ey+eh, ex:ex+ew]
        _, threshold = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            cv2.circle(frame, (ex + center[0], ey + center[1]), int(radius), (0, 255, 0), 2)

            if ex < frame.shape[1] // 2:  # Left eye
                prev_pupil = prev_left_pupil
                prev_left_pupil = center
            else:  # Right eye
                prev_pupil = prev_right_pupil
                prev_right_pupil = center

            if prev_pupil is not None:
                dx = center[0] - prev_pupil[0]
                dy = center[1] - prev_pupil[1]

                new_direction = detect_movement(dx, dy)
                if new_direction != current_direction:
                    check_eye_movement()
                    current_direction = new_direction
                    direction_label.config(text=f"{current_direction}")

    return frame

def show_frame():
    ret, frame = camera.read()
    if ret:
        frame = detect_eyes(frame)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, show_frame)

show_frame()
root.mainloop()
