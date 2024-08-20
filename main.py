import cv2
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, Response

app = FastAPI()
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
templates = Jinja2Templates(directory="templates")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
prev_left_pupil = None
prev_right_pupil = None

def detect_movement(dx, dy):
    #have to adjusted threshold for more sensitive movement detection
    if abs(dx) > 1.5 * abs(dy):  # More horizontal movement
        return "Right" if dx > 0 else "Left"
    elif abs(dy) > 1.5 * abs(dx):  # More vertical movement
        return "Down" if dy > 0 else "Up"
    else:
        return "Straight"  # Little or no significant movement

def detect_eyes(frame):
    global prev_left_pupil, prev_right_pupil

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    direction = "Straight"

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

                direction = detect_movement(dx, dy)
                
                print(f"Detected movement: {direction}, dx: {dx}, dy: {dy}")

    return frame, direction

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_with_eyes, _ = detect_eyes(frame)
            _, buffer = cv2.imencode('.jpg', frame_with_eyes)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/eye_movement')
def eye_movement():
    def event_stream():
        while True:
            success, frame = camera.read()
            if not success:
                break
            _, direction = detect_eyes(frame)
            yield f"data: {direction}\n\n"
            time.sleep(0.5)
    return Response(event_stream(), media_type='text/event-stream')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
