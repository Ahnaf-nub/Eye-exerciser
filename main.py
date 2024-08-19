from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

movement_logs = []

def detect_movement(dx, dy):
    if abs(dx) > 2 * abs(dy):  # More horizontal movement
        return "Right" if dx > 0 else "Left"
    elif abs(dy) > 2 * abs(dx):  # More vertical movement
        return "Down" if dy > 0 else "Up"
    else:
        return "Straight"  # Little or no significant movement

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    global movement_logs
    movement_logs.clear()

    cap = cv2.VideoCapture(0)
    try:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        prev_left_pupil = None
        prev_right_pupil = None

        for (ex, ey, ew, eh) in eyes:
            roi_gray = gray[ey:ey+eh, ex:ex+ew]
            _, threshold = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))

                if ex < frame.shape[1] // 2:  # Left side of the frame
                    current_pupil = "left"
                    prev_pupil = prev_left_pupil
                    prev_left_pupil = center
                else:  # Right side of the frame
                    current_pupil = "right"
                    prev_pupil = prev_right_pupil
                    prev_right_pupil = center

                if prev_pupil is not None:
                    dx = center[0] - prev_pupil[0]
                    dy = center[1] - prev_pupil[1]

                    direction = detect_movement(dx, dy)
                    movement_logs.append(f"{current_pupil.capitalize()} pupil moving {direction}")
    finally:
        cap.release()

    return templates.TemplateResponse("index.html", {"request": request, "logs": movement_logs})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
