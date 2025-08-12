from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()

HAZARD_CLASSES = ["person", "cat", "dog", "bird"]
latest_alert = ""

# Load MediaPipe Object Detector (uses built-in model)
base_options = python.BaseOptions(model_asset_path=mp.__file__.replace("__init__.py", "tasks/test_data/efficientdet_lite0.tflite"))
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

def list_cameras(max_test=5):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def camera_loop(cam_index):
    global latest_alert
    cap = cv2.VideoCapture(cam_index)
    mp_image = vision.Image
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to MediaPipe format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect objects
        detection_result = detector.detect(mp_img)
        alert_text = ""
        for detection in detection_result.detections:
            cls_name = detection.categories[0].category_name
            score = detection.categories[0].score
            if cls_name.lower() in HAZARD_CLASSES:
                alert_text = f"⚠ {cls_name.upper()} DETECTED!"
                bbox = detection.bounding_box
                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        latest_alert = alert_text
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>Road Safety Detection</title>
    </head>
    <body style="background:#222; color:white; text-align:center;">
        <h1>🐾 Road Safety Detection</h1>
        <label>Select Camera:</label>
        <select id="camSelect"></select>
        <button onclick="startFeed()">Start</button>
        <h2 id="alertText" style="color:red;"></h2>
        <br>
        <img id="videoFeed" width="640" height="480" style="border:2px solid white;">
        <script>
            async function loadCameras() {
                let res = await fetch('/list_cameras');
                let cams = await res.json();
                let sel = document.getElementById('camSelect');
                cams.forEach(idx => {
                    let opt = document.createElement('option');
                    opt.value = idx;
                    opt.innerText = "Camera " + idx;
                    sel.appendChild(opt);
                });
            }
            function startFeed() {
                let idx = document.getElementById('camSelect').value;
                document.getElementById('videoFeed').src = '/video?cam_index=' + idx;
                setInterval(async () => {
                    let alertRes = await fetch('/alert');
                    let alertTxt = await alertRes.text();
                    document.getElementById('alertText').innerText = alertTxt;
                }, 500);
            }
            loadCameras();
        </script>
    </body>
    </html>
    """

@app.get("/list_cameras")
def get_cameras():
    return JSONResponse(list_cameras())

@app.get("/alert")
def get_alert():
    return latest_alert

@app.get("/video")
def video_feed(cam_index: int = Query(0)):
    return StreamingResponse(camera_loop(cam_index),
                             media_type="multipart/x-mixed-replace; boundary=frame")
