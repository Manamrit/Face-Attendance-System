"""
app.py
------
Flask web dashboard for the Face Recognition Attendance System.

Pages:
  /            -> Live recognition feed (marks attendance automatically)
  /register    -> Register a new person by capturing photos from the webcam
  /attendance  -> View attendance records for any date

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.

Design note:
    The webcam is accessed directly by this Python backend (via OpenCV),
    NOT through the browser's camera API. This keeps things simple and
    matches how the original desktop version worked -- it just streams
    the video into the browser instead of opening a popup window.
    A background thread continuously reads frames so multiple pages
    (live feed + registration + photo capture) can share one camera.
"""

import os
import csv
import time
import pickle
import threading
from datetime import datetime

import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, request, jsonify

DATASET_DIR = "dataset"
ENCODINGS_PATH = os.path.join("encodings", "encodings.pickle")
ATTENDANCE_DIR = "attendance"
MATCH_TOLERANCE = 0.5

app = Flask(__name__)

# ---- Shared state between the background camera thread and Flask routes ----
lock = threading.Lock()
latest_raw_frame = None          # unmodified frame -- used when saving registration photos
latest_annotated_frame = None    # frame with name boxes drawn -- used for the live feed
known_encodings = []
known_names = []
already_marked_today = set()
today_csv_path = None


def load_known_faces():
    """Loads saved face encodings from disk into memory."""
    global known_encodings, known_names
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    else:
        known_encodings, known_names = [], []


def get_today_csv_path():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")


def load_already_marked(csv_path):
    marked = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    marked.add(row[0])
    return marked


def mark_attendance(name):
    """Logs a name to today's CSV, but only once per day."""
    global today_csv_path, already_marked_today
    csv_path = get_today_csv_path()
    if csv_path != today_csv_path:
        today_csv_path = csv_path
        already_marked_today = load_already_marked(csv_path)

    if name in already_marked_today:
        return

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Time", "Date"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")])

    already_marked_today.add(name)


def camera_loop():
    """
    Runs forever in a background thread. Continuously reads webcam frames,
    runs face recognition every other frame (for speed), draws boxes/names,
    and stores both the raw and annotated frames for the Flask routes to use.
    """
    global latest_raw_frame, latest_annotated_frame

    cap = cv2.VideoCapture(0)
    process_this_frame = True
    face_locations, face_names = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        with lock:
            latest_raw_frame = frame.copy()

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < MATCH_TOLERANCE:
                        name = known_names[best_match_index]
                        mark_attendance(name)
                face_names.append(name)

        process_this_frame = not process_this_frame

        annotated = frame.copy()
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            cv2.rectangle(annotated, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(annotated, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        with lock:
            latest_annotated_frame = annotated

        time.sleep(0.01)


def gen_stream(get_frame_fn):
    """Generic MJPEG generator used by both the live feed and registration feed."""
    while True:
        frame = get_frame_fn()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


def get_annotated_frame():
    with lock:
        return latest_annotated_frame.copy() if latest_annotated_frame is not None else None


def get_raw_frame():
    with lock:
        return latest_raw_frame.copy() if latest_raw_frame is not None else None


# ---------------------------- Routes ----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_stream(get_annotated_frame),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/register_feed")
def register_feed():
    return Response(gen_stream(get_raw_frame),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture_photo", methods=["POST"])
def capture_photo():
    name = request.form.get("name", "").strip()
    count = request.form.get("count", "1")

    if not name:
        return jsonify({"success": False, "error": "Name is required"})

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    frame = get_raw_frame()
    if frame is None:
        return jsonify({"success": False, "error": "No camera frame available yet"})

    img_path = os.path.join(person_dir, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    return jsonify({"success": True, "path": img_path})


@app.route("/encode", methods=["POST"])
def encode():
    """Re-generates encodings.pickle from everything currently in dataset/."""
    known_enc, known_nm = [], []

    if not os.path.exists(DATASET_DIR):
        return jsonify({"success": False, "error": "No dataset found"})

    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(person_folder, image_name)
            try:
                image = face_recognition.load_image_file(image_path)

                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

                image = np.ascontiguousarray(image, dtype=np.uint8)

                boxes = face_recognition.face_locations(image, model="hog")
                if not boxes:
                    continue

                encs = face_recognition.face_encodings(image, boxes)
                for e in encs:
                    known_enc.append(e)
                    known_nm.append(person_name)
            except Exception:
                continue

    os.makedirs("encodings", exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_enc, "names": known_nm}, f)

    load_known_faces()
    return jsonify({"success": True, "count": len(known_enc)})


@app.route("/attendance")
def attendance():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    csv_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            rows = list(reader)

    return render_template("attendance.html", rows=rows, date_str=date_str)


if __name__ == "__main__":
    load_known_faces()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    app.run(debug=False, threaded=True)
