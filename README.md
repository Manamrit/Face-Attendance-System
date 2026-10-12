# AI-Based Attendance System Using Face Recognition

A real-time attendance system that recognizes registered faces via webcam
and automatically logs attendance to a CSV file, timestamped, once per
person per day.

## Tech Stack
- **Python 3.9+**
- **OpenCV** – webcam capture, drawing, image handling
- **face_recognition** (built on dlib) – face detection + 128-d face encodings
- **NumPy** – distance calculations for matching faces
- **CSV** – simple attendance storage (can be swapped for SQLite/MySQL later)

## Project Structure
```
face_attendance_system/
│
├── dataset/                  # Captured face images, organized by person
│   ├── John_Doe/
│   │   ├── 1.jpg
│   │   └── ...
│   └── Jane_Smith/
│
├── encodings/
│   └── encodings.pickle      # All known face encodings + names (generated)
│
├── attendance/
│   └── Attendance_2026-09-12.csv   # Auto-generated daily attendance log
│
├── add_faces.py               # Step 1: Register a new person via webcam
├── encode_faces.py            # Step 2: Convert images into face encodings
├── attendance_system.py       # Step 3: Run live recognition + attendance marking
└── requirements.txt
```

## How It Works (the core idea)

1. **Registration (`add_faces.py`)**
   Captures a handful of photos of a person's face and stores them under
   `dataset/<name>/`.

2. **Encoding (`encode_faces.py`)**
   For each stored photo, `face_recognition` detects the face and converts
   it into a **128-dimensional vector** — a unique numeric "fingerprint" of
   that face's geometry (distance between eyes, nose shape, jawline, etc.).
   These vectors + the person's name are saved into one `encodings.pickle`
   file.

3. **Live Recognition (`attendance_system.py`)**
   The webcam feed is processed frame by frame:
   - Detect any faces in the frame (HOG-based detector — fast on CPU).
   - Compute the 128-d encoding of each detected face.
   - Compare it (Euclidean distance) against every known encoding.
   - If the closest match is below a threshold (`0.5` here), it's
     considered a match → name is shown on screen and attendance is
     logged with a timestamp.
   - A person is only logged **once per day** (checked against the day's
     CSV before writing).

## Setup & Run

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register one or more people
python add_faces.py

# 4. Generate face encodings from the registered photos
python encode_faces.py

# 5. Run the live attendance system
python attendance_system.py
```

Repeat steps 3–4 for every new person you want to register. Press `q` to
quit the live window.

## Possible Improvements (great to mention in an interview/viva)
- Replace CSV storage with **SQLite/MySQL** for a real database backend.
- Add a **Flask/Streamlit web UI** instead of an OpenCV popup window.
- Use a **CNN-based detector** instead of HOG for better accuracy in low
  light (trade-off: slower without a GPU).
- Add **liveness detection** (blink detection) to prevent someone holding
  up a photo to spoof attendance.
- Deploy on a **Raspberry Pi + camera module** for a real classroom/office
  kiosk setup.
- Send automatic email/SMS attendance summary at end of day.

## How to Explain This Project in an Interview
If asked "walk me through this project," a strong answer is:

> "I built a face recognition attendance system using OpenCV for video
> capture and the `face_recognition` library, which is built on dlib's
> ResNet-based face recognition model. Each face is converted into a
> 128-dimensional embedding, and I compare new faces against stored
> embeddings using Euclidean distance with a similarity threshold. When a
> match is found, I log the person's name and timestamp to a CSV, with a
> check to prevent duplicate entries for the same day. I designed it in
> three modular stages — registration, encoding, and live inference — so
> each part can be tested and replaced independently, e.g., swapping CSV
> for a proper database."

This shows you understand: **the CV pipeline, embeddings/distance metrics,
practical engineering decisions (thresholds, duplicate handling), and
system design (modularity)** — all things interviewers listen for.
