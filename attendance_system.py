"""
attendance_system.py
---------------------
STEP 3 of the project (the main app).

Purpose:
    Opens the webcam, detects faces in real time, compares them against
    the known encodings from encodings/encodings.pickle, and if a match
    is found, marks that person "Present" in today's attendance CSV file
    (attendance/Attendance_<date>.csv). Each person is marked only ONCE per day.

Run:
    python attendance_system.py

Press 'q' to quit.
"""

import os
import pickle
import csv
from datetime import datetime

import cv2
import face_recognition
import numpy as np

ENCODINGS_PATH = os.path.join("encodings", "encodings.pickle")
ATTENDANCE_DIR = "attendance"
MATCH_TOLERANCE = 0.5  # lower = stricter matching. 0.6 is default, 0.5 is a bit stricter.


def load_known_faces():
    if not os.path.exists(ENCODINGS_PATH):
        raise FileNotFoundError(
            "encodings.pickle not found. Run add_faces.py then encode_faces.py first."
        )
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def get_today_csv_path():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")


def get_already_marked(csv_path):
    """Returns the set of names already marked present today, so we don't duplicate rows."""
    marked = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    marked.add(row[0])
    return marked


def mark_attendance(name, csv_path, already_marked):
    if name in already_marked:
        return  # already marked today, do nothing

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Time", "Date"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")])

    already_marked.add(name)
    print(f"Attendance marked for: {name}")


def run_attendance_system():
    known_encodings, known_names = load_known_faces()
    csv_path = get_today_csv_path()
    already_marked = get_already_marked(csv_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not access webcam.")
        return

    print("Starting attendance system. Press 'q' to quit.\n")

    # To keep it fast, process every other frame and a downscaled image
    process_this_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if process_this_frame:
            # Resize to 1/4 size for faster face detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compare this face to all known faces
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                name = "Unknown"

                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < MATCH_TOLERANCE:
                        name = known_names[best_match_index]
                        mark_attendance(name, csv_path, already_marked)

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Draw results back on the full-size frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Attendance System - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Attendance saved in: {csv_path}")


if __name__ == "__main__":
    run_attendance_system()
