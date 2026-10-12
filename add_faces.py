"""
add_faces.py
------------
STEP 1 of the project.

Purpose:
    Registers a new person into the system by capturing several photos
    of their face using the webcam and saving them into:
        dataset/<person_name>/1.jpg, 2.jpg, ...

Run:
    python add_faces.py

You will be asked to type the person's name, then the webcam will open.
Press SPACE to capture a photo (capture 5-10 photos from different angles
for best accuracy). Press ESC to finish early.
"""

import cv2
import os

def add_new_face():
    name = input("Enter the person's name (no spaces, e.g. John_Doe): ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    # Folder where this person's face images will be stored
    person_dir = os.path.join("dataset", name)
    os.makedirs(person_dir, exist_ok=True)

    # Start webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    # Haar cascade is only used here to DRAW a box for the user's guidance
    # while capturing - the actual recognition later uses face_recognition,
    # which is more accurate.
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    target_images = 8

    print("\nInstructions:")
    print(" - Look at the camera, move your head slightly between shots")
    print(" - Press SPACE to capture a photo")
    print(" - Press ESC to stop early\n")

    while count < target_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Captured: {count}/{target_images}  (SPACE=capture, ESC=quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Register Face - " + name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            img_path = os.path.join(person_dir, f"{count + 1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {count} images saved for '{name}'.")
    print("Next step: run encode_faces.py to process this person's face.")


if __name__ == "__main__":
    add_new_face()
