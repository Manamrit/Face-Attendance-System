"""
encode_faces.py
---------------
STEP 2 of the project.
"""

import os
import pickle
import cv2
import numpy as np
import face_recognition

DATASET_DIR = "dataset"
ENCODINGS_PATH = os.path.join("encodings", "encodings.pickle")


def encode_dataset():
    known_encodings = []
    known_names = []

    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("No data found in dataset/. Run add_faces.py first.")
        return

    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"Processing images for: {person_name}")

        for image_name in os.listdir(person_folder):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"  Skipping non-image file: {image_name}")
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

                if len(boxes) == 0:
                    print(f"  Warning: no face found in {image_name}, skipping.")
                    continue

                encodings = face_recognition.face_encodings(image, boxes)

                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person_name)

            except Exception as e:
                print(f"  Error processing {image_name}: {e} -- skipping this file.")
                continue

    data = {"encodings": known_encodings, "names": known_names}
    os.makedirs("encodings", exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"\nDone. {len(known_encodings)} face encodings saved to {ENCODINGS_PATH}")


if __name__ == "__main__":
    encode_dataset()