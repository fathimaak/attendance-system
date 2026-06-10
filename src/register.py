import cv2
import pickle
import os
import numpy as np
from deepface import DeepFace

ENCODINGS_FILE = "data/encodings.pkl"
MODEL_NAME = "ArcFace"  # Research-backed choice over FaceNet/dlib

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {"names": [], "encodings": []}

def save_encodings(data):
    os.makedirs("data", exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

def get_embedding(frame_rgb):
    """Extract 512D ArcFace embedding from a face image."""
    try:
        result = DeepFace.represent(
            img_path=frame_rgb,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend="opencv"
        )
        return np.array(result[0]["embedding"]), "ok"
    except Exception as e:
        return None, f"No face detected clearly. Try again."

def capture_and_register(student_name):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Cannot open camera.")
        return

    data = load_encodings()
    print(f"\nRegistering: {student_name}")
    print("Camera is open. Look directly at camera.")
    print("Press ENTER to capture | type q + ENTER to quit\n")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        cv2.putText(frame, "Look at camera - Press ENTER in terminal",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Register Student", frame)
        cv2.waitKey(30)

        user_input = input("ENTER to capture (q to quit): ").strip().lower()

        if user_input == 'q':
            print("Cancelled.")
            break

        # Convert BGR → RGB for DeepFace
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding, message = get_embedding(rgb)

        if embedding is None:
            print(message)
            continue

        data["names"].append(student_name)
        data["encodings"].append(embedding)
        save_encodings(data)

        print(f"\nRegistered {student_name} successfully!")
        print(f"Embedding size: {len(embedding)}D")  # Should print 512D for ArcFace
        print(f"Total students registered: {len(set(data['names']))}")
        break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter student name: ").strip()
    if name:
        capture_and_register(name)
    else:
        print("Name cannot be empty.")