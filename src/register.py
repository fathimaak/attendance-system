import cv2
import numpy as np
from deepface import DeepFace
from database import init_db, add_student

MODEL_NAME = "ArcFace"

def get_embedding(frame_rgb):
    try:
        result = DeepFace.represent(
            img_path=frame_rgb,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend="opencv"
        )
        return np.array(result[0]["embedding"]), "ok"
    except Exception:
        return None, "No face detected clearly. Try again."

def capture_and_register(student_name):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Cannot open camera.")
        return

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding, message = get_embedding(rgb)

        if embedding is None:
            print(message)
            continue

        add_student(student_name, embedding)
        print(f"\nRegistered {student_name} successfully in database!")
        break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_db()
    name = input("Enter student name: ").strip()
    if name:
        capture_and_register(name)
    else:
        print("Name cannot be empty.")