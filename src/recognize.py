import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from database import init_db, get_all_students, mark_attendance

MODEL_NAME = "ArcFace"
THRESHOLD = 0.40

def find_match(embedding, names, embeddings):
    best_match = "Unknown"
    best_score = 1.0

    for name, stored_enc in zip(names, embeddings):
        dist = cosine(embedding, stored_enc)
        if dist < best_score:
            best_score = dist
            if dist < THRESHOLD:
                best_match = name

    confidence = round((1 - best_score) * 100, 1)
    return best_match, confidence

def run_recognition():
    init_db()
    print("Loading registered faces from database...")
    names, embeddings = get_all_students()
    print(f"Loaded {len(set(names))} registered students.\n")

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Cannot open camera.")
        return

    print("Camera open. Recognized faces auto-mark attendance.")
    print("Press ENTER to check status, q + ENTER to quit.\n")
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.represent(
                img_path=rgb,
                model_name=MODEL_NAME,
                enforce_detection=True,
                detector_backend="opencv"
            )
            embedding = np.array(result[0]["embedding"])
            name, confidence = find_match(embedding, names, embeddings)

            face_region = result[0]["facial_area"]
            x, y, w, h = (face_region["x"], face_region["y"],
                          face_region["w"], face_region["h"])

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({confidence}%)"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if name != "Unknown":
                newly_marked = mark_attendance(name, confidence)
                if newly_marked:
                    print(f"✅ Attendance marked: {name} ({confidence}%)")

        except Exception:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Attendance - Recognition", frame)
        cv2.waitKey(30)

        if frame_count % 20 == 0:
            user = input("ENTER to continue, q to quit: ").strip().lower()
            if user == 'q':
                break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()