import cv2
import face_recognition

def run_detection():
    video = cv2.VideoCapture(0)  # 0 = default webcam

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Flip BGR → RGB for face_recognition
        rgb = frame[:, :, ::-1]

        # Detect face locations in the frame
        locations = face_recognition.face_locations(rgb)

        # Draw box around each detected face
        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()