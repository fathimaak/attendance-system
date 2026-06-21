import cv2
import face_recognition

def run_detection():
    video = cv2.VideoCapture(0)  # 0 = default webcam

    while True:
        ret, frame = video.read()
        if not ret:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = small[:, :, ::-1]
        locations = face_recognition.face_locations(rgb_small)
        locations = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in locations]

        # Draw box around each detected face
        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()