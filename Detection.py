import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            coords = []
            for lm in results.pose_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                coords.append((x, y))

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            min_x, max_x = max(min(xs) - 20, 0), min(max(xs) + 20, w)
            min_y, max_y = max(min(ys) - 20, 0), min(max(ys) + 20, h)

            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(coords) and end_idx < len(coords):
                    cv2.line(frame, coords[start_idx], coords[end_idx], (0, 200, 255), 2)

            for (x, y) in coords:
                cv2.circle(frame, (x, y), 3, (0, 165, 255), -1)

        cv2.imshow('person-detect_v2', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
