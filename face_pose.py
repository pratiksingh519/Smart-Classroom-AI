import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            h, w, _ = frame.shape

            for lm in face.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame,
                           (x,y),
                           1,
                           (0,255,0),
                           -1)

    cv2.imshow("Face Tracking", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()