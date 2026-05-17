import cv2
import mediapipe as mp

from ultralytics import YOLO

from tik import track_students
from attention import get_attention
from phone_detector import detect_phone

# --------------------------------
# LOAD YOLO
# --------------------------------

model = YOLO("yolov8n.pt")

# --------------------------------
# MEDIAPIPE FACE MESH
# --------------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10
)

# --------------------------------
# CAMERA
# --------------------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # --------------------------------
    # YOLO DETECTION
    # --------------------------------

    results = model(frame)

    detections = []

    phone_detected = False

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1

            # --------------------------------
            # PERSON
            # --------------------------------

            if cls == 0 and confidence > 0.5:

                detections.append(
                    (
                        [x1, y1, width, height],
                        confidence,
                        "student"
                    )
                )

            # --------------------------------
            # PHONE
            # --------------------------------

            if detect_phone(cls):

                phone_detected = True

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "PHONE",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    # --------------------------------
    # TRACK STUDENTS
    # --------------------------------

    tracks = track_students(detections, frame)

    # --------------------------------
    # FACE PROCESSING
    # --------------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)

    attention_state = "Unknown"

    if face_results.multi_face_landmarks:

        for face_landmarks in face_results.multi_face_landmarks:

            # --------------------------------
            # NOSE
            # --------------------------------

            nose = face_landmarks.landmark[1]

            nose_x = int(nose.x * frame_width)
            nose_y = int(nose.y * frame_height)

            # --------------------------------
            # LEFT EYE
            # --------------------------------

            left_eye = face_landmarks.landmark[33]

            left_eye_x = int(left_eye.x * frame_width)

            # --------------------------------
            # RIGHT EYE
            # --------------------------------

            right_eye = face_landmarks.landmark[263]

            right_eye_x = int(right_eye.x * frame_width)

            # --------------------------------
            # ATTENTION LOGIC
            # --------------------------------

            attention_state = get_attention(
                nose_x,
                nose_y,
                left_eye_x,
                right_eye_x,
                frame_width,
                frame_height
            )

            # --------------------------------
            # DRAW FULL FACE LANDMARKS
            # --------------------------------

            for landmark in face_landmarks.landmark:

                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

            # --------------------------------
            # DRAW NOSE POINT
            # --------------------------------

            cv2.circle(
                frame,
                (nose_x, nose_y),
                5,
                (255, 0, 0),
                -1
            )

    # --------------------------------
    # DRAW TRACKING
    # --------------------------------

    for track in tracks:

        track_id = track["id"]

        x1, y1, x2, y2 = track["box"]

        # --------------------------------
        # COLORS
        # --------------------------------

        if attention_state == "Attentive":

            color = (0, 255, 0)

        elif attention_state == "Head Down":

            color = (0, 255, 255)

        else:

            color = (0, 0, 255)

        # --------------------------------
        # BOX
        # --------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # --------------------------------
        # LABEL
        # --------------------------------

        text = f"ID {track_id} - {attention_state}"

        if phone_detected:

            text += " | PHONE"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # --------------------------------
    # TITLE
    # --------------------------------

    cv2.putText(
        frame,
        "CLASSROOM AI MONITOR",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # --------------------------------
    # SHOW FRAME
    # --------------------------------

    cv2.imshow("Classroom AI", frame)

    # --------------------------------
    # EXIT
    # --------------------------------

    if cv2.waitKey(1) == ord('q'):
        break

# --------------------------------
# RELEASE
# --------------------------------

cap.release()

cv2.destroyAllWindows()