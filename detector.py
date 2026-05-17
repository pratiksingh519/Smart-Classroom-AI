from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Run AI detection
    results = model(frame)

    # Loop through detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Get class ID
            cls = int(box.cls[0])

            # PERSON
            if cls == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0,255,0),
                              2)

                cv2.putText(frame,
                            "Student",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0),
                            2)

            # LAPTOP
            elif cls == 63:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0,255,255),
                              2)

                cv2.putText(frame,
                            "Laptop",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,255),
                            2)

            # PHONE
            elif cls == 67:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0,0,255),
                              2)

                cv2.putText(frame,
                            "Phone",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2)

    cv2.imshow("AI Classroom", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()