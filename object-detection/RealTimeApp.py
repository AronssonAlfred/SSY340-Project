import cv2
from ultralytics import YOLO
import time

# Load the trained YOLOv8 model
model = YOLO('object-detection/runs/detect/train7/weights/best.pt')

def main():
    # Access the webcam (0 means default camera)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to quit the application.")

    # Loop to read frames from the camera and run inference
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Run inference on the frame
        results = model.predict(frame, stream=True)  # stream=True to enable real-time results

        # Draw detections on the frame
        for result in results:
            for box in result.boxes:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the frame with detections
        cv2.imshow('Face Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()