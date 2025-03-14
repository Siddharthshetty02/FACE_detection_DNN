import cv2
import numpy as np

# Load the pre-trained DNN model
prototxt_path = "DNN_face_detection/deploy.prototxt"
model_path = "DNN_face_detection/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    
    # Prepare the frame for the DNN model
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw rectangle around face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Display confidence score
            text = f"{confidence*100:.2f}%"
            cv2.putText(frame, text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("DNN Face Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
