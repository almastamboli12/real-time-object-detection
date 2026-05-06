import cv2

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'mobilenet_iter_73000.caffemodel'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x1, y1, x2, y2) = box.astype("int")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
