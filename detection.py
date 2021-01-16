import cv2


def detect(frame, cascade):
    return cascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )


def draw_rectangles(faces, frame):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
