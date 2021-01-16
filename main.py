import cv2
import dataset, recognition

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FPS, 24)  # Частота кадров
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)  # Ширина кадров в видеопотоке.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Высота кадров в видеопотоке.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


def start():
    action = input('[1]: Добавить пользователя\n[2]: Запустить\n')
    if action == '1':
        dataset.add_user(cap, faceCascade)
    if action == '2':
        recognition.start(cap, faceCascade)

if __name__ == '__main__':
    start()