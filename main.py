import os
import shutil

import cv2
import dataset, recognition

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FPS, 24)  # Частота кадров
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)  # Ширина кадров в видеопотоке.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Высота кадров в видеопотоке.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


def delete_data():
    if os.path.exists("info"): shutil.rmtree("info")
    if os.path.exists("dataset"): shutil.rmtree("dataset")
    if os.path.exists("trainer"): shutil.rmtree("trainer")
    print("Данные удалены")

def start():
    print('[1]: Добавить пользователя\n[2]: Запустить\n[3]: Удалить данные\n')
    action = input("Выберите действие: ")
    if action == '1':
        dataset.add_user(cap, faceCascade)
    if action == '2':
        recognition.start(cap, faceCascade)
    if action == '3':
        delete_data()

    start()


if __name__ == '__main__':
    start()
