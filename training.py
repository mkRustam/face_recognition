import pickle

import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath)
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split("_")[1])

        faceSamples.append(img_numpy)
        ids.append(id)

    return faceSamples, ids


def get_model():
    if os.path.exists("trainer/trainer.yml"):
        print("Loading model...")
        recognizer.read('trainer/trainer.yml')
    else:
        print("Training model...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        if not os.path.exists("trainer"):
            os.mkdir("trainer")
        recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    return recognizer
