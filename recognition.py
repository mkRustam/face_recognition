import cv2, detection, dataset, training, os
from PIL import ImageFont, ImageDraw, Image
import numpy as np


def start(cap, cascade):
    recognizer = training.get_model()

    # recognition
    users_info = dataset.users_info_load()
    print(users_info)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detection.detect(gray, cascade)

        pil_im = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_im)
        # Choose a font
        font = ImageFont.truetype("Roboto-Regular.ttf", 50)
        fontConf = ImageFont.truetype("Roboto-Regular.ttf", 25)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # If confidence is less them 100 ==> "0" : perfect match
            if confidence > 100:
                confidence = 0
            if confidence > 60:
                name = users_info.get(str(id))
                confidence = "  {0}%".format(round(confidence))
            else:
                name = "Неизвестно"
                confidence = "  {0}%".format(round(confidence))

            # Draw the text
            draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=2)
            draw.text((x + 5, y - 5), str(name), font=font)
            draw.text((x + 5, y + h - 5), str(confidence), font=font, fill=(255, 0, 0))

        cv2.imshow('camera', np.array(pil_im))
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()