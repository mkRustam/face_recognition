import cv2, detection, dataset, training, os


def start(cap, cascade):
    recognizer = training.get_model()

    # recognition
    font = cv2.FONT_HERSHEY_SIMPLEX
    users_info = dataset.users_info_load()
    print(users_info)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detection.detect(gray, cascade)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # If confidence is less them 100 ==> "0" : perfect match
            if confidence < 100:
                name = users_info.get(str(id))
                print(id)
                print(name)
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                name = "Неизвестно"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(name), (x + 5, y - 5),
                        font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5),
                        font, 1, (255, 255, 0), 1)
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()