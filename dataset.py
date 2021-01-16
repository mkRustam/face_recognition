import pickle

import cv2, detection
import os


def users_info_load():
    if not os.path.exists("info/"):
        os.mkdir("info")

    if os.path.exists("info/users.pickle"):
        with open('info/users.pickle', 'rb') as f:
            return pickle.load(f)
    else:
        return dict()


def users_info_save(info):
    if not os.path.exists("info/"):
        os.mkdir("info")
    with open('info/users.pickle', 'wb') as f:
        pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)


# Save each 5th frame as sample
def add_user(cap, cascade):
    user_id = input("Идентификатор пользователя: ")
    user_name = input("Имя пользователя: ")
    input("Медленно покрутите головой в течение 4 секунд")
    count = 10
    skip = 10  # Skip some frames to avoid duplicates
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detection.detect(gray, cascade)
        detection.draw_rectangles(faces, img)
        if len(faces) > 0 and skip <= 0:
            skip = 5
            for (x, y, w, h) in faces:
                if not os.path.exists("dataset/"):
                    os.mkdir("dataset/")
                cv2.imwrite("dataset/user_" + str(user_id) + '_' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            count -= 1
        else:
            skip -= 1
        cv2.imshow('image', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break
        elif count <= 0:
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()

    users_info = users_info_load()
    users_info[user_id] = user_name
    users_info_save(users_info)
