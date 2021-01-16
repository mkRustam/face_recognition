### Installing OpenCV (Dont use 'conda install')
pip install opencv-python <br/>
pip install opencv-contrib-python

### Download haarcascades from here
https://github.com/opencv/opencv/tree/master/data/haarcascades

### Steps
Запустить main.py.<br/>
[1] Добавить пользователя<br/>
    Заполнение датасета фотографиями путем считывания каждого 6го кадра с камеры. В среднем по 10 изображений на 1 пользователя<br/>
[2] Запустить<br/>
    Тренировка модели на полученной выборке фотографий и запуск системы распознавания через камеру<br/>

### Folders
dataset - здесь хранятся полученные из пункта 1 изображения в формате user_<userId>_<count>
info/users.pickle - информация о датасете в виде словаря в формате: '<userId>':'<userName>'
trainer/trainer.yml - натренированная модель
cascades - набор каскадов Хаара