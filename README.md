# scidev22-cv

Ссылка на ноутбук с Resnet реализацией: https://colab.research.google.com/drive/1q8qRRO-CWc7u5CjQn0MlkJdKw_sl9Kx6?usp=sharing
Ссылка на ноутбук с базовой реализацией: https://colab.research.google.com/drive/1uJ2zxUpdfP1wJ4lsu0M2UR_oLmzueUCa?usp=sharing

Чтобы запустить проект требуется запустить файл `main.py`. В качестве аргумента можно передать:
* `--batch` размер батча. Стандартное значение - 16.
* `--epoch` число эпох. Стандартное значение - 100.
* `--block` вид блока. Стандартное значение - 'resnet'. Могут быть три вида блока: resnet, mobilenet, standart
* `--lr` скорость обучение. Стандартное значение - 1e-3
* `--state_path` пуnь до папки, куда будут сохраняться состояния. Стандартное значение - "model states/"
* `path` путь до папки c датасетом. Обязательный аргумент.
