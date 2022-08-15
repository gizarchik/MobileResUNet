# MobileResUNet

Решение задачи сегментации с помощью Unet c различными блоками на выбор (базовый блок из Unet, bottleneck блок с ResNet,
MobileNet блок с декомпозированной сверткой).

Используемый набор данных COCOStuff10k: https://github.com/nightrome/cocostuff10k

Ссылка на ноутбук с Resnet реализацией: https://colab.research.google.com/drive/1q8qRRO-CWc7u5CjQn0MlkJdKw_sl9Kx6?usp=sharing
TODO: Ссылка на ноутбук с Mobilenet реализацией:\
Ссылка на ноутбук с базовой реализацией: https://colab.research.google.com/drive/1uJ2zxUpdfP1wJ4lsu0M2UR_oLmzueUCa?usp=sharing

Чтобы начать обучение требуется запустить файл `main.py`. В качестве аргумента можно передать:
* `--batch` размер батча. Стандартное значение - 16.
* `--epoch` число эпох. Стандартное значение - 100.
* `--block` вид блока. Стандартное значение - 'resnet'. Могут быть три вида блока: resnet, mobilenet, standart
* `--lr` скорость обучение. Стандартное значение - 1e-3
* `--state_path` пуnь до папки, куда будут сохраняться состояния. Стандартное значение - "model states/"
* `path` путь до папки c датасетом. Обязательный аргумент.
