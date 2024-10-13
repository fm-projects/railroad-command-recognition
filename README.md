# Команда fmprojects — Интеллектуальный пульт составителя

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![nltk](https://img.shields.io/badge/nltk-gray?style=for-the-badge)
![librosa](https://img.shields.io/badge/librosa-purple?style=for-the-badge)

## О проекте
Команда fmprojects представляет модель для классификации голосовых команд посредством автоматического распознавания речи. 
Помимо этого, реализованы и протестированы алгоритмы подавления внешних шумов, постпроцессинг результатов распознавания и фильтрация случайной речи.

Распознавание речи реализовано с помощью модели Vosk Small, которая способна при ограниченных ресурсах определять слова с высокой точностью. Архитектура нейросети основана на DNN-HMM. Подавление шума происходит посредством анализа и обработки спектрограмм аудиофайлов.

Технические особенности и уникальность: 
- обрезание незначащей части аудио
- поиск наиболее подходящего класса с помощью модифицированного расстояния Хэмминга
- определение числительных и спряжение соответствующих слов  
- в результате оптимизаций обработка одного аудио занимает менее 100 мс

## Установка
1) Скачайте репозиторий с GitHub
```git clone https://github.com/fm-projects/railroad-command-recognition.git```
2) Запустите скрипт `start.sh`, либо же `inference.py` с заранее установленными баблиотеками из `requirements.txt`

## Структура проекта
- `src` - папка с исходным кодом проекта
  - `start.sh` - bash-скрипт для запуска проекта (включает в себя установку нужных модулей)
  - `inference.py` - файл с инференсом модели
  - `model.py` - файл с моделью
  - `utils.py` - файл, содержащий реализацию фильтрации случайной речи, обрезание аудио, проверку порядка слов в команде, автоматическое спряжение и остальные фичи
- `EDA.ipynb` - ноутбук с первичным исследованием данных
- `ASR_inference.ipynb` - ноутбук с запуском моделей распознавания речи, применением постпроцессинга и остальными фичами
- `vosk-model-small-ru-0.22.zip` - файл с моделью распознавания речи


