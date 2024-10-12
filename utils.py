import numpy as np 
import typing as tp
import librosa
import os
from nltk.stem.snowball import SnowballStemmer


number_dict = {
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяноста": 90,
    "сто": 100
}

correct_id2label = {
    0: "отказ",
    1: "отмена",
    2: "подтверждение",
    3: "начать осаживание",
    4: "осадить на (количество) вагон",
    5: "продолжаем осаживание",
    6: "зарядка тормозной магистрали",
    7: "вышел из межвагонного пространства",
    8: "продолжаем роспуск",
    9: "растянуть автосцепки",
    10: "протянуть на (количество) вагон",
    11: "отцепка",
    12: "назад на башмак",
    13: "захожу в межвагонное,пространство",
    14: "остановка",
    15: "вперед на башмак",
    16: "сжать автосцепки",
    17: "назад с башмака",
    18: "тише",
    19: "вперед с башмака",
    20: "прекратить зарядку тормозной магистрали",
    21: "тормозить",
    22: "отпустить",
}

_id2label = {
    "отказ": 0,
    "отмена": 1,
    "подтверждение": 2,
    "начать осаживание": 3,
    "осадить на вагон": 4,
    "продолжаем осаживание": 5,
    "зарядка тормозной магистрали": 6,
    "вышел из меж вагонного пространства": 7,
    "продолжаем роспуск": 8,
    "растянуть авто сцепки": 9,
    "протянуть на вагон": 10,
    "от сцепка": 11,
    "назад на башмак": 12,
    "захожу в меж вагонного пространство": 13,
    "остановка": 14,
    "вперёд на башмак": 15,
    "сжать авто сцепки": 16,
    "назад с башмака": 17,
    "тише": 18,
    "вперёд с башмака": 19,
    "прекратить зарядку тормозной магистрали": 20,
    "тормозить": 21,
    "отпустить": 22,
}

_id2label = dict((v, k) for k, v in _id2label.items())

stemmer = SnowballStemmer("russian") 
filtered_labels = [list(filter(lambda x: x != "(количество)", 
                        _id2label[i].split()))
                    for i in range(23)]
stemmed_labels = [{stemmer.stem(word) for word in label}
                  for label in filtered_labels]


def find_nearest_label(sent: str, stemmed_labels: list[str]) -> tp.Tuple[int, str]:
    processed_sent = {stemmer.stem(word) for word in sent.split()}
    intersects = [len(processed_sent.intersection(label)) for label in stemmed_labels]
    max_common_id = np.argmax(intersects)
    return max_common_id, correct_id2label[max_common_id]


def extract_number(label: list[str]) -> tp.Tuple[int, str]:
    res_list = []
    res_num = 0
    for num_str in list(number_dict.keys())[::-1]:
        if num_str in label:
            res_num += number_dict[num_str]
            res_list.append(num_str)
    return (res_num, ' '.join(res_list))


def get_vagon_form(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "вагон"
    elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
        return "вагона"
    else:
        return "вагонов"


def postprocess_vagon(str_label: str) -> str:
    label = str_label.split()
    voice_num, voice_str = extract_number(label)
    vagon_form = get_vagon_form(voice_num)
    label[-2] = voice_str
    label[-1] = vagon_form
    return ' '.join(label)
