import numpy as np 
import typing as tp
import librosa
import os
from nltk.stem.snowball import SnowballStemmer


_id2label = dict((v, k) for k, v in _id2label.items())

stemmer = SnowballStemmer("russian") 
filtered_labels = [list(filter(lambda x: x != "(количество)", 
                        _id2label[i].split()))
                    for i in range(23)]
stemmed_labels = [{stemmer.stem(word) for word in label}
                  for label in filtered_labels]


def find_nearest_label(sent: str, stemmed_labels: list[str]) -> tp.Tuple[int, str]:
    """
    Находит ближайшую метку для данного предложения.

    Аргументы:
    sent (str): Входное предложение.
    stemmed_labels (list[str]): Список стеммированных меток.

    Возвращает:
    Tuple[int, str]: Кортеж, содержащий индекс метки с максимальным пересечением и саму метку.
    """
    processed_sent = {stemmer.stem(word) for word in sent.split()}
    intersects = [len(processed_sent.intersection(label)) for label in stemmed_labels]
    max_common_id = np.argmax(intersects)
    return max_common_id, correct_id2label[max_common_id]


def extract_number(label: list[str]) -> tp.Tuple[int, str]:
    """
    Извлекает число из строки с предсказанным текстом.

    Аргументы:
    label (list[str]): Строка с предсказаниями, превращенная в список.

    Возвращает:
    tp.Tuple[int, str]: Кортеж, содержащий сумму чисел и строку с объединенными числовыми метками.
    """
    res_list = []
    res_num = 0
    for num_str in list(number_dict.keys())[::-1]:
        if num_str in label:
            res_num += number_dict[num_str]
            res_list.append(num_str)
    return (res_num, ' '.join(res_list))


def get_vagon_form(n: int) -> str:
    """
    Возвращает правильную форму слова "вагон" в зависимости от числа.

    Аргументы:
    n (int): Число вагонов.

    Возвращает:
    str: Правильная форма слова "вагон" ("вагон", "вагона" или "вагонов").
    """
    if n % 10 == 1 and n % 100 != 11:
        return "вагон"
    elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
        return "вагона"
    else:
        return "вагонов"


def postprocess_vagon(str_label: str) -> str:
    """
    Обрабатывает строку метки вагона, извлекая номер и форму вагона, и заменяет соответствующие части строки.

    Аргументы:
    str_label (str): Исходная строка метки вагона.

    Возвращает:
    str: Обработанная строка метки вагона с заменёнными номером и формой.
    """
    label = str_label.split()
    voice_num, voice_str = extract_number(label)
    vagon_form = get_vagon_form(voice_num)
    label[-2] = voice_str
    label[-1] = vagon_form
    return ' '.join(label)
