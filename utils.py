import numpy as np 
import typing as tp
import librosa
import os
from nltk.stem.snowball import SnowballStemmer
from dicts_const import _id2label


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


def check_word_order(pred_text: str, label: str) -> bool:
    """
    Проверка на правильность порядка слов в предложении

    Аргументы:
    pred_text (str): предсказанный текст.
    label (str): текст сопоставленной метки.

    Возвращает:
    bool: True, если порядок слов в предложении верный, иначе False.
    """
    label_words = label.split()
    pred_text_words = pred_text.split()
    ptr = 0
    for i in range(len(label_words)):
        if label_words[i] in pred_text_words:
            if ptr > pred_text_words.index(label_words[i]):
                return False
            ptr = pred_text_words.index(label_words[i])
    return True


def filter_accidental_speech(sent: str) -> bool:
    """
    Функция для фильтрации случайной речи

    Аргументы:
    sent (str): распознанное предложение

    Возвращает:
    bool: True, если предложение является случайной речью, иначе False
    """
    if ('[unk]' in sent) or (sent.strip() == ''):
        return True
    return False
