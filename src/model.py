from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from src.dicts_const import _label2id, numbers_dict, correct_id2label, _id2label, number_dict
from nltk.stem.snowball import SnowballStemmer
import typing as tp
import os
from os import listdir
from os.path import isfile, join
# from utils import *


SetLogLevel(-2)
import librosa
import numpy as np
import wave
import io

def get_random_audio_and_remove_silence(file_path: str, top_db: int = 12, safe_zone: float = 0.25) -> wave:
    audio_data, sr = librosa.load(file_path, sr=None)
    non_silent_intervals = librosa.effects.split(audio_data, top_db=top_db)

    
    safe_zone = int(sr * safe_zone)
    unique_indexes = set()
    for start, end in non_silent_intervals:
        unique_indexes.update(range(max(0, start - safe_zone), min(len(audio_data), end + safe_zone)))
    processed_audio = audio_data[list(unique_indexes)]

    # Создаем объект BytesIO для хранения аудиоданных в памяти
    byte_io = io.BytesIO()
    
    # Записываем данные в формате WAV в объект BytesIO
    with wave.open(byte_io, 'wb') as wave_file:
        wave_file.setnchannels(1)  # Установите количество каналов (1 для моно)
        wave_file.setsampwidth(2)   # Установите ширину выборки (2 байта для int16)
        wave_file.setframerate(sr)   # Установите частоту дискретизации
        wave_file.writeframes((processed_audio * 32767).astype(np.int16).tobytes())  # Преобразуем в int16 и записываем

    # Перемещаем указатель в начало объекта BytesIO
    byte_io.seek(0)

    # Открываем объект BytesIO как wave
    wave_obj = wave.open(byte_io, 'rb')
    
    return wave_obj

def extract_number(number_list, label: list[str]) -> tp.Tuple[int, str]:
    res_list = []
    res_num = 0
    for num_str in number_list[::-1]:
        if num_str in label:
            res_num += number_dict[num_str]
            res_list.append(num_str)
    return res_num, ' '.join(res_list)

def get_vagon_form(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "вагон"
    elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
        return "вагона"
    else:
        return "вагонов"

def postprocess_vagon(number_list, str_label: str) -> str:
    label = str_label.split()
    voice_num, voice_str = extract_number(number_list, label)
    vagon_form = get_vagon_form(voice_num)

    if label[0] in ['осадить', 'протянуть']:
        label = [label[0]] + ["на"] + [voice_str] + [vagon_form]
    else:
        label = ["на"] + [voice_str] + [vagon_form]

    return ' '.join(label)

def vosk_asr(model, wf: wave, classes: str) -> tp.Tuple[str, str, str]:
    # wf = wave.open(path, 'rb')
    rec = KaldiRecognizer(model, wf.getframerate(), classes)
    res = ""
    partial_res = ""
    final_res = ""

    while True:
        data = wf.readframes(10000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res += list(json.loads(rec.Result()).values())[0] + " "
        else:
            partial_res += str(json.loads(rec.PartialResult()))

    final_res = list(json.loads(rec.FinalResult()).values())[0]
    return res, partial_res, final_res

def find_nearest_label(stemmer, sent: str, stemmed_labels: list[set[str]]) -> tp.Tuple[int, str]:
    processed_sent = {stemmer.stem(word) for word in sent.split()}
    intersects = [len(processed_sent.intersection(label)) for label in stemmed_labels]
    max_common_id = np.argmax(intersects)
    return max_common_id, correct_id2label[max_common_id]

class YourModelClass:
    def __init__(self, model_path="vosk-model-small-ru-0.22"):
        self.model = Model(lang="ru-RU", model_path=model_path)
        # self.audio_paths = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        # self.audio_dir = audio_dir
        # self.process_dir = process_dir
        self.classes = f'{list(_label2id.keys()) + list(numbers_dict.values()) + ["[unk]"]}'.replace("'", '"')
        self.number_list = list(number_dict.keys())
        self.vagon_dict = {i: get_vagon_form(i) for i in range(1, 101)}
        self._id2label = dict((v, k) for k, v in _id2label.items())
        self.stemmer = SnowballStemmer("russian")
        self.filtered_labels = [list(filter(lambda x: x != "(количество)", self._id2label[i].split())) for i in range(23)]
        self.stemmed_labels = [{self.stemmer.stem(word) for word in label} for label in self.filtered_labels]
    
    def preprocess_many_audio(self):
        for row in self.audio_paths:
            in_path = f"{self.audio_dir}/{row}"
            out_path = f"{self.process_dir}/{row}"
            get_random_audio_and_remove_silence(in_path, out_path)


    def predict(self, audio_path: str, is_reduce=False) -> dict:
        if is_reduce:
            audio = get_random_audio_and_remove_silence(audio_path)
        else:
            audio = wave.open(audio_path, 'rb')
        res, partial_res, final_res = vosk_asr(self.model, audio, self.classes)
        if res == "":
            res = final_res

        nearest_label_id, nearest_label = find_nearest_label(self.stemmer, res, self.stemmed_labels)
        if nearest_label_id in [4, 10]:
            prediction_text = postprocess_vagon(self.number_list, res)
        else:
            prediction_text = nearest_label

        extracted_number = extract_number(self.number_list, prediction_text.split())[0]

        return Prediction(prediction_text, int(nearest_label_id), int(extracted_number))

    # def big_predict(self, submission_file_path: str):
    #     predictions = []
    #     for row in [f for f in listdir(self.process_dir) if isfile(join(self.process_dir, f))]:
    #         path = f"{self.process_dir}/{row}"
    #         prediction = self.predict(path)
    #         predictions.append(prediction)

    #     submission_df = pd.DataFrame(predictions)
    #     submission_df.to_json(submission_file_path, index=False, orient="records")


class Prediction:
    def __init__(self, text, label, attribute) -> None:
        self.res_dict = {'text': text, 'label': label, 'attribute': attribute}

    def get(self, type, default):
        if self.res_dict[type] == None: return default
        return self.res_dict[type]


# def parse_args():
#     parser = argparse.ArgumentParser(description="Audio Inference Script")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the Vosk model")
#     parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
#     parser.add_argument("--audio_dir", type=str, required=True, help="Path to the audio directory")
#     parser.add_argument("--process_dir", type=str, required=True, help="Path to the processed audio directory")
#     parser.add_argument("--submission_file_path", type=str, required=True, help="Path to save the submission file")
#     return parser.parse_args()


if __name__ == '__main__':
    # import time
    # start_time = time.time()
    for i in ['02_11_2023', '03_07_2023', '11_10_2023', '15_11_2023', '21_11_2023']:
        inference = YourModelClass()
        results = []
        for audio_path in os.listdir("../rzd/ESC_DATASET_v1.2/luga/" + i):
            result = inference.predict(os.path.join("../rzd/ESC_DATASET_v1.2/luga/" + i, audio_path))
            result = {
                "audio": os.path.basename(audio_path),          # Audio file base name
                "text": result.get("text", -1),             # Predicted text
                "label": result.get("label", -1),           # Text class
                "attribute": result.get("attribute", -1),   # Predicted attribute (if any, or -1)
            }
            results.append(result)
        with open(
            os.path.join("", "submission" + i + ".json"), "w", encoding="utf-8"
        ) as outfile:
            json.dump(results, outfile)
    # diff = time.time() - start_time
    # print("time:", diff)
    # print("one time:", diff / 610)

    # import resource

    # peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    # print(f'Peak RAM usage: {peak_ram} MB')