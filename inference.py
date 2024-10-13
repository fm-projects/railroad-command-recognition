from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from dicts_const import _label2id, numbers_dict, correct_id2label, _id2label, number_dict
from nltk.stem.snowball import SnowballStemmer
import typing as tp
import os
from os import listdir
from os.path import isfile, join
import argparse


SetLogLevel(-2)
def get_random_audio_and_remove_silence(file_path: str, out_path: str, top_db: int = 12, safe_zone: float = 0.25) -> None:
    """
    Load an audio file, remove silence, and save the processed audio.

    Args:
        file_path (str): Path to the input audio file.
        out_path (str): Path to save the processed audio file.
        top_db (int, optional): Threshold (in decibels) below reference to consider as silence. Defaults to 12.
        safe_zone (float, optional): Duration (in seconds) to include before and after non-silent intervals. Defaults to 0.25.
    """
    audio_data, sr = librosa.load(file_path, sr=None)
    non_silent_intervals = librosa.effects.split(audio_data, top_db=top_db)
    
    safe_zone = int(sr * safe_zone)
    unique_indexes = set()
    for start, end in non_silent_intervals:
        unique_indexes.update(range(max(0, start - safe_zone), min(len(audio_data), end + safe_zone)))
    processed_audio = audio_data[list(unique_indexes)]

    sf.write(out_path, processed_audio, sr)

def extract_number(number_list, label: list[str]) -> tp.Tuple[int, str]:
    """
    Extract a number from a list of strings.

    Args:
        label (list[str]): List of strings containing the number.

    Returns:
        tp.Tuple[int, str]: Extracted number and its string representation.
    """
    res_list = []
    res_num = 0
    for num_str in number_list[::-1]:
        if num_str in label:
            res_num += number_dict[num_str]
            res_list.append(num_str)
    return res_num, ' '.join(res_list)

def get_vagon_form(n: int) -> str:
    """
    Get the correct form of the word "вагон" based on the number.

    Args:
        n (int): Number of wagons.

    Returns:
        str: Correct form of the word "вагон".
    """
    if n % 10 == 1 and n % 100 != 11:
        return "вагон"
    elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
        return "вагона"
    else:
        return "вагонов"

def postprocess_vagon(number_list, str_label: str) -> str:
    """
    Post-process the label to include the correct form of the word "вагон".

    Args:
        str_label (str): Input label string.

    Returns:
        str: Post-processed label string.
    """
    label = str_label.split()
    voice_num, voice_str = extract_number(number_list, label)
    vagon_form = get_vagon_form(voice_num)

    if label[0] in ['осадить', 'протянуть']:
        label = [label[0]] + ["на"] + [voice_str] + [vagon_form]
    else:
        label = ["на"] + [voice_str] + [vagon_form]

    return ' '.join(label)

def vosk_asr(model, path: str, classes: str) -> tp.Tuple[str, str, str]:
    """
    Perform speech recognition using Vosk.

    Args:
        path (str): Path to the audio file.
        classes (str): Classes for the recognizer.

    Returns:
        tp.Tuple[str, str, str]: Recognized text, partial result, and final result.
    """
    wf = wave.open(path, 'rb')
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
    """
    Find the nearest label to the given sentence.

    Args:
        sent (str): Input sentence.
        stemmed_labels (list[set[str]]): List of stemmed labels.

    Returns:
        tp.Tuple[int, str]: Index of the nearest label and the corresponding label.
    """
    processed_sent = {stemmer.stem(word) for word in sent.split()}
    intersects = [len(processed_sent.intersection(label)) for label in stemmed_labels]
    max_common_id = np.argmax(intersects)
    return max_common_id, correct_id2label[max_common_id]

class AudioInference:
    def __init__(self, model_path: str, data_path: str, audio_dir: str, process_dir: str):
        self.model = Model(lang="ru-RU", model_path=model_path)
        self.audio_paths = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        self.audio_dir = audio_dir
        self.process_dir = process_dir
        self.classes = f'{list(_label2id.keys()) + list(numbers_dict.values()) + ["[unk]"]}'.replace("'", '"')
        self.number_list = list(number_dict.keys())
        self.vagon_dict = {i: get_vagon_form(i) for i in range(1, 101)}
        self._id2label = dict((v, k) for k, v in _id2label.items())
        self.stemmer = SnowballStemmer("russian")
        self.filtered_labels = [list(filter(lambda x: x != "(количество)", self._id2label[i].split())) for i in range(23)]
        self.stemmed_labels = [{self.stemmer.stem(word) for word in label} for label in self.filtered_labels]
    
    def preprocess_audio(self):
        for row in self.audio_paths:
            in_path = f"{self.audio_dir}/{row}"
            out_path = f"{self.process_dir}/{row}"
            get_random_audio_and_remove_silence(in_path, out_path)

    def predict(self, audio_path: str) -> dict:
        res, partial_res, final_res = vosk_asr(self.model, audio_path, self.classes)
        if res == "":
            res = final_res

        nearest_label_id, nearest_label = find_nearest_label(self.stemmer, res, self.stemmed_labels)
        if nearest_label_id in [4, 10]:
            prediction_text = postprocess_vagon(self.number_list, res)
        else:
            prediction_text = nearest_label

        extracted_number = extract_number(self.number_list, prediction_text.split())[0]

        return {
            "audio": os.path.basename(audio_path),
            "text": prediction_text,
            "label": nearest_label_id,
            "attribute": extracted_number
        }

    def run_inference(self, submission_file_path: str):
        predictions = []
        for row in [f for f in listdir(self.process_dir) if isfile(join(self.process_dir, f))]:
            path = f"{self.process_dir}/{row}"
            prediction = self.predict(path)
            predictions.append(prediction)

        submission_df = pd.DataFrame(predictions)
        submission_df.to_json(submission_file_path, index=False, orient="records")


def parse_args():
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Vosk model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to the audio directory")
    parser.add_argument("--process_dir", type=str, required=True, help="Path to the processed audio directory")
    parser.add_argument("--submission_file_path", type=str, required=True, help="Path to save the submission file")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    
    inference = AudioInference(
        model_path=args.model_path,
        data_path=args.data_path,
        audio_dir=args.audio_dir,
        process_dir=args.process_dir
    )
    inference.preprocess_audio()
    inference.run_inference(args.submission_file_path)