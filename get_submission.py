from __future__ import annotations

import os
import argparse
import json

from src.model import YourModelClass


class Predictor:
    """Class for your model's predictions.

    You are free to add your own properties and methods
    or modify existing ones, but the output submission
    structure must be identical to the one presented.

    Examples:
        >>> python -m get_submission --src input_dir --dst output_dir
    """

    def __init__(self):
        self.model = YourModelClass()

    def __call__(self, audio_path: str):
        prediction = self.model.predict(audio_path)
        result = {
            "audio": os.path.basename(audio_path),          # Audio file base name
            "text": prediction.get("text", -1),             # Predicted text
            "label": prediction.get("label", -1),           # Text class
            "attribute": prediction.get("attribute", -1),   # Predicted attribute (if any, or -1)
        }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get submission.")
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the source audio files.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to the output submission.",
    )
    args = parser.parse_args()
    predictor = Predictor()

    results = []
    for audio_path in os.listdir(args.src):
        result = predictor(os.path.join(args.src, audio_path))
        results.append(result)

    with open(
        os.path.join(args.dst, "submission.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile)
