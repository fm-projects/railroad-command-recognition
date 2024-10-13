#!/bin/bash

# Install requirements
pip install -r requirements.txt

# Define arguments
MODEL_PATH="vosk-model-small-ru-0.22"
DATA_PATH="rzd/ESC_DATASET_v1.2/luga/02_11_2023"
AUDIO_DIR="rzd/ESC_DATASET_v1.2/luga/02_11_2023"
PROCESS_DIR="denoise/luga/02_11_2023"
SUB_FILE="submit.json"


# Start inference
python inference.py --model_path "$MODEL_PATH" --data_path "$DATA_PATH" --audio_dir "$AUDIO_DIR" --process_dir "$PROCESS_DIR" --submission_file_path "$SUB_FILE"