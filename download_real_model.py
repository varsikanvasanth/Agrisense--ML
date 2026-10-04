"""Download the public PlantVillage EfficientNetV2S TFLite model used by AgriSense.

Source: animeshakr/plant-disease-efficientnetv2s on Hugging Face.
The model card documents 38 PlantVillage classes and a reported 99.57% test accuracy.
Those metrics belong to the model author; AgriSense does not claim them as its own evaluation.
"""
from pathlib import Path
import json
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)
REPO = 'animeshakr/plant-disease-efficientnetv2s'
MODEL_FILE = 'model_float16_quant.tflite'
LABEL_FILE = 'class_indices.json'

print('Downloading real PlantVillage image model (~42 MB)...')
model_path = hf_hub_download(repo_id=REPO, filename=MODEL_FILE, local_dir=MODEL_DIR)
label_path = hf_hub_download(repo_id=REPO, filename=LABEL_FILE, local_dir=MODEL_DIR)
# Normalize labels into the name expected by the Flask app.
labels = json.loads(Path(label_path).read_text(encoding='utf-8'))
(Path(MODEL_DIR) / 'class_indices.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
print('Model:', model_path)
print('Labels:', label_path)
print('Real image model is ready.')
