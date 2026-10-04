# AgriSense — Final Real-Model College Project

**Machine Learning Based Crop Growth, Pest Risk Prediction and Smart Agricultural Advisory System**

Chennai Institute of Technology • Department of Information Technology • PBL – Machine Learning • Academic Year 2026–27

## What is actually real here?

### 1. Leaf image branch — real pretrained PlantVillage model
AgriSense uses the public `animeshakr/plant-disease-efficientnetv2s` EfficientNetV2S TFLite model. Its model card states that it was trained on the PlantVillage color dataset, covers 38 classes, and reports 99.57% test accuracy. Those benchmark numbers are **the publisher's reported metrics**, not a metric measured by this project.

The model expects 384×384 RGB images and returns probabilities for 38 PlantVillage classes.

### 2. Field branch — existing Random Forest demo model
The field pest-risk model is trained from `data/AgriSense_demo_dataset.csv`, which is the supplied **synthetic/demo dataset**. Therefore its confidence is a model confidence on that demo distribution; it must not be presented as field-validated agricultural accuracy.

### 3. Image + field output
The website presents both branches together and performs **decision fusion**: image disease evidence + field pest-risk result + weather/soil advisory. This is deliberately **not** falsely described as a jointly trained multimodal neural network because the available image dataset does not contain matching temperature/NPK/weather records for each leaf image.

## One-time setup

From the project folder:

```powershell
pip install -r requirements.txt
python download_real_model.py
python app.py
```

Open:

`http://127.0.0.1:5000`

The model download is about 42 MB. Internet is required only for the one-time model download and for live Open-Meteo weather in the browser.

## If the model download fails

The project intentionally does **not** create a fake image model. Retry with an internet connection or download the model files from the public Hugging Face repository and place them in `models/`:

- `model_float16_quant.tflite`
- `class_indices.json`

## Image prediction

Upload a clear leaf photograph. The result includes:

- predicted PlantVillage class
- crop
- disease/healthy status
- top confidence
- top-3 predictions
- prevention guidance

PlantVillage is a controlled/lab-style benchmark. A field photograph with different lighting, backgrounds, occlusion or symptoms can perform differently. The image result is therefore an AI screening result, not a guaranteed field diagnosis.

## Live weather

The browser requests location only when the user presses **Use My Location**. Current temperature, humidity, precipitation and wind are fetched from Open-Meteo and copied into the prediction form.

## Project structure

```text
AgriSense_Final_Real/
├── app.py
├── download_real_model.py
├── train_image_model.py
├── train_models.py
├── requirements.txt
├── run_windows.bat
├── README.md
├── data/
│   ├── AgriSense_demo_dataset.csv
│   └── plantvillage/          # optional local training data
├── models/
│   └── agrisense_pest_risk_model.pkl
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

## Optional: train your own image model

If you later obtain the PlantVillage images as class folders, use:

```powershell
python train_image_model.py --epochs 8
```

Expected layout:

```text
data/plantvillage/
  Tomato___Bacterial_spot/
  Tomato___healthy/
  Potato___Late_blight/
  ...
```

A locally trained model can then replace the public TFLite inference branch, but only report accuracy after actually evaluating it on a held-out test set.

## Academic honesty note

Do not write “100% accurate.” No ML system can guarantee that. For the report, distinguish clearly between:

- publisher-reported PlantVillage benchmark metrics;
- AgriSense's synthetic field-model results; and
- real-world field performance, which requires independent field validation.

Fertilizer and crop-protection guidance is decision support. Verify soil-based fertilizer decisions and any pesticide use with local agricultural recommendations and product labels.


## Recommended final PBL setup: 3-crop trained model

For a focused project, AgriSense can be trained only on **Tomato, Maize/Corn and Potato**. These crops/classes are represented in PlantVillage; Rice/Paddy is not part of the standard PlantVillage 38-class dataset.

### 1. Download the three crop folders

Requires Git on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_3crop_dataset.ps1
```

Then prepare the folders:

```powershell
powershell -ExecutionPolicy Bypass -File .\prepare_3crop_dataset.ps1
```

### 2. Train the actual 3-crop model

Fast CPU-friendly training:

```powershell
python train_image_model.py --epochs 5 --max_images_per_class 300
```

More complete training:

```powershell
python train_image_model.py --epochs 10
```

The script creates a stratified 80/10/10 train/validation/test split and evaluates the held-out test set. It saves:

- `models/plant_disease_3crop.keras`
- `models/class_names_3crop.json`
- `models/3crop_metrics.json`

The website automatically prefers this **locally trained 3-crop model** once it exists. Therefore the prediction shown by the website comes from the model you actually trained, rather than from a hard-coded result.

### Classes

The exact disease classes are taken from the downloaded PlantVillage folders. Tomato has multiple disease/healthy classes; Corn/Maize has disease/healthy classes; Potato has disease/healthy classes. The project does not invent unsupported disease labels.

### Academic honesty

Never write 100% accuracy. The exact accuracy in your report should be copied from `models/3crop_metrics.json` after your actual training run. PlantVillage images are a benchmark dataset and field photographs can behave differently because of lighting, background, camera quality and symptom variation.
