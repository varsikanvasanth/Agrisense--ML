from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from pathlib import Path
import json
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------
# EXISTING FIELD MODEL
# ---------------------------------------------------------
MODEL = joblib.load(
    ROOT / 'models' / 'agrisense_pest_risk_model.pkl'
)

DATA = pd.read_csv(
    ROOT / 'data' / 'AgriSense_demo_dataset.csv'
)

# ---------------------------------------------------------
# REAL LOCALLY TRAINED 3-CROP IMAGE MODEL
# ---------------------------------------------------------
TRAINED_IMAGE_MODEL_PATH = (
    ROOT / 'models' / 'plant_disease_3crop.keras'
)

TRAINED_CLASS_NAMES_PATH = (
    ROOT / 'models' / 'class_names_3crop.json'
)

# ---------------------------------------------------------
# OPTIONAL OLD TFLITE MODEL
# ---------------------------------------------------------
IMAGE_MODEL_PATH = (
    ROOT / 'models' / 'model_float16_quant.tflite'
)

CLASS_NAMES_PATH = (
    ROOT / 'models' / 'class_indices.json'
)

# ---------------------------------------------------------
# PEST MAP
# ---------------------------------------------------------
PEST_MAP = {
    'Paddy': [
        'Brown Planthopper',
        'Stem Borer',
        'Leaf Folder'
    ],

    'Tomato': [
        'Fruit Borer',
        'Whitefly',
        'Leaf Miner'
    ],

    'Maize': [
        'Fall Armyworm',
        'Stem Borer',
        'Aphids'
    ],

    'Groundnut': [
        'Leaf Miner',
        'Aphids',
        'Thrips'
    ],

    'Sugarcane': [
        'Early Shoot Borer',
        'Top Borer',
        'Whitefly'
    ],

    'Potato': [
        'Potato Tuber Moth',
        'Aphids',
        'White Grub'
    ]
}

# ---------------------------------------------------------
# CROP ADVISORY
# ---------------------------------------------------------
CROP_ADVISORY = {

    'Paddy': {
        'fertilizer':
            'Use soil-test-based NPK. Split nitrogen applications rather than applying all nitrogen at once.',

        'timing':
            'Common nutrient stages are basal, active tillering and panicle initiation; follow local extension guidance.',

        'prevention':
            'Maintain spacing, field sanitation, balanced nitrogen and regular scouting for planthoppers and stem borers.'
    },

    'Tomato': {
        'fertilizer':
            'Prioritize balanced NPK and avoid excess nitrogen during flowering/fruiting. Use organic matter according to soil test.',

        'timing':
            'Basal nutrients at planting followed by split feeding during vegetative growth, flowering and fruit development as locally recommended.',

        'prevention':
            'Use clean planting material, remove badly infected leaves, avoid prolonged leaf wetness and scout leaf undersides.'
    },

    'Maize': {
        'fertilizer':
            'Use soil-test-based NPK with split nitrogen, especially during early vegetative growth.',

        'timing':
            'Common stages include basal, early vegetative/top-dressing and pre-tasseling; follow local recommendations.',

        'prevention':
            'Scout whorls for armyworm, remove volunteer plants and maintain balanced nutrition and irrigation.'
    },

    'Groundnut': {
        'fertilizer':
            'Use soil-test-based nutrient management; calcium and sulphur can matter depending on soil and crop stage.',

        'timing':
            'Apply basal nutrients at establishment and any recommended top/foliar nutrients at the locally advised stage.',

        'prevention':
            'Use clean seed, sanitation, crop rotation and regular inspection for leaf miner, thrips and aphids.'
    },

    'Sugarcane': {
        'fertilizer':
            'Use soil-test-based NPK and avoid excessive nitrogen. Split nutrients according to crop stage and local package of practices.',

        'timing':
            'Basal nutrients at planting followed by scheduled split applications during early and active growth as locally recommended.',

        'prevention':
            'Remove infested shoots, maintain sanitation and monitor early shoot borer/top borer symptoms.'
    },

    'Potato': {
        'fertilizer':
            'Use soil-test-based balanced NPK. Avoid excessive nitrogen because it can encourage excessive vegetative growth.',

        'timing':
            'Apply basal nutrients at planting followed by crop-stage-based nutrient management according to local agricultural recommendations.',

        'prevention':
            'Use healthy seed tubers, maintain field sanitation, avoid prolonged leaf wetness and regularly inspect leaves for disease symptoms.'
    }
}

# ---------------------------------------------------------
# DISEASE PREVENTION
# ---------------------------------------------------------
DISEASE_PREVENTION = {

    'healthy':
        'No disease class was detected by the image model. Continue scouting, balanced nutrition, sanitation and good irrigation management.',

    'bacterial':
        'Reduce prolonged leaf wetness, improve airflow, remove severely affected material and use clean planting material. Follow local extension guidance.',

    'blight':
        'Remove severely affected leaves, improve airflow, avoid overhead irrigation when possible and monitor frequently.',

    'rust':
        'Improve airflow, remove heavily affected debris and monitor new growth. Use locally approved control measures only after confirmation.',

    'mildew':
        'Improve ventilation, avoid excessive humidity around foliage and remove severely affected tissue where appropriate.',

    'spot':
        'Remove badly affected leaves, avoid splashing water between plants and maintain sanitation.',

    'mosaic':
        'Check for viral/vector symptoms, use clean planting material and follow integrated pest management.',

    'virus':
        'Use clean planting material, manage insect vectors with IPM and remove severely symptomatic plants when advised.',

    'scab':
        'Maintain field sanitation, remove infected debris and avoid conditions that keep foliage wet for long periods.'
}


# ---------------------------------------------------------
# SAFE FLOAT CONVERSION
# ---------------------------------------------------------
def clean_float(name, default=0):
    try:
        return float(request.form.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# ---------------------------------------------------------
# CROP ADVICE
# ---------------------------------------------------------
def crop_advice(crop, risk, likely_pest):

    base = CROP_ADVISORY.get(
        crop,
        {
            'fertilizer':
                'Use a soil-test-based balanced fertilizer programme.',

            'timing':
                'Follow the crop-specific nutrient schedule recommended by your local agricultural extension service.',

            'prevention':
                'Use field sanitation, balanced nutrition, irrigation management and regular scouting.'
        }
    )

    # Potato currently does not have a trained field-risk model.
    if risk == 'Not available':
        return [
            'Field pest-risk prediction is not available for Potato because the current Random Forest field model was trained on the original five field crops.',

            base['fertilizer'],

            base['timing'],

            base['prevention']
        ]

    risk_actions = {

        'Low':
            'Risk is currently low in the field-risk model. Continue regular monitoring.',

        'Medium':
            f'Increase scouting frequency and inspect for {likely_pest}.',

        'High':
            f'Inspect promptly for {likely_pest}. Confirm diagnosis before any pesticide use and follow product-label/local agricultural advice.'
    }

    return [
        risk_actions.get(
            risk,
            'Continue regular field monitoring and follow local agricultural recommendations.'
        ),

        base['fertilizer'],

        base['timing'],

        base['prevention']
    ]


# ---------------------------------------------------------
# DISEASE ACTION
# ---------------------------------------------------------
def disease_action(label):

    low = label.lower().replace('___', ' ')

    for key, msg in DISEASE_PREVENTION.items():

        if key in low:
            return msg

    return (
        'Confirm the image result with a qualified agricultural expert '
        'before treatment; use sanitation, irrigation management and '
        'IPM appropriate to the crop.'
    )


# ---------------------------------------------------------
# REAL IMAGE MODEL PREDICTION
# ---------------------------------------------------------
def image_predict(image_file):

    if image_file is None:
        return None

    # -----------------------------------------------------
    # PREFER OUR LOCALLY TRAINED 3-CROP MODEL
    # -----------------------------------------------------
    if (
        TRAINED_IMAGE_MODEL_PATH.exists()
        and TRAINED_CLASS_NAMES_PATH.exists()
    ):

        try:

            import tensorflow as tf
            from PIL import Image

            model = tf.keras.models.load_model(
                TRAINED_IMAGE_MODEL_PATH
            )

            labels = json.loads(
                TRAINED_CLASS_NAMES_PATH.read_text(
                    encoding='utf-8'
                )
            )

            image_file.stream.seek(0)

            img = (
                Image.open(image_file)
                .convert('RGB')
                .resize((224, 224))
            )

            arr = np.expand_dims(
                np.asarray(
                    img,
                    dtype=np.float32
                ) / 255.0,
                0
            )

            probs = model.predict(
                arr,
                verbose=0
            )[0]

            order = np.argsort(probs)[::-1][:3]

            top = [
                {
                    'label': labels[int(i)],
                    'confidence': round(
                        float(probs[i]) * 100,
                        1
                    )
                }

                for i in order
            ]

            label = top[0]['label']

            parts = label.split(
                '___',
                1
            )

            crop = (
                parts[0]
                .replace('_', ' ')
            )

            disease = (
                parts[1]
                .replace('_', ' ')
                if len(parts) > 1
                else label
            )

            return {
                'available': True,

                'mode':
                    'agrisense-trained-3crop',

                'label':
                    label,

                'crop':
                    crop,

                'disease':
                    disease,

                'confidence':
                    top[0]['confidence'],

                'top3':
                    top,

                'prevention':
                    disease_action(label),

                'model_note':
                    'AgriSense locally trained image model using Tomato, Maize/Corn and Potato PlantVillage classes.'
            }

        except Exception as exc:

            return {
                'available': False,

                'mode':
                    'trained-model-error',

                'message':
                    f'Local 3-crop model could not run: {exc}'
            }

    # -----------------------------------------------------
    # OLD OPTIONAL TFLITE MODEL
    # -----------------------------------------------------
    if (
        not IMAGE_MODEL_PATH.exists()
        or not CLASS_NAMES_PATH.exists()
    ):

        return {
            'available': False,

            'mode':
                'training-required',

            'message':
                'Train the 3-crop model first using train_image_model.py.'
        }

    try:

        import tensorflow as tf
        from PIL import Image

        interpreter = tf.lite.Interpreter(
            model_path=str(IMAGE_MODEL_PATH)
        )

        interpreter.allocate_tensors()

        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

        image_file.stream.seek(0)

        img = (
            Image.open(image_file)
            .convert('RGB')
            .resize((384, 384))
        )

        arr = np.expand_dims(
            np.asarray(
                img,
                dtype=np.float32
            ),
            0
        )

        interpreter.set_tensor(
            inp['index'],
            arr
        )

        interpreter.invoke()

        probs = interpreter.get_tensor(
            out['index']
        )[0]

        labels = json.loads(
            CLASS_NAMES_PATH.read_text(
                encoding='utf-8'
            )
        )

        order = np.argsort(
            probs
        )[::-1][:3]

        top = [
            {
                'label':
                    labels[str(int(i))],

                'confidence':
                    round(
                        float(probs[i]) * 100,
                        1
                    )
            }

            for i in order
        ]

        label = top[0]['label']

        parts = label.split(
            '___',
            1
        )

        crop = (
            parts[0]
            .replace('_', ' ')
        )

        disease = (
            parts[1]
            .replace('_', ' ')
            if len(parts) > 1
            else label
        )

        return {
            'available': True,

            'mode':
                'public-pretrained-plantvillage',

            'label':
                label,

            'crop':
                crop,

            'disease':
                disease,

            'confidence':
                top[0]['confidence'],

            'top3':
                top,

            'prevention':
                disease_action(label),

            'model_note':
                'Public PlantVillage image model.'
        }

    except Exception as exc:

        return {
            'available': False,

            'mode':
                'real-model-error',

            'message':
                f'Real image model could not run: {exc}'
        }


# ---------------------------------------------------------
# IMAGE + FIELD FUSION
# ---------------------------------------------------------
def fusion_message(
    image_result,
    selected_crop,
    risk
):

    if not image_result or not image_result.get('available'):

        if selected_crop == 'Potato':

            return (
                'Potato image prediction is available, '
                'but the current Random Forest field-risk model '
                'was not trained for Potato.'
            )

        return (
            'Field-only risk result; upload a leaf image '
            'for image-based disease evidence.'
        )

    image_crop = (
        image_result
        .get('crop', '')
        .lower()
        .replace('pepper, bell', 'pepper')
    )

    selected = selected_crop.lower()

    # -----------------------------------------------------
    # TOMATO
    # -----------------------------------------------------
    if (
        selected == 'tomato'
        and 'tomato' in image_crop
    ):

        match = (
            'Image crop and selected crop agree (Tomato).'
        )

    # -----------------------------------------------------
    # MAIZE / CORN
    # -----------------------------------------------------
    elif (
        selected == 'maize'
        and 'corn' in image_crop
    ):

        match = (
            'Image crop and selected crop agree (Maize/Corn).'
        )

    # -----------------------------------------------------
    # POTATO
    # -----------------------------------------------------
    elif selected == 'potato':

        match = (
            'Image branch detected Potato. '
            'The Potato image model is active; '
            'the current field-risk model does not include Potato.'
        )

    # -----------------------------------------------------
    # OTHER
    # -----------------------------------------------------
    else:

        match = (
            f'Check crop selection: image branch detected '
            f'{image_result.get("crop")}, while field branch '
            f'used {selected_crop}.'
        )

    return (
        f'{match} '
        f'Image disease evidence: '
        f'{image_result.get("disease")}. '
        f'Field pest-risk status: {risk}. '
        f'This is decision fusion, not a separately trained '
        f'multimodal neural model.'
    )


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
@app.route('/')
def home():

    # IMPORTANT:
    # These are the three crops supported by the
    # locally trained image model.
    crops = [
        'Maize',
        'Potato',
        'Tomato'
    ]

    return render_template(
        'index.html',
        crops=crops
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    crop = request.form.get(
        'crop',
        'Tomato'
    )

    soil = request.form.get(
        'soil',
        'Loamy'
    )

    water = request.form.get(
        'water',
        'Medium'
    )

    temp = clean_float(
        'temperature',
        28
    )

    humidity = clean_float(
        'humidity',
        68
    )

    rainfall = clean_float(
        'rainfall',
        60
    )

    ph = clean_float(
        'ph',
        6.5
    )

    nitrogen = clean_float(
        'nitrogen',
        75
    )

    phosphorus = clean_float(
        'phosphorus',
        45
    )

    potassium = clean_float(
        'potassium',
        60
    )

    # -----------------------------------------------------
    # IMAGE PREDICTION
    # -----------------------------------------------------
    photo = request.files.get(
        'crop_photo'
    )

    image_result = (
        image_predict(photo)
        if photo and photo.filename
        else None
    )

    # -----------------------------------------------------
    # POTATO
    # -----------------------------------------------------
    # The current Random Forest model was NOT trained with
    # Potato. Therefore, do NOT send Potato to MODEL.predict().
    if crop == 'Potato':

        risk = 'Not available'

        confidence = 0

        growth_stage = 'Image-based crop diagnosis'

        pest = 'Field model not available for Potato'

        advice = crop_advice(
            crop,
            risk,
            pest
        )

        fusion = fusion_message(
            image_result,
            crop,
            risk
        )

        return jsonify({

            'success':
                True,

            'crop':
                crop,

            'risk':
                risk,

            'confidence':
                confidence,

            'growth_stage':
                growth_stage,

            'likely_pest':
                pest,

            'advice':
                advice,

            'image_result':
                image_result,

            'fusion':
                fusion,

            'input_summary': {

                'temperature':
                    temp,

                'humidity':
                    humidity,

                'rainfall':
                    rainfall,

                'ph':
                    ph,

                'nitrogen':
                    nitrogen,

                'phosphorus':
                    phosphorus,

                'potassium':
                    potassium,

                'soil':
                    soil,

                'water':
                    water
            }
        })

    # -----------------------------------------------------
    # MAIZE / TOMATO FIELD MODEL
    # -----------------------------------------------------
    row = pd.DataFrame([{

        'Crop':
            crop,

        'Temperature_C':
            temp,

        'Humidity_percent':
            humidity,

        'Rainfall_mm':
            rainfall,

        'Soil_Type':
            soil,

        'Soil_pH':
            ph,

        'Nitrogen_mgkg':
            nitrogen,

        'Phosphorus_mgkg':
            phosphorus,

        'Potassium_mgkg':
            potassium,

        'Water_Availability':
            water

    }])

    try:

        risk = MODEL.predict(
            row
        )[0]

        probs = MODEL.predict_proba(
            row
        )[0]

        confidence = round(
            float(max(probs)) * 100,
            1
        )

    except Exception as exc:

        return jsonify({

            'success':
                False,

            'error':
                f'Field model prediction failed: {exc}'
        }), 500

    # -----------------------------------------------------
    # GROWTH STAGE ESTIMATION
    # -----------------------------------------------------
    score = (
        0.025 * rainfall
        + 0.012 * humidity
        + 0.55 * (water == 'High')
        + 0.25 * (nitrogen > 70)
    )

    if score < 2.2:

        stage = 'Germination'

    elif score < 3.5:

        stage = 'Vegetative'

    elif score < 4.8:

        stage = 'Flowering'

    elif score < 6.2:

        stage = 'Fruiting'

    else:

        stage = 'Maturity'

    # -----------------------------------------------------
    # PEST SELECTION
    # -----------------------------------------------------
    pests = PEST_MAP.get(
        crop,
        ['Common crop pests']
    )

    if risk == 'High':

        pest = pests[0]

    elif humidity > 70:

        pest = pests[1]

    else:

        pest = pests[2]

    # -----------------------------------------------------
    # ADVISORY
    # -----------------------------------------------------
    advice = crop_advice(
        crop,
        risk,
        pest
    )

    # -----------------------------------------------------
    # FUSION
    # -----------------------------------------------------
    fusion = fusion_message(
        image_result,
        crop,
        risk
    )

    # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------
    return jsonify({

        'success':
            True,

        'crop':
            crop,

        'risk':
            risk,

        'confidence':
            confidence,

        'growth_stage':
            stage,

        'likely_pest':
            pest,

        'advice':
            advice,

        'image_result':
            image_result,

        'fusion':
            fusion,

        'input_summary': {

            'temperature':
                temp,

            'humidity':
                humidity,

            'rainfall':
                rainfall,

            'ph':
                ph,

            'nitrogen':
                nitrogen,

            'phosphorus':
                phosphorus,

            'potassium':
                potassium,

            'soil':
                soil,

            'water':
                water
        }
    })


# ---------------------------------------------------------
# API STATS
# ---------------------------------------------------------
@app.route('/api/stats')
def stats():

    return jsonify({

        'records':
            len(DATA),

        'classes':
            DATA[
                'Pest_Risk'
            ].value_counts().to_dict(),

        'crops':
            [
                'Maize',
                'Potato',
                'Tomato'
            ]
    })


# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------
if __name__ == '__main__':

    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000
    )