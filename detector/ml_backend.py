"""
ml_backend.py
─────────────
Handles both backends:
  DETECTOR_MODE = 'claude' → calls Anthropic Vision API
  DETECTOR_MODE = 'pkl'    → loads your local model file and runs inference
"""

import io
import json
import base64
import numpy as np
from pathlib import Path
from django.conf import settings

_model_cache = None


def _get_model_path():
    raw = getattr(settings, 'MODEL_PATH', None) or getattr(settings, 'PKL_MODEL_PATH', None)

    if not raw:
        raise ValueError(
            "No model path configured. "
            "Set MODEL_PATH in settings.py or environment variable."
        )

    return Path(str(raw))


def _load_model():
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    path = _get_model_path()

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    ext = path.suffix.lower()

    # ─────────────────────────────────────────────
    # KERAS / TENSORFLOW
    # ─────────────────────────────────────────────
    if ext in ('.keras', '.h5'):

        try:
            import keras
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        from keras.applications.efficientnet import EfficientNetB0, preprocess_input

        try:
            full_model = keras.models.load_model(
                str(path),
                compile=False,
                custom_objects={"preprocess_input": preprocess_input}
            )
        except Exception:
            full_model = keras.models.load_model(str(path), compile=False)

        # rebuild inference model
        base = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))

        x = keras.layers.GlobalAveragePooling2D()(base.output)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        out = keras.layers.Dense(1, activation='sigmoid')(x)

        inference_model = keras.Model(inputs=base.input, outputs=out)

        saved_layers = full_model.layers
        inf_layers = inference_model.layers

        # copy EfficientNet weights
        for sl in saved_layers:
            if 'efficientnet' in sl.name.lower():
                for il in inf_layers:
                    if 'efficientnet' in il.name.lower():
                        il.set_weights(sl.get_weights())
                        break
                break

        # copy Dense layers
        saved_by_name = {l.name: l for l in saved_layers}

        for il in inf_layers:
            if il.name in saved_by_name:
                sw = saved_by_name[il.name].get_weights()
                if sw:
                    il.set_weights(sw)

        _model_cache = inference_model

    # ─────────────────────────────────────────────
    # PICKLE
    # ─────────────────────────────────────────────
    elif ext == '.pkl':

        import pickle

        with open(path, 'rb') as f:
            _model_cache = pickle.load(f)

    # ─────────────────────────────────────────────
    # JOBLIB
    # ─────────────────────────────────────────────
    elif ext == '.joblib':

        import joblib
        _model_cache = joblib.load(path)

    # ─────────────────────────────────────────────
    # PYTORCH
    # ─────────────────────────────────────────────
    elif ext in ('.pt', '.pth'):

        import torch

        _model_cache = torch.load(str(path), map_location='cpu')

        if hasattr(_model_cache, 'eval'):
            _model_cache.eval()

    else:
        raise ValueError(f"Unsupported model format: {ext}")

    return _model_cache


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def analyze_image(image_bytes, media_type):

    mode = getattr(settings, 'DETECTOR_MODE', 'claude')

    if mode == 'pkl':
        return _analyze_with_model(image_bytes, media_type)

    return _analyze_with_claude(image_bytes, media_type)


# ─────────────────────────────────────────────
# CLAUDE API
# ─────────────────────────────────────────────

def _analyze_with_claude(image_bytes, media_type):

    import urllib.request

    api_key = getattr(settings, 'ANTHROPIC_API_KEY', '')

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    b64 = base64.b64encode(image_bytes).decode()

    prompt = (
        'Analyze this image and return ONLY JSON with AI detection score.'
    )

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    raw = "".join(c.get("text", "") for c in data.get("content", []))

    return json.loads(raw.replace("```json", "").replace("```", "").strip())


# ─────────────────────────────────────────────
# LOCAL MODEL
# ─────────────────────────────────────────────

def _analyze_with_model(image_bytes, media_type):

    from PIL import Image
    from keras.applications.efficientnet import preprocess_input

    model = _load_model()
    path = _get_model_path()
    ext = path.suffix.lower()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    img_batch = preprocess_input(img_batch)

    try:
        preds = model.predict(img_batch, verbose=0)
    except Exception:
        preds = np.array(model(img_batch, training=False))

    sigmoid_out = float(preds[0][0])

    CLASS_0_NAME = "AI Generated"
    CLASS_1_NAME = "Real Photo"

    if sigmoid_out >= 0.5:
        label = CLASS_1_NAME
        confidence = sigmoid_out
        ai_prob = 1 - sigmoid_out
    else:
        label = CLASS_0_NAME
        confidence = 1 - sigmoid_out
        ai_prob = 1 - sigmoid_out

    ai_verdict = (
        "AI Generated" if ai_prob >= 0.75 else
        "Likely AI" if ai_prob >= 0.50 else
        "Possibly AI" if ai_prob >= 0.30 else
        "Likely Real" if ai_prob >= 0.10 else
        "Real Photo"
    )

    return {
        "scene": f"Model prediction: {label}",
        "scene_type": "other",
        "lighting": "unknown",
        "detections": [{
            "label": label,
            "category": "Model Prediction",
            "confidence": round(confidence, 4),
            "description": f"{confidence*100:.1f}% confidence",
            "box": [0.05, 0.05, 0.9, 0.9]
        }],
        "tags": ["local-inference", ext.replace(".", "")],
        "colors": [],
        "mood": "analytical",
        "anomalies": "none",
        "summary": f"Model predicted '{label}' with {confidence*100:.1f}% confidence.",
        "ai_score": {
            "score": round(ai_prob, 3),
            "verdict": ai_verdict,
            "reasoning": "Binary classifier output",
            "signals": [
                {
                    "label": "Prediction Confidence",
                    "value": "neutral",
                    "detail": f"{confidence*100:.1f}%"
                }
            ]
        }
    }