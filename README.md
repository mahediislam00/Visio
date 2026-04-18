# VISIO AI — Django Backend

## Setup

```bash
pip install -r requirements.txt
```

---

## Option A — Use Claude Vision API (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export DETECTOR_MODE=claude
python manage.py runserver
```

Open http://localhost:8000

---

## Option B — Use your own .pkl model

1. Copy your model file into:
   ```
   detector/ml_models/model.pkl
   ```
   Or set a custom path:
   ```bash
   export PKL_MODEL_PATH=/path/to/your/model.pkl
   ```

2. Open `detector/ml_backend.py` and edit `_analyze_with_pkl()` to match your model's API:
   ```python
   # sklearn example
   proba = model.predict_proba(img_array)[0]

   # PyTorch example
   import torch
   tensor = torch.from_numpy(img_array).permute(2,0,1).unsqueeze(0).float() / 255
   proba = torch.softmax(model(tensor), dim=1)[0].detach().numpy()

   # Custom detector returning bboxes
   detections = model.detect(img_array)
   ```

3. Run:
   ```bash
   export DETECTOR_MODE=pkl
   python manage.py runserver
   ```

---

## What the backend expects your model to return

The UI expects this JSON shape from `/analyze/`:

```json
{
  "scene": "...",
  "scene_type": "indoor|outdoor|...",
  "lighting": "natural|artificial|...",
  "detections": [
    {
      "label": "Dog",
      "category": "Animal",
      "confidence": 0.94,
      "description": "Large brown dog",
      "box": [0.1, 0.2, 0.4, 0.5]   ← normalized [x, y, w, h] 0–1
    }
  ],
  "tags": ["outdoor", "animal", "nature"],
  "colors": ["brown", "green", "blue"],
  "mood": "peaceful",
  "anomalies": "none",
  "summary": "Two sentence summary.",
  "ai_score": {
    "score": 0.12,
    "verdict": "Real Photo",
    "reasoning": "Natural noise grain and lighting inconsistencies...",
    "signals": [
      {"label": "Noise Pattern", "value": "negative", "detail": "Film-like grain present"}
    ]
  }
}
```

Edit `_analyze_with_pkl()` in `ml_backend.py` to map your model's output to this shape.

---

## Project structure

```
visio_django/
├── manage.py
├── requirements.txt
├── visio_django/
│   ├── settings.py      ← set DETECTOR_MODE and paths here
│   └── urls.py
└── detector/
    ├── views.py         ← /analyze/ endpoint
    ├── ml_backend.py    ← ← EDIT THIS for your .pkl model
    ├── ml_models/
    │   └── model.pkl    ← drop your model here
    └── templates/
        └── detector/
            └── index.html
```
