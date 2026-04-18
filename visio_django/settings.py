from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'change-me-in-production'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'detector',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'visio_django.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': ['django.template.context_processors.request']},
}]

WSGI_APPLICATION = 'visio_django.wsgi.application'

STATIC_URL = '/static/'

# ── Your Anthropic API key (used only if MODE = 'claude') ──
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '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')

# ── Backend mode: 'claude' | 'pkl' ──
# Set to 'pkl' to use your own model, 'claude' to use Claude Vision API
DETECTOR_MODE = os.environ.get('DETECTOR_MODE', 'pkl')

# ── Path to your model file (used only if DETECTOR_MODE = 'pkl') ──
# Supports .keras, .h5, .pkl, .pt, .pth, .joblib
# Change the filename below to match your file.
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    str(BASE_DIR / 'detector' / 'ml_models' / 'model.keras')
)

# Legacy alias — kept for backward compatibility
PKL_MODEL_PATH = os.environ.get('PKL_MODEL_PATH', MODEL_PATH)
