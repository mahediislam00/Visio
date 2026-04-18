import json
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render

from .ml_backend import analyze_image


def index(request):
    """Serve the main UI."""
    return render(request, 'detector/index.html')


@csrf_exempt
@require_http_methods(["POST"])
def analyze(request):
    """
    POST /analyze/
    Accepts multipart form with an 'image' file field.
    Returns JSON analysis result.
    """
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image file provided.'}, status=400)

        if image_file.size > 20 * 1024 * 1024:
            return JsonResponse({'error': 'Image too large (max 20MB).'}, status=400)

        image_bytes = image_file.read()
        media_type  = image_file.content_type or 'image/jpeg'

        result = analyze_image(image_bytes, media_type)
        return JsonResponse(result)

    except FileNotFoundError as e:
        return JsonResponse({'error': str(e)}, status=500)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Analysis failed: {str(e)}'}, status=500)
