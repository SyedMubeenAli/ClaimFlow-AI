"""
Prepares a stored damage photo for a vision model: fix rotation, shrink to at most
1024 pixels on the long side, and re-encode as a plain JPEG. Re-encoding drops
metadata and anything hidden in the original file, and keeps the request small.
"""

from __future__ import annotations

import io
from pathlib import Path

from .llm import ImageInput


def prepare_image(path: Path, max_side: int = 1024) -> ImageInput:
    from PIL import Image, ImageOps  # Pillow ships with Streamlit

    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
    return ImageInput(buffer.getvalue(), "image/jpeg")
