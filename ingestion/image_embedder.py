"""
Generates CLIP embeddings directly from images, so retrieval can
match on visual content even when OCR text is poor or missing
(e.g. charts, diagrams, photos with no text at all).
"""

import torch
import open_clip
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration

_caption_processor = None
_caption_model = None

_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load once at module level — loading per-call would be very slow
_model, _, _preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
_model.to(_device)
_model.eval()


def embed_image(image_path):
    """
    Returns a normalized CLIP embedding (as a list of floats) for
    the image at image_path.
    """
    image = Image.open(image_path).convert("RGB")
    image_input = _preprocess(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = _model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().tolist()


def embed_text(text):
    """
    Returns a normalized CLIP embedding for a text string, in the
    SAME vector space as embed_image — so you can compare the two
    directly (e.g. cosine similarity) for text-to-image search.
    """
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer([text]).to(_device)

    with torch.no_grad():
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().tolist()


def _load_caption_model():
    global _caption_processor, _caption_model
    if _caption_model is None:
        _caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _caption_model.to(_device)
        _caption_model.eval()


def caption_image(image_path):
    """
    Returns a natural-language caption describing the image, using BLIP.
    """
    _load_caption_model()
    image = Image.open(image_path).convert("RGB")
    inputs = _caption_processor(image, return_tensors="pt").to(_device)

    with torch.no_grad():
        out = _caption_model.generate(**inputs, max_new_tokens=40)

    caption = _caption_processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    import numpy as np
    from ocr.pipeline import run_ocr

    img_path = "output/sample2_page1_img1.jpeg"

    print("=== PATH 1: OCR TEXT ===")
    ocr_text = run_ocr(img_path, output_path="output/embedder_test_annotated.png", show_debug=False)
    print(f"OCR text ({len(ocr_text)} chars): {ocr_text[:150]}")

    print("\n=== PATH 2: CLIP VISUAL EMBEDDING ===")
    img_emb = embed_image(img_path)
    queries = [
        "a diagram about privacy and risk",
        "a photo of a cat",
        "a bar chart showing sales data",
    ]
    for q in queries:
        txt_emb = embed_text(q)
        sim = np.dot(img_emb, txt_emb)
        print(f"  '{q}' -> similarity {sim:.4f}")

    print("\n=== PATH 3: CAPTION ===")
    caption = caption_image(img_path)
    print(f"Caption: {caption}")