"""
OCR pipeline: image preprocessing (crop, deskew, denoise, contrast, threshold)
followed by EasyOCR with a Tesseract fallback.
"""

import cv2
import numpy as np
import easyocr
from matplotlib import pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# Utility functions
# -------------------------------

def show(title, img, cmap="gray"):
    """Quick display for debugging"""
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def upscale(img, scale=2):
    """Upscale image with bicubic interpolation"""
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def crop_to_content(img, thresh=250):
    """Crop away large white borders"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(cv2.bitwise_not(cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]))
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


def deskew(img):
    """Estimate skew angle and rotate"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, angle


def preprocess(img):
    """Full preprocessing pipeline"""
    img = crop_to_content(img)

    img, angle = deskew(img)
    print(f"[INFO] Deskew angle corrected: {angle:.2f}°")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    binary = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    return binary


# -------------------------------
# OCR Runner
# -------------------------------

def run_ocr(image_path, output_path="output/output_annotated.png", show_debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    if min(h, w) < 1200:
        scale = 1600 / min(h, w)
        img = upscale(img, scale=scale)

    processed = preprocess(img)
    if show_debug:
        show("Preprocessed", processed)

    reader = easyocr.Reader(['en'])
    results = reader.readtext(processed)

    clean_results = [(text.strip(), prob) for (_, text, prob) in results if prob > 0.4 and len(text.strip()) > 2]

    output = img.copy()
    for (bbox, text, prob) in results:
        (top_left, _, bottom_right, _) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(output, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(output, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if show_debug:
        show("OCR Result", cv2.cvtColor(output, cv2.COLOR_BGR2RGB), cmap=None)
    cv2.imwrite(output_path, output)

    final_text = "\n".join([t for t, _ in clean_results])
    if len(final_text) < 30:
        final_text = pytesseract.image_to_string(processed).strip()

    return final_text