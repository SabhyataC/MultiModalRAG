from ocr.pipeline import run_ocr

if __name__ == "__main__":
    image_path = "data/samp_img.png"
    text = run_ocr(image_path, show_debug=False)

    print("\n=== OCR TEXT OUTPUT ===")
    print(text)