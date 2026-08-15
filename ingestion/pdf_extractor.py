import os
import fitz  # PyMuPDF
from ocr.pipeline import run_ocr
import pdfplumber
import glob

def extract_from_pdf(pdf_path, image_output_dir="output"):
    """
    Extracts text and embedded images page-by-page from a PDF, and
    runs OCR on every extracted image.

    Returns a list of dicts, one per page:
        {
            "page_num": int,
            "text": str,                # native PDF text layer
            "image_paths": [str, ...],  # saved image file paths
            "ocr_text": str             # combined OCR text from this page's images
        }
    """
    os.makedirs(image_output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        image_paths = []
        ocr_texts = []
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            out_name = f"{pdf_name}_page{page_num}_img{img_index}.{image_ext}"
            out_path = os.path.join(image_output_dir, out_name)
            with open(out_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(out_path)

            # Run OCR on this image
            annotated_path = os.path.join(image_output_dir, f"annotated_{out_name}.png")
            try:
                img_ocr_text = run_ocr(out_path, output_path=annotated_path, show_debug=False).strip()
                if img_ocr_text and _is_garbled_ocr(img_ocr_text):
                    print(f"[WARN] Skipping likely-garbled OCR from {out_path}")
                elif img_ocr_text:
                    ocr_texts.append(img_ocr_text)
            except Exception as e:
                print(f"[WARN] OCR failed on {out_path}: {e}")

        pages.append({
            "page_num": page_num,
            "text": text,
            "image_paths": image_paths,
            "ocr_text": "\n".join(ocr_texts)
        })

    doc.close()
    return pages


_COMMON_ENGLISH_WORDS = {
    "the", "and", "of", "in", "for", "to", "is", "are", "with", "on",
    "by", "as", "that", "this", "a", "an", "or", "from", "at", "be",
    "which", "it", "these", "such", "can", "will", "their", "its",
}


def _is_garbled_ocr(text, min_stopword_ratio=0.08, max_long_word_ratio=0.15):
    """
    Real prose is dense with short common function words ("the", "and",
    "of", etc). OCR garbage from diagrams/tables tends to merge words
    together (missing spaces), which erases these function words and
    inflates average word length -- even though the merged text is
    still fully alphabetic, so a symbol/alpha-ratio check misses it.
    """
    words = [w.strip(".,;:()[]\"'") for w in text.lower().split()]
    words = [w for w in words if w]
    if len(words) < 5:
        return False  # too short to judge reliably

    stopword_count = sum(1 for w in words if w in _COMMON_ENGLISH_WORDS)
    stopword_ratio = stopword_count / len(words)

    long_word_count = sum(1 for w in words if len(w) > 12)
    long_word_ratio = long_word_count / len(words)

    return stopword_ratio < min_stopword_ratio or long_word_ratio > max_long_word_ratio

def extract_tables_from_pdf(pdf_path):
    """
    Extracts tables page-by-page from a PDF.

    Returns a list of dicts, one per page:
        {"page_num": int, "tables": [table, ...]}
    where each table is a list of rows, and each row is a list of cell strings.
    """
    tables_by_page = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            tables_by_page.append({"page_num": page_num, "tables": tables})

    return tables_by_page


import glob

if __name__ == "__main__":
    pdf_files = glob.glob("data/*.pdf")
    print(f"Found {len(pdf_files)} PDFs: {pdf_files}\n")

    for pdf_path in pdf_files:
        print(f"\n{'='*50}")
        print(f"PROCESSING: {pdf_path}")
        print('='*50)

        try:
            result = extract_from_pdf(pdf_path)
            for page in result:
                print(f"\n--- Page {page['page_num']} ---")
                print("PDF TEXT:", page["text"][:150] or "(empty)")
                print("IMAGES:", page["image_paths"] or "(none)")
                print("OCR TEXT:", page["ocr_text"][:150] or "(empty)")

            table_result = extract_tables_from_pdf(pdf_path)
            found_tables = [p for p in table_result if p["tables"]]
            print(f"\nTables found on pages: {[p['page_num'] for p in found_tables] or 'none'}")

        except Exception as e:
            print(f"[ERROR] Failed processing {pdf_path}: {e}")