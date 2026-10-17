# MultiModalRAG

A local, fully offline retrieval-augmented generation system that answers questions over PDFs using **text, tables, and images**. Every answer carries citations, and each citation is validated against what was actually retrieved.

## Features

- **Multi-path ingestion**: native PDF text, embedded images, OCR on images (EasyOCR + Tesseract with preprocessing), and table extraction in one pass.
- **True multimodal retrieval**: text goes through a sentence-transformer space, images through CLIP. An image can be retrieved by a text query even when its OCR output is garbled.
- **Hybrid text search**: dense retrieval + BM25, fused with Reciprocal Rank Fusion, then reranked with a cross-encoder.
- **Cited answers**: every `[n]` citation in the answer is checked against the real retrieved sources, and invalid ones are flagged.
- **OCR garbage filtering**: garbled chunks are detected with a stopword-ratio and long-word-ratio heuristic and dropped before indexing.
- **Local generation**: runs on Ollama, with no API keys and no data leaving your machine.
- **Evaluation harness**: a QA set with recall@k and an LLM-judged faithfulness metric.
- **Streamlit demo**: PDF upload, per-document filter, cited answers, and retrieved text and images.

## Architecture

```
                        INGESTION
 PDF ──▶ native text ──────────────┐
     ├─▶ embedded images ─▶ OCR ───┼─▶ chunker (500 chars, 75 overlap, garble filter)
     └─▶ tables (pdfplumber) ──────┘            │
              │                                  ▼
              └─▶ CLIP (ViT-B-32)        MiniLM-L6-v2 embeddings
                        │                        │
                        ▼                        ▼
                 Chroma: images            Chroma: text
                (512-dim, CLIP space)     (384-dim, MiniLM space)

                        RETRIEVAL
 query ─┬─▶ dense text search ─┐
        ├─▶ BM25 ──────────────┴─▶ RRF fusion ─▶ cross-encoder rerank ─▶ top text chunks
        └─▶ CLIP image search ───────────────────────────────────────────▶ top images

                        GENERATION
 chunks + images ─▶ numbered-source prompt ─▶ Ollama (llama3.2:3b)
                 ─▶ citation validation ─▶ answer + resolved Sources
```

**Key design decisions**

| Decision | Why |
|---|---|
| Two separate Chroma collections | CLIP (512-dim) and MiniLM (384-dim) embeddings can't share a collection; they're linked by `source_pdf` / `page_num` metadata |
| RRF to fuse dense + BM25 | Their scores are on incompatible scales; rank-based fusion avoids calibration |
| Image results kept as a separate list | The cross-encoder can't score images, and CLIP distances aren't comparable to its scores |
| BM25 index rebuilt per call | Simple and fast enough at current corpus size; caching is the upgrade path |
| Garble filter applied after chunking | Garbled text also arrives through the PDF's native text layer, which an OCR-only filter misses |
| Citation validator checks structure only | It confirms a cited `[n]` maps to a real source; whether the claim is accurate is measured by the faithfulness eval |

## Tech stack

Python 3.12, PyMuPDF, pdfplumber, EasyOCR, Tesseract, OpenCLIP (ViT-B-32), BLIP-base (captioning), sentence-transformers (all-MiniLM-L6-v2), Chroma, rank-bm25, cross-encoder/ms-marco-MiniLM-L-6-v2, Ollama (llama3.2:3b), Streamlit.

## Getting started

### Prerequisites

- Python 3.12
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on your `PATH`
- [Ollama](https://ollama.com) installed

### Setup

```bash
git clone <your-repo-url>   # VERIFY
cd MultiModalRAG
python -m venv venv
venv\Scripts\activate       # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

### Run the demo

```bash
streamlit run app.py
```

Upload a PDF in the UI, then ask questions. Use the document filter to restrict search to one file.

### Run the pipeline from the command line

Modules are run with `-m` from the project root so cross-package imports resolve:

```bash
python -m ingestion.pdf_extractor
python -m generation.pipeline
```

## Evaluation <!-- VERIFY script name and location -->

The eval uses a QA set generated from the ingested PDFs. Metrics:

- **recall@k**: whether the source chunk for each question appears in the top-k retrieved results.
- **faithfulness rate**: an LLM judge produces a `VERDICT: YES/NO` on whether each answer is supported by its retrieved context.

| Metric | Result |
|---|---|
| recall@k | 1.0 |
| faithfulness rate | 1.0 |

> **Caveat:** these results are on 8 questions over a single PDF, so treat them as a sanity check, not a benchmark. Getting there involved fixing the eval itself (a faulty verdict parser, a QA set containing questions about documents that were never ingested) as well as the pipeline (OCR garbage filtering).

## Project layout

```
MultiModalRAG/
├── app.py                  # Streamlit demo
├── ocr/
│   └── pipeline.py         # preprocessing (deskew, CLAHE, adaptive threshold) + OCR
├── ingestion/
│   ├── pdf_extractor.py    # text, images, OCR, tables in one pass
│   ├── image_embedder.py   # CLIP embeddings + BLIP captions
│   ├── chunker.py          # chunking, metadata, garble filter
│   └── vector_store.py     # Chroma collections
├── retrieval/
│   ├── dense_search.py     # text + image dense search
│   ├── bm25_search.py      # BM25 keyword search
│   ├── reranker.py         # cross-encoder reranking
│   └── hybrid.py           # fusion + rerank orchestration
├── generation/
│   ├── llm.py              # prompt building + Ollama call
│   ├── citations.py        # citation validation + Sources section
│   └── pipeline.py         # end-to-end entry point
├── data/                   # input PDFs
└── output/                 # extracted images, intermediate output
```

## Known limitations

- **Images aren't seen by the LLM.** Retrieved images are cited by filename and page, but their pixels aren't passed to the model. A vision model such as LLaVA is the natural upgrade.
- **BLIP captions are weak on diagram-heavy images**, which is why CLIP embeddings carry most of the visual retrieval.
- **Generation is slow on CPU.** There's no GPU requirement, but CPU-only Ollama is the main latency source.
- **Small local model.** llama3.2:3b can make inaccurate claims about real sources, and the citation validator can't catch that.
- **Small evaluation set** (see above).