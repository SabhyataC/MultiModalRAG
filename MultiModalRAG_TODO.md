# MultiModalRAG — Build Plan

## Phase 1: Foundation + Ingestion
- [T] Clean up repo structure (separate folders: ingestion/, ocr/, data/, output/)
- [T] Write requirements.txt / set up virtual environment
- [T] Remove dead commented-out code from main.py
- [T] Rewrite PDF text extraction using PyMuPDF
- [T] Route each PDF page through text extraction + image extraction
- [T] Wire extracted images into existing OCR pipeline
- [T] Add table extraction (pdfplumber or camelot)
- [T] Test on all 3 sample PDFs in /data

## Phase 2: True Multimodal Retrieval
- [T] Add CLIP (or similar) for direct image embeddings
- [T] Add image captioning (BLIP or similar) for diagrams/charts
- [T] Confirm dual-path works: OCR text + visual embedding, independently

## Phase 3: Chunking + Indexing
- [ ] Implement semantic/recursive chunking for text
- [ ] Attach metadata (page number, source image path) to every chunk
- [ ] Set up vector store (Chroma or Qdrant)
- [ ] Store text-chunk embeddings and image embeddings (same or linked collections)

## Phase 4: Retrieval
- [ ] Implement dense (embedding) search
- [ ] Implement BM25 keyword search
- [ ] Add cross-encoder reranker
- [ ] Query fan-out across text + image collections, merge results

## Phase 5: Generation
- [ ] Wire retrieved chunks into LLM call (local via Ollama, or API)
- [ ] Enforce citation of page/image source in every answer
- [ ] Test end-to-end: PDF in → question → cited answer out

## Phase 6: What Makes It "Major"
- [ ] Build 20-30 QA eval set
- [ ] Run RAGAS (or custom eval) for retrieval precision/recall + faithfulness
- [ ] Build demo UI (Streamlit or Gradio) — upload PDF, ask questions, show sources
- [ ] Write README covering architecture + design tradeoffs
