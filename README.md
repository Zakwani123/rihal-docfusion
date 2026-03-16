# DocFusion: Operation Intelligent Documents

**Author:** Mohammed Al Zakwani  
**Competition:** Rihal CodeStacker 2026 — ML Challenge

## Overview

An end-to-end document processing pipeline that extracts structured information from scanned receipts and detects forged or tampered documents. The system combines OCR-based text extraction with machine learning anomaly detection, packaged in a deployable Streamlit web interface.

## Project Structure
```
ML/
├── solution.py          # Core pipeline — DocFusionSolution class (train + predict)
├── app.py               # Streamlit web UI for receipt analysis
├── 01_eda.ipynb         # Exploratory Data Analysis across all 3 datasets
├── check_submission.py  # Competition harness validator
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container configuration for deployment
├── dummy_data/          # Competition smoke-test data
└── README.md            # This file
```

## Approach

### Level 1 — Document Understanding & EDA

Explored all three datasets (SROIE, Find-It-Again, CORD) in `01_eda.ipynb`. Analyzed OCR text quality, field distributions, layout variations, and the characteristics of forged receipts. Key finding: forged receipts in Find-It-Again were created using copy-paste techniques in Paint/Paint3D/GIMP, affecting mostly Total/Payment and Product fields.

### Level 2 — Structured Information Extraction

Built a multi-strategy extraction pipeline using Tesseract OCR:

- **Vendor:** Two-pass approach. First checks against known vendors learned during training. Falls back to heuristic scoring based on business keywords (e.g., "Sdn Bhd", "Enterprise"), capitalization patterns, and line position. Filters out person names, addresses, and document headers.
- **Date:** Regex matching across 11 date format patterns (DD/MM/YYYY, YYYY-MM-DD, DD.MM.YY, etc.). Prioritizes lines containing the word "date" before scanning the full text.
- **Total:** Six-priority extraction system designed to handle real-world OCR noise. Prioritizes rounded/nett totals over subtotals, handles OCR misspellings (e.g., "tatal", "tota!"), cleans spacing errors in numbers (e.g., "39. 80" → "39.80"), and skips subtotal/tax/rounding/discount lines.

### Level 3 — Anomaly Detection & Web UI

**Part A — Anomaly Detection:**  
Trained a Random Forest classifier on the Find-It-Again dataset (577 receipts, 94 forged / 483 genuine). The model uses 10 features extracted from each receipt's OCR text: text length, line count, field presence (vendor/date/total), total value, and character distribution ratios (digit, uppercase, special characters). Uses `class_weight='balanced'` to handle the class imbalance. Falls back to a rule-based scoring system if no trained model is available.

**Part B — Web UI:**  
Streamlit dashboard (app.py) with a dark theme UI. Users upload a receipt image and see extracted fields displayed in color-coded cards — teal for vendor, blue for date, amber for total, red for missing fields. The anomaly verdict appears as a status panel with a pulsing indicator dot (green for genuine, red for suspicious). Each analysis includes a generated summary explaining why the receipt was flagged or cleared. Suspicious receipts also get a red border on the image preview. Raw OCR output is available in an expandable section.

### Level 4 — Harness Integration

The `DocFusionSolution` class implements the exact `train()` / `predict()` interface required by the autograder. The `train()` method learns known vendors and total statistics from `train.jsonl`, and trains the anomaly detection model on Find-It-Again data. The `predict()` method processes each test record through the full extraction and anomaly detection pipeline, outputting predictions in the required JSONL format. Passes `check_submission.py` validation.

### Bonus — Containerization

Dockerfile uses `python:3.13-slim` base image with Tesseract OCR installed. Runs the Streamlit app on port 8501. Tested locally with `docker build` and `docker run`.

### Bonus — Cloud Deployment

The application is deployed on Streamlit Community Cloud and accessible at:

**[Live Demo](https://rihal-docfusion-hphdakzj7u4oltepqshejz.streamlit.app/)**

## Accuracy

Tested on 100 SROIE receipts:

| Field  | Accuracy |
|--------|----------|
| Date   | 69%      |
| Total  | 62%      |
| Vendor | 32%      |

Anomaly detection tested on Find-It-Again samples with correct classification of forged and genuine receipts.

## Design Decisions

- **Tesseract over EasyOCR:** Both engines were tested on SROIE receipts during EDA (see notebook). Accuracy was comparable — Tesseract scored 8/10 on dates vs EasyOCR's 7/10, and both scored similarly on totals and vendors. The deciding factor was speed: Tesseract processes a receipt in under a second while EasyOCR takes several seconds per image and requires a GPU for reasonable performance. Since the competition benchmarks inference latency and the judge environment may not have a GPU, Tesseract was the safer and faster choice. The tradeoff is that Tesseract produces noisier OCR output (e.g., "Total" becomes "Tota!", "15.00" becomes "1i5.u0"), which is why the extraction pipeline has extensive error tolerance built in.
- **Rule-based extraction over deep learning:** With limited labeled data and strict time/memory constraints, hand-crafted extraction rules with OCR error tolerance proved more reliable and interpretable than trying to fine-tune a model.
- **Random Forest for anomaly detection:** Works well on small datasets (577 samples), handles class imbalance natively with balanced weighting, fast inference, and small model footprint. No GPU required.
- **Feature-based anomaly detection over image-based:** The forged regions in Find-It-Again are tiny (often single characters), making pixel-level detection impractical with basic tools. Text-level statistical features capture the downstream effects of tampering.
- **Accuracy limitations:** The main bottleneck is OCR quality, not extraction logic. On receipts where Tesseract produces clean text, extraction accuracy is high. The failures (especially vendor at 32%) are almost entirely cases where OCR garbles the text beyond recognition — characters swapped, words truncated, lines merged. A more sophisticated OCR engine or an image-based extraction model (like LayoutLM) would likely improve results, but at the cost of significantly higher inference time and model size.

## Datasets Used

- **SROIE 2019** — 626 English scanned receipts with vendor, date, address, and total labels. Primary dataset for extraction development.
- **Find-It-Again (L3i Lab)** — 577 genuine and forged receipts. Ground truth for anomaly detection training. Forgeries created with Paint/Paint3D/GIMP using CPI (Copy-Paste-Insert) techniques.
- **CORD (HuggingFace)** — 1,000 diverse receipt layouts. Used for understanding layout diversity and OCR challenges.

## Setup

### Prerequisites

- Python 3.13+
- Tesseract OCR installed ([download here](https://github.com/tesseract-ocr/tesseract))

### Installation
```bash
pip install -r requirements.txt
```

### Tesseract Path

The solution auto-detects the platform. On Windows it uses the default Tesseract install path. On Linux/Docker it uses the system default.

## Usage

### Validate Submission
```bash
python check_submission.py --submission . --data ./dummy_data
```

### Run Web UI
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t docfusion .
docker run -p 8501:8501 docfusion
```

Then open http://localhost:8501
