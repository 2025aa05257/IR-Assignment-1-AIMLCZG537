# BITS Pilani – Information Retrieval (AIMLCZG537)
## Assignment 1 — End-to-End IR System
**Streamlit URL:** https://ir-assignment-1-aimlczg537-group46.streamlit.app/

**GitHub:** https://github.com/2025aa05257/IR-Assignment-1-AIMLCZG537

| Field | Details |
|-------|---------|
| Course | Information Retrieval (AIMLCZG537) |
| Semester | S2 2025-26 |
| Instructor | Dr. Reddy Rani V, BITS Pilani |
| Max Marks | 10 |
| Deadline | 15th June 2026, 23:59 IST |
| Member 1 | KADARMOHIDEEN N — 2025aa05257 |
| Member 2 | SAUMYARANJAN MOHANTY — 2025aa05252 |
| Member 3 | DIDDIGE ABHINAV — 2025aa05251 |

---

## Overview

An end-to-end Information Retrieval system built with **Streamlit**, applied to the
**Bank Transaction Fraud Detection** dataset (1,000,000 rows × 26 columns).
All IR algorithms (inverted index, biword/positional index, BST, B-Tree, k-gram,
edit distance, Soundex) are implemented from scratch in pure Python.

---

## Files in This Repository

| File | Description |
|------|-------------|
| `app.py` | Complete Streamlit application (1,236 lines, self-contained) |
| `bank_fraud_sample.csv` | 2,000-row sample of the dataset (< 25 MB, ready to use) |
| `README.md` | This file |

> **Note:** The full `bank_fraud.csv` (1M rows, ~100 MB) is NOT in the repo.
> Use `bank_fraud_sample.csv` which is already included.

---

## Prerequisites

- Python 3.9+  |  pip (or conda / Anaconda)

---

## Installation

```bash
pip install streamlit nltk pandas
python -m nltk.downloader punkt stopwords wordnet punkt_tab
```

---

## Running the App

### Option A — Clone from GitHub

```bash
git clone https://github.com/2025aa05257/IR-Assignment-1-AIMLCZG537.git
cd IR-Assignment-1-AIMLCZG537
streamlit run app.py
```

### Option B — Local folder (Windows PowerShell)

```powershell
cd C:\Users\kadarmohideen\ir_assignment
streamlit run app.py
```

App opens at **http://localhost:8501**

> The app automatically finds `bank_fraud_sample.csv` when it is in the **same folder as app.py**.
> No path configuration needed.

---

## Application Modules

| Module | What It Does |
|--------|-------------|
| **A. Dataset & Workflow** | Dataset metrics, raw CSV preview, IR document representation, system workflow diagram |
| **B. Text Preprocessing** | Step-by-step pipeline: tokenize → lowercase → stop-words → hyphen handling → stemming vs lemmatization; inverted index viewer |
| **C. Phrase Query Processing** | Biword index vs Positional index — live query, false positive demo, comparison table |
| **D. Dictionary Search (BST & B-Tree)** | Live benchmark: build time, comparisons, timing; statistical summary |
| **E. Tolerant Retrieval** | 4 techniques: Wildcard (k-gram k=2), Spelling correction (edit distance ≤2), Interactive DP table, Phonetic Soundex |
| **G. Inference & Discussion** | Compulsory Q&A, limitations, future improvements, rubric self-assessment |

---

## Key Design Decisions

- **Lemmatization > Stemming** — Porter corrupts fraud terms (`"phishing"→"phish"`). WordNet preserves them.
- **Positional index > Biword** — Biword produces false positives for phrases like `"card cloning"`.
- **B-Tree > BST** — ~55% fewer comparisons; guaranteed `O(log_t n)` height.
- **NLTK + pure-Python fallback** — App works even without NLTK installed.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `bank_fraud_sample.csv not found` | Place CSV in the **same folder as app.py** |
| `ModuleNotFoundError: streamlit` | `pip install streamlit nltk pandas` |
| `LookupError: NLTK resource` | `python -m nltk.downloader punkt stopwords wordnet punkt_tab` |
| File upload >25 MB error | Use `bank_fraud_sample.csv` (already in repo, ~1 MB) |
| `ValueError: Duplicate column names` | Fixed — DP table uses position-prefixed labels (h0, h3…) |
| `ValueError: All arrays must be same length` | Fixed — rubric table has 9 entries in every column |
| Streamlit "Deploy" button error | Ignore — only needed for Streamlit Community Cloud; app runs fine locally |

---

## Rubric

| Component | Marks | Status |
|-----------|-------|--------|
| Streamlit end-to-end workflow | 1.0 | ✅ Done |
| Text preprocessing | 1.5 | ✅ Done |
| Stemming vs Lemmatization | 1.0 | ✅ Done |
| Phrase query (Biword + Positional) | 1.5 | ✅ Done |
| BST and B-Tree comparison | 1.5 | ✅ Done |
| Tolerant retrieval | 1.5 | ✅ Done |
| Experimental evidence & inferences | 1.0 | ✅ Done |
| Virtual lab usage | 1.0 | ✅ Run on BITS Lab portal |
| **Total** | **10** | **10/10 ready** |
