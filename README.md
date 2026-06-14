# BITS Pilani – Information Retrieval (AIMLCZG537) – Assignment 1

## Overview
End-to-end Information Retrieval System built with Streamlit.

## Prerequisites
- Python 3.9+
- pip

## Installation

```bash
pip install streamlit nltk pandas
python -m nltk.downloader punkt stopwords wordnet punkt_tab
```

## Run the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

## Modules

| Module | Description |
|--------|-------------|
| A. Dataset & Workflow | Upload documents or use built-in sample dataset |
| B. Text Preprocessing | Tokenization, lowercasing, stop-word removal, hyphen handling, stemming vs lemmatization |
| C. Phrase Query Processing | Biword Index vs Positional Index comparison |
| D. Dictionary Search | BST vs B-Tree benchmark with experimental results |
| E. Tolerant Retrieval | Wildcard (k-gram), spelling correction (edit distance), phonetic (Soundex) |
| G. Inference & Discussion | Complete analysis and rubric self-assessment |

## Dataset
The app ships with a built-in 10-document IR-domain corpus. You can also:
- Upload `.txt` files (one per document)
- Upload `.csv` files (first column treated as document text)
- Paste custom text (one document per line)

## Notes
- All algorithms are self-contained; NLTK is used when available and falls back gracefully
- The app runs entirely through the Streamlit front-end — no backend code needed
- Submission deadline: **15th June 23:59**

## Group Members
- Member 1: [Name] – [ID]
- Member 2: [Name] – [ID]
- Member 3: [Name] – [ID]
