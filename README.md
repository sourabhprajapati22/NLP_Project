# Myntra Sentiment Analysis Project

This project implements an end-to-end pipeline for scraping, processing, analyzing, and classifying product reviews from Myntra. Its goal is to perform sentiment and aspect-based analysis on reviews so that products can be automatically recommended (or not) based on customer sentiment. The project leverages web scraping, data cleaning, natural language processing (NLP), transformer-based sentence embeddings, and machine learning classification.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Setup and Installation](#setup-and-installation)
- [Process and Execution](#process-and-execution)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Web Scraping](#2-web-scraping)
  - [3. Data Exploration & Preprocessing](#3-data-exploration--preprocessing)
  - [4. Sentiment & Aspect Analysis](#4-sentiment--aspect-analysis)
  - [5. Embedding Generation](#5-embedding-generation)
  - [6. Model Training](#6-model-training)
    - [Models Used](#models-used)
  - [7. Inference Pipeline](#7-inference-pipeline)
- [Final Results](#final-results)
- [Known Issues and Future Improvements](#known-issues-and-future-improvements)
- [License](#license)

---

## Project Overview

The project is structured as an integrated solution with the following capabilities:

- **Web Scraping:** Extract product pages and reviews from Myntra using Playwright and BeautifulSoup.  
- **Data Cleaning:** Clean raw HTML data by removing unwanted characters, HTML tags, URLs, and by replacing unconventional emoji characters with descriptive text.
- **Sentiment Analysis:** Apply a pre-trained BERT-based sentiment analysis pipeline alongside a zero-shot classifier (BART model) to extract aspect-based sentiment information.
- **Embedding Generation:** Convert cleaned reviews into fixed-length embeddings using the Sentence Transformer model `"sentence-transformers/all-MiniLM-L6-v2"`.
- **Classification:** Train multiple machine learning models (Logistic Regression, Decision Tree, Random Forest) on the review embeddings to classify sentiment.
- **Inference:** Run an end-to-end pipeline to scrape new reviews, process them, generate embeddings, classify sentiment, and finally output product recommendation results.

---

## Features

- **Automated Web Scraping:**  
  - Use of Playwright for browser automation.
  - BeautifulSoup for HTML parsing.
  - Pagination handling and dynamic content loading.
- **Data Preprocessing:**  
  - Cleaning review texts by removing HTML, URLs, and extraneous special characters.
  - Replacing non-standard emoji characters with sentiment descriptive labels.
- **Dual Sentiment Analysis:**  
  - Public pre-trained sentiment analysis using a BERT-based model.
  - Aspect-based sentiment classification using a zero-shot classifier.
- **Embeddings and Classification:**  
  - Generation of sentence embeddings to capture semantic meaning.
  - Training and evaluation of Logistic Regression, Decision Tree, and Random Forest classifiers.
- **End-to-End Inference:**  
  - Integrated pipeline that scrapes, cleans, embeds, classifies, and produces a recommendation.

---

## File Structure

```
main_project/
├── myntra_sentiment_analysis/
│   ├── scraping/
│   │   ├── myntra_scraper.py
│   │   └── embedding.py
│   ├── notebooks/
│   │   ├── data_exploration.ipynb
│   │   └── model_training.ipynb
│   ├── data/
│   │   ├── raw/
│   │   │   ├── reviews_same.jsonl
│   │   │   ├── reviews_diff.jsonl
│   │   │   └── top20_product_aspect.jsonl
│   │   └── processed/
│   │       ├── reviews_same.csv
│   │       └── review_embeddings_flat.jsonl
│   ├── models/
│   │   └── baseline/
│   │       ├── random_forest_model.joblib
│   │       ├── decision_tree_model.joblib
│   │       └── logistic_regression_model.joblib
├── inference.py
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### Prerequisites

- Python 3.8
- Conda
- Internet access

### 1. Environment Setup

```bash
conda create --prefix ./venv python==3.8 -y
conda activate ./venv
pip install -r requirements.txt
pip install playwright ipykernel beautifulsoup4 tqdm pandas matplotlib transformers emoji torch hf_xet textblob
playwright install chromium
```

---

## Process and Execution

### 2. Web Scraping

Run:
```bash
python main_project/myntra_sentiment_analysis/scraping/myntra_scraper.py
```
- Extract product links and review HTML
- Saves `reviews_same.jsonl`, `reviews_diff.jsonl`

### 3. Data Exploration & Preprocessing

Open and run:
```bash
jupyter notebook main_project/myntra_sentiment_analysis/notebooks/data_exploration.ipynb
```
- Clean reviews
- Replace emojis
- Save to `reviews_same.csv`

### 4. Sentiment & Aspect Analysis

- Run sentiment analysis using HuggingFace pipeline
- Apply zero-shot aspect classifier (`facebook/bart-large-mnli`)
- Save results to `top20_product_aspect.jsonl`

### 5. Embedding Generation

Run:
```bash
python main_project/myntra_sentiment_analysis/scraping/embedding.py
```
- Generates sentence embeddings
- Saves to `review_embeddings_flat.jsonl`

### 6. Model Training

Open:
```bash
jupyter notebook main_project/myntra_sentiment_analysis/notebooks/model_training.ipynb
```
- Train Logistic Regression, Decision Tree, Random Forest
- Evaluate using F1-score and Confusion Matrix
- Save models to `/models/baseline/`

#### Models Used

- **Logistic Regression:** Simple linear classifier, fast and interpretable.
- **Decision Tree:** Non-linear model that splits features by rules, good for interpretability.
- **Random Forest:** Ensemble of decision trees for higher accuracy and robustness.
- All models are trained using review embeddings generated from the transformer model.

### 7. Inference Pipeline

Run:
```bash
python inference.py
```
- Processes new reviews
- Generates embeddings
- Loads trained models
- Outputs whether the product is **recommended** or **not recommended**

---

## Final Results

After training and evaluating the models, the following performance metrics were observed:

### Logistic Regression:
- Accuracy: ~92%
- F1-Score (Positive): 0.96
- F1-Score (Neutral): 0.62
- F1-Score (Negative): 0.61

### Decision Tree:
- Accuracy: ~85%
- F1-Score (Positive): 0.92
- F1-Score (Neutral): 0.59
- F1-Score (Negative): 0.58

### Random Forest:
- Accuracy: ~93%
- F1-Score (Positive): 0.97
- F1-Score (Neutral): 0.65
- F1-Score (Negative): 0.66

### Recommendation Decision:
If the majority of predicted sentiments for a product’s reviews are positive, the product is marked as:
** Recommended for the customer**  
Otherwise, it is:
** Not recommended to the customer**

---

## Known Issues and Future Improvements

- Improve scraping robustness for website changes
- Add Docker container for deployment
- Include support for multilingual reviews

---

## License

This project is licensed under the MIT License.

