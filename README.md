# Review-Informed Skincare Product Discovery Using NLP and Data Analytics

![Domain](https://img.shields.io/badge/Domain-Beauty%20%26%20Wellness-coral)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Stage](https://img.shields.io/badge/Stage-Week%203%20NLP%20Analysis-success)
![Status](https://img.shields.io/badge/Status-Week%203%20Analysis%20Completed-brightgreen)

## Project Overview

This project explores how Python-based data analysis and Natural Language Processing (NLP) can be used to understand customer feedback on skincare products.

The work focuses on collecting customer reviews, preparing review text, exploring rating and review patterns, estimating sentiment, and extracting informative terms and phrases using TF-IDF.

The project is being developed as part of the **YuvaIntern Data Science with Python Analyst Internship**. Week 3 builds on the earlier project definition and data preparation work by applying NLP-oriented analysis to a product-specific review dataset.

> **Scope note:** The Week 3 dataset contains 162 unique reviews collected for one The Derma Co product. Findings describe this collected review sample only; they should not be generalized to all The Derma Co products, all skincare products, or the skincare market.

## Business Problem

Star ratings provide a quick summary of customer satisfaction, but they do not explain the reasons behind a customer's experience. Written reviews can contain useful details about product experience, concerns, and expectations, but manually examining every review can be time-consuming.

**Problem question:** How can customer review text and star ratings be analyzed using NLP and data analytics to summarize feedback and identify informative words and phrases for a selected skincare product?

## Week 3 Objectives

1. Collect customer reviews for the selected skincare product.
2. inspect the collected dataset for record counts, unique review identifiers, and missing values.
3. Prepare review titles and review text for NLP analysis.
4. Perform exploratory data analysis on ratings, review lengths, review dates, verified-buyer status, and word usage.
5. Apply a multilingual pretrained model to estimate Positive, Neutral, and Negative sentiment.
6. Compare predicted sentiment with star ratings as an exploratory analysis.
7. Apply TF-IDF to identify informative words and two-word phrases across reviews.
8. Examine important terms within predicted sentiment groups.
9. Export analysis-ready datasets and document methods, findings, and limitations.

## Selected Product and Data Source

### Product analyzed

**The Derma Co 2% Salicylic Acid + 2% Niacinamide Anti-Acne Face Wash**

Product page: https://thedermaco.com/products/2-sali-cinamide-anti-acne-face-wash-with-2-salicylic-acid-2-niacinamide-150-ml-pack-of-2

### Review source and collection approach

The review data was collected from the product's publicly displayed Judge.me review section using a Python-based browser automation and HTML parsing workflow.

The scraper used review-card elements and fields exposed in the review widget, including:

- Review identifier
- Reviewer name
- Star rating
- Review title
- Review text
- Review date
- Verified-buyer indicator
- Helpful and not-helpful vote counts
- Product name and product URL

The collection workflow was implemented using a standalone Python scraper script and a Jupyter notebook. The standalone script was used to avoid browser automation event-loop issues encountered when running Playwright directly inside Windows JupyterLab.

> **Data-use note:** Public visibility does not automatically grant permission to redistribute review content. Check the source website's current terms and applicable rules before publishing raw review text. Avoid publishing reviewer-identifying information. The repository should contain only data that is permitted to be shared, preferably sanitized or aggregated.

## Dataset Overview

The collected dataset contains **162 unique review records**.

### Raw dataset fields

| Field | Description | Initial use |
|---|---|---|
| `review_id` | Review identifier from the source | Record identification and duplicate checks |
| `product_name` | Product name | Product context |
| `reviewer_name` | Displayed reviewer name | Not required for aggregate analysis; treat as personal data |
| `rating` | Star rating | Rating distribution and comparison |
| `review_title` | Short review heading | Text analysis |
| `review_text` | Written review | Main NLP input |
| `review_date` | Review date | Review timeline analysis |
| `verified_buyer` | Verified-buyer indicator | Descriptive comparison |
| `helpful_votes` | Helpful vote count | Optional review usefulness analysis |
| `not_helpful_votes` | Not-helpful vote count | Optional review usefulness analysis |
| `product_url` | Product page URL | Source traceability |

The raw file is retained separately from the cleaned analysis dataset. Some columns, such as reviewer name and product URL, are not needed for the main NLP analysis. Do not publish reviewer-identifying data.

### Collection checks

| Check | Observed result |
|---|---:|
| Collected review records | 162 |
| Unique review identifiers | 162 |
| Duplicate review identifiers | 0 |

### Initial rating distribution

| Star rating | Review count | Percentage |
|---:|---:|---:|
| 1 star | 4 | 2.47% |
| 2 stars | 0 | 0.00% |
| 3 stars | 0 | 0.00% |
| 4 stars | 26 | 16.05% |
| 5 stars | 132 | 81.48% |
| **Total** | **162** | **100.00%** |

The collected sample is strongly concentrated in 5-star reviews. This is a description of the collected sample and may reflect review-platform and self-selection effects.

## Environment Setup: Conda and JupyterLab

The project was developed in a separate Conda environment named `skincare-ds`.

### 1. Activate the environment

Open **Anaconda Prompt** and run:

```bash
conda activate skincare-ds
```

### 2. Install the project packages

Install the packages required for the analysis in the active environment:

```bash
conda install -y pandas numpy matplotlib seaborn scikit-learn
python -m pip install transformers torch sentencepiece protobuf tiktoken
```

For browser-based review collection, install Playwright and the browser dependencies if they are not already available:

```bash
python -m pip install playwright beautifulsoup4
python -m playwright install
```

Package versions can change. Use the versions installed in the active environment and verify imports before running the notebooks.

### 3. Register the environment as a Jupyter kernel

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name skincare-ds --display-name "Python (skincare-ds)"
```

### 4. Launch JupyterLab from the project directory

Change to the project folder (update the path if it is stored elsewhere):

```bash
cd /d "C:\Users\muppu\OneDrive\Desktop\SkinCare_Review"
jupyter lab
```

In JupyterLab, select:

**Kernel → Change Kernel → Python (skincare-ds)**

### 5. Verify the environment

Run this in a notebook cell:

```python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import transformers
import torch

print("Python executable:", sys.executable)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Transformers:", transformers.__version__)
print("PyTorch:", torch.__version__)
print("Environment is ready!")
```

The Python executable should point to the `skincare-ds` Conda environment.

> **Environment note:** The sentiment notebook uses a pretrained Hugging Face model. Its first run requires an internet connection to download the model files. The model is cached locally after download. A Hugging Face token is not required for downloading this public model under normal conditions.

## Week 3 Workflow

### 1. Data Collection

The product review section was accessed and review records were extracted. The final collection contained 162 records with 162 unique review identifiers.

### 2. Text Preprocessing

The preprocessing notebook:

- Loaded the raw review CSV.
- Created a working copy of the data.
- Handled missing review titles and review text.
- Combined review titles and review bodies into a single text field.
- Prepared cleaned text while retaining English and Romanized Hinglish words.
- Exported the preprocessed dataset.

The combined review field was used as the main text input for subsequent NLP analysis.

### 3. Exploratory Data Analysis

The EDA notebook examined:

- Dataset dimensions, columns, and data types.
- Missing values and duplicate review identifiers.
- Star-rating distribution.
- Review length in words.
- Review-date trends.
- Verified-buyer distribution.
- Review length by star rating.
- Frequent words and phrases.

The EDA stage generated visualizations and an exported analysis dataset.

### 4. Sentiment Analysis

A multilingual pretrained Transformer model was used:

**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`

The model produces one of three sentiment classes:

- Positive
- Neutral
- Negative

The notebook stores the predicted class and the model's confidence score for each review. It also creates sentiment counts and percentages, visualizes the distribution, compares predicted sentiment with star ratings, and displays review examples for inspection.

**Interpretation note:** Model confidence is not model accuracy. The comparison with star ratings is exploratory and does not constitute a validated evaluation. Romanized Hinglish and code-mixed text may be misclassified. Predictions should be manually inspected before making conclusions.

### 5. TF-IDF and Keyword Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) was applied using Scikit-learn's `TfidfVectorizer`.

The notebook uses:

- Unigrams and bigrams (`ngram_range=(1, 2)`).
- A minimum document frequency of 2.
- A maximum vocabulary size of 5,000 features.
- Mean TF-IDF scores to rank terms across the review collection.
- Separate keyword rankings for predicted sentiment groups when sufficient reviews are available.
- Up to three high-scoring terms or phrases for each review.

TF-IDF highlights terms that are informative within this dataset. It does not independently understand context, identify causal factors, or validate customer-feedback themes.

## Week 3 Outputs

The notebooks generate the following datasets:

| Output file | Description |
|---|---|
| `thedermaco_all_reviews.csv` | Raw collected review data |
| `TheDermaCo_Preprocessed_Reviews.csv` | Cleaned review dataset with combined text |
| `TheDermaCo_EDA_Analysis.csv` | Dataset used/exported for EDA |
| `TheDermaCo_Sentiment_Analysis.csv` | Reviews with predicted sentiment and confidence |
| `TheDermaCo_TFIDF_Top_Keywords.csv` | Overall TF-IDF term and phrase ranking |
| `TheDermaCo_TFIDF_Review_Keywords.csv` | Review-level top TF-IDF terms |
| `TheDermaCo_TFIDF_Sentiment_Keywords.csv` | TF-IDF term rankings by predicted sentiment group |

The raw and processed CSVs may contain review text or reviewer information. Do not publish them unless redistribution is permitted and personal information has been removed. Prefer publishing aggregate tables, charts, and sanitized examples.

## Repository Structure

The following structure is recommended for GitHub. Update filenames if your local notebook names differ.

```text
skincare-product-review-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 00_Data_Collection.ipynb
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Sentiment_Analysis.ipynb
│   └── 04_TFIDF_Keyword_Extraction.ipynb
│
├── scripts/
│   └── 00_scrape_all_reviews.py
│
├── outputs/
│   ├── TheDermaCo_TFIDF_Top_Keywords.csv
│   ├── TheDermaCo_TFIDF_Review_Keywords.csv
│   └── TheDermaCo_TFIDF_Sentiment_Keywords.csv
│
└── reports/
    └── Week3_Skincare_Data_Sourcing_and_Initial_Analysis_Report_Gnaneswar.docx
```

Do not commit virtual environments, Hugging Face model caches, downloaded model weights, temporary files, credentials, or raw review data that you are not permitted to redistribute.

## Initial Findings

The collected review sample contains 162 unique review identifiers. The rating distribution is dominated by 5-star reviews (81.48%), followed by 4-star reviews (16.05%). Four reviews are rated 1 star, and no 2-star or 3-star reviews were present in this collected sample.

These rating patterns describe the collected records only. Sentiment predictions and TF-IDF rankings should be interpreted after inspecting their actual notebook outputs and representative review examples. No sentiment percentages or keyword findings are stated here because those results must be taken from the executed analysis outputs.

## Risks and Limitations

| Challenge | Why it matters | Response / interpretation |
|---|---|---|
| Single-product sample | Findings may not represent other products | Limit conclusions to the selected product and collected reviews |
| Review-platform selection bias | People who post reviews may differ from all buyers | Avoid generalizing to all customers |
| Small dataset | 162 reviews provide limited coverage | Treat results as exploratory |
| Highly positive rating distribution | Negative examples are relatively few | Report group sizes alongside comparisons |
| Romanized Hinglish | The model may not fully understand code-mixed language | Manually inspect representative predictions |
| Pretrained sentiment model | Predictions may not match skincare-specific meanings | Treat outputs as estimates, not ground truth |
| TF-IDF limitations | Important terms are not necessarily meaningful themes | Read terms in the context of the original reviews |
| Reviewer information | Names and other identifiers may be personal data | Exclude identifying fields from public releases |
| Source terms and permissions | Public access does not guarantee unrestricted reuse | Check current website terms before publishing collected content |

## Tools and Technologies

- **Python** — programming language
- **Pandas** — data loading, cleaning, and analysis
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — TF-IDF vectorization
- **Hugging Face Transformers** — pretrained sentiment model
- **PyTorch** — model execution
- **SentencePiece** — tokenizer support
- **Playwright** — browser automation for review collection
- **JupyterLab** — interactive notebook environment
- **Conda** — Python environment management

## Current Status

**Week 3 — Data collection, preprocessing, EDA, sentiment analysis, and TF-IDF workflow completed.**

Completed in this stage:

- Collected 162 review records with 162 unique review identifiers.
- Prepared review text for NLP analysis.
- Created an EDA workflow and exported its dataset.
- Applied a multilingual pretrained sentiment model.
- Generated sentiment labels and confidence scores.
- Compared predicted sentiment with star ratings.
- Applied TF-IDF to extract unigrams and bigrams.
- Generated overall and sentiment-group keyword outputs.
- Exported CSV results for further interpretation.

The analysis findings should be finalized using the actual executed notebook outputs and manually inspected review examples.

## Next Steps

1. Review the top TF-IDF terms and phrases in their original review context.
2. Manually inspect a sample of positive, neutral, and negative sentiment predictions, including Romanized Hinglish reviews.
3. Summarize recurring customer feedback themes supported by the review text.
4. Prepare the Week 3 report with methodology, visualizations, findings, and limitations.
5. If the project is extended, collect reviews from additional skincare products using consistent collection and inclusion rules, then compare products cautiously.

## References

- The Derma Co product page: https://thedermaco.com/products/2-sali-cinamide-anti-acne-face-wash-with-2-salicylic-acid-2-niacinamide-150-ml-pack-of-2
- Hugging Face model: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
- Scikit-learn TF-IDF documentation: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
- Pandas documentation: https://pandas.pydata.org/docs/
- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/stable/
- Seaborn documentation: https://seaborn.pydata.org/
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/
- JupyterLab documentation: https://jupyterlab.readthedocs.io/
