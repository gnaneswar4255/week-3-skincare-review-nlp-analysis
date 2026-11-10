# Review-Informed Skincare Product Discovery Using NLP and Data Analytics

![Domain](https://img.shields.io/badge/Domain-Beauty%20%26%20Wellness-coral)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Stage](https://img.shields.io/badge/Stage-Week%203%20Data%20Cleaning-success)
![Status](https://img.shields.io/badge/Status-Preprocessing%20Pipeline%20Developed-brightgreen)

## Project Overview

This project explores how Python-based data analysis and Natural Language Processing (NLP) can be used to examine customer reviews and star ratings for beauty and skincare-related products.

The goal is to identify patterns in customer feedback, understand rating distributions, and prepare review data for later review-theme analysis. The project is being developed as part of the **YuvaIntern Data Science with Python Analyst Internship**.

The Week 3 work moves the project from Week 2 dataset inspection and initial exploratory data analysis into a documented data-cleaning and preprocessing pipeline. It covers data profiling, missing-value handling, duplicate detection, rating and timestamp validation, text normalization, outlier assessment, quality auditing, and export of analysis-ready datasets.

> **Scope note:** The selected source contains broad beauty-product reviews. It has not been verified as a skincare-only dataset. Product metadata must be checked and a defensible skincare subset selected before making skincare-category claims.

## Business Problem

Average star ratings compress different customer experiences into one number. Written reviews may explain the reasons behind those ratings, but manually reading large volumes of feedback can be slow and inconsistent.

**Problem question:** How can product ratings and written reviews be cleaned, organized, and prepared for reliable analysis of customer feedback across skincare-related product categories?

## Week 3 Objectives

1. Preserve the original dataset and create a separate working copy for cleaning.
2. Profile the working sample and assess its structure, data types, missing values, and duplicate records.
3. Develop a step-by-step data-cleaning and preprocessing pipeline using Python.
4. Handle missing titles, review bodies, and image metadata using field-specific decisions.
5. Detect exact duplicate rows and check review identifiers where a stable review ID is available.
6. Validate rating values and parse review timestamps into a consistent datetime format.
7. Normalize review text conservatively while preserving potentially meaningful sentiment information.
8. Calculate review word counts and flag unusually short or long reviews for inspection.
9. Create a quality audit and cleaning log to document the transformations.
10. Export a cleaned master dataset and an NLP-ready subset for later analysis.
11. Explain the selected Python libraries, cleaning decisions, limitations, and reproducibility steps.
12. Document the complete preprocessing workflow with a flowchart and diagnostic visualizations.

## Data Source

### Selected working dataset

**Amazon Beauty Reviews dataset — hosted on Hugging Face**

Dataset page: https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset

The dataset is accessed as a CSV file using the Hugging Face filesystem URI:

```python
dataset_path = (
    "hf://datasets/jhan21/amazon-beauty-reviews-dataset/"
    "amazon_beauty_reviews_dataset.csv"
)
```

The Week 3 pipeline loads the first **10,000 records** as a manageable working sample, consistent with the earlier Week 2 exploration.

### Alternative source considered

**Amazon Reviews'23 — All_Beauty category, McAuley Lab**

- Dataset overview: https://amazon-reviews-2023.github.io/main.html
- Dataset card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

This is another large-scale review resource. Its All_Beauty category is also broader than skincare, so product metadata must be inspected before treating records as skincare reviews.

### Data-use note

Technical accessibility does not automatically establish unrestricted reuse. Review the dataset card, provenance, licence, and current terms before redistribution or publishing raw data. The full raw dataset is not included in this repository.

## Environment Setup: Conda and JupyterLab

The project was developed in a separate Conda environment to keep its Python packages isolated from other projects.

### 1. Create the Conda environment

Open **Anaconda Prompt** and run:

```bash
conda create --solver=classic -n skincare-ds python=3.12 pandas numpy matplotlib jupyterlab ipykernel -y
```

### 2. Activate the environment

```bash
conda activate skincare-ds
```

The prompt should begin with:

```text
(skincare-ds)
```

### 3. Install the project dependencies

From the repository root, run:

```bash
python -m pip install -r requirements.txt
```

The requirements include Pandas, NumPy, Matplotlib, Hugging Face filesystem support, and Jupyter tools.

### 4. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name skincare-ds --display-name "Python (skincare-ds)"
```

### 5. Launch JupyterLab from the project folder

Change to the repository directory (update the path if the folder is stored elsewhere):

```bash
cd /d "C:\path\to\week-3-amazon-beauty-data-cleaning"
jupyter lab
```

In JupyterLab, open the Week 3 notebook and select:

**Kernel → Change Kernel → Python (skincare-ds)**

### 6. Verify the active environment

Run this in a notebook cell:

```python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fsspec
import huggingface_hub

print("Python executable:", sys.executable)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Environment is ready!")
```

The Python executable should point to the `skincare-ds` environment, for example:

```text
C:\Users\muppu\anaconda3\envs\skincare-ds\python.exe
```

Package versions may change when the environment is recreated. Check the verification output rather than assuming versions will always match.

## Loading the Dataset

The notebook and standalone script use the Hugging Face filesystem URI:

```python
import pandas as pd

dataset_path = (
    "hf://datasets/jhan21/amazon-beauty-reviews-dataset/"
    "amazon_beauty_reviews_dataset.csv"
)

df_raw = pd.read_csv(dataset_path, nrows=10000)
```

The `nrows=10000` argument limits the initial load to 10,000 records.

If the `hf://` URI cannot be read, verify that `fsspec` and `huggingface_hub` are installed in the active environment. Restart the kernel after installing packages and retry.

Alternatively, download the CSV from the dataset page and set `LOCAL_CSV` in `pipeline.py` to the local CSV path.

## Dataset Structure

The Week 3 working sample contains **10,000 raw rows and 10 columns** before cleaning. The pipeline inspects the actual column names and data types rather than assuming that every source version has an identical schema.

The documented source fields include:

| Field | Initial use in the project |
|---|---|
| `rating` | Star-rating analysis and validation against the 1–5 range |
| `title` | Review heading and text analysis |
| `text` | Written review content and NLP |
| `images` | Image metadata; missing entries are labelled `No image uploaded` when this field exists |
| `asin` | Product identifier |
| `parent_asin` | Parent product identifier |
| `user_id` | Reviewer identifier; not a unique review identifier |
| `timestamp` | Review timestamp; requires parsing and validation |
| `helpful_vote` | Helpful-vote count, if relevant |
| `verified_purchase` | Purchase-verification indicator |

The source schema is checked in the notebook and script. Missing expected fields are reported rather than silently created.

## Week 3 Data Cleaning and Preprocessing Pipeline

The pipeline preserves the raw DataFrame as `df_raw` and applies transformations to a separate working copy, `df`. This allows raw-versus-cleaned comparisons and protects the original loaded sample.

### 1. Data profiling

The initial profiling stage inspects:

- Number of rows and columns
- Column names and data types
- Missing-value counts
- Exact duplicate rows
- Descriptive statistics

### 2. Column-name standardization

Column names are stripped of surrounding whitespace, converted to lowercase, and normalized to `snake_case`. This creates consistent names for later code.

### 3. Missing-value handling

Missing values are handled according to the role of each field:

| Field / issue | Cleaning decision | Reason |
|---|---|---|
| Missing review title | Fill with an empty string | The review body may still contain useful information |
| Missing or blank review body | Retain in the cleaned master; exclude from NLP-ready data if combined text is empty | Missing text cannot support text analysis |
| Missing image metadata | Replace empty or null representations with `No image uploaded`, when an `images` column exists | Makes the absence of an uploaded image explicit |
| Missing rating | Flag; do not invent a rating | Rating-based analysis requires a valid score |
| Missing timestamp | Flag if it cannot be parsed | Time-based analysis requires a valid datetime |

The pipeline distinguishes a missing review body from an empty combined review. A review may have a title even when its body is missing.

### 4. Duplicate detection

The pipeline checks exact duplicate rows and checks a stable review identifier only if a `review_id` column exists.

The source schema may not provide a review-level ID. **`user_id` is a reviewer identifier, not a unique review ID**, so it is not used to remove duplicate reviews.

### 5. Rating validation

Ratings are converted to numeric values and checked against the expected 1–5 range. Missing, non-numeric, or out-of-range values are flagged. The pipeline does not replace invalid ratings with guessed values.

### 6. Timestamp parsing

The original timestamp field is preserved. A separate UTC datetime field is created, and unparseable timestamps are flagged for review.

### 7. Review-text normalization

The text-cleaning function:

- Removes HTML tags
- Normalizes repeated whitespace
- Trims leading and trailing spaces
- Creates cleaned title and body fields
- Combines cleaned title and review body for downstream text analysis

Punctuation, numbers, negation words, and non-English characters are not aggressively removed at this stage because they may contain useful product or sentiment information. More specific tokenization can be designed later for the selected NLP method.

### 8. Review-length assessment and outlier handling

The pipeline calculates word and character counts. It uses the Interquartile Range (IQR) rule to flag unusually long or short reviews.

**Outliers are flagged, not automatically deleted.** A long review may contain detailed product feedback, so length alone is not sufficient evidence that a record is invalid.

### 9. Consistency checks

The pipeline inspects distinct values and constant fields such as purchase-verification and product metadata. Constant fields are documented for interpretation; they are not automatically removed because a constant value may reflect the working sample or source limitations.

### 10. Analysis-ready datasets

Two analysis subsets are created from the cleaned master:

- **NLP-ready dataset:** records with non-empty combined cleaned review text.
- **Valid-rating NLP dataset:** NLP-ready records that also have a valid rating.

The cleaned master is retained separately so that filtering for one analysis does not discard records needed for another.

## Quality Audit: Observed Results

The uploaded execution audit for the 10,000-row working sample reported the following results:

| Quality measure | Observed result |
|---|---:|
| Raw row count | 10,000 |
| Raw column count | 10 |
| Raw missing cells | 2 |
| Raw exact duplicate rows | 13 |
| Cleaned master row count | 9,987 |
| Cleaned master column count | 21 |
| Missing titles after handling | 0 |
| Empty cleaned review bodies | 2 |
| Invalid or missing ratings | 0 |
| Invalid timestamps | 0 |
| Empty combined review text | 1 |
| Review-length outliers flagged | 728 |
| NLP-ready rows | 9,986 |
| Valid-rating NLP rows | 9,986 |

The 13 exact duplicate rows were removed from the cleaned master. The 728 review-length outliers were retained and flagged for inspection.

The difference between the two empty-body records and the one empty combined review is possible because a record may have a title even when its review body is empty. NLP eligibility is based on combined cleaned text.

These results describe the uploaded 10,000-row working sample and should be regenerated if the dataset, sample size, or cleaning logic changes.

## Visualizations

The pipeline creates diagnostic visualizations to support the quality assessment.

### 1. Rating distribution

A Matplotlib bar chart displays the number of reviews for each rating. It helps identify rating imbalance that should be considered before rating-group comparisons or sentiment modelling.

### 2. Review word-count distribution

A boxplot displays the spread of review word counts and highlights observations beyond the whiskers. These points are treated as potential outliers for inspection, not automatically invalid records.

### 3. Preprocessing flowchart

The flowchart summarizes the sequence from raw-data loading and profiling through cleaning, validation, text normalization, outlier flagging, auditing, and export.

## Running the Pipeline

### Option A: Run the standalone Python script

From the repository root:

```bash
python pipeline.py
```

The script attempts to load the dataset from Hugging Face by default.

If you downloaded the CSV locally, open `pipeline.py` and set:

```python
LOCAL_CSV = "amazon_beauty_reviews_dataset.csv"
```

Then run the script again.

The script creates a `week3_outputs/` directory in the current working directory containing the generated outputs.

### Option B: Run the Jupyter notebook

1. Open JupyterLab from the `skincare-ds` environment.
2. Open the Week 3 notebook in the `notebooks/` folder.
3. Select the `Python (skincare-ds)` kernel.
4. Choose **Kernel → Restart Kernel and Run All Cells**.
5. Confirm that all cells finish without errors.
6. Check the generated CSV files and compare their audit values with the report.

## Repository Contents

Update this list to match the files actually committed to GitHub.

```text
week-3-amazon-beauty-data-cleaning/
│
├── README.md
├── requirements.txt
├── pipeline.py
├── .gitignore
│
├── data/
│   └── processed/
│       ├── amazon_beauty_cleaned_master_10k.csv
│       ├── amazon_beauty_nlp_ready_10k.csv
│       ├── amazon_beauty_quality_audit_10k.csv
│       └── amazon_beauty_cleaning_log_10k.csv
│
├── notebooks/
│   └── Week3_Amazon_Beauty_Data_Cleaning_Pipeline.ipynb
│
├── reports/
│   └── Week3_Amazon_Beauty_Data_Cleaning_Report.pdf

```

Do not commit the full raw dataset, private credentials, Hugging Face access tokens, or unnecessary reviewer identifiers. Review processed CSV columns before publishing them.

## Risks and Limitations

| Challenge | Why it matters | Planned response |
|---|---|---|
| Broad beauty dataset | Includes products that may not be skincare | Inspect product metadata and document skincare inclusion rules |
| First 10,000 rows only | May not represent the complete dataset | Treat results as sample-level observations |
| Missing review text | Cannot support text analysis | Retain in master; exclude empty combined text from NLP subset |
| Duplicate records | Can affect review counts | Check exact rows and stable review IDs; do not use reviewer ID as review ID |
| Rating imbalance | Can affect comparisons and modelling | Report class distribution and evaluate appropriate methods later |
| Long or short reviews | Length alone does not prove a review is invalid | Flag for inspection rather than automatically deleting |
| Mixed language, slang, and sarcasm | Basic NLP may misinterpret review meaning | Preserve original text and inspect representative examples |
| Self-selection and platform bias | Reviewers may not represent all customers | Limit conclusions to the available review sample |
| Licence and provenance | Public access does not guarantee unrestricted reuse | Check current source terms before redistribution |

## Link to Week 1 Research Questions and Hypotheses

| Week 1 question / hypothesis | Week 3 progress | Remaining work |
|---|---|---|
| RQ1: Common review themes | Text fields cleaned and NLP-ready subset created | Extract and validate frequent words, phrases, and themes |
| H1: Themes differ across skincare categories | General beauty review data cleaned | Validate skincare products and categories before comparison |
| H2: Lower- and higher-rated reviews differ | Ratings validated and joined with cleaned review text | Compare review themes across rating groups |
| H3: Simple sentiment may disagree with manual interpretation | Text normalization prepared | Compare sentiment predictions with a manually reviewed sample if sentiment modelling is used |
| H4: Ratings plus review text provide richer summaries | Cleaned text and ratings are available | Evaluate review-theme summaries alongside ratings |

**The Week 3 cleaning work does not confirm or reject these hypotheses.** Category-level and NLP analysis remain future work.

## Current Status

**Week 3 — Data cleaning and preprocessing pipeline developed.**

Completed in this stage:

- Preserved the raw 10,000-row sample and created a separate working copy.
- Inspected dataset structure, types, missing values, and duplicates.
- Standardized column names.
- Handled missing review titles and image metadata where available.
- Detected and removed 13 exact duplicate rows.
- Validated rating values and timestamps.
- Normalized review text using conservative cleaning.
- Calculated review word counts and flagged 728 length outliers without deleting them.
- Created cleaned master and NLP-ready datasets.
- Exported a quality audit and cleaning log.
- Prepared rating and review-length diagnostic visualizations.
- Documented the workflow, decisions, tools, limitations, and reproducibility steps.

**Not completed in Week 3:** validated skincare-only product selection, category-level hypothesis testing, review-theme extraction, sentiment modelling, or confirmation of the Week 1 hypotheses.

## Next Steps

The next stage will focus on validating skincare-related products using product metadata and documenting the category-selection rules. Once a defensible skincare subset is available, the project can proceed to exploratory text analysis, keyword and review-theme extraction, rating-group comparisons, and sentiment analysis where appropriate.

## References

- Hugging Face dataset: https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset
- Amazon Reviews'23 documentation: https://amazon-reviews-2023.github.io/main.html
- Amazon Reviews'23 dataset card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Pandas documentation: https://pandas.pydata.org/docs/
- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/stable/
- JupyterLab documentation: https://jupyterlab.readthedocs.io/
- Hugging Face Hub documentation: https://huggingface.co/docs/huggingface_hub/
