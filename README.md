# Week 3 — Amazon Beauty Reviews: Data Cleaning and Preprocessing

This project continues the **Review-Informed Skincare Product Discovery Using NLP and Data Analytics** work from Week 2.

The notebook and Python script use the Amazon Beauty Reviews dataset hosted on Hugging Face. The first 10,000 records are used as a working sample so the data can be inspected and cleaned in a manageable way.

The purpose of Week 3 is to identify data-quality issues, apply suitable cleaning methods, document each decision, and prepare the review data for later analysis and NLP.

---

## Dataset

**Dataset:** Amazon Beauty Reviews  
**Source:** Hugging Face  
**Dataset link:** https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset

The dataset contains customer review information such as ratings, review titles, review text, product identifiers, timestamps, helpful votes, and purchase-verification information. The actual columns are inspected in the notebook before cleaning.

**Working sample:** First 10,000 records.

**Important scope note:** The source covers the broader *All Beauty* category. It is not a verified skincare-only dataset. Product metadata must be checked and skincare products must be identified before making skincare-specific conclusions.

---

## Tools and Libraries Used

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading, inspecting, cleaning, filtering, and exporting review data |
| NumPy | Numerical operations and conditional checks |
| Regular Expressions (`re`) | Removing HTML tags and normalizing whitespace |
| Matplotlib | Visualizing rating distribution and review-length outliers |
| JupyterLab | Running and documenting the preprocessing steps |
| fsspec | Reading the dataset through its Hugging Face `hf://` path |
| huggingface_hub | Supporting access to Hugging Face-hosted datasets |

---

## Project Structure

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

Only add files that have actually been generated and checked. The raw source CSV is not included in this suggested public repository structure.

---

## Week 3 — Data Cleaning and Preprocessing Steps

### 1. Load the Amazon Beauty Reviews sample

The dataset is loaded from the Hugging Face CSV location. The `nrows=10000` setting limits the working sample to the first 10,000 records. The original DataFrame is preserved as `df_raw`, and a separate copy is used for cleaning.

### 2. Inspect the dataset

The notebook checks the dataset shape, column names, data types, non-missing counts, missing values, and descriptive statistics. This provides an initial view of the data before any cleaning decisions are applied.

### 3. Standardize column names

Column names are normalized to lowercase `snake_case`. This makes the column references consistent and reduces errors caused by spaces, capitalization, or special characters.

### 4. Handle missing values

Missing values are checked field by field rather than deleting every row that contains a missing value.

- Missing review titles are filled with an empty string.
- Missing review bodies are retained in the cleaned master dataset and excluded from the NLP-ready subset when the combined review text is empty.
- Missing image entries are labelled `No image uploaded` when an `images` column is present.
- Missing or invalid ratings are flagged; no rating is invented.

### 5. Detect duplicate records

The pipeline checks exact duplicate rows and checks duplicate review identifiers only if a `review_id` field exists. The `user_id` field is not treated as a unique review identifier because one reviewer may submit multiple reviews.

Confirmed duplicate rows are removed and the number removed is recorded in the cleaning log.

### 6. Validate rating values

Ratings are converted to numeric values and checked against the expected range of 1 to 5. Missing, non-numeric, or out-of-range ratings are flagged for review and excluded from rating-dependent analysis.

### 7. Validate review timestamps

The timestamp field is converted into a UTC datetime column. Values that cannot be parsed are flagged. The original timestamp field is retained so that the source value is not lost.

### 8. Clean and normalize review text

The text-cleaning function removes HTML tags and normalizes repeated whitespace. It creates cleaned title and review-body columns, then combines them into a single text field for later NLP work.

Punctuation, numbers, negation terms, and non-English characters are preserved at this stage because they may contain useful information for sentiment analysis.

### 9. Check short reviews and outliers

The pipeline calculates review word counts and character counts. Very short reviews and unusually long reviews are flagged for inspection.

The Interquartile Range (IQR) method is used to identify review-length outliers. These records are not automatically deleted because long reviews may contain useful product experiences and detailed feedback.

### 10. Inspect inconsistent and constant fields

The notebook inspects values in fields such as `verified_purchase`, `asin`, and `parent_asin`, when available. Constant fields are reported for interpretation rather than automatically removed. A constant value in the working sample does not necessarily mean the field is incorrect in the full source.

### 11. Create analysis-ready datasets

The pipeline creates separate outputs:

- **Cleaned master dataset:** retains the cleaned records and quality flags.
- **NLP-ready dataset:** contains records with non-empty combined review text.
- **Valid-rating subset:** contains NLP-ready records with valid ratings.

Keeping these datasets separate makes the inclusion criteria clear and avoids unnecessary deletion from the cleaned master.

### 12. Generate the quality audit

The quality audit reports raw and cleaned dataset measurements separately. It includes row and column counts, missing values, duplicate rows, invalid ratings, invalid timestamps, empty review text, flagged outliers, and the number of records in the analysis-ready subsets.

### 13. Create visualizations

The project includes the following visual checks:

- **Rating distribution:** displays the number of reviews for each star rating.
- **Review word-count boxplot:** shows the spread of review lengths and potential outliers.
- **Preprocessing flowchart:** summarizes the sequence of data-cleaning steps.

The charts describe the working sample only and should not be treated as proof of market-wide customer preferences.

### 14. Export the cleaned outputs

The pipeline exports the cleaned master dataset, NLP-ready dataset, quality audit, and cleaning log as CSV files. It also saves the rating distribution and review word-count charts.

---

## How to Run the Project

### 1. Install the required libraries

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### 2. Run the Python pipeline

```bash
python pipeline.py
```

By default, the script attempts to load the dataset using the Hugging Face `hf://` path.

If you have downloaded the CSV locally, open `pipeline.py` and set `LOCAL_CSV` to the downloaded file path:

```python
LOCAL_CSV = "amazon_beauty_reviews_dataset.csv"
```

Then run the script again.

### 3. Run the Jupyter notebook

Open the notebook in JupyterLab and run the cells from top to bottom. For a clean execution, restart the kernel before selecting **Run All Cells**.

---

## Output Files

| File | Description |
|---|---|
| `amazon_beauty_cleaned_master_10k.csv` | Cleaned master dataset with derived fields and quality flags |
| `amazon_beauty_nlp_ready_10k.csv` | Reviews with non-empty combined text |
| `amazon_beauty_quality_audit_10k.csv` | Summary of raw and cleaned data-quality checks |
| `amazon_beauty_cleaning_log_10k.csv` | Steps performed, issues checked, actions, and reasons |
| `rating_distribution.png` | Bar chart of review ratings |
| `review_word_count_boxplot.png` | Boxplot of review word counts |

The Python script saves its generated files inside the `week3_outputs/` folder.

---

## Important Notes and Limitations

1. The working sample contains the first 10,000 rows, not the complete dataset.
2. The source covers the broader All Beauty category and is not exclusively skincare.
3. Skincare-specific analysis requires a documented product-category filtering method.
4. Customer reviews are self-selected and may contain platform, brand, and reviewer biases.
5. Review text can include spelling variation, slang, sarcasm, and multilingual expressions.
6. Outliers are flagged for inspection and are not automatically removed.
7. Reviewer identifiers should not be published unless they are necessary and permitted.
8. Check the dataset's current licence and terms before redistributing the data or processed records.

---

## References

- Hugging Face — Amazon Beauty Reviews: https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset
- Pandas documentation: https://pandas.pydata.org/docs/
- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/stable/

---

**Week 3 — Data Cleaning and Preprocessing Pipeline Development**  
**Muppur Gnaneswar**
