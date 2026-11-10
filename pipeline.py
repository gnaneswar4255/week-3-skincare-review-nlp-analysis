"""
WEEK 3: AMAZON BEAUTY REVIEWS DATA CLEANING PIPELINE
Project: Review-Informed Skincare Product Discovery Using NLP and Data Analytics

Run:
    python pipeline.py

Optional:
    1. Put the downloaded CSV in this folder and set LOCAL_CSV below.
    2. Otherwise, the script attempts to read the Hugging Face hf:// URI.
       Install fsspec and huggingface_hub if needed.

Outputs:
    amazon_beauty_cleaned_master_10k.csv
    amazon_beauty_nlp_ready_10k.csv
    amazon_beauty_quality_audit_10k.csv
    amazon_beauty_cleaning_log_10k.csv
    rating_distribution.png
    review_word_count_boxplot.png
"""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------

DATASET_URI = (
    "hf://datasets/jhan21/amazon-beauty-reviews-dataset/"
    "amazon_beauty_reviews_dataset.csv"
)

# If you downloaded the CSV, enter its filename or full path here.
# Example: LOCAL_CSV = "amazon_beauty_reviews_dataset.csv"
LOCAL_CSV = None

SAMPLE_SIZE = 10_000
OUTPUT_DIR = Path("week3_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 120)


# ---------------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------------

def load_data():
    """Load the first SAMPLE_SIZE records from local CSV or Hugging Face."""
    if LOCAL_CSV:
        csv_path = Path(LOCAL_CSV)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Local CSV not found: {csv_path.resolve()}\n"
                "Update LOCAL_CSV with the correct file path."
            )
        print(f"Loading local dataset: {csv_path.resolve()}")
        data = pd.read_csv(csv_path, nrows=SAMPLE_SIZE)
    else:
        print("Loading dataset from Hugging Face...")
        try:
            data = pd.read_csv(DATASET_URI, nrows=SAMPLE_SIZE)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the Hugging Face dataset.\n"
                "Download the CSV from:\n"
                "https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset\n"
                "Then set LOCAL_CSV to the downloaded CSV path.\n"
                "For hf:// support, install: pip install fsspec huggingface_hub"
            ) from exc

    if data.empty:
        raise ValueError("The loaded dataset contains no rows.")

    return data


df_raw = load_data()
print("Raw sample shape:", df_raw.shape)
print("\nOriginal columns:")
print(df_raw.columns.tolist())


# ---------------------------------------------------------------------
# 3. INITIAL PROFILE
# ---------------------------------------------------------------------

print("\nInitial data types:")
print(df_raw.dtypes)

print("\nMissing values before cleaning:")
print(df_raw.isna().sum())

print("\nExact duplicate rows before cleaning:", int(df_raw.duplicated().sum()))


# ---------------------------------------------------------------------
# 4. PRESERVE RAW DATA AND CREATE WORKING COPY
# ---------------------------------------------------------------------

df = df_raw.copy(deep=True)
cleaning_log = []


def log_step(step, issue, action, rationale):
    cleaning_log.append({
        "step": step,
        "issue_or_check": issue,
        "action_taken": action,
        "rationale": rationale
    })


# ---------------------------------------------------------------------
# 5. STANDARDIZE COLUMN NAMES
# ---------------------------------------------------------------------

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

expected_columns = [
    "rating", "title", "text", "images", "asin", "parent_asin",
    "user_id", "timestamp", "helpful_vote", "verified_purchase"
]
missing_columns = [col for col in expected_columns if col not in df.columns]

print("\nExpected columns missing from loaded file:", missing_columns)

log_step(
    "Column standardization",
    "Column labels may contain inconsistent capitalization or spacing",
    "Normalized labels to lowercase snake_case",
    "Consistent names reduce coding errors and make the pipeline reusable."
)


# ---------------------------------------------------------------------
# 6. HANDLE MISSING VALUES
# ---------------------------------------------------------------------

for col in ["title", "text"]:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)

missing_before_handling = df.isna().sum()

if "title" in df.columns:
    df["title"] = df["title"].fillna("")

# If the source contains an images field, replace missing/empty image
# representations with the requested readable label.
if "images" in df.columns:
    image_text = df["images"].astype("string").str.strip().str.lower()
    no_image_values = {"", "[]", "nan", "none", "null", "<na>"}

    images_missing = (
        df["images"].isna()
        | image_text.isin(no_image_values)
    )

    no_image_count = int(images_missing.sum())
    df.loc[images_missing, "images"] = "No image uploaded"

    print(f"\nRecords labelled 'No image uploaded': {no_image_count}")
else:
    no_image_count = None
    print("\nNo images column is present in this dataset.")

log_step(
    "Missing-value handling",
    "Missing titles, review bodies, and image metadata",
    "Filled missing titles with empty strings; retained missing review bodies for audit; "
    "replaced missing/empty image entries with 'No image uploaded' when images column exists",
    "A missing title does not make an otherwise complete review unusable. "
    "A missing review body cannot support text analysis. The image label makes missing image data explicit."
)

print("\nMissing values before handling:")
print(missing_before_handling)
print("\nMissing values after title/image handling:")
print(df.isna().sum())


# ---------------------------------------------------------------------
# 7. DUPLICATE DETECTION AND HANDLING
# ---------------------------------------------------------------------

# This source schema does not provide review_id in the documented field list.
# user_id identifies a reviewer, not an individual review. Never deduplicate
# review records using user_id.

exact_duplicates = int(df.duplicated().sum())
id_col = "review_id" if "review_id" in df.columns else None

if id_col:
    duplicate_review_ids = int(df[id_col].duplicated().sum())
    before = len(df)
    df = df.drop_duplicates(subset=[id_col], keep="first").copy()
    removed_id_duplicates = before - len(df)
else:
    duplicate_review_ids = None
    removed_id_duplicates = 0
    print("\nNo review_id column is available; user_id was not used as a review identifier.")

before_exact = len(df)
df = df.drop_duplicates().copy()
removed_exact_duplicates = before_exact - len(df)

log_step(
    "Duplicate handling",
    f"{exact_duplicates} exact duplicate rows found; "
    + ("review_id duplicates checked" if id_col else "no stable review_id field is available"),
    f"Removed {removed_id_duplicates} duplicate review IDs and "
    f"{removed_exact_duplicates} exact duplicate rows",
    "Exact duplicate rows can overweight repeated feedback. user_id identifies a reviewer "
    "and is not a unique review key."
)

print("\nExact duplicate rows detected before cleaning:", exact_duplicates)
print("Duplicate review identifiers:", duplicate_review_ids)
print("Exact duplicate rows removed:", removed_exact_duplicates)


# ---------------------------------------------------------------------
# 8. VALIDATE RATINGS
# ---------------------------------------------------------------------

if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["invalid_rating"] = (
        df["rating"].isna() | ~df["rating"].between(1, 5)
    )
else:
    df["invalid_rating"] = True

invalid_rating_count = int(df["invalid_rating"].sum())

log_step(
    "Rating validation",
    "Ratings may be missing, non-numeric, or outside the 1–5 scale",
    "Converted ratings to numeric and flagged missing/out-of-range values",
    "Rating-based analysis requires valid values; invalid ratings are excluded only "
    "from rating-dependent analysis."
)

print("\nInvalid or missing ratings:", invalid_rating_count)


# ---------------------------------------------------------------------
# 9. PARSE TIMESTAMPS
# ---------------------------------------------------------------------

if "timestamp" in df.columns:
    df["review_datetime"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    numeric_ts = pd.to_numeric(df["timestamp"], errors="coerce")
    numeric_mask = numeric_ts.notna()

    if numeric_mask.any():
        parsed_numeric = pd.to_datetime(
            numeric_ts[numeric_mask],
            unit="ms",
            errors="coerce",
            utc=True
        )
        df.loc[numeric_mask, "review_datetime"] = parsed_numeric

    df["invalid_timestamp"] = df["review_datetime"].isna()
else:
    df["review_datetime"] = pd.NaT
    df["invalid_timestamp"] = True

invalid_timestamp_count = int(df["invalid_timestamp"].sum())

log_step(
    "Timestamp validation",
    "Timestamp representation may be text or epoch milliseconds",
    "Parsed into UTC datetime and flagged unparseable values",
    "A consistent datetime type supports time-based analysis and reproducible filtering."
)

print("Invalid timestamps:", invalid_timestamp_count)


# ---------------------------------------------------------------------
# 10. NORMALIZE REVIEW TEXT
# ---------------------------------------------------------------------

def clean_review_text(value):
    """Remove HTML fragments and normalize whitespace conservatively."""
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#\d+;|&amp;", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


for col in ["title", "text"]:
    if col in df.columns:
        df[col + "_clean"] = df[col].map(clean_review_text)

title_clean = (
    df["title_clean"].fillna("")
    if "title_clean" in df.columns
    else pd.Series("", index=df.index)
)
text_clean = (
    df["text_clean"].fillna("")
    if "text_clean" in df.columns
    else pd.Series("", index=df.index)
)

df["combined_review"] = (title_clean + " " + text_clean).str.strip()
df["text_is_empty"] = df["combined_review"].eq("")
df["text_word_count"] = (
    df["combined_review"].str.split().str.len().fillna(0).astype(int)
)
df["text_char_count"] = df["combined_review"].str.len()

log_step(
    "Text normalization",
    "HTML fragments, irregular whitespace, and blank review text",
    "Removed HTML tags, normalized whitespace, and created cleaned title/body fields",
    "Conservative normalization improves consistency without deleting potentially meaningful sentiment cues."
)


# ---------------------------------------------------------------------
# 11. FLAG SHORT REVIEWS AND WORD-COUNT OUTLIERS
# ---------------------------------------------------------------------

df["very_short_review"] = df["text_word_count"] <= 2

q1 = df["text_word_count"].quantile(0.25)
q3 = df["text_word_count"].quantile(0.75)
iqr = q3 - q1

lower_fence = max(0, q1 - 1.5 * iqr)
upper_fence = q3 + 1.5 * iqr

df["review_length_outlier"] = (
    (df["text_word_count"] < lower_fence)
    | (df["text_word_count"] > upper_fence)
)

outlier_count = int(df["review_length_outlier"].sum())

log_step(
    "Text length assessment",
    f"Review word-count IQR fences: lower={lower_fence:.2f}, upper={upper_fence:.2f}",
    f"Flagged {outlier_count} length outliers; retained them in the master dataset",
    "IQR is a screening method. Long reviews may be informative and should not be removed without content review."
)

print("\nWord-count Q1:", q1)
print("Word-count Q3:", q3)
print("IQR:", iqr)
print("Lower fence:", lower_fence)
print("Upper fence:", upper_fence)
print("Length outliers flagged:", outlier_count)


# ---------------------------------------------------------------------
# 12. INSPECT CONSTANT FIELDS
# ---------------------------------------------------------------------

for col in ["verified_purchase", "asin", "parent_asin"]:
    if col in df.columns:
        print(f"\n{col} value counts (top 10):")
        print(df[col].value_counts(dropna=False).head(10))

constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

print("\nConstant columns:", constant_columns)

log_step(
    "Consistency checks",
    "Categorical inconsistency and constant fields",
    "Inspected distinct values and flagged constant columns for interpretation",
    "Constant fields may have limited analytical value but can still be valid metadata; "
    "retain until the project question is finalized."
)


# ---------------------------------------------------------------------
# 13. CREATE ANALYSIS-READY DATASETS
# ---------------------------------------------------------------------

cleaned_master = df.copy()

nlp_ready = cleaned_master.loc[
    ~cleaned_master["text_is_empty"]
].copy()

rating_ready = nlp_ready.loc[
    ~nlp_ready["invalid_rating"]
].copy()

print("\nCleaned master shape:", cleaned_master.shape)
print("NLP-ready shape:", nlp_ready.shape)
print("Valid-rating NLP shape:", rating_ready.shape)

log_step(
    "Analysis subsets",
    "Different analyses require different minimum fields",
    "Created NLP-ready and valid-rating subsets without deleting records from the master dataset",
    "This avoids unnecessary data loss and makes inclusion criteria explicit."
)


# ---------------------------------------------------------------------
# 14. CREATE QUALITY AUDIT
# ---------------------------------------------------------------------

audit_rows = [
    ("Raw row count", len(df_raw)),
    ("Raw column count", len(df_raw.columns)),
    ("Raw missing cells", int(df_raw.isna().sum().sum())),
    ("Raw exact duplicate rows", int(df_raw.duplicated().sum())),
    ("Cleaned master row count", len(cleaned_master)),
    ("Cleaned master column count", len(cleaned_master.columns)),
    (
        "Missing title after handling",
        int(cleaned_master["title"].isna().sum())
        if "title" in cleaned_master else np.nan
    ),
    (
        "Empty cleaned review bodies",
        int(cleaned_master["text_clean"].eq("").sum())
        if "text_clean" in cleaned_master else np.nan
    ),
    ("Invalid or missing ratings", int(cleaned_master["invalid_rating"].sum())),
    ("Invalid timestamps", int(cleaned_master["invalid_timestamp"].sum())),
    ("Empty combined review text", int(cleaned_master["text_is_empty"].sum())),
    ("Review length outliers flagged", int(cleaned_master["review_length_outlier"].sum())),
    ("NLP-ready rows", len(nlp_ready)),
    ("Valid-rating NLP rows", len(rating_ready)),
]

if no_image_count is not None:
    audit_rows.append(("Records labelled No image uploaded", no_image_count))

audit = pd.DataFrame(audit_rows, columns=["measure", "count"])

print("\nQUALITY AUDIT")
print(audit.to_string(index=False))


# ---------------------------------------------------------------------
# 15. CREATE CHARTS
# ---------------------------------------------------------------------

if "rating" in cleaned_master.columns:
    plt.figure(figsize=(8, 5))
    cleaned_master["rating"].value_counts(dropna=False).sort_index().plot(kind="bar")
    plt.title("Rating distribution in working sample")
    plt.xlabel("Rating")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rating_distribution.png", dpi=200)
    plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot(
    cleaned_master["text_word_count"],
    orientation="vertical"
)
plt.title("Review word-count distribution")
plt.ylabel("Words per review")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "review_word_count_boxplot.png", dpi=200)
plt.close()


# ---------------------------------------------------------------------
# 16. EXPORT CSV OUTPUTS
# ---------------------------------------------------------------------

cleaned_master.to_csv(
    OUTPUT_DIR / "amazon_beauty_cleaned_master_10k.csv",
    index=False
)
nlp_ready.to_csv(
    OUTPUT_DIR / "amazon_beauty_nlp_ready_10k.csv",
    index=False
)
audit.to_csv(
    OUTPUT_DIR / "amazon_beauty_quality_audit_10k.csv",
    index=False
)
pd.DataFrame(cleaning_log).to_csv(
    OUTPUT_DIR / "amazon_beauty_cleaning_log_10k.csv",
    index=False
)

print("\nPipeline completed successfully.")
print(f"All output files are saved in: {OUTPUT_DIR.resolve()}")
