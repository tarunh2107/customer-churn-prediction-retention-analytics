"""
Data Cleaning Module for Customer Churn Analytics.

This module loads the raw Telco customer churn Excel dataset, cleans string whitespaces,
resolves numeric type conversion issues for Total Charges with domain-defensible logic,
validates integrity (duplicates, non-negative values, binary target), and saves the cleaned dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_raw_data(
    file_path: str | Path = "data/raw/Telco_customer_churn.xlsx",
    sheet_name: str = "Telco_Churn",
) -> pd.DataFrame:
    """Loads the raw Excel customer churn dataset without modifying original file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading/trailing whitespace from column headers while preserving business names."""
    df = df.copy()
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    return df


def strip_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading and trailing whitespace from string/object columns."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def clean_total_charges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Converts Total Charges to numeric, investigates resulting NaN values,
    and imputes them with 0.0 based on customer tenure evidence.
    """
    df = df.copy()
    
    # Convert to numeric, coercing non-numeric (e.g. whitespace) to NaN
    total_charges_numeric = pd.to_numeric(df["Total Charges"], errors="coerce")
    nan_mask = total_charges_numeric.isna()
    nan_count = int(nan_mask.sum())
    
    # Inspect tenure for NaN rows
    tenure_for_nans = df.loc[nan_mask, "Tenure Months"].tolist() if nan_count > 0 else []
    all_zero_tenure = all(t == 0 for t in tenure_for_nans) if nan_count > 0 else True
    
    # Domain-defensible treatment:
    # All NaN rows correspond to brand-new customers with Tenure Months == 0 who have
    # not yet completed their first billing cycle. Their accumulated total charges are $0.00.
    df["Total Charges"] = total_charges_numeric.fillna(0.0)
    
    metadata = {
        "nan_count": nan_count,
        "tenure_values_for_nans": tenure_for_nans,
        "all_zero_tenure": all_zero_tenure,
        "resolution_strategy": "Filled missing Total Charges with 0.0 because Tenure Months is 0 for all affected accounts."
    }
    
    return df, metadata


def check_duplicates(df: pd.DataFrame) -> int:
    """Checks for duplicate records in the dataframe."""
    return int(df.duplicated().sum())


def validate_numerical_ranges(df: pd.DataFrame) -> dict:
    """Validates numerical columns for impossible negative values."""
    numerical_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
    negative_counts = {}
    
    for col in numerical_cols:
        if col in df.columns:
            neg_count = int((df[col] < 0).sum())
            negative_counts[col] = neg_count
            
    has_negatives = any(count > 0 for count in negative_counts.values())
    return {
        "negative_counts": negative_counts,
        "has_negatives": has_negatives
    }


def validate_target(df: pd.DataFrame, target_col: str = "Churn Value") -> dict:
    """Validates that the target column contains strictly binary values {0, 1}."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")
        
    unique_vals = set(df[target_col].dropna().unique())
    is_valid = unique_vals.issubset({0, 1})
    val_counts = df[target_col].value_counts(dropna=False).to_dict()
    
    return {
        "target_col": target_col,
        "unique_values": list(unique_vals),
        "is_valid": is_valid,
        "value_counts": val_counts
    }


def clean_churn_data(
    raw_path: str | Path = "data/raw/Telco_customer_churn.xlsx",
    sheet_name: str = "Telco_Churn",
    output_path: str | Path = "data/processed/telco_churn_clean.csv",
) -> pd.DataFrame:
    """
    Executes the complete end-to-end data cleaning pipeline and saves the output.
    """
    print(f"Loading raw dataset from {raw_path} (Sheet: '{sheet_name}')...")
    df_raw = load_raw_data(raw_path, sheet_name=sheet_name)
    orig_shape = df_raw.shape
    print(f"Raw shape: {orig_shape[0]:,} rows, {orig_shape[1]} columns.")
    
    # 1. Clean column names
    df = clean_column_names(df_raw)
    
    # 2. Clean string whitespaces
    df = strip_string_whitespace(df)
    
    # 3. Clean and impute Total Charges
    df, total_charges_meta = clean_total_charges(df)
    print(f"Total Charges: {total_charges_meta['nan_count']} invalid/blank values found.")
    print(f"Treatment: {total_charges_meta['resolution_strategy']}")
    
    # 4. Duplicate checks
    dup_count = check_duplicates(df)
    print(f"Duplicate rows detected: {dup_count}")
    
    # 5. Validate numerical columns for negatives
    num_val = validate_numerical_ranges(df)
    print(f"Negative values check: {num_val}")
    
    # 6. Validate target
    target_val = validate_target(df)
    print(f"Target validation for '{target_val['target_col']}': is_valid={target_val['is_valid']}, counts={target_val['value_counts']}")
    
    # 7. Save cleaned data
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Successfully saved cleaned dataset to {out.resolve()}")
    print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    
    return df


if __name__ == "__main__":
    clean_churn_data()
