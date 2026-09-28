"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
Module: src/preprocessing.py
Description: Data loading, cleaning, missing value handling, and categorical/numerical transformation pipelines.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.utils import get_logger, DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR

logger = get_logger("Preprocessing")

def load_raw_data(file_name: str = "india_housing_prices.csv") -> pd.DataFrame:
    """
    Loads raw housing prices dataset from data/raw/ directory or root.
    """
    path = os.path.join(DATA_RAW_DIR, file_name)
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(DATA_RAW_DIR), "..", file_name)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Raw dataset file not found at {path} or {alt_path}")
            
    logger.info(f"Loading raw dataset from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded raw dataset with shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data verification, deduplication, and sanity cleaning.
    """
    logger.info("Performing data cleaning and validation...")
    initial_rows = len(df)
    
    # 1. Remove duplicate rows if any
    df = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate records.")
    else:
        logger.info("Zero duplicate records found.")
        
    # 2. Check and handle missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if not missing_cols.empty:
        logger.warning(f"Missing values detected:\n{missing_cols}")
        # Numeric columns impute with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        # Categorical columns impute with mode
        for col in df.select_dtypes(include=['object', 'string']).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        logger.info("No missing values found across all 23 columns.")
        
    # 3. Strip whitespace from string columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # 4. Standardize types
    int_cols = ['ID', 'BHK', 'Size_in_SqFt', 'Year_Built', 'Floor_No', 'Total_Floors', 
                'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    float_cols = ['Price_in_Lakhs', 'Price_per_SqFt']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    logger.info(f"Data cleaning complete. Clean shape: {df.shape}")
    return df

def build_preprocessor(numeric_features: list, categorical_features: list, 
                       scaler_type: str = "robust") -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
    
    Args:
        numeric_features: List of continuous/numerical column names.
        categorical_features: List of categorical column names.
        scaler_type: 'robust' for RobustScaler (outlier resistant) or 'standard' for StandardScaler.
        
    Returns:
        ColumnTransformer pipeline.
    """
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    
    num_pipeline = Pipeline(steps=[
        ('scaler', scaler)
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def run_preprocessing_pipeline() -> pd.DataFrame:
    """
    Full preprocessing execution pipeline: loads raw data, cleans, and saves to data/processed/.
    """
    df_raw = load_raw_data()
    df_cleaned = clean_data(df_raw)
    
    output_path = os.path.join(DATA_PROCESSED_DIR, "housing_cleaned.csv")
    df_cleaned.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved successfully to {output_path}")
    return df_cleaned

if __name__ == "__main__":
    run_preprocessing_pipeline()
