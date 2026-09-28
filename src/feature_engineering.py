"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
Module: src/feature_engineering.py
Description: Feature engineering pipeline, domain-based target creation (Good Investment & 5-Year Future Price),
             and train/test split generation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from src.utils import get_logger, DATA_PROCESSED_DIR
from src.preprocessing import run_preprocessing_pipeline

logger = get_logger("FeatureEngineering")

# State economic tiers for dynamic appreciation modeling
TIER1_STATES = {'Maharashtra', 'Delhi', 'Karnataka', 'Telangana', 'Tamil Nadu', 'Gujarat'}
TIER2_STATES = {'Uttar Pradesh', 'Haryana', 'West Bengal', 'Punjab', 'Rajasthan', 'Kerala', 'Andhra Pradesh'}

def calculate_dynamic_cagr(df: pd.DataFrame) -> pd.Series:
    """
    Computes a realistic, location and property-type driven 5-year Compound Annual Growth Rate (CAGR).
    
    Factors considered:
    1. State Economic Tier: Tier 1 (8.5%), Tier 2 (7.5%), Tier 3 (6.8%)
    2. Property Type: Villa (+1.2%), Independent House (+0.8%), Apartment (+0.0%)
    3. Public Transport: High (+0.8%), Medium (+0.4%), Low (-0.4%)
    4. Infrastructure Density (Schools + Hospitals): High >= 12 (+0.5%), Medium >= 8 (+0.2%), Low (-0.2%)
    5. Property Age: <= 7 years (+0.5%), >= 25 years (-0.6%), else (0.0%)
    6. Amenities Diversity: >= 4 amenities (+0.4%)
    """
    # 1. State Base CAGR
    base_cagr = np.where(
        df['State'].isin(TIER1_STATES), 0.085,
        np.where(df['State'].isin(TIER2_STATES), 0.075, 0.068)
    )
    
    # 2. Property Type Adjustment
    prop_adj = np.where(
        df['Property_Type'] == 'Villa', 0.012,
        np.where(df['Property_Type'] == 'Independent House', 0.008, 0.000)
    )
    
    # 3. Public Transport Adjustment
    trans_adj = np.where(
        df['Public_Transport_Accessibility'] == 'High', 0.008,
        np.where(df['Public_Transport_Accessibility'] == 'Medium', 0.004, -0.004)
    )
    
    # 4. Infrastructure Count Adjustment
    infra_count = df['Nearby_Schools'] + df['Nearby_Hospitals']
    infra_adj = np.where(
        infra_count >= 12, 0.005,
        np.where(infra_count >= 8, 0.002, -0.002)
    )
    
    # 5. Age Adjustment
    age_adj = np.where(
        df['Age_of_Property'] <= 7, 0.005,
        np.where(df['Age_of_Property'] >= 25, -0.006, 0.000)
    )
    
    # 6. Amenity Count Adjustment
    amenity_cnt = df['Amenity_Count'] if 'Amenity_Count' in df.columns else df['Amenities'].apply(
        lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
    )
    amenity_adj = np.where(amenity_cnt >= 4, 0.004, 0.000)
    
    total_cagr = base_cagr + prop_adj + trans_adj + infra_adj + age_adj + amenity_adj
    return pd.Series(np.clip(total_cagr, 0.05, 0.13), index=df.index, name="Dynamic_CAGR")

def calculate_investment_score(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Computes the Domain Investment Composite Index (0 - 100 points) and the binary 'Good_Investment' label.
    
    Scoring Rubric:
    - Valuation Attractiveness (Price vs Locality Median): Up to 35 pts
    - Growth Potential (Projected CAGR): Up to 20 pts
    - Connectivity & Infrastructure: Up to 25 pts
    - Specification & Desirability: Up to 20 pts
    
    Threshold: Score >= 60 -> Good_Investment = 1, else 0.
    """
    score = np.zeros(len(df), dtype=float)
    
    # 1. Valuation Attractiveness
    ratio = df['Price_to_Locality_Ratio']
    score += np.where(ratio <= 0.85, 35, np.where(ratio <= 1.00, 25, np.where(ratio <= 1.10, 10, 0)))
    
    # 2. Growth Potential
    cagr = df['Dynamic_CAGR']
    score += np.where(cagr >= 0.095, 20, np.where(cagr >= 0.080, 15, np.where(cagr >= 0.070, 10, 0)))
    
    # 3. Connectivity & Infrastructure
    score += np.where(df['Public_Transport_Accessibility'] == 'High', 15,
                      np.where(df['Public_Transport_Accessibility'] == 'Medium', 8, 0))
    
    both_infra = (df['Nearby_Schools'] >= 5) & (df['Nearby_Hospitals'] >= 5)
    sum_infra = (df['Nearby_Schools'] + df['Nearby_Hospitals']) >= 8
    score += np.where(both_infra, 10, np.where(sum_infra, 5, 0))
    
    # 4. Specification & Desirability
    score += np.where(df['BHK'] >= 3, 10, np.where(df['BHK'] == 2, 6, 0))
    score += np.where(df['Parking_Space'] == 'Yes', 5, 0)
    score += np.where(df['Security'] == 'Yes', 5, 0)
    score += np.where(df['Availability_Status'] == 'Ready_to_Move', 5, 0)
    
    investment_score = pd.Series(score, index=df.index, name="Investment_Score")
    good_investment = pd.Series((score >= 60).astype(int), index=df.index, name="Good_Investment")
    return investment_score, good_investment

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs full feature engineering on the housing dataset.
    """
    logger.info("Starting feature engineering pipeline...")
    df = df.copy()
    
    # 1. Price per Sq Ft in INR
    df['Price_per_SqFt_INR'] = (df['Price_in_Lakhs'] * 100000.0) / df['Size_in_SqFt']
    
    # 2. Locality & City Level Price Benchmarks
    locality_median = df.groupby('Locality')['Price_per_SqFt_INR'].transform('median')
    city_median = df.groupby('City')['Price_per_SqFt_INR'].transform('median')
    
    df['Locality_Median_Price_per_SqFt'] = locality_median
    df['City_Median_Price_per_SqFt'] = city_median
    df['Price_to_Locality_Ratio'] = df['Price_per_SqFt_INR'] / locality_median
    df['Price_to_City_Ratio'] = df['Price_per_SqFt_INR'] / city_median
    
    # 3. Amenity Parsing & Indicators
    amenity_tokens = ['Gym', 'Playground', 'Garden', 'Clubhouse', 'Pool']
    for token in amenity_tokens:
        df[f'Has_{token}'] = df['Amenities'].astype(str).apply(lambda x: 1 if token in x else 0)
        
    df['Amenity_Count'] = df[[f'Has_{token}' for token in amenity_tokens]].sum(axis=1)
    
    # 4. Infrastructure and School Density Scores
    df['Total_Infra_Count'] = df['Nearby_Schools'] + df['Nearby_Hospitals']
    df['School_Density_Score'] = (df['Nearby_Schools'] / (df['Size_in_SqFt'] / 1000.0)).round(4)
    df['Hospital_Density_Score'] = (df['Nearby_Hospitals'] / (df['Size_in_SqFt'] / 1000.0)).round(4)
    df['Infra_Balance_Ratio'] = (df['Nearby_Schools'] / np.maximum(df['Nearby_Hospitals'], 1)).round(3)
    
    # 5. Spatial & Architectural Features
    df['Space_Per_BHK'] = (df['Size_in_SqFt'] / df['BHK']).round(2)
    df['Floor_Ratio'] = np.clip(df['Floor_No'] / np.maximum(df['Total_Floors'], 1), 0.0, 5.0).round(3)
    df['Is_Top_Floor'] = (df['Floor_No'] >= df['Total_Floors']).astype(int)
    df['Is_Ground_Floor'] = (df['Floor_No'] == 0).astype(int)
    
    # 6. Age Categorization
    df['Age_Category'] = pd.cut(
        df['Age_of_Property'],
        bins=[-1, 5, 12, 22, 100],
        labels=['New (<5y)', 'Modern (5-12y)', 'Mid-Age (13-22y)', 'Established (>22y)']
    ).astype(str)
    
    # 7. Target Engineering
    logger.info("Computing CAGR and Target Variables...")
    df['Dynamic_CAGR'] = calculate_dynamic_cagr(df)
    
    # Regression Targets: Fixed 8% CAGR vs Dynamic Multi-Factor CAGR
    df['Future_Price_5Y_Fixed'] = (df['Price_in_Lakhs'] * ((1.0 + 0.08) ** 5)).round(4)
    df['Future_Price_5Y_Dynamic'] = (df['Price_in_Lakhs'] * ((1.0 + df['Dynamic_CAGR']) ** 5)).round(4)
    df['Projected_5Y_Profit_Lakhs'] = (df['Future_Price_5Y_Dynamic'] - df['Price_in_Lakhs']).round(4)
    df['Projected_5Y_ROI_Pct'] = (((df['Future_Price_5Y_Dynamic'] - df['Price_in_Lakhs']) / df['Price_in_Lakhs']) * 100.0).round(2)
    
    # Classification Target: Good Investment
    inv_score, good_inv = calculate_investment_score(df)
    df['Investment_Score'] = inv_score
    df['Good_Investment'] = good_inv
    
    logger.info(f"Feature engineering complete. Total columns: {df.shape[1]}")
    logger.info(f"Good_Investment class balance: {df['Good_Investment'].value_counts(normalize=True).to_dict()}")
    return df

def save_engineered_datasets(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Saves the full engineered dataset and generates train/test splits for both classification and regression.
    """
    full_path = os.path.join(DATA_PROCESSED_DIR, "housing_engineered_full.csv")
    df.to_csv(full_path, index=False)
    logger.info(f"Saved full engineered dataset to {full_path}")
    
    # Train / Test split for Classification (Stratified on Good_Investment)
    train_clf, test_clf = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['Good_Investment']
    )
    
    train_clf.to_csv(os.path.join(DATA_PROCESSED_DIR, "train_classification.csv"), index=False)
    test_clf.to_csv(os.path.join(DATA_PROCESSED_DIR, "test_classification.csv"), index=False)
    logger.info(f"Classification splits saved: Train={train_clf.shape}, Test={test_clf.shape}")
    
    # Train / Test split for Regression
    train_reg, test_reg = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    train_reg.to_csv(os.path.join(DATA_PROCESSED_DIR, "train_regression.csv"), index=False)
    test_reg.to_csv(os.path.join(DATA_PROCESSED_DIR, "test_regression.csv"), index=False)
    logger.info(f"Regression splits saved: Train={train_reg.shape}, Test={test_reg.shape}")

def run_feature_engineering_pipeline():
    """
    Loads cleaned data, runs feature engineering, and exports splits.
    """
    cleaned_path = os.path.join(DATA_PROCESSED_DIR, "housing_cleaned.csv")
    if not os.path.exists(cleaned_path):
        df_cleaned = run_preprocessing_pipeline()
    else:
        df_cleaned = pd.read_csv(cleaned_path)
        
    df_engineered = engineer_features(df_cleaned)
    save_engineered_datasets(df_engineered)
    return df_engineered

if __name__ == "__main__":
    run_feature_engineering_pipeline()
