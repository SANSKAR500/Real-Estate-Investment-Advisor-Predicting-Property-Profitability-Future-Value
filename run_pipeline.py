"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
Module: run_pipeline.py
Description: Master end-to-end execution pipeline.
             Orchestrates Preprocessing -> Feature Engineering -> 20-Question EDA -> Classification -> Regression -> MLflow Logging.
"""

import sys
import time
from src.utils import get_logger, BASE_DIR
from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering_pipeline
from src.eda import run_all_eda
from src.train_classification import train_and_evaluate as run_classification_training
from src.train_regression import train_and_evaluate as run_regression_training

logger = get_logger("MasterPipeline")

def run_full_pipeline():
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("🚀 STARTING REAL ESTATE INVESTMENT ADVISOR END-TO-END PIPELINE")
    logger.info("================================================================================")
    
    # Step 1: Data Preprocessing
    logger.info("\n[STEP 1/5] Running Data Cleaning & Preprocessing...")
    df_cleaned = run_preprocessing_pipeline()
    
    # Step 2: Feature Engineering & Target Construction
    logger.info("\n[STEP 2/5] Running Feature Engineering & Target Creation...")
    df_engineered = run_feature_engineering_pipeline()
    
    # Step 3: Exploratory Data Analysis (20 Questions across 4 Sections)
    logger.info("\n[STEP 3/5] Running Comprehensive 20-Question EDA...")
    insights = run_all_eda(df_engineered)
    
    # Step 4: Classification Modeling & MLflow Tracking
    logger.info("\n[STEP 4/5] Training & Evaluating Classification Models (Target: Good_Investment)...")
    clf_results = run_classification_training()
    
    # Step 5: Regression Modeling & MLflow Tracking
    logger.info("\n[STEP 5/5] Training & Evaluating Regression Models (Target: Future_Price_5Y_Dynamic)...")
    reg_results = run_regression_training()
    
    elapsed = time.time() - start_time
    logger.info("\n================================================================================")
    logger.info(f"✅ REAL ESTATE INVESTMENT ADVISOR PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    logger.info("================================================================================")
    logger.info("\n📊 CLASSIFICATION SUMMARY:")
    for model_name, metrics in clf_results.items():
        logger.info(f"  • {model_name:25s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
    logger.info("\n📈 REGRESSION SUMMARY:")
    for model_name, metrics in reg_results.items():
        logger.info(f"  • {model_name:25s} | RMSE: ₹{metrics['rmse']:6.2f}L | MAE: ₹{metrics['mae']:6.2f}L | R²: {metrics['r2_score']:.4f} | MAPE: {metrics['mape']*100:.2f}%")
        
    logger.info(f"\n📂 Saved Models: {BASE_DIR}/models/")
    logger.info(f"🖼️ Saved Plots: {BASE_DIR}/plots/")
    logger.info(f"📊 MLflow Runs: sqlite:///{BASE_DIR}/mlflow.db (Start UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db)")

if __name__ == "__main__":
    run_full_pipeline()
