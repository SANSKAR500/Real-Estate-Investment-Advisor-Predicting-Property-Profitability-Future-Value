"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
Module: src/train_regression.py
Description: Regression model training (Ridge, Random Forest, XGBoost),
             evaluation metrics (RMSE, MAE, R², MAPE), residual analysis, MLflow experiment tracking, and model registration.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.utils import get_logger, BASE_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, MLRUNS_DIR, set_plot_style

logger = get_logger("TrainRegression")

FEATURE_NUMERIC = [
    'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt_INR', 'Year_Built',
    'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals',
    'Amenity_Count', 'Total_Infra_Count', 'School_Density_Score', 'Hospital_Density_Score',
    'Space_Per_BHK', 'Floor_Ratio', 'Is_Top_Floor', 'Is_Ground_Floor',
    'Has_Gym', 'Has_Playground', 'Has_Garden', 'Has_Clubhouse', 'Has_Pool'
]

FEATURE_CATEGORICAL = [
    'State', 'City', 'Property_Type', 'Furnished_Status',
    'Public_Transport_Accessibility', 'Parking_Space', 'Security',
    'Facing', 'Owner_Type', 'Availability_Status', 'Age_Category'
]

TARGET_COL = 'Future_Price_5Y_Dynamic'

def load_data():
    """Loads pre-split train and test sets for regression."""
    train_path = os.path.join(DATA_PROCESSED_DIR, "train_regression.csv")
    test_path = os.path.join(DATA_PROCESSED_DIR, "test_regression.csv")
    
    logger.info(f"Loading regression train set from {train_path}...")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loading regression test set from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    X_train = train_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_train = train_df[TARGET_COL]
    
    X_test = test_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_test = test_df[TARGET_COL]
    
    return X_train, y_train, X_test, y_test

def get_feature_preprocessor():
    """Builds a scikit-learn ColumnTransformer for regression inputs."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, FEATURE_NUMERIC),
            ('cat', categorical_transformer, FEATURE_CATEGORICAL)
        ]
    )
    return preprocessor

def plot_and_save_actual_vs_predicted(y_true, y_pred, model_name: str) -> str:
    """Plots and saves Actual vs Predicted scatter plot."""
    set_plot_style()
    plt.figure(figsize=(8, 7))
    sample_indices = np.random.choice(len(y_true), size=min(3000, len(y_true)), replace=False)
    y_t_sample = np.array(y_true)[sample_indices]
    y_p_sample = np.array(y_pred)[sample_indices]
    
    plt.scatter(y_t_sample, y_p_sample, alpha=0.35, color='#1f77b4', edgecolors='none', s=25)
    min_val = min(y_t_sample.min(), y_p_sample.min())
    max_val = max(y_t_sample.max(), y_p_sample.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label="Ideal Line (y = x)")
    
    r2 = r2_score(y_true, y_pred)
    plt.title(f"Actual vs Predicted 5-Year Price: {model_name}\n(R² = {r2:.4f})", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Actual Future Price (₹ Lakhs)")
    plt.ylabel("Predicted Future Price (₹ Lakhs)")
    plt.legend(loc="upper left")
    save_path = os.path.join(PLOTS_DIR, f"regression_actual_vs_pred_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_and_save_residuals(y_true, y_pred, model_name: str) -> str:
    """Plots and saves Residual distribution."""
    set_plot_style()
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='#2ca02c', bins=50)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.title(f"Residuals Distribution: {model_name}\n(Mean = {residuals.mean():.3f}, Std = {residuals.std():.3f})", 
              fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Residual (Actual - Predicted in ₹ Lakhs)")
    plt.ylabel("Frequency")
    save_path = os.path.join(PLOTS_DIR, f"regression_residuals_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def train_and_evaluate():
    """
    Main training workflow for regression with MLflow tracking.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    db_path = os.path.join(BASE_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Real_Estate_Investment_Advisor_Regression")
    
    X_train, y_train, X_test, y_test = load_data()
    logger.info(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    
    preprocessor = get_feature_preprocessor()
    logger.info("Fitting preprocessor pipeline on training features...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save preprocessor artifact
    preprocessor_path = os.path.join(MODELS_DIR, "regression_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved regression preprocessor to {preprocessor_path}")
    
    models = {
        "Ridge_Regression": Ridge(alpha=1.0, random_state=42),
        "Random_Forest_Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost_Regressor": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float("inf")
    best_model = None
    
    for model_name, model in models.items():
        logger.info(f"\n--- Training {model_name} ---")
        with mlflow.start_run(run_name=model_name):
            # Fit model
            model.fit(X_train_transformed, y_train)
            
            # Predict
            y_pred = model.predict(X_test_transformed)
            
            # Metrics
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            metrics = {
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "mape": mape
            }
            results[model_name] = metrics
            
            logger.info(f"Results for {model_name}: RMSE=₹{rmse:.4f} Lakhs, MAE=₹{mae:.4f} Lakhs, R²={r2:.4f}, MAPE={mape*100:.2f}%")
            
            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            # Plots
            actual_pred_plot = plot_and_save_actual_vs_predicted(y_test, y_pred, model_name)
            residuals_plot = plot_and_save_residuals(y_test, y_pred, model_name)
            mlflow.log_artifact(actual_pred_plot)
            mlflow.log_artifact(residuals_plot)
            mlflow.log_artifact(preprocessor_path)
            
            # Save champion
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_model = model
                
    logger.info(f"Champion Regression Model: {best_model_name} with R² Score: {best_r2:.4f}")
    
    # Save champion model to disk
    champion_path = os.path.join(MODELS_DIR, "regression_model.joblib")
    joblib.dump(best_model, champion_path)
    logger.info(f"Champion regression model saved to {champion_path}")
    
    # Save full inference pipeline (preprocessor + model)
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    pipeline_path = os.path.join(MODELS_DIR, "regression_pipeline.joblib")
    joblib.dump(full_pipeline, pipeline_path)
    logger.info(f"Full inference regression pipeline saved to {pipeline_path}")
    
    # Metadata summary
    meta = {
        "model_type": "Regression",
        "champion_model": best_model_name,
        "metrics": results[best_model_name],
        "all_model_results": results,
        "features": {
            "numeric": FEATURE_NUMERIC,
            "categorical": FEATURE_CATEGORICAL
        },
        "target": TARGET_COL
    }
    with open(os.path.join(MODELS_DIR, "regression_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    return results

if __name__ == "__main__":
    train_and_evaluate()
