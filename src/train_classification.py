"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
Module: src/train_classification.py
Description: Classification model training (Logistic Regression, Random Forest, XGBoost),
             evaluation metrics, confusion matrix logging, MLflow experiment tracking, and model registration.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.utils import get_logger, BASE_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, MLRUNS_DIR, set_plot_style

logger = get_logger("TrainClassification")

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

TARGET_COL = 'Good_Investment'

def load_data():
    """Loads pre-split train and test sets for classification."""
    train_path = os.path.join(DATA_PROCESSED_DIR, "train_classification.csv")
    test_path = os.path.join(DATA_PROCESSED_DIR, "test_classification.csv")
    
    logger.info(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loading testing data from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    X_train = train_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_train = train_df[TARGET_COL]
    
    X_test = test_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_test = test_df[TARGET_COL]
    
    return X_train, y_train, X_test, y_test

def get_feature_preprocessor():
    """Builds a scikit-learn ColumnTransformer for classification inputs."""
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

def plot_and_save_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    """Plots and saves confusion matrix."""
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Recommended (0)', 'Good Investment (1)'],
                yticklabels=['Not Recommended (0)', 'Good Investment (1)'])
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    save_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_and_save_roc_curve(y_true, y_prob, model_name: str) -> str:
    """Plots and saves ROC curve."""
    set_plot_style()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f"ROC Curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC-AUC Curve: {model_name}", fontsize=13, fontweight='bold', pad=10)
    plt.legend(loc="lower right")
    save_path = os.path.join(PLOTS_DIR, f"roc_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def train_and_evaluate():
    """
    Main training workflow for classification with MLflow tracking.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    db_path = os.path.join(BASE_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Real_Estate_Investment_Advisor_Classification")
    
    X_train, y_train, X_test, y_test = load_data()
    logger.info(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    
    preprocessor = get_feature_preprocessor()
    logger.info("Fitting preprocessor pipeline on training features...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save preprocessor artifact
    preprocessor_path = os.path.join(MODELS_DIR, "classification_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved classification preprocessor to {preprocessor_path}")
    
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random_Forest_Classifier": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost_Classifier": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None
    
    for model_name, model in models.items():
        logger.info(f"\n--- Training {model_name} ---")
        with mlflow.start_run(run_name=model_name):
            # Fit model
            model.fit(X_train_transformed, y_train)
            
            # Predict
            y_pred = model.predict(X_test_transformed)
            y_prob = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc
            }
            results[model_name] = metrics
            
            logger.info(f"Results for {model_name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={auc:.4f}")
            logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            
            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            # Plots
            cm_plot_path = plot_and_save_confusion_matrix(y_test, y_pred, model_name)
            roc_plot_path = plot_and_save_roc_curve(y_test, y_prob, model_name)
            mlflow.log_artifact(cm_plot_path)
            mlflow.log_artifact(roc_plot_path)
            mlflow.log_artifact(preprocessor_path)
            
            # Save champion
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_model = model
                
    logger.info(f"Champion Classification Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    # Save champion model to disk
    champion_path = os.path.join(MODELS_DIR, "classification_model.joblib")
    joblib.dump(best_model, champion_path)
    logger.info(f"Champion classification model saved to {champion_path}")
    
    # Save full inference pipeline (preprocessor + model)
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    pipeline_path = os.path.join(MODELS_DIR, "classification_pipeline.joblib")
    joblib.dump(full_pipeline, pipeline_path)
    logger.info(f"Full inference classification pipeline saved to {pipeline_path}")
    
    # Metadata summary
    meta = {
        "model_type": "Classification",
        "champion_model": best_model_name,
        "metrics": results[best_model_name],
        "all_model_results": results,
        "features": {
            "numeric": FEATURE_NUMERIC,
            "categorical": FEATURE_CATEGORICAL
        },
        "target": TARGET_COL
    }
    with open(os.path.join(MODELS_DIR, "classification_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    return results

if __name__ == "__main__":
    train_and_evaluate()
