"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
Module: src/utils.py
Description: Logging configuration, path management, plotting utilities, and metric helpers.
"""

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# Ensure all directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, PLOTS_DIR, MODELS_DIR, NOTEBOOKS_DIR, MLRUNS_DIR]:
    os.makedirs(path, exist_ok=True)

# Logger setup
def get_logger(name: str = "RealEstateAdvisor") -> logging.Logger:
    """Configures and returns a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# Plotting theme helper
def set_plot_style():
    """Sets modern, publication-grade aesthetics for matplotlib and seaborn plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans, Arial, Helvetica",
        "font.family": "sans-serif",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })
