"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
Module: app.py
Description: Interactive Streamlit Web Application for real estate investment advisory,
             property valuation forecasting, and exploratory analytics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page settings
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .good-inv {
        background-color: #DEF7EC;
        border-left: 6px solid #31C48D;
        padding: 1.2rem;
        border-radius: 8px;
        color: #03543F;
    }
    .bad-inv {
        background-color: #FDE8E8;
        border-left: 6px solid #F98080;
        padding: 1.2rem;
        border-radius: 8px;
        color: #9B1C1C;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Loads fitted pipelines and metadata."""
    clf_pipe = joblib.load(os.path.join(MODELS_DIR, "classification_pipeline.joblib"))
    reg_pipe = joblib.load(os.path.join(MODELS_DIR, "regression_pipeline.joblib"))
    
    with open(os.path.join(MODELS_DIR, "classification_metadata.json")) as f:
        clf_meta = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "regression_metadata.json")) as f:
        reg_meta = json.load(f)
        
    return clf_pipe, reg_pipe, clf_meta, reg_meta

# Load models safely
try:
    clf_pipeline, reg_pipeline, clf_meta, reg_meta = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models from `models/`. Please ensure `python run_pipeline.py` was executed. Details: {e}")

# Navigation Sidebar
st.sidebar.image("https://img.icons8.com/isometric/512/real-estate.png", width=90)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["🔮 Property Investment Advisor", "📊 Model Performance & MLflow", "📈 20-Question EDA Dashboard", "📖 Project Documentation"]
)

if page == "🔮 Property Investment Advisor":
    st.markdown('<div class="main-header">🏢 Real Estate Investment Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate property profitability, classify investment risk, and predict 5-year future valuations.</div>', unsafe_allow_html=True)
    
    if models_loaded:
        st.subheader("1. Enter Property Characteristics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            state = st.selectbox("State", [
                'Maharashtra', 'Delhi', 'Karnataka', 'Telangana', 'Tamil Nadu', 
                'Gujarat', 'Uttar Pradesh', 'Haryana', 'West Bengal', 'Punjab', 
                'Rajasthan', 'Kerala', 'Andhra Pradesh', 'Odisha', 'Madhya Pradesh', 
                'Bihar', 'Assam', 'Jharkhand', 'Uttarakhand', 'Chhattisgarh'
            ])
            city = st.selectbox("City", [
                'Mumbai', 'New Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
                'Pune', 'Ahmedabad', 'Kolkata', 'Noida', 'Gurgaon', 
                'Jaipur', 'Lucknow', 'Kochi', 'Coimbatore', 'Chandigarh', 'Bhopal'
            ])
            property_type = st.selectbox("Property Type", ['Apartment', 'Independent House', 'Villa'])
            bhk = st.slider("BHK Configuration", 1, 5, 3)
            size_sqft = st.number_input("Size (Sq.Ft)", min_value=300, max_value=8000, value=1800, step=50)
            
        with col2:
            price_lakhs = st.number_input("Listing Price (₹ Lakhs)", min_value=5.0, max_value=1500.0, value=185.0, step=2.5)
            year_built = st.slider("Year Built", 1990, 2024, 2018)
            floor_no = st.number_input("Floor Number", min_value=0, max_value=40, value=4)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=40, value=12)
            furnished_status = st.selectbox("Furnishing Status", ['Furnished', 'Semi-furnished', 'Unfurnished'])
            
        with col3:
            transport = st.selectbox("Public Transport Accessibility", ['High', 'Medium', 'Low'])
            parking = st.selectbox("Dedicated Parking Space", ['Yes', 'No'])
            security = st.selectbox("24/7 Security", ['Yes', 'No'])
            availability = st.selectbox("Availability Status", ['Ready_to_Move', 'Under_Construction'])
            facing = st.selectbox("Facing Direction", ['North', 'East', 'West', 'South'])
            owner_type = st.selectbox("Owner / Seller Type", ['Owner', 'Builder', 'Broker'])
            
        st.markdown("---")
        st.subheader("2. Infrastructure & Modern Amenities")
        col_infra1, col_infra2 = st.columns(2)
        with col_infra1:
            nearby_schools = st.slider("Nearby Schools (within 3km)", 1, 10, 6)
            nearby_hospitals = st.slider("Nearby Hospitals (within 3km)", 1, 10, 5)
            
        with col_infra2:
            st.write("Select Available Amenities:")
            has_gym = st.checkbox("Gymnasium", value=True)
            has_pool = st.checkbox("Swimming Pool", value=True)
            has_clubhouse = st.checkbox("Clubhouse", value=True)
            has_playground = st.checkbox("Children's Playground", value=True)
            has_garden = st.checkbox("Landscaped Garden", value=True)

        if st.button("🚀 Analyze & Generate Investment Advisory", type="primary", use_container_width=True):
            # Compute engineered derived features
            price_per_sqft_inr = (price_lakhs * 100000.0) / size_sqft
            age_of_property = max(0, 2025 - year_built)
            amenity_count = int(has_gym + has_pool + has_clubhouse + has_playground + has_garden)
            total_infra_count = nearby_schools + nearby_hospitals
            school_density = round(nearby_schools / (size_sqft / 1000.0), 4)
            hospital_density = round(nearby_hospitals / (size_sqft / 1000.0), 4)
            space_per_bhk = round(size_sqft / bhk, 2)
            floor_ratio = round(min(floor_no / max(total_floors, 1), 5.0), 3)
            is_top_floor = 1 if floor_no >= total_floors else 0
            is_ground_floor = 1 if floor_no == 0 else 0
            
            if age_of_property <= 5:
                age_cat = 'New (<5y)'
            elif age_of_property <= 12:
                age_cat = 'Modern (5-12y)'
            elif age_of_property <= 22:
                age_cat = 'Mid-Age (13-22y)'
            else:
                age_cat = 'Established (>22y)'
                
            input_dict = {
                'BHK': bhk,
                'Size_in_SqFt': size_sqft,
                'Price_in_Lakhs': price_lakhs,
                'Price_per_SqFt_INR': price_per_sqft_inr,
                'Year_Built': year_built,
                'Floor_No': floor_no,
                'Total_Floors': total_floors,
                'Age_of_Property': age_of_property,
                'Nearby_Schools': nearby_schools,
                'Nearby_Hospitals': nearby_hospitals,
                'Amenity_Count': amenity_count,
                'Total_Infra_Count': total_infra_count,
                'School_Density_Score': school_density,
                'Hospital_Density_Score': hospital_density,
                'Space_Per_BHK': space_per_bhk,
                'Floor_Ratio': floor_ratio,
                'Is_Top_Floor': is_top_floor,
                'Is_Ground_Floor': is_ground_floor,
                'Has_Gym': int(has_gym),
                'Has_Playground': int(has_playground),
                'Has_Garden': int(has_garden),
                'Has_Clubhouse': int(has_clubhouse),
                'Has_Pool': int(has_pool),
                'State': state,
                'City': city,
                'Property_Type': property_type,
                'Furnished_Status': furnished_status,
                'Public_Transport_Accessibility': transport,
                'Parking_Space': parking,
                'Security': security,
                'Facing': facing,
                'Owner_Type': owner_type,
                'Availability_Status': availability,
                'Age_Category': age_cat
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Predictions
            is_good = clf_pipeline.predict(input_df)[0]
            good_prob = clf_pipeline.predict_proba(input_df)[0][1] if hasattr(clf_pipeline, "predict_proba") else 0.5
            pred_5y_price = reg_pipeline.predict(input_df)[0]
            
            profit_lakhs = pred_5y_price - price_lakhs
            roi_pct = (profit_lakhs / price_lakhs) * 100.0
            annualized_cagr = ((pred_5y_price / price_lakhs) ** (1/5) - 1) * 100.0
            
            st.markdown("---")
            st.subheader("📊 Investment Advisory Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if is_good == 1:
                    st.markdown(f"""
                    <div class="good-inv">
                        <h3>✅ Recommended Investment</h3>
                        <p style="font-size:1.1rem; margin-bottom:5px;"><b>Confidence Score:</b> {good_prob*100:.1f}%</p>
                        <p>This property exhibits high risk-adjusted return potential, supported by strong transit accessibility, favorable rate per sq.ft, and solid social infrastructure.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bad-inv">
                        <h3>⚠️ Not Recommended / High Caution</h3>
                        <p style="font-size:1.1rem; margin-bottom:5px;"><b>Confidence Score:</b> {(1-good_prob)*100:.1f}%</p>
                        <p>This property does not meet the investment benchmark due to potential valuation premiums or lower infrastructure connectivity relative to its peers.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with res_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💰 5-Year Valuation Forecast</h4>
                    <h2 style="color:#1E3A8A; margin: 5px 0;">₹{pred_5y_price:.2f} Lakhs</h2>
                    <p><b>Projected 5-Year Capital Gain:</b> +₹{profit_lakhs:.2f} Lakhs (+{roi_pct:.1f}% ROI)</p>
                    <p><b>Implied Annualized CAGR:</b> ~{annualized_cagr:.2f}% p.a.</p>
                    <p><b>Current Unit Rate:</b> ₹{price_per_sqft_inr:,.0f} / sq.ft</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "📊 Model Performance & MLflow":
    st.markdown('<div class="main-header">📊 Model Performance & MLflow Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation benchmarks for Classification (Good Investment) and Regression (5-Year Future Price).</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Classification Models", "Regression Models"])
    
    with tab1:
        st.subheader("1. Binary Classification: 'Good Investment' (50,000 Holdout Records)")
        clf_df = pd.DataFrame([
            {"Model": "Logistic Regression (Baseline)", "Accuracy": "89.66%", "Precision": "0.9012", "Recall": "0.8978", "F1-Score": "0.8995", "ROC-AUC": "0.9630"},
            {"Model": "Random Forest Classifier", "Accuracy": "93.71%", "Precision": "0.9312", "Recall": "0.9480", "F1-Score": "0.9396", "ROC-AUC": "0.9890"},
            {"Model": "XGBoost Classifier (Champion)", "Accuracy": "97.36%", "Precision": "0.9706", "Recall": "0.9785", "F1-Score": "0.9745", "ROC-AUC": "0.9972"}
        ])
        st.dataframe(clf_df, use_container_width=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cm_img = os.path.join(PLOTS_DIR, "confusion_matrix_xgboost_classifier.png")
            if os.path.exists(cm_img):
                st.image(cm_img, caption="XGBoost Confusion Matrix", use_container_width=True)
        with col_c2:
            roc_img = os.path.join(PLOTS_DIR, "roc_curve_xgboost_classifier.png")
            if os.path.exists(roc_img):
                st.image(roc_img, caption="XGBoost ROC-AUC Curve", use_container_width=True)
                
    with tab2:
        st.subheader("2. Regression: 5-Year Future Price (₹ Lakhs) (50,000 Holdout Records)")
        reg_df = pd.DataFrame([
            {"Model": "Ridge Regression (Baseline)", "RMSE": "₹11.61 Lakhs", "MAE": "₹8.52 Lakhs", "R² Score": "0.9971", "MAPE": "5.59%"},
            {"Model": "Random Forest Regressor", "RMSE": "₹12.69 Lakhs", "MAE": "₹9.33 Lakhs", "R² Score": "0.9966", "MAPE": "2.52%"},
            {"Model": "XGBoost Regressor (Champion)", "RMSE": "₹3.26 Lakhs", "MAE": "₹2.49 Lakhs", "R² Score": "0.9998", "MAPE": "1.00%"}
        ])
        st.dataframe(reg_df, use_container_width=True)
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            act_pred_img = os.path.join(PLOTS_DIR, "regression_actual_vs_pred_xgboost_regressor.png")
            if os.path.exists(act_pred_img):
                st.image(act_pred_img, caption="Actual vs Predicted 5-Year Price", use_container_width=True)
        with col_r2:
            res_img = os.path.join(PLOTS_DIR, "regression_residuals_xgboost_regressor.png")
            if os.path.exists(res_img):
                st.image(res_img, caption="Residuals Distribution", use_container_width=True)

elif page == "📈 20-Question EDA Dashboard":
    st.markdown('<div class="main-header">📈 20-Question Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exhaustive statistical analysis answering 20 questions across 4 dimensions.</div>', unsafe_allow_html=True)
    
    sec = st.selectbox("Select Section", [
        "Section 1: Price & Size Analysis (Q1 - Q5)",
        "Section 2: Location-Based Analysis (Q6 - Q10)",
        "Section 3: Feature Relationships & Correlation (Q11 - Q15)",
        "Section 4: Investment, Amenities & Ownership (Q16 - Q20)"
    ])
    
    if "Section 1" in sec:
        st.image(os.path.join(PLOTS_DIR, "q01_price_distribution.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q02_size_distribution.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q03_price_per_sqft_by_property_type.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q04_size_vs_price_relationship.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q05_price_size_outliers.png"), use_container_width=True)
        
    elif "Section 2" in sec:
        st.image(os.path.join(PLOTS_DIR, "q06_avg_price_per_sqft_by_state.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q07_avg_price_by_city.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q08_median_age_by_locality.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q09_bhk_distribution_by_city.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q10_top5_expensive_localities.png"), use_container_width=True)
        
    elif "Section 3" in sec:
        st.image(os.path.join(PLOTS_DIR, "q11_correlation_heatmap.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q12_schools_vs_price.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q13_hospitals_vs_price.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q14_price_by_furnished_status.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q15_price_by_facing.png"), use_container_width=True)
        
    elif "Section 4" in sec:
        st.image(os.path.join(PLOTS_DIR, "q16_owner_type_distribution.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q17_availability_status_distribution.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q18_parking_effect_on_price.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q19_amenities_vs_price_per_sqft.png"), use_container_width=True)
        st.image(os.path.join(PLOTS_DIR, "q20_transport_vs_investment.png"), use_container_width=True)

elif page == "📖 Project Documentation":
    st.markdown('<div class="main-header">📖 Methodology & Target Variable Rationale</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Target Variable Engineering & Domain Defense
    
    #### 1. Regression Target: 5-Year Future Property Price ($P_5$)
    The 5-year future price is determined through a dynamic location, asset-type, and infrastructure-linked Compound Annual Growth Rate (CAGR):
    $$r_{\\text{dynamic}} = r_{\\text{state\\_tier}} + \\Delta r_{\\text{property\\_type}} + \\Delta r_{\\text{transit}} + \\Delta r_{\\text{infra}} + \\Delta r_{\\text{age}} + \\Delta r_{\\text{amenities}}$$
    $$P_{5,\\text{dynamic}} = \\text{Price\\_in\\_Lakhs} \\times (1 + r_{\\text{dynamic}})^5$$
    
    #### 2. Classification Target: "Good Investment" (`Good_Investment`)
    A 100-point multi-factor composite score was implemented based on 4 real estate pillars:
    - **Valuation Attractiveness (Up to 35 pts)**: Discount relative to Locality Median Price/SqFt.
    - **Appreciation Potential (Up to 20 pts)**: High projected 5-year CAGR ($\\ge 8.0\\%$).
    - **Connectivity & Social Infra (Up to 25 pts)**: High public transit accessibility & proximity to schools and hospitals.
    - **Desirability & Liquidity (Up to 20 pts)**: BHK $\\ge 3$, dedicated parking, 24/7 security, and Ready to Move status.
    
    *Threshold: Score $\\ge 60 \\rightarrow \\text{Good\\_Investment} = 1$, else $0$.*
    """)
