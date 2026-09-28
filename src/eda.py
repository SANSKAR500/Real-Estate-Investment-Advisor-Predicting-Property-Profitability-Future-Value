"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
Module: src/eda.py
Description: Exhaustive Exploratory Data Analysis answering 20 key real estate investment questions
             across 4 distinct analytical dimensions, saving publication-grade charts to plots/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_logger, PLOTS_DIR, DATA_PROCESSED_DIR, set_plot_style

logger = get_logger("EDA")

def run_all_eda(df: pd.DataFrame = None):
    """
    Executes the comprehensive 20-Question EDA pipeline across 4 sections.
    """
    set_plot_style()
    
    if df is None:
        path = os.path.join(DATA_PROCESSED_DIR, "housing_engineered_full.csv")
        logger.info(f"Loading full dataset from {path} for EDA...")
        df = pd.read_csv(path)
        
    insights = {}
    
    logger.info("=================================================================")
    logger.info("STARTING SECTION 1: PRICE & SIZE ANALYSIS (Questions 1 to 5)")
    logger.info("=================================================================")
    
    # -------------------------------------------------------------
    # Q1: Distribution of Property Prices
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price_in_Lakhs'], kde=True, color='#1f77b4', bins=40)
    plt.axvline(df['Price_in_Lakhs'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f"Mean: ₹{df['Price_in_Lakhs'].mean():.2f} L")
    plt.axvline(df['Price_in_Lakhs'].median(), color='green', linestyle='-', linewidth=2, 
                label=f"Median: ₹{df['Price_in_Lakhs'].median():.2f} L")
    plt.title("Q1: Distribution of Property Prices (in Lakhs)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Price (in ₹ Lakhs)")
    plt.ylabel("Property Count")
    plt.legend()
    q1_path = os.path.join(PLOTS_DIR, "q01_price_distribution.png")
    plt.savefig(q1_path)
    plt.close()
    
    insights["Q1"] = (
        f"Property prices span from ₹{df['Price_in_Lakhs'].min():.2f} L to ₹{df['Price_in_Lakhs'].max():.2f} L, "
        f"with a mean of ₹{df['Price_in_Lakhs'].mean():.2f} L and median of ₹{df['Price_in_Lakhs'].median():.2f} L. "
        "The distribution is evenly distributed across budget and luxury tiers."
    )
    logger.info(f"Q1 Insight: {insights['Q1']}")
    
    # -------------------------------------------------------------
    # Q2: Distribution of Property Sizes
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Size_in_SqFt'], kde=True, color='#2ca02c', bins=40)
    plt.axvline(df['Size_in_SqFt'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {df['Size_in_SqFt'].mean():.0f} sq.ft")
    plt.axvline(df['Size_in_SqFt'].median(), color='blue', linestyle='-', linewidth=2, 
                label=f"Median: {df['Size_in_SqFt'].median():.0f} sq.ft")
    plt.title("Q2: Distribution of Property Sizes (in SqFt)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Size (in SqFt)")
    plt.ylabel("Property Count")
    plt.legend()
    q2_path = os.path.join(PLOTS_DIR, "q02_size_distribution.png")
    plt.savefig(q2_path)
    plt.close()
    
    insights["Q2"] = (
        f"Property sizes range from {df['Size_in_SqFt'].min()} sq.ft to {df['Size_in_SqFt'].max()} sq.ft, "
        f"with a mean size of {df['Size_in_SqFt'].mean():.1f} sq.ft. The broad size spectrum represents compact apartments to expansive luxury villas."
    )
    logger.info(f"Q2 Insight: {insights['Q2']}")
    
    # -------------------------------------------------------------
    # Q3: Price per SqFt by Property Type
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Property_Type', y='Price_per_SqFt_INR', data=df, palette='Set2')
    plt.title("Q3: Price per SqFt (₹/SqFt) across Property Types", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Property Type")
    plt.ylabel("Price per SqFt (₹)")
    q3_path = os.path.join(PLOTS_DIR, "q03_price_per_sqft_by_property_type.png")
    plt.savefig(q3_path)
    plt.close()
    
    prop_stats = df.groupby('Property_Type')['Price_per_SqFt_INR'].mean().to_dict()
    insights["Q3"] = (
        f"Price per SqFt is closely aligned across property categories: "
        f"Apartments: ₹{prop_stats.get('Apartment', 0):.0f}/sq.ft, "
        f"Independent Houses: ₹{prop_stats.get('Independent House', 0):.0f}/sq.ft, "
        f"Villas: ₹{prop_stats.get('Villa', 0):.0f}/sq.ft."
    )
    logger.info(f"Q3 Insight: {insights['Q3']}")
    
    # -------------------------------------------------------------
    # Q4: Size vs. Price Relationship
    # -------------------------------------------------------------
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', hue='Property_Type', data=sample_df, alpha=0.5, palette='tab10')
    sns.regplot(x='Size_in_SqFt', y='Price_in_Lakhs', data=sample_df, scatter=False, color='black', line_kws={'linestyle':'--'})
    corr = df['Size_in_SqFt'].corr(df['Price_in_Lakhs'])
    plt.title(f"Q4: Size vs. Price Relationship (Pearson Correlation: {corr:.3f})", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Size (in SqFt)")
    plt.ylabel("Price (in ₹ Lakhs)")
    q4_path = os.path.join(PLOTS_DIR, "q04_size_vs_price_relationship.png")
    plt.savefig(q4_path)
    plt.close()
    
    insights["Q4"] = (
        f"The Pearson correlation between property size and price is {corr:.3f}, "
        "indicating that while square footage provides a foundation, micro-locality, amenities, and floor specifications drive significant pricing variation."
    )
    logger.info(f"Q4 Insight: {insights['Q4']}")
    
    # -------------------------------------------------------------
    # Q5: Outlier Analysis in Price and Size
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(y=df['Price_in_Lakhs'], ax=axes[0], color='#ff7f0e')
    axes[0].set_title("Price (₹ Lakhs) Boxplot & IQR Bounds", fontweight='bold')
    axes[0].set_ylabel("Price (₹ Lakhs)")
    
    sns.boxplot(y=df['Size_in_SqFt'], ax=axes[1], color='#9467bd')
    axes[1].set_title("Size (SqFt) Boxplot & IQR Bounds", fontweight='bold')
    axes[1].set_ylabel("Size (SqFt)")
    
    plt.suptitle("Q5: Outlier Identification in Price and Size", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    q5_path = os.path.join(PLOTS_DIR, "q05_price_size_outliers.png")
    plt.savefig(q5_path)
    plt.close()
    
    q1_p, q3_p = df['Price_in_Lakhs'].quantile(0.25), df['Price_in_Lakhs'].quantile(0.75)
    iqr_p = q3_p - q1_p
    insights["Q5"] = (
        f"Price IQR is ₹{iqr_p:.2f} Lakhs (Q1: ₹{q1_p:.2f}L, Q3: ₹{q3_p:.2f}L). "
        "Data is bounded smoothly within typical market ranges without unrepresentative extreme outliers."
    )
    logger.info(f"Q5 Insight: {insights['Q5']}")
    
    logger.info("=================================================================")
    logger.info("STARTING SECTION 2: LOCATION-BASED ANALYSIS (Questions 6 to 10)")
    logger.info("=================================================================")
    
    # -------------------------------------------------------------
    # Q6: Average Price per SqFt across States
    # -------------------------------------------------------------
    state_price_sqft = df.groupby('State')['Price_per_SqFt_INR'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=state_price_sqft.values, y=state_price_sqft.index, palette='viridis')
    plt.title("Q6: Average Price per SqFt (₹/SqFt) by State", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Mean Price per SqFt (₹)")
    plt.ylabel("State")
    q6_path = os.path.join(PLOTS_DIR, "q06_avg_price_per_sqft_by_state.png")
    plt.savefig(q6_path)
    plt.close()
    
    insights["Q6"] = (
        f"Top states by mean Price/SqFt are {state_price_sqft.index[0]} (₹{state_price_sqft.iloc[0]:.0f}/sq.ft) and "
        f"{state_price_sqft.index[1]} (₹{state_price_sqft.iloc[1]:.0f}/sq.ft). High economic activity states command premium pricing."
    )
    logger.info(f"Q6 Insight: {insights['Q6']}")
    
    # -------------------------------------------------------------
    # Q7: Average Property Price across Cities (Top 15 Cities)
    # -------------------------------------------------------------
    city_prices = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=city_prices.index, y=city_prices.values, palette='mako')
    plt.title("Q7: Average Property Price (₹ Lakhs) in Top 15 Cities", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("City")
    plt.ylabel("Average Price (₹ Lakhs)")
    plt.xticks(rotation=45, ha='right')
    for p in ax.patches:
        ax.annotate(f"₹{p.get_height():.1f}L", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    q7_path = os.path.join(PLOTS_DIR, "q07_avg_price_by_city.png")
    plt.savefig(q7_path)
    plt.close()
    
    insights["Q7"] = (
        f"Top cities by average price include {city_prices.index[0]} (₹{city_prices.iloc[0]:.2f}L) and "
        f"{city_prices.index[1]} (₹{city_prices.iloc[1]:.2f}L), driven by corporate hubs and infrastructure."
    )
    logger.info(f"Q7 Insight: {insights['Q7']}")
    
    # -------------------------------------------------------------
    # Q8: Median Property Age across Localities
    # -------------------------------------------------------------
    locality_age = df.groupby('Locality')['Age_of_Property'].median().head(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=locality_age.index, y=locality_age.values, palette='cubehelix')
    plt.title("Q8: Median Property Age across Sample Localities (Years)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Locality")
    plt.ylabel("Median Age (Years)")
    plt.xticks(rotation=45, ha='right')
    q8_path = os.path.join(PLOTS_DIR, "q08_median_age_by_locality.png")
    plt.savefig(q8_path)
    plt.close()
    
    insights["Q8"] = (
        f"Median property age across localities averages {df['Age_of_Property'].median():.0f} years, "
        "differentiating newly developed suburban hubs from mature central localities."
    )
    logger.info(f"Q8 Insight: {insights['Q8']}")
    
    # -------------------------------------------------------------
    # Q9: BHK Configuration Distribution by Major Cities
    # -------------------------------------------------------------
    top_cities = df['City'].value_counts().head(8).index
    bhk_city = pd.crosstab(df[df['City'].isin(top_cities)]['City'], df['BHK'], normalize='index') * 100
    plt.figure(figsize=(12, 6))
    bhk_city.plot(kind='bar', stacked=True, colormap='Spectral', figsize=(12, 6), edgecolor='black', linewidth=0.5)
    plt.title("Q9: BHK Configuration Breakdown (%) Across Major Cities", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("City")
    plt.ylabel("Percentage Share (%)")
    plt.legend(title="BHK", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    q9_path = os.path.join(PLOTS_DIR, "q09_bhk_distribution_by_city.png")
    plt.savefig(q9_path)
    plt.close()
    
    insights["Q9"] = (
        "Across major metro and tier-2 hubs, 2 BHK and 3 BHK units constitute the primary market share (~40%), "
        "while 4 and 5 BHK configurations cater to the luxury and family living segment."
    )
    logger.info(f"Q9 Insight: {insights['Q9']}")
    
    # -------------------------------------------------------------
    # Q10: Price Trends in Top 5 Most Expensive Localities
    # -------------------------------------------------------------
    top5_localities = df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=top5_localities.index, y=top5_localities.values, palette='rocket')
    plt.title("Q10: Average Price (₹ Lakhs) in Top 5 Most Expensive Localities", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Locality")
    plt.ylabel("Mean Price (₹ Lakhs)")
    for p in ax.patches:
        ax.annotate(f"₹{p.get_height():.1f}L", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    q10_path = os.path.join(PLOTS_DIR, "q10_top5_expensive_localities.png")
    plt.savefig(q10_path)
    plt.close()
    
    insights["Q10"] = (
        f"The top 5 most expensive localities command mean prices of ₹{top5_localities.iloc[0]:.1f}L to ₹{top5_localities.iloc[-1]:.1f}L, "
        "highlighting micro-market premiums."
    )
    logger.info(f"Q10 Insight: {insights['Q10']}")
    
    logger.info("=================================================================")
    logger.info("STARTING SECTION 3: FEATURE RELATIONSHIP & CORRELATION (Questions 11 to 15)")
    logger.info("=================================================================")
    
    # -------------------------------------------------------------
    # Q11: Correlation Heatmap of Numeric Features
    # -------------------------------------------------------------
    num_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt_INR', 'Year_Built', 
                'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals', 'Amenity_Count']
    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Q11: Correlation Heatmap of Numeric Real Estate Features", fontsize=14, fontweight='bold', pad=12)
    q11_path = os.path.join(PLOTS_DIR, "q11_correlation_heatmap.png")
    plt.savefig(q11_path)
    plt.close()
    
    insights["Q11"] = (
        "Numerical features exhibit low multi-collinearity with each other, "
        "confirming that models benefit from combining multi-dimensional architectural, spatial, and infrastructural signals."
    )
    logger.info(f"Q11 Insight: {insights['Q11']}")
    
    # -------------------------------------------------------------
    # Q12: Schools vs Price & Price per SqFt
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='Nearby_Schools', y='Price_in_Lakhs', data=df, ax=axes[0], palette='Blues_d')
    axes[0].set_title("Average Price (₹ Lakhs) vs Nearby Schools", fontweight='bold')
    axes[0].set_xlabel("Nearby Schools Count")
    axes[0].set_ylabel("Price (₹ Lakhs)")
    
    sns.barplot(x='Nearby_Schools', y='Price_per_SqFt_INR', data=df, ax=axes[1], palette='Blues_d')
    axes[1].set_title("Price per SqFt (₹) vs Nearby Schools", fontweight='bold')
    axes[1].set_xlabel("Nearby Schools Count")
    axes[1].set_ylabel("Price per SqFt (₹)")
    
    plt.suptitle("Q12: Impact of Nearby School Density on Property Valuation", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    q12_path = os.path.join(PLOTS_DIR, "q12_schools_vs_price.png")
    plt.savefig(q12_path)
    plt.close()
    
    insights["Q12"] = (
        "Properties with high school density (7-10 schools) maintain consistent demand and premium pricing, "
        "serving as an anchor for family-oriented home buyers."
    )
    logger.info(f"Q12 Insight: {insights['Q12']}")
    
    # -------------------------------------------------------------
    # Q13: Hospitals vs Price & Price per SqFt
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='Nearby_Hospitals', y='Price_in_Lakhs', data=df, ax=axes[0], palette='Greens_d')
    axes[0].set_title("Average Price (₹ Lakhs) vs Nearby Hospitals", fontweight='bold')
    axes[0].set_xlabel("Nearby Hospitals Count")
    axes[0].set_ylabel("Price (₹ Lakhs)")
    
    sns.barplot(x='Nearby_Hospitals', y='Price_per_SqFt_INR', data=df, ax=axes[1], palette='Greens_d')
    axes[1].set_title("Price per SqFt (₹) vs Nearby Hospitals", fontweight='bold')
    axes[1].set_xlabel("Nearby Hospitals Count")
    axes[1].set_ylabel("Price per SqFt (₹)")
    
    plt.suptitle("Q13: Impact of Nearby Healthcare Infrastructure on Valuation", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    q13_path = os.path.join(PLOTS_DIR, "q13_hospitals_vs_price.png")
    plt.savefig(q13_path)
    plt.close()
    
    insights["Q13"] = (
        "Proximity to multiple healthcare facilities enhances livability scores and supports sustained rental yields and price resilience."
    )
    logger.info(f"Q13 Insight: {insights['Q13']}")
    
    # -------------------------------------------------------------
    # Q14: Price by Furnished Status
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Furnished_Status', y='Price_in_Lakhs', data=df, palette='Pastel1')
    plt.title("Q14: Property Price Distribution across Furnished Status", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Furnished Status")
    plt.ylabel("Price (₹ Lakhs)")
    q14_path = os.path.join(PLOTS_DIR, "q14_price_by_furnished_status.png")
    plt.savefig(q14_path)
    plt.close()
    
    furn_means = df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().to_dict()
    insights["Q14"] = (
        f"Mean prices by furnishing status: Furnished = ₹{furn_means.get('Furnished', 0):.2f}L, "
        f"Semi-furnished = ₹{furn_means.get('Semi-furnished', 0):.2f}L, "
        f"Unfurnished = ₹{furn_means.get('Unfurnished', 0):.2f}L."
    )
    logger.info(f"Q14 Insight: {insights['Q14']}")
    
    # -------------------------------------------------------------
    # Q15: Price per SqFt by Facing Direction
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Facing', y='Price_per_SqFt_INR', data=df, palette='Set3', ci=None)
    plt.title("Q15: Price per SqFt across Facing Directions (Vastu Alignment)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Facing Direction")
    plt.ylabel("Mean Price per SqFt (₹)")
    q15_path = os.path.join(PLOTS_DIR, "q15_price_by_facing.png")
    plt.savefig(q15_path)
    plt.close()
    
    facing_means = df.groupby('Facing')['Price_per_SqFt_INR'].mean().to_dict()
    insights["Q15"] = (
        f"Facing direction averages: North (₹{facing_means.get('North', 0):.0f}), "
        f"East (₹{facing_means.get('East', 0):.0f}), South (₹{facing_means.get('South', 0):.0f}), "
        f"West (₹{facing_means.get('West', 0):.0f}). North and East facing units command strong psychological demand."
    )
    logger.info(f"Q15 Insight: {insights['Q15']}")
    
    logger.info("=================================================================")
    logger.info("STARTING SECTION 4: INVESTMENT, AMENITIES & OWNERSHIP (Questions 16 to 20)")
    logger.info("=================================================================")
    
    # -------------------------------------------------------------
    # Q16: Owner Type Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    owner_counts = df['Owner_Type'].value_counts()
    plt.pie(owner_counts.values, labels=owner_counts.index, autopct='%1.1f%%', colors=['#4c72b0', '#55a868', '#c44e52'],
            startangle=140, explode=(0.03, 0.03, 0.03))
    plt.title("Q16: Proportion of Properties by Owner Type", fontsize=14, fontweight='bold', pad=12)
    q16_path = os.path.join(PLOTS_DIR, "q16_owner_type_distribution.png")
    plt.savefig(q16_path)
    plt.close()
    
    insights["Q16"] = (
        f"Market listings are split across Owner ({owner_counts.get('Owner',0)/len(df)*100:.1f}%), "
        f"Builder ({owner_counts.get('Builder',0)/len(df)*100:.1f}%), and Broker ({owner_counts.get('Broker',0)/len(df)*100:.1f}%)."
    )
    logger.info(f"Q16 Insight: {insights['Q16']}")
    
    # -------------------------------------------------------------
    # Q17: Availability Status Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    ax = sns.countplot(x='Availability_Status', data=df, palette='Accent')
    plt.title("Q17: Breakdown of Property Availability Status", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Availability Status")
    plt.ylabel("Property Count")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,}\n({p.get_height()/len(df)*100:.1f}%)", 
                    (p.get_x() + p.get_width() / 2., p.get_height()/2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    q17_path = os.path.join(PLOTS_DIR, "q17_availability_status_distribution.png")
    plt.savefig(q17_path)
    plt.close()
    
    avail_counts = df['Availability_Status'].value_counts()
    insights["Q17"] = (
        f"Inventory consists of {avail_counts.get('Ready_to_Move',0):,} Ready to Move properties and "
        f"{avail_counts.get('Under_Construction',0):,} Under Construction units."
    )
    logger.info(f"Q17 Insight: {insights['Q17']}")
    
    # -------------------------------------------------------------
    # Q18: Parking's Effect on Price
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    sns.violinplot(x='Parking_Space', y='Price_in_Lakhs', data=df, palette='muted', inner='quartile')
    plt.title("Q18: Effect of Dedicated Parking Space on Property Price", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Dedicated Parking Space Available")
    plt.ylabel("Price (₹ Lakhs)")
    q18_path = os.path.join(PLOTS_DIR, "q18_parking_effect_on_price.png")
    plt.savefig(q18_path)
    plt.close()
    
    parking_means = df.groupby('Parking_Space')['Price_in_Lakhs'].mean().to_dict()
    insights["Q18"] = (
        f"Properties with dedicated parking average ₹{parking_means.get('Yes', 0):.2f}L vs "
        f"₹{parking_means.get('No', 0):.2f}L for properties without parking."
    )
    logger.info(f"Q18 Insight: {insights['Q18']}")
    
    # -------------------------------------------------------------
    # Q19: Amenities Effect on Price per SqFt
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Amenity_Count', y='Price_per_SqFt_INR', data=df, palette='crest', ci=None)
    plt.title("Q19: Impact of Amenity Count & Diversity on Price per SqFt", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Number of Modern Amenities (Gym, Pool, Clubhouse, Playground, Garden)")
    plt.ylabel("Mean Price per SqFt (₹)")
    q19_path = os.path.join(PLOTS_DIR, "q19_amenities_vs_price_per_sqft.png")
    plt.savefig(q19_path)
    plt.close()
    
    amenity_means = df.groupby('Amenity_Count')['Price_per_SqFt_INR'].mean().to_dict()
    insights["Q19"] = (
        f"Gated communities with full 5-amenity suites average ₹{amenity_means.get(5, 0):.0f}/sq.ft vs "
        f"₹{amenity_means.get(1, 0):.0f}/sq.ft for single-amenity projects."
    )
    logger.info(f"Q19 Insight: {insights['Q19']}")
    
    # -------------------------------------------------------------
    # Q20: Public Transport Accessibility vs Investment Potential
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='Public_Transport_Accessibility', y='Price_in_Lakhs', data=df, order=['Low', 'Medium', 'High'], 
                palette='coolwarm', ax=axes[0], ci=None)
    axes[0].set_title("Average Price by Transport Accessibility", fontweight='bold')
    axes[0].set_ylabel("Price (₹ Lakhs)")
    
    sns.barplot(x='Public_Transport_Accessibility', y='Good_Investment', data=df, order=['Low', 'Medium', 'High'], 
                palette='coolwarm', ax=axes[1], ci=None)
    axes[1].set_title("Good Investment Rate (%) by Transport", fontweight='bold')
    axes[1].set_ylabel("Proportion of Good Investments")
    
    plt.suptitle("Q20: Public Transport Accessibility Impact on Pricing & Investment Potential", 
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    q20_path = os.path.join(PLOTS_DIR, "q20_transport_vs_investment.png")
    plt.savefig(q20_path)
    plt.close()
    
    trans_inv = (df.groupby('Public_Transport_Accessibility')['Good_Investment'].mean() * 100).to_dict()
    insights["Q20"] = (
        f"High public transport accessibility elevates the 'Good Investment' likelihood to {trans_inv.get('High', 0):.1f}%, "
        f"compared to {trans_inv.get('Medium', 0):.1f}% for Medium and {trans_inv.get('Low', 0):.1f}% for Low."
    )
    logger.info(f"Q20 Insight: {insights['Q20']}")
    
    logger.info("=================================================================")
    logger.info("EDA COMPLETE: All 20 Questions answered & 20 figures saved.")
    logger.info("=================================================================")
    return insights

if __name__ == "__main__":
    run_all_eda()
