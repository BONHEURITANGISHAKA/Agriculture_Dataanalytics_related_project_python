#THIS PROJECT CONTAIN THREE PROJECT BOTH WITH ITS DATASET: 
1.FISH_FARM_IOT_ANALYSIS
2.AGRICULTURE_SUPPLY_CHAIN_ANALYSIS
3.PLANT_DISEASES_PREDICTION



1. # Fish_Farm_IoT_Analysis
“Fish Farm IoT Sensor Data Analysis with Python”

# Fish Farm IoT Data Analysis: Optimizing Aquaculture Operations

## Project Overview

This project presents a comprehensive data analysis of IoT sensor data from a fish farm, focusing on key water quality parameters: water temperature, pH, and Total Dissolved Solids (TDS). Leveraging Python with libraries such as Pandas, Matplotlib, and Seaborn, this analysis aims to identify trends, detect anomalies, and derive actionable insights to optimize fish farm operations, ensure fish health, and enhance productivity.

## Key Features

-   **Data Understanding & Preprocessing:** Thorough inspection of dataset structure, data types, and initial statistics, followed by cleaning (datetime conversion, missing value, and duplicate handling).
-   **Exploratory Data Analysis (EDA) with Modern Visuals:** In-depth analysis of water temperature, pH, and TDS distributions using attractive histograms, box plots, and density plots. Visualization of parameter relationships through pair plots.
-   **Time-Series Trend Analysis:** Engaging time-series plots to visualize the temporal patterns and fluctuations of water temperature, pH, and TDS, revealing critical operational dynamics.
-   **Anomaly Detection:** Implementation of a statistical method (IQR-based outlier detection) to identify unusual readings in water parameters, indicating potential system malfunctions or critical conditions affecting fish health.
-   **Actionable Insights & Recommendations:** Formulation of clear, business-impact-focused recommendations based on analytical findings, aimed at optimizing operational protocols, preventing issues, and improving overall farm efficiency and profitability.

## Business Impact & Value Proposition

This project demonstrates the ability to transform raw IoT data into strategic business insights. By identifying critical thresholds and potential operational inefficiencies, it proposes solutions that can lead to:

-   **Reduced Fish Stress & Mortality:** Proactive management of water quality parameters.
-   **Optimized Resource Utilization:** Efficient feeding and water management strategies.
-   **Enhanced Operational Efficiency:** Implementation of automated alert systems and root cause analysis protocols.
-   **Increased Yield & Profitability:** By maintaining optimal environmental conditions for fish.

This analysis showcases strong skills in data manipulation, statistical analysis, modern data visualization, and the ability to derive commercially relevant insights from complex datasets – all crucial for roles in data science and analytics, particularly within the aquaculture or IoT sectors.

## Technologies Used

-   **Python:** Programming language
-   **Pandas:** Data manipulation and analysis
-   **Matplotlib:** Data visualization
-   **Seaborn:** Enhanced statistical data visualization



2  Agriculture Supply Chain Analysis and Demand Forecasting

## Project Overview
This project aims to perform a comprehensive analysis of an agriculture supply chain dataset. It encompasses data preprocessing, exploratory data analysis (EDA) with modern visualizations, feature engineering, and the development and evaluation of machine learning models for demand forecasting. The ultimate goal is to derive actionable insights and recommendations to improve supply chain efficiency.

## Dataset
The dataset contains information related to various aspects of an agriculture supply chain, including warehouse details, product types, stock levels, monthly demand, delivery frequencies, lead times, supplier ratings, storage capacities, and months.

## Methodology
1.  **Data Preprocessing and Cleaning:** Handled missing values (none found), duplicate rows (none found), and reviewed data types. Categorical features (`Warehouse_ID`, `Region`, `Product`, `Month`) were one-hot encoded.
2.  **Exploratory Data Analysis (EDA):** Visualized distributions of numerical and categorical variables, identified correlations, and explored patterns relevant to supply chain dynamics using histograms, box plots, bar plots, and correlation heatmaps.
3.  **Feature Engineering:** Created new features such as `Stock_to_Demand_Ratio`, `Storage_Utilization`, and `Demand_Lead_Time_Ratio` to enhance predictive power.
4.  **Model Development and Evaluation:** Defined `Monthly_Demand_units` as the target variable. Trained and evaluated `LinearRegression`, `RandomForestRegressor`, and `XGBRegressor` models for demand forecasting.

## Key Findings from EDA
*   **Warehouse and Region Distribution:** The dataset shows a fairly balanced distribution of records across different warehouses and regions.
*   **Product Distribution:** Maize, Potatoes, Beans, and Sorghum are the primary products, with Maize being slightly more frequent.
*   **Monthly Demand Patterns:** The average monthly demand shows some seasonal variation, with peaks and troughs across different months.
*   **Stock Levels:** Stock levels vary across warehouses, indicating potential differences in storage strategies or demand in those locations.
*   **Numerical Variable Distributions:** Most numerical variables appear to be relatively well-distributed, without extreme outliers that would severely distort basic statistical measures.
*   **Correlation:** A heatmap revealed weak correlations between most numerical features, suggesting that demand might be influenced by a complex interplay of factors not strongly linear with single features.

## Model Performance Summary
All models demonstrated poor predictive performance on the test set, as indicated by negative R-squared scores, which suggests they perform worse than simply predicting the mean of the target variable.

*   **Linear Regression:** MSE = 79474.62, R-squared = -0.08
*   **Random Forest Regressor:** MSE = 79754.87, R-squared = -0.08
*   **XGBoost Regressor:** MSE = 90817.95, R-squared = -0.23

The poor performance is likely due to the limited feature set, the complex nature of agricultural demand patterns, the dataset's size, and the lack of explicit time-series handling.

## Actionable Insights and Recommendations
Despite the current model limitations, the EDA and general supply chain principles allow for several recommendations:

1.  **Inventory Management for High-Variance Products:** Products like Potatoes and Sorghum, which may exhibit higher monthly demand variability, require more careful inventory adjustments and safety stock planning to prevent stockouts or excessive holding costs.
2.  **Optimized Delivery Frequencies:** Given lead times, explore strategies for more frequent, smaller deliveries to reduce storage costs and spoilage risk, especially for perishable goods.
3.  **Regional and Warehouse-Specific Strategies:** Implement tailored inventory and logistics approaches for each region and warehouse, as variations in stock levels and demand suggest a one-size-fits-all strategy is inefficient.
4.  **Supplier Relationship Management:** Maintain strong relationships with highly-rated suppliers to ensure reliability during peak demand or disruptions.


#3.Plant Disease Prediction using Machine Learning
Project Overview
This project focuses on building and evaluating machine learning models to predict the presence of plant disease based on environmental factors such as temperature, humidity, rainfall, and soil pH. The goal is to demonstrate a complete data science workflow, from exploratory data analysis and preprocessing to model training, evaluation, and interpretation. This project showcases proficiency in data analysis, machine learning fundamentals, and effective communication of results.

Dataset
The dataset plant_disease_dataset.csv contains environmental measurements and a target variable indicating the presence (1) or absence (0) of plant disease. It includes the following features:

temperature (float): Average temperature
humidity (float): Average humidity
rainfall (float): Amount of rainfall
soil_pH (float): Soil pH level
disease_present (int): Target variable (0: No Disease, 1: Disease Present)
Methodology
The project followed a structured approach:

Data Loading and Inspection: Loaded the dataset into a Pandas DataFrame and performed initial checks for structure, data types, and missing values using df.info() and df.describe().

Exploratory Data Analysis (EDA): Conducted a thorough EDA to understand the data distribution, relationships, and potential issues:

Visualized the distribution of the target variable (disease_present) to identify class imbalance.
Generated histograms with KDE for numerical features (temperature, humidity, rainfall, soil_pH) to observe their distributions.
Used box plots to examine the relationship between each numerical feature and the disease_present target.
Plotted a correlation matrix heatmap to understand pairwise relationships between features.
Identified potential outliers in numerical features using box plots.
Data Preprocessing: Prepared the data for machine learning models:

Separated features (X) from the target variable (y).
Identified a significant class imbalance in the disease_present variable (approximately 76% 'No Disease' vs. 24% 'Disease Present').
Performed a stratified train-test split (80% train, 20% test) to ensure consistent class distribution across sets.
Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to balance the classes, mitigating bias towards the majority class.
Scaled numerical features using StandardScaler to standardize their range, which is crucial for distance-based and gradient-descent algorithms.
Model Training and Evaluation: Trained and evaluated three different classification models:

K-Nearest Neighbors (KNN) Classifier
Random Forest Classifier
XGBoost Classifier Each model's performance was assessed using key metrics: Accuracy, Precision, Recall, F1-Score, and ROC AUC. Confusion matrices were generated, and ROC curves were plotted for visual comparison.
Feature Importance Analysis: Analyzed feature importance for Random Forest and XGBoost models to understand which environmental factors were most influential in predicting disease presence.

Key Findings
Class Imbalance: The initial dataset exhibited a significant class imbalance, which was effectively addressed using SMOTE during preprocessing.
Model Performance: Both XGBoost Classifier and Random Forest Classifier demonstrated strong and comparable performance, significantly outperforming the K-Nearest Neighbors model.
XGBoost achieved the highest ROC AUC (0.8158), indicating excellent discriminatory power.
Random Forest also performed very well with a high ROC AUC (0.8092) and F1-Score (0.6584).
Feature Importance: soil_pH, rainfall, and humidity were consistently identified as the most important features by both Random Forest and XGBoost models, suggesting their critical role in predicting plant disease. temperature had a comparatively lower impact.
Skills Demonstrated
Data Analysis & Visualization: Pandas, NumPy, Matplotlib, Seaborn
Machine Learning: Scikit-learn (Logistic Regression, RandomForestClassifier, KNeighborsClassifier, StandardScaler, train_test_split), Imbalanced-learn (SMOTE), XGBoost
Model Evaluation: Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrices
Data Preprocessing: Handling class imbalance, feature scaling
Problem Solving & Critical Thinking: Interpreting model results, understanding limitations, and proposing next steps.
Future Enhancements
Hyperparameter Tuning: Optimize the best-performing models (XGBoost and Random Forest) using techniques like GridSearchCV or RandomizedSearchCV.
Further Feature Engineering: Explore creating new features from existing ones to potentially improve model performance.
Deployment: Consider deploying the best model as an API for real-time predictions.
