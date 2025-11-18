Project Portfolio Summary
This portfolio contains three distinct data analysis projects, each with its own dataset:

Fish Farm IoT Analysis: Optimizing aquaculture through sensor data.

Agriculture Supply Chain Analysis: Exploring logistics and forecasting demand.

Plant Disease Prediction: Using machine learning to classify disease from environmental data.

1. Fish Farm IoT Analysis
Objective: Analyze IoT sensor data (water temperature, pH, TDS) to optimize fish farm operations and ensure fish health.

Key Activities & Results:

Data Processing: Cleaned and prepared time-series sensor data.

Trend & Anomaly Detection: Used visualizations and statistical methods (IQR) to identify parameter fluctuations and critical outliers that could indicate equipment failure or harmful conditions for fish.

Actionable Insights: Provided data-driven recommendations to reduce fish mortality and improve resource efficiency by maintaining optimal water quality.

Technologies: Python, Pandas, Matplotlib, Seaborn.

2. Agriculture Supply Chain Analysis
Objective: Analyze supply chain data to uncover inefficiencies and build a model for demand forecasting.

Key Activities & Results:

Exploratory Data Analysis (EDA): Visualized data to understand stock levels, demand patterns, and product distribution across warehouses and regions.

Feature Engineering: Created new features like Stock_to_Demand_Ratio to better capture supply chain dynamics.

Model Forecasting: Trained multiple models (Linear Regression, Random Forest, XGBoost) to predict Monthly_Demand. The models performed poorly, indicating the complex, non-linear nature of agricultural demand.

Business Recommendations: Despite model limitations, the EDA yielded valuable insights for improving inventory management for high-variance products and creating region-specific logistics strategies.

Technologies: Python, Pandas, Scikit-learn, XGBoost.

3. Plant Disease Prediction
Objective: Build a machine learning classifier to predict plant disease from environmental conditions (temperature, humidity, rainfall, soil pH).

Key Activities & Results:

Handled Class Imbalance: The dataset was initially skewed (76% healthy plants). Used SMOTE to synthetically generate data for the diseased class, preventing model bias.

Model Training & Evaluation: Compared K-Nearest Neighbors, Random Forest, and XGBoost classifiers. XGBoost and Random Forest performed best.

Key Drivers Identified: Feature importance analysis revealed soil_pH, rainfall, and humidity as the most critical factors for predicting disease, more so than temperature.

Successful Prediction: Achieved strong model performance (ROC AUC > 0.81), demonstrating a reliable link between environmental factors and disease presence.

Technologies: Python, Pandas, Scikit-learn, XGBoost, SMOTE.
