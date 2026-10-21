# TriMind AI 🤖
**Three Challenges. Three Predictions. One AI Platform.**

A unified Streamlit intelligence platform that turns raw business data into decisions. This project fulfills the Data Science and Machine Learning course requirements by building three distinct ML applications (Regression & Classification) and deploying them on Streamlit Cloud.

🔗 **Live Application URL:** [https://trimindai777760.streamlit.app](https://trimindai777760.streamlit.app)

---

## 📌 Project Details

### 1. Sales Analytics & Profit Prediction (Regression)
*   **Business Problem:** A retail company wants to understand its sales performance, identify profitable products, and predict the expected profit from future orders.
*   **Dataset:** Retail order data (Includes Sales, Discount, Region, Category, Profit, etc.).
*   **Models Used:** Linear Regression, Random Forest Regressor, Neural Network (MLP).
*   **Best Model:** Random Forest Regressor (Evaluated on R², RMSE, MAE).

### 2. Customer Churn Prediction (Classification)
*   **Business Problem:** A subscription-based company is losing customers and wants to identify which customers are at high risk of leaving to offer retention incentives.
*   **Dataset:** Telecom customer data (Includes Tenure, Monthly Charges, Contract Type, Satisfaction Score, Churn status).
*   **Models Used:** Logistic Regression, Random Forest, Neural Network (MLP).
*   **Best Model:** Logistic Regression / Random Forest (Evaluated on F1-Score and ROC-AUC to handle class imbalance).

### 3. HR Analytics & Employee Attrition (Classification)
*   **Business Problem:** An organization wants to understand why employees are leaving and predict attrition risk using ethical ML practices.
*   **Dataset:** HR workforce data (Includes Age, Monthly Income, Job Satisfaction, OverTime, Attrition status).
*   **Models Used:** Logistic Regression, Random Forest, Neural Network (MLP).
*   **Best Model:** Random Forest Classifier (Evaluated on F1-Score, Precision, and Recall).

---

## 🛠️ Tools & Technologies
*   **Language:** Python
*   **Data Handling & EDA:** Pandas, NumPy, Matplotlib, Seaborn, Plotly
*   **Machine Learning:** Scikit-Learn
*   **Web Framework:** Streamlit
*   **Model Saving:** Joblib
*   **Deployment:** Streamlit Community Cloud, GitHub

---

## 🧹 Data Cleaning & Feature Engineering
*   Converted date columns to datetime objects and extracted `Order_Year` and `Order_Month`.
*   Handled missing values by imputing medians for numerical columns and modes for categorical columns.
*   Removed duplicates and dropped ID columns to prevent data leakage.
*   Encoded target variables (Yes/No to 1/0) and applied One-Hot Encoding to categorical features.

---

## 📊 Exploratory Data Analysis (EDA) & Visualizations
*   Built **Interactive Plotly charts** (Line charts for trends, Bar charts for comparisons, Pie charts for distributions).
*   Created **KPI Dashboards** to show high-level metrics (Total Sales, Churn Rate, Attrition Rate).
*   Generated **Correlation Heatmaps** and **Box Plots** using Matplotlib/Seaborn to detect outliers and relationships.

---

## 🤖 Machine Learning Pipeline & Evaluation
*   Used `ColumnTransformer` and `Pipeline` to prevent data leakage during preprocessing.
*   Trained and compared multiple models.
*   Evaluated using **Regression Metrics** (R², RMSE, MAE) and **Classification Metrics** (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
*   Implemented **Threshold Analysis** for Churn to balance Precision and Recall based on business needs.
*   Saved the best performing models as `.pkl` files inside the `models/` directory.

---

## ✨ Streamlit Application Features
*   **Home Page:** Interactive UI with animated visuals and room navigation.
*   **Dataset Explorer:** Filter data by region, category, and download the filtered CSV.
*   **Dashboards:** KPI cards, dynamic charts, and interactive tabs.
*   **Model Evaluation:** Live training progress, model comparison tables, and confusion matrices.
*   **Live Prediction:** Dynamic forms to input data and get real-time predictions with probability scores.
*   **Auto HTML Reports:** Downloadable HTML reports containing model metrics and charts.
*   **Responsible ML:** Ethical guidelines and fairness evaluation shown on the HR page.

---

## 💡 Business Recommendations (Findings & Actions)

1.  **Sales - Discount Strategy:** High discounts result in negative profit. *Action:* Cap discounts at 20% to maintain profitability.
2.  **Sales - Regional Focus:** The West region has the highest profit margin. *Action:* Increase marketing budget in the West.
3.  **Churn - Contract Type:** Month-to-month contracts have the highest churn. *Action:* Offer a 15% discount for switching to annual contracts.
4.  **Churn - Support:** Customers with multiple support tickets churn more. *Action:* Improve customer success response time.
5.  **HR - Overtime:** Employees working overtime have a 3x higher attrition rate. *Action:* Hire more staff to reduce overtime.
6.  **HR - Income:** Lower-income bands show high attrition. *Action:* Review compensation for entry-level roles.

---

## 💻 Local Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/attarehmanrao-source/TriMind-AI.git