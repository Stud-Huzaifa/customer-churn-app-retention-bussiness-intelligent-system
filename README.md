# 🚀 Customer Churn Prediction & Retention Intelligence System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?style=for-the-badge\&logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge\&logo=streamlit)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-success?style=for-the-badge)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-2088FF?style=for-the-badge\&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

An end-to-end **Machine Learning Engineering** project that predicts customer churn using customer demographics, purchasing behavior, engagement metrics, and support interactions.

The project demonstrates the complete ML lifecycle—from **Business Understanding → Exploratory Data Analysis → Feature Engineering → Model Development → Hyperparameter Tuning → Streamlit Deployment → CI/CD → Monitoring**.

---

# 📌 Project Overview

Customer retention is one of the most important challenges for online businesses. Losing existing customers reduces revenue and increases acquisition costs.

This project develops a machine learning solution that predicts customers who are likely to churn before they become inactive, allowing businesses to launch targeted retention campaigns instead of offering discounts to every customer.

Rather than focusing only on model accuracy, this project demonstrates a complete production-inspired machine learning workflow including:

* 📊 Business Understanding & EDA
* 🧹 Data Cleaning & Feature Engineering
* 🤖 Machine Learning Model Development
* 📈 Model Evaluation & Hyperparameter Tuning
* 🖥 Interactive Streamlit Dashboard
* 📦 Modular Python Package
* 🔄 CI/CD Pipeline
* 📡 Input Validation & Monitoring

---

# 💼 Business Problem

Imagine an online fashion retailer with thousands of customers.

Every month, some customers stop purchasing without warning.

Traditional marketing campaigns send promotions to every customer, wasting both time and budget.

The goal of this project is to identify customers at risk of churn **before** they leave so that the business can:

* Improve customer retention
* Reduce marketing costs
* Increase customer lifetime value
* Prioritize high-risk customers
* Support data-driven decision making

---

# 🎯 Project Objectives

* Analyze customer purchasing behavior.
* Discover factors influencing churn.
* Engineer meaningful predictive features.
* Train and compare multiple ML models.
* Optimize the best-performing model.
* Deploy predictions through Streamlit.
* Build a modular, production-inspired project structure.
* Demonstrate lightweight CI/CD and monitoring.

---

# 🏗️ Solution Architecture

```text
Customer Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Gradient Boosting Model
      │
      ▼
Churn Probability
      │
      ▼
Risk Segmentation
      │
      ▼
Interactive Streamlit Dashboard
```

---

# 📂 Dataset

The dataset represents customer activity from an online fashion retailer.

| Category             | Features                                                |
| -------------------- | ------------------------------------------------------- |
| Customer Information | Age, Gender, Location                                   |
| Purchase Behavior    | Purchase Frequency, Total Spending, Average Order Value |
| Engagement           | Website Visits, Email Open Rate                         |
| Customer Service     | Customer Support Tickets                                |
| Engineered Features  | Customer Tenure, Log Features, Missing Indicators       |

**Dataset Size**

* Approximately **50,000 customer records**
* **49,584** records after cleaning
* Binary classification problem (Customer Churn)

---
# 🔍 Exploratory Data Analysis (EDA)

Before building any machine learning model, a comprehensive Exploratory Data Analysis (EDA) was performed to understand customer behavior, assess data quality, identify hidden patterns, and detect potential issues that could impact model performance.

### Data Quality Assessment

The following data quality checks were completed:

* ✅ Removed **416 duplicate customer records**
* ✅ Handled missing values appropriately
* ✅ Identified **310 future registration dates** and excluded them from tenure calculations
* ✅ Investigated outliers instead of blindly removing them
* ✅ Analyzed feature distributions and correlations
* ✅ Identified and removed potential data leakage features

---

## 📊 Key Business Insights

The analysis revealed several important customer behavior patterns:

### 🛒 Purchase Frequency

Purchase Frequency emerged as the strongest indicator of customer loyalty.

Customers who purchased more frequently consistently showed:

* Higher total spending
* Higher website engagement
* Better email interaction
* Lower customer support requests

---

### 🌐 Website Engagement

Website activity strongly correlated with customer retention.

Highly engaged customers:

* Purchased more frequently
* Spent significantly more
* Opened more marketing emails
* Had lower churn risk

---

### 📧 Email Engagement

Email Open Rate acted as a useful engagement signal.

Customers interacting with marketing campaigns were generally more active and less likely to churn.

Instead of removing missing values, an **Email_Open_Rate_Missing** indicator feature was engineered to preserve potentially useful information.

---

### 🎧 Customer Support Activity

Customers with a higher number of support tickets tended to exhibit greater churn risk.

This suggests unresolved issues or dissatisfaction may contribute to customer attrition.

---

### 👥 Demographics

Features such as **Age**, **Gender**, and **Location** provided useful customer context but had considerably lower predictive power compared to behavioral variables.

---

# ⚙️ Feature Engineering

Several new features were engineered to improve predictive performance while preserving business meaning.

| Engineered Feature      | Purpose                         |
| ----------------------- | ------------------------------- |
| Customer_Tenure_Days    | Measure customer lifetime       |
| Total_Spending_Log      | Reduce skewness                 |
| Average_Order_Value_Log | Normalize spending distribution |
| Email_Open_Rate_Missing | Preserve missing information    |

---

# 🚨 Data Leakage Prevention

One of the most important decisions during preprocessing was preventing **data leakage**.

The feature:

```text
Days_Since_Last_Purchase
```

was used to define the churn target.

Including this feature during training would allow the model to indirectly learn the answer, resulting in unrealistically high performance.

To ensure fair evaluation and real-world applicability, this feature was intentionally excluded from the final training dataset.

---

# 📈 Machine Learning Pipeline

The complete workflow followed a production-inspired pipeline:

```text
Raw Customer Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Train/Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Hyperparameter Optimization
        │
        ▼
Final Prediction
```

The preprocessing pipeline includes:

* Median imputation for numerical features
* Most frequent imputation for categorical features
* Standard Scaling
* One-Hot Encoding
* Automatic handling of unseen categories during prediction

This ensures consistent preprocessing during both training and deployment.

---
# 🤖 Model Development & Results

To identify the most suitable algorithm for customer churn prediction, five supervised machine learning models were trained and evaluated using the same preprocessing pipeline.

### Models Evaluated

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors (KNN)

Rather than selecting a model based solely on accuracy, evaluation focused on metrics that are more appropriate for churn prediction, including **Precision**, **Recall**, **F1 Score**, and **ROC-AUC**.

---

# 📊 Model Performance

| Model                   |  Accuracy  |  Precision |   Recall   |  F1 Score  |  ROC-AUC  |
| ----------------------- | :--------: | :--------: | :--------: | :--------: | :-------: |
| Logistic Regression     |   76.76%   |   65.98%   |   61.07%   |   63.43%   |   0.831   |
| Decision Tree           |   69.09%   |   53.10%   |   54.51%   |   53.79%   |   0.654   |
| Random Forest           |   76.78%   |   66.55%   |   59.59%   |   62.88%   |   0.826   |
| **Gradient Boosting** ⭐ | **77.59%** | **66.78%** | **63.86%** | **65.29%** | **0.837** |
| KNN                     |   73.46%   |   60.48%   |   56.49%   |   58.42%   |   0.768   |

---

# 🏆 Final Model Selection

Among all evaluated models, **Gradient Boosting** delivered the best overall performance.

It achieved:

* Highest Accuracy
* Highest Recall
* Highest F1 Score
* Highest ROC-AUC

More importantly, it demonstrated strong generalization on unseen customer data, making it the most suitable model for deployment.

---

# ⚙️ Hyperparameter Optimization

To further improve performance, **Optuna** was used for automated hyperparameter optimization.

### Optimization Details

* **Framework:** Optuna
* **Trials:** 50
* **Objective:** Maximize F1 Score

### Best Parameters

```python
{
    "n_estimators": 152,
    "learning_rate": 0.0277,
    "max_depth": 4,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "subsample": 0.711
}
```

Although Optuna identified a better parameter combination, the performance improvement was marginal. This indicates that the baseline Gradient Boosting model was already well-optimized for the available dataset.

---

# 📉 Feature Importance

The trained model identified the following features as the strongest predictors of customer churn:

| Rank | Feature                  |
| ---- | ------------------------ |
| 🥇   | Purchase Frequency       |
| 🥈   | Website Visits           |
| 🥉   | Email Open Rate          |
| 4    | Customer Support Tickets |
| 5    | Total Spending (Log)     |
| 6    | Customer Tenure          |
| 7    | Average Order Value      |
| 8    | Age                      |

### Key Insight

Behavioral features consistently outperformed demographic features, highlighting that **how customers interact with the business** is far more predictive of churn than **who the customers are**.

---

# 🎯 Customer Risk Segmentation

Instead of returning only binary predictions, the application converts churn probabilities into business-friendly customer segments.

| Churn Probability | Risk Level     |
| ----------------- | -------------- |
| < 30%             | 🟢 Low Risk    |
| 30% – 70%         | 🟡 Medium Risk |
| > 70%             | 🔴 High Risk   |

This allows marketing teams to prioritize retention efforts and allocate resources more effectively.

---

# 💼 Business Value

The final solution enables businesses to:

* Identify high-risk customers before they churn.
* Launch targeted retention campaigns.
* Improve marketing efficiency.
* Reduce customer acquisition costs.
* Increase customer lifetime value.
* Make data-driven retention decisions instead of relying on intuition.

---

# 📌 Key Achievements

✔ End-to-End Machine Learning Pipeline

✔ Business-Oriented Feature Engineering

✔ Data Leakage Prevention

✔ Hyperparameter Optimization with Optuna

✔ Explainable Feature Importance

✔ Customer Risk Segmentation

✔ Deployable Streamlit Application

✔ Production-Inspired Project Structure

---
# 🖥️ Streamlit Application

To make the machine learning model accessible to non-technical users, an interactive **Streamlit** web application was developed.

### Features

* 📂 Upload customer dataset (CSV)
* ✅ Automatic schema validation
* 🤖 Predict customer churn
* 📊 Generate churn probability
* 🎯 Risk segmentation (Low / Medium / High)
* 📈 Business summary dashboard
* 📥 Download prediction results

The application uses the same preprocessing pipeline and trained model used during development, ensuring consistent predictions.

---

# 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Hyperparameter_Tuning.ipynb
│   └── 04_Prediction.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
├── reports/
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🔄 CI/CD & Monitoring

To improve reliability and maintainability, the project follows a lightweight MLOps approach.

### Continuous Integration

* GitHub Actions
* Automated smoke tests
* Dependency installation
* Prediction pipeline validation
* Streamlit application syntax checks

### Monitoring

The application includes basic monitoring through:

* Input schema validation
* Missing column detection
* Invalid value checks
* Prediction probability monitoring
* Customer risk distribution summary

These checks help ensure the application behaves reliably when new customer datasets are uploaded.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git

cd customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

# 📚 Tech Stack

| Category              | Technologies        |
| --------------------- | ------------------- |
| Programming           | Python              |
| Data Analysis         | Pandas, NumPy       |
| Machine Learning      | Scikit-learn        |
| Hyperparameter Tuning | Optuna              |
| Deployment            | Streamlit           |
| Visualization         | Matplotlib, Seaborn |
| Model Persistence     | Joblib              |
| Version Control       | Git & GitHub        |
| CI/CD                 | GitHub Actions      |

---

# 🔮 Future Improvements

* Docker containerization
* FastAPI REST API
* MLflow experiment tracking
* Evidently AI for data drift monitoring
* AWS deployment
* Batch prediction pipeline
* Automated model retraining
* Database integration
* Authentication and role-based access

---

# 🤝 AI-Assisted Development

This project was developed as a learning-focused Machine Learning Engineering project.

AI-assisted tools were used to accelerate portions of the Python module organization, Streamlit interface, and project structuring. However, all critical aspects of the project—including business understanding, exploratory data analysis, feature engineering, preprocessing decisions, model evaluation, debugging, data leakage prevention, deployment strategy, and overall architecture—were reviewed, understood, and validated throughout development.

The objective was not simply to generate code, but to understand and implement an end-to-end machine learning solution following modern development practices.

---

# 👨‍💻 Author

**Muhammad Huzaifa**

Machine Learning Engineer • Data Scientist • Business Automation Specialist

📧 Connect with me on LinkedIn to discuss Machine Learning, Data Science, MLOps, and Business Automation.

If you found this project useful or interesting, consider giving it a ⭐ on GitHub.

---

# 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more information.

---

## ⭐ Final Thoughts

This project demonstrates a complete end-to-end Machine Learning Engineering workflow—from business understanding and exploratory data analysis to model deployment and lightweight MLOps practices.

Rather than focusing solely on predictive performance, the project emphasizes building a practical, maintainable, and business-oriented solution capable of supporting real-world customer retention strategies.
