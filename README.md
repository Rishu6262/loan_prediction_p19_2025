# 🥇 Gold Price Prediction Using Machine Learning

## 📌 Project Overview

Gold is one of the most valuable and widely traded assets in the world. Its price is influenced by various economic, financial, and geopolitical factors such as stock market performance, oil prices, silver prices, and currency exchange rates.

This project aims to predict gold prices using Machine Learning techniques by analyzing historical financial market data. The system learns patterns from past market behavior and provides accurate predictions of gold prices based on key influencing factors.

The project demonstrates the practical application of Data Science, Machine Learning, and Financial Analytics in solving real-world forecasting problems.

---

# ❓ Why I Chose This Project?

Gold price prediction is a real-world financial forecasting problem that is highly relevant to investors, traders, financial analysts, and researchers.

I selected this project to:

* Understand financial market behavior.
* Learn predictive analytics techniques.
* Apply Machine Learning algorithms to real-world data.
* Improve data preprocessing and feature engineering skills.
* Build an end-to-end Machine Learning application.

This project helped me gain practical experience in financial data analysis and predictive modeling.

---

# 🚀 Project Objectives

* Predict gold prices using Machine Learning algorithms.
* Analyze relationships between gold prices and economic indicators.
* Perform exploratory data analysis on financial data.
* Compare multiple regression models.
* Identify the best-performing prediction model.
* Deploy the model through an interactive application.

---

# 📊 Dataset Information

### Dataset Name

Gold Price Dataset

### Dataset Size

* 2290 Records

### Features

| Feature | Description                     |
| ------- | ------------------------------- |
| Date    | Trading Date                    |
| SPX     | S&P 500 Stock Market Index      |
| GLD     | Gold Price (Target Variable)    |
| USO     | United States Oil Fund Price    |
| SLV     | Silver Price                    |
| EUR/USD | Euro to US Dollar Exchange Rate |

---

# 🔍 Understanding the Features

### SPX

Represents the S&P 500 stock market index and indicates overall market performance.

### USO

Represents oil prices which often influence inflation and commodity markets.

### SLV

Represents silver prices, which are closely related to gold prices.

### EUR/USD

Represents currency exchange rates that affect global commodity pricing.

### GLD

Represents gold price and serves as the target variable for prediction.

---

# 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Streamlit

### Machine Learning

* Regression Models
* Model Evaluation
* Feature Engineering

---

# 📂 Project Structure

```bash
Gold_Price_Prediction/
│
├── app.py
├── model.pkl
├── Gold Loan.csv
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── Gold_Price_Prediction.ipynb
│
└── assets/
    └── screenshots/
```

---

# 🔎 Exploratory Data Analysis (EDA)

The following analyses were performed:

### Data Inspection

* Dataset Shape
* Data Types
* Missing Values

### Statistical Analysis

* Mean
* Median
* Standard Deviation
* Correlation Analysis

### Visualizations

* Gold Price Trend
* Oil Price Trend
* Silver Price Trend
* Correlation Heatmap
* Distribution Plots
* Pair Plots

---

# 📈 Data Preprocessing

The following preprocessing steps were applied:

### Data Cleaning

* Checked Missing Values
* Removed Inconsistent Records

### Feature Selection

Selected the most relevant features:

* SPX
* USO
* SLV
* EUR/USD

### Train-Test Split

```python
train_test_split()
```

Used to split data into training and testing datasets.

---

# 🤖 Machine Learning Models Used

## 1. Linear Regression

Advantages:

* Simple and Interpretable
* Fast Training
* Good Baseline Model

---

## 2. Decision Tree Regressor

Advantages:

* Captures Nonlinear Relationships
* Easy Interpretation

---

## 3. Random Forest Regressor

Advantages:

* High Accuracy
* Reduced Overfitting
* Robust Performance

---

# ⚙️ Model Training

The dataset was trained using supervised learning techniques.

Training Steps:

1. Load Dataset
2. Perform Data Cleaning
3. Conduct EDA
4. Select Features
5. Split Dataset
6. Train Models
7. Evaluate Models
8. Save Best Model

---

# 📊 Model Evaluation Metrics

The models were evaluated using:

### Mean Absolute Error (MAE)

Measures average prediction error.

### Mean Squared Error (MSE)

Measures squared prediction errors.

### Root Mean Squared Error (RMSE)

Provides error in original units.

### R² Score

Measures how well the model explains variance.

---

# 🏆 Best Model Selection

The best model was selected based on:

* Highest R² Score
* Lowest MAE
* Lowest RMSE
* Better Generalization Performance

Models Compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* K-Nearest Neighbors (KNN)

---

# 📉 Business Impact

This system can help:

* Investors
* Traders
* Financial Analysts
* Researchers

by providing data-driven insights into gold price movements.

---

# 💻 Streamlit Web Application

A user-friendly Streamlit application was developed.

### User Inputs

* SPX Value
* Oil Price (USO)
* Silver Price (SLV)
* EUR/USD Exchange Rate

### Output

* Predicted Gold Price

The prediction is generated instantly based on the trained model.

---

# ▶️ Installation Guide

### Clone Repository

```bash
git clone https://github.com/yourusername/gold-price-prediction.git
```

### Navigate to Project Folder

```bash
cd gold-price-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

# 🎯 Learning Outcomes

Through this project, I learned:

* Financial Data Analysis
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Regression Modeling
* Model Evaluation
* Hyperparameter Tuning
* Streamlit Deployment
* End-to-End Machine Learning Workflow

---

# 🔮 Future Improvements

* Real-Time Gold Price API Integration
* Deep Learning Forecasting Models
* Time Series Analysis
* Interactive Dashboard
* Investment Recommendation System
* Market Trend Prediction

---

# 📜 Disclaimer

This project is developed for educational and research purposes only.

The predicted gold prices are generated using machine learning models trained on historical market data. The results should not be considered financial, trading, or investment advice.

Users should perform their own research before making any financial decisions.

---

# ✅ Conclusion

This project demonstrates how Machine Learning can be applied to financial market data for predicting gold prices. By analyzing stock market indices, oil prices, silver prices, and currency exchange rates, the system successfully identifies patterns that influence gold price movements. The project showcases the practical implementation of predictive analytics, regression modeling, and financial forecasting using Machine Learning.

---

# 👨‍💻 Author

**Rishu Gurjar**

Aspiring Data Science | Machine Learning Enthusiast | Python Developer

### Skills

* Python
* SQL
* Machine Learning
* Data Analysis
* Streamlit
* Scikit-Learn

Connect with me on LinkedIn and GitHub to explore more Data Science and Machine Learning projects.
