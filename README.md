# 🏠 Airbnb Price Prediction in NYC

## 📌 Overview

This project investigates the factors that affect Airbnb listing prices in New York City and builds predictive models to estimate prices based on listing features. The goal is to uncover insights that benefit hosts, travelers, and the platform itself, while also tackling a real-world machine learning problem end-to-end.

---

## 🎯 Objectives

- Understand what drives Airbnb pricing in NYC.
- Build regression models to predict listing prices from features.
- Perform hypothesis testing to explore pricing behavior.
- Use unsupervised learning to segment listings into meaningful clusters.

---

## 📊 Dataset

- **Source**: [Kaggle: NYC Airbnb Open Data](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata/data)
- **Size**: Over 100,000 rows, 26 features.
- **Key Features**:
  - `neighbourhood`, `room_type`, `instant_bookable`, `cancellation_policy`, `reviews`, `minimum_nights`, `price`, `service_fee`, etc.

---

## 🧹 Data Preprocessing

- Removed duplicates and irrelevant features.
- Handled missing values (NaNs) by:
  - Dropping columns with excessive NaNs.
  - Filling others with the mean/median.
- Cleaned formatting (e.g. removed dollar signs from price).
- Converted data types (e.g., `price` to numeric, `date` to timestamp).

---

## 📈 Exploratory Data Analysis (EDA)

- Visualized the distribution of listings across neighborhoods.
- Analyzed price distributions across:
  - Room types
  - Availability
  - Review counts
- Identified key drivers of price:
  - Room type, number of reviews, neighborhood.
- Detected anomalies in minimum night stays and price outliers.

---

## 🤖 Modeling

### 🔹 Linear Regression (Baseline)
- Target: `log(price)`
- Features: 12 selected from EDA
- **RMSE**: 0.729 | **R²**: -0.0013 (very low explanatory power)

### 🔹 Gradient Boosted Trees
- Baseline: RMSE = 0.7278, R² = 0.0029
- After tuning:
  - `n_estimators=500`, `learning_rate=0.01`
  - RMSE = 0.7284, R² = 0.0014

### 🔹 Random Forest Regression
- Baseline: RMSE = 0.7268, R² = 0.0057
- After tuning:
  - `n_estimators=100`, `max_depth=15`
  - RMSE = 0.7229, R² = 0.0163

### 🔹 XGBoost (Best Model)
- Grid Search Parameters:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [3, 5, 7, 10]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
  - `subsample`, `colsample_bytree`
- **Best RMSE**: 0.7008 | **Best R²**: 0.0756

---

## 📦 Unsupervised Learning

### K-Means Clustering

- Objective: Discover latent groups in listings.
- Used Elbow Method → chose 5 clusters.
- Cluster interpretation:
  - Balanced popular listings
  - High availability, low demand
  - Outliers / legacy listings
  - Budget listings
  - Seasonal premium listings

---

## 🔬 Hypothesis Testing

### 1. Instant Bookable Listings
- H₀: No price difference
- H₁: Instant bookable listings are more expensive
- Result: **p = 0.4178** → Fail to reject H₀

### 2. Listings with More Reviews
- H₀: No price difference
- H₁: Listings with more reviews have higher prices
- Result: **p = 0.0437** → Reject H₀ at 5% level

---

## 💡 Insights

- Listings with many reviews are priced higher → social proof matters.
- Room type and neighborhood strongly impact price.
- K-means clustering reveals strategic pricing and availability patterns.
- Models struggled due to missing visual/aesthetic features and booking behavior data.

---

## 👥 Authors

- **Lesley Zhao** (lesleyzh@seas.upenn.edu)
- **Jianing Cai** (cjianing@seas.upenn.edu)
