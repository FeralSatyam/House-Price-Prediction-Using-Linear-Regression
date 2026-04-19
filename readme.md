# California Housing Price Prediction

## Project Overview

This project implements **linear regression with gradient descent from scratch** to predict median house prices in California districts using the 1990 census data. The entire implementation uses only NumPy and pandas — no scikit-learn for the core algorithm.

This was completed as a hands-on extension of Andrew Ng's Machine Learning Specialization (Week 2), focusing on:
- Multiple linear regression
- Batch gradient descent
- Feature scaling and standardization
- One-hot encoding for categorical variables
- Model evaluation metrics (R², RMSE, MAE)

## Dataset

**Source:** [California Housing Prices on Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

**Size:** 20,640 districts | 10 features

**Features:**
- `longitude`, `latitude` - Geographic coordinates
- `housing_median_age` - Median age of houses in district
- `total_rooms`, `total_bedrooms` - Room counts
- `population`, `households` - Population metrics
- `median_income` - District median income
- `ocean_proximity` - Categorical (NEAR BAY, `<1H OCEAN`, INLAND, NEAR OCEAN, ISLAND)
- `median_house_value` - Target variable (capped at $500,001)

## Model Architecture

**Linear Regression Hypothesis:**
```
price = w₁x₁ + w₂x₂ + ... + w₁₃x₁₃ + b
```

**Cost Function (Mean Squared Error):**
```
J(w,b) = (1/m) * Σ(ŷ - y)²
```

**Gradient Descent Update Rules:**
```
w = w - α * (2/m) * Xᵀ·(ŷ - y)
b = b - α * (2/m) * Σ(ŷ - y)
```

## Implementation Details

### Data Preprocessing

1. **Missing Values:** Dropped 207 rows with missing `total_bedrooms` (1% of data)
2. **One-Hot Encoding:** Converted `ocean_proximity` to 5 binary columns, dropped `ISLAND` category to avoid dummy variable trap
3. **Train-Test Split:** 80/20 stratified split with shuffling (16,346 train, 4,087 test)
4. **Feature Scaling:** Standardization (mean=0, std=1) applied to 8 numeric features using training statistics
5. **Target Scaling:** Standardization applied to `median_house_value` using training statistics

### Gradient Descent from Scratch

```python
def gradient_descent(X_train, y_train, w, b, alpha, n):
    m = len(X_train)
    cost_history = []
    
    for i in range(n):
        predictions = X_train @ w + b
        errors = predictions - y_train
        
        w_gradient = (2/m) * (X_train.T @ errors)
        b_gradient = (2/m) * np.sum(errors)
        
        w = w - alpha * w_gradient
        b = b - alpha * b_gradient
        
        cost = (1/m) * np.sum(errors ** 2)
        cost_history.append(cost)
    
    return w, b, cost_history
```

### Model Evaluation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.64 | Model explains 64% of price variance |
| **RMSE** | $68,707 | Typical prediction error |
| **MAE** | $50,000 | Average absolute error |

## Results

### Feature Weights (Interpretation)

| Feature | Weight | Effect on Price |
|---------|--------|-----------------|
| median_income | +0.65 | Strong positive |
| total_bedrooms | +0.35 | Positive |
| is_inland | -0.28 | Strong negative |
| latitude | -0.42 | Negative |
| longitude | -0.42 | Negative |

### Visualizations

**Predictions vs Actual:**
- Points cluster around the diagonal
- Some over-prediction at lower price ranges
- Under-prediction at the $500k cap

**Cost Convergence:**
- Cost decreased from ~1.0 to ~0.32 over 500 iterations
- Smooth convergence with learning rate α = 0.1

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/FeralSatyam/House-Price-Prediction-Using-Linear-Regression
cd housing-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook house-price-pred.ipynb
```


## Key Learnings

1. **Feature scaling is critical** — Without it, gradient descent either explodes or converges extremely slowly
2. **Target scaling matters** — Unscaled y_train leads to overflow errors
3. **Vectorization > Loops** — Matrix operations are 100x faster than Python loops
4. **Test data must use training statistics** — Using test set mean/std causes data leakage
5. **Dummy variable trap** — Dropping one category avoids perfect multicollinearity
6. **Cost curve debugging** — Plotting cost vs iterations reveals learning rate problems



## Limitations & Future Work

**Current Limitations:**
- Linear model cannot capture non-linear relationships
- Capped target variable ($500k) introduces bias
- No regularization (would help with multicollinearity)

**Potential Improvements:**
- Add polynomial features (e.g., income², rooms²)
- Implement Ridge/Lasso regularization (Week 3 material)
- Try feature engineering (rooms per household, bedrooms per room)
- Experiment with different learning rate schedules


## License

This project is for educational purposes. Dataset used under fair use for learning.

---

**Key Insight:** Understanding how gradient descent works under the hood makes you a better machine learning engineer. Libraries abstract away the details, but knowing the math helps you debug when things go wrong.
