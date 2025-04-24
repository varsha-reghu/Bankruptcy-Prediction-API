# Bankruptcy-Prediction-API

# 🏦 Bankruptcy Prediction API

**Bankruptcy Prediction API** is a FastAPI-based web service designed to predict the likelihood of a company going bankrupt. It uses two machine learning models — a **Random Forest** and an **Ensemble model** — trained on financial indicators. The API receives a JSON input of financial ratios and returns a binary prediction along with the probability of bankruptcy.

---

## 🚀 Features

- 🔍 Predict bankruptcy risk using:
  - ✅ Random Forest model
  - ✅ Ensemble model (combined predictors)
- 📈 Returns prediction (0 = Not Bankrupt, 1 = Bankrupt)
- 📊 Also returns model confidence as a probability
- 🛡️ Input validation with Pydantic
- ⚡ Built with FastAPI for high performance and interactive docs

---

## 🧠 Models Used

- `best_rf_model.pkl` — Random Forest Classifier trained on key financial indicators
- `ensemble_model.pkl` — A stacked or voting classifier for improved generalization

---

## 📦 Input Features

| Feature Name                                           | Description |
|--------------------------------------------------------|-------------|
| Borrowing_dependency                                   | Degree of borrowing reliance |
| Net_Income_to_Stockholders_Equity                      | Return on equity |
| Net_Value_Growth_Rate                                  | Growth rate of company's net worth |
| Net_Value_Per_Share_A                                  | Value per share |
| Interest_Expense_Ratio                                 | Interest expense over revenue |
| Interest_bearing_debt_interest_rate                    | Rate on interest-bearing debts |
| Persistent_EPS_in_the_Last_Four_Seasons                | Earnings per share trend |
| Total_debt_Total_net_worth                             | Leverage ratio |
| Non_industry_income_and_expenditure_revenue            | Financial activities outside core business |
| Net_profit_before_tax_Paid_in_capital                  | Profitability indicator |

---

## 🖥️ API Endpoints

### 🔹 Root
```
GET /
```
**Returns**: Welcome message and basic instructions.

---

### 🔹 Predict using Random Forest
```
POST /predict/rf
```

**Request JSON**:
```json
{
  "Borrowing_dependency": 0.1,
  "Net_Income_to_Stockholders_Equity": 0.25,
  "Net_Value_Growth_Rate": 0.03,
  "Net_Value_Per_Share_A": 1.5,
  "Interest_Expense_Ratio": 0.05,
  "Interest_bearing_debt_interest_rate": 0.06,
  "Persistent_EPS_in_the_Last_Four_Seasons": 2.0,
  "Total_debt_Total_net_worth": 0.7,
  "Non_industry_income_and_expenditure_revenue": 0.02,
  "Net_profit_before_tax_Paid_in_capital": 0.1
}
```

**Response**:
```json
{
  "model": "RandomForest",
  "prediction": 0,
  "probability": 0.13
}
```

---

### 🔹 Predict using Ensemble Model
```
POST /predict/ensemble
```
Same input as above, but returns prediction from the Ensemble model.

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bankruptcy-prediction-api.git
   cd bankruptcy-prediction-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   uvicorn Bankapp:app --reload
   ```

4. Open your browser at:  
   👉 `http://127.0.0.1:8000`  
   👉 `http://127.0.0.1:8000/docs` for Swagger UI

---

## 📂 File Structure

```
project/
├── Bankapp.py                  # Main FastAPI application
├── best_rf_model.pkl           # Trained Random Forest model
├── ensemble_model.pkl          # Trained Ensemble model
├── out_of_sample.csv           # Sample input data for testing
├── filtered_out_of_sample.csv  # Cleaned test dataset
└── README.md                   # Project documentation
```

---

## 👨‍💻 Contributors

- Renya Ann Regi  
- Varsha Reghu  
- [Add your team members here]

---

## 📄 License

MIT License – feel free to use, modify, and share this project for educational purposes.

