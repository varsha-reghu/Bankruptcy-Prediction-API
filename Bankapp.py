from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# File paths for the saved models
RF_MODEL_PATH = "best_rf_model.pkl"
ENSEMBLE_MODEL_PATH = "ensemble_model.pkl"

# Load the persisted models
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# Initialize the FastAPI app
app = FastAPI(
    title="Bankruptcy Prediction API",
    description="API for predicting bankruptcy using trained models (RandomForest and Ensemble).",
    version="1.0"
)

# Define the exact feature names used during training
FEATURES = [
    ' Borrowing dependency', 
    " Net Income to Stockholder's Equity", 
    ' Net Value Growth Rate', 
    ' Net Value Per Share (A)', 
    ' Interest Expense Ratio', 
    ' Interest-bearing debt interest rate', 
    ' Persistent EPS in the Last Four Seasons', 
    ' Total debt/Total net worth', 
    ' Non-industry income and expenditure/revenue', 
    ' Net profit before tax/Paid-in capital'
]

# Define the request schema
class PredictionRequest(BaseModel):
    Borrowing_dependency: float
    Net_Income_to_Stockholders_Equity: float
    Net_Value_Growth_Rate: float
    Net_Value_Per_Share_A: float
    Interest_Expense_Ratio: float
    Interest_bearing_debt_interest_rate: float
    Persistent_EPS_in_the_Last_Four_Seasons: float
    Total_debt_Total_net_worth: float
    Non_industry_income_and_expenditure_revenue: float
    Net_profit_before_tax_Paid_in_capital: float

# Define the response schema
class PredictionResponse(BaseModel):
    model: str
    prediction: int
    probability: float

# Utility function to align input data with the model's feature names
def align_features(input_data: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {
        'Borrowing_dependency': ' Borrowing dependency',
        'Net_Income_to_Stockholders_Equity': " Net Income to Stockholder's Equity",
        'Net_Value_Growth_Rate': ' Net Value Growth Rate',
        'Net_Value_Per_Share_A': ' Net Value Per Share (A)',
        'Interest_Expense_Ratio': ' Interest Expense Ratio',
        'Interest_bearing_debt_interest_rate': ' Interest-bearing debt interest rate',
        'Persistent_EPS_in_the_Last_Four_Seasons': ' Persistent EPS in the Last Four Seasons',
        'Total_debt_Total_net_worth': ' Total debt/Total net worth',
        'Non_industry_income_and_expenditure_revenue': ' Non-industry income and expenditure/revenue',
        'Net_profit_before_tax_Paid_in_capital': ' Net profit before tax/Paid-in capital'
    }
    # Rename columns to match model feature names
    input_data = input_data.rename(columns=column_mapping)
    return input_data[FEATURES]

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Bankruptcy Prediction API. Use the /predict endpoint to get predictions."}

# Prediction endpoint for the Random Forest model
@app.post("/predict/rf", response_model=PredictionResponse)
def predict_rf(request: PredictionRequest):
    try:
        # Convert the request data to a DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Align features with the model's feature names
        input_data = align_features(input_data)

        # Make predictions using the Random Forest model
        prediction = rf_model.predict(input_data)[0]
        probability = rf_model.predict_proba(input_data)[0][1]  # Probability for the positive class

        return PredictionResponse(model="RandomForest", prediction=int(prediction), probability=float(probability))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Random Forest prediction: {e}")

# Prediction endpoint for the Ensemble model
@app.post("/predict/ensemble", response_model=PredictionResponse)
def predict_ensemble(request: PredictionRequest):
    try:
        # Convert the request data to a DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Align features with the model's feature names
        input_data = align_features(input_data)

        # Make predictions using the Ensemble model
        prediction = ensemble_model.predict(input_data)[0]
        probability = ensemble_model.predict_proba(input_data)[0][1]  # Probability for the positive class

        return PredictionResponse(model="Ensemble", prediction=int(prediction), probability=float(probability))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Ensemble prediction: {e}")
