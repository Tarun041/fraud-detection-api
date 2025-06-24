from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load('model/fraud_model.pkl')

# Use the same mapping as training
transaction_type_mapping = {
    "TRANSFER": 1,
    "CASH_OUT": 0
}

# Define input schema
class Transaction(BaseModel):
    amount: float
    transaction_type: str  # keep it string in the input
    oldbalanceOrg: float
    newbalanceOrig: float

@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()

    # Convert transaction_type to numeric
    if data['transaction_type'] in transaction_type_mapping:
        data['transaction_type'] = transaction_type_mapping[data['transaction_type']]
    else:
        return {"error": f"Unknown transaction_type '{data['transaction_type']}'"}

    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return {"fraud": bool(prediction[0])}
