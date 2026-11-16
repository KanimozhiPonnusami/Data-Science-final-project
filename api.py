
from fastapi import FastAPI, HTTPException
from predictor import predict_churn

app = FastAPI(
    title="Retail Customer Churn Prediction API",
    description="Predict customer churn using a trained ML pipeline",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Retail Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: dict):

    try:
        result = predict_churn(customer)

        return {
            "success": True,
            "prediction": result
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs."
        )
