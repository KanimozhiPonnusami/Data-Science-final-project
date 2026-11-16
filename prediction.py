
import joblib
import pandas as pd

bundle = joblib.load("retail_churn_model.pkl")

model = bundle["model"]
FEATURE_COLUMNS = bundle["feature_columns"]
NUMERIC_FEATURES = bundle["numeric_features"]


def predict_churn(customer_input):

    if not isinstance(customer_input, dict):
        raise ValueError("Input must be a dictionary.")

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in customer_input
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    unexpected = [
        col for col in customer_input
        if col not in FEATURE_COLUMNS
    ]

    if unexpected:
        raise ValueError(
            f"Unexpected features: {unexpected}"
        )

    input_df = pd.DataFrame(
        [customer_input],
        columns=FEATURE_COLUMNS
    )
    for col in NUMERIC_FEATURES:
        if pd.isna(input_df.loc[0, col]):
            continue

        try:
            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="raise"
            )
        except (ValueError, TypeError):
            raise ValueError(
                f"{col} must contain a numeric value."
            )

    probability = model.predict_proba(input_df)[0][1]

    if probability >= 0.70:
        risk = "High Risk"
    elif probability >= 0.40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "churn_probability": round(
            float(probability), 4
        ),
        "churn_probability_percent": round(
            float(probability) * 100, 2
        ),
        "risk_category": risk
    }


if __name__ == "__main__":

    sample = {
        col: None for col in FEATURE_COLUMNS
    }

    print("Expected input fields:", FEATURE_COLUMNS)
