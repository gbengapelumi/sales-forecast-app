from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved LightGBM model
model = joblib.load("forecasting_model.pkl")


# Feature engineering functions
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe["sales_lag_" + str(lag)] = dataframe.groupby(["store", "item"])[
            "sales"
        ].transform(lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe["sales_roll_mean_" + str(window)] = dataframe.groupby(
            ["store", "item"]
        )["sales"].transform(
            lambda x: x.shift(1)
            .rolling(window=window, min_periods=10, win_type="triang")
            .mean()
        ) + random_noise(
            dataframe
        )
    return dataframe


# Create the feature engineering function
def create_features(df):
    # Date-based features
    df["month"] = df.date.dt.month
    df["day_of_month"] = df.date.dt.day
    df["day_of_year"] = df.date.dt.dayofyear
    df["week_of_year"] = df.date.dt.isocalendar().week  # Fixed line
    df["day_of_week"] = df.date.dt.dayofweek
    df["year"] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df["is_month_start"] = df.date.dt.is_month_start.astype(int)
    df["is_month_end"] = df.date.dt.is_month_end.astype(int)

    # Lag features
    df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

    # Rolling mean features
    df = roll_mean_features(df, [365, 546, 730])

    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            # Get form data
            store = int(request.form["store"])
            item = int(request.form["item"])
            date = request.form["date"]
            sales = float(request.form["sales"])  # Convert sales to float

            # Create a DataFrame from the form data
            data = {
                "store": [store],
                "item": [item],
                "date": [pd.to_datetime(date)],
                "sales": [sales],
            }
            df = pd.DataFrame(data)

            # Ensure the correct data types
            df["sales"] = df["sales"].astype(float)

            # Apply feature engineering
            df = create_features(df)

            # Convert lag features to numeric
            lag_columns = [
                f"sales_lag_{lag}"
                for lag in [91, 98, 105, 112, 119, 126, 182, 364, 546, 728]
            ]
            for col in lag_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Columns used for training (same as your notebook's 'cols' list)
            cols = [col for col in df.columns if col not in ["date", "sales", "year"]]

            # Make predictions
            predictions = model.predict(df[cols])
            predictions = np.expm1(predictions)  # Reverse log1p

            # Render the template with the prediction results
            return render_template("home.html", results=predictions[0])

        except Exception as e:
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
