import os
import math
import columns_prep
from flask import Flask, render_template
from linear_regression_gradient_descent import predict_price as predict_price_gradient
from linear_regression_analytical import predict_price as predict_price_analytical
from linear_regression_libraries import predict_price as predict_price_libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# set the path to the templates folder
template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=template_path)

@app.route("/")
def results():
    # copy dataset
    df = columns_prep.df.copy()
    district_cols = [c for c in df.columns if c.startswith("district_")]

    # use all available properties as test samples
    test_samples = df.copy()

    results = []

    for _, row in test_samples.iterrows():
        # prepare input dictionary for each property
        x_test = {c: 0 for c in df.columns if c != "price_num"}
        x_test["surface_num"] = row["surface_num"]
        x_test["no_of_rooms"] = row["no_of_rooms"]
        x_test["bias"] = 1
        for d in district_cols:
            x_test[d] = row[d]

        # generate predictions from all models
        pred_gd = predict_price_gradient(x_test)
        pred_anal = predict_price_analytical(x_test)
        pred_lib = predict_price_libraries(x_test)

        results.append({
            "district": next((d.replace("district_", "") for d in district_cols if row[d] == 1), "Bemowo"),
            "surface": row["surface_num"],
            "rooms": row["no_of_rooms"],
            "actual": row["price_num"],
            "gradient_descent": pred_gd,
            "analytical": pred_anal,
            "sklearn": pred_lib
        })

    # compute metrics for all models
    y_true = [r["actual"] for r in results]
    metrics = {}
    for name, preds in {
        "Gradient Descent": [r["gradient_descent"] for r in results],
        "Analytical": [r["analytical"] for r in results],
        "Sklearn": [r["sklearn"] for r in results]
    }.items():
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(y_true, preds)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_true, preds)
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    # render results page
    return render_template("results.html", results=results, metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True)
