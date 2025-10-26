from linear_regression_gradient_descent import predict_price as predict_price_gradient
from linear_regression_analytical import predict_price as predict_price_analytical
from linear_regression_libraries import predict_price as predict_price_libraries
import columns_prep
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math



df = columns_prep.df.copy()
district_cols = [c for c in df.columns if c.startswith("district_")]

# choose 100 random rows
test_indices = random.sample(range(len(df)), 100)
test_samples = df.iloc[test_indices]

results = []




for _, row in test_samples.iterrows():
    # prepare input
    x_test = {c: 0 for c in df.columns if c != "price_num"}
    x_test["surface_num"] = row["surface_num"]
    x_test["no_of_rooms"] = row["no_of_rooms"]
    x_test["bias"] = 1
    for d in district_cols:
        x_test[d] = row[d]

    # predictions
    pred_gd = predict_price_gradient(x_test)
    pred_anal = predict_price_analytical(x_test)
    pred_lib = predict_price_libraries(x_test)

    results.append({
        "actual": row["price_num"],
        "gradient_descent": pred_gd,
        "analytical": pred_anal,
        "sklearn": pred_lib
    })



# print first 10 results
for r in results[:100]:
    print(f"Actual: {r['actual']:,.2f} | GD: {r['gradient_descent']:,.2f} | Anal: {r['analytical']:,.2f} | Sklearn: {r['sklearn']:,.2f}")



y_true = [r["actual"] for r in results]
y_pred_gd = [r["gradient_descent"] for r in results]
y_pred_anal = [r["analytical"] for r in results]
y_pred_lib = [r["sklearn"] for r in results]

def print_regression_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {label} ---")
    print(f"MAE: {mae:,.2f}")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²: {r2:.4f}\n")

print_regression_metrics(y_true, y_pred_gd, "Gradient Descent")
print_regression_metrics(y_true, y_pred_anal, "Analytical")
print_regression_metrics(y_true, y_pred_lib, "Sklearn LinearRegression")
