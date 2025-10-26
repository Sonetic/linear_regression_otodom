import columns_prep
import numpy as np

def predict_price(x_test):
    # copy dataset
    df = columns_prep.df.copy()
    district_cols = [c for c in df.columns if c.startswith("district_")]

    # create interactions
    for d in district_cols:
        df[f"{d}_x_surface"] = df[d] * df["surface_num"]
        df[f"{d}_x_rooms"] = df[d] * df["no_of_rooms"]

    feature_cols = [c for c in df.columns if c != "price_num"]
    X = df[feature_cols].astype(float).values
    y = df["price_num"].values.reshape(-1, 1).astype(float)

    numeric_cols = ["surface_num", "no_of_rooms"]
    for d in district_cols:
        numeric_cols.append(f"{d}_x_surface")
        numeric_cols.append(f"{d}_x_rooms")

    numeric_idx = [feature_cols.index(c) for c in numeric_cols]

    # standardization
    X_mean = np.mean(X[:, numeric_idx], axis=0)
    X_std = np.std(X[:, numeric_idx], axis=0)
    X_std[X_std == 0] = 0.01
    X[:, numeric_idx] = (X[:, numeric_idx] - X_mean) / X_std

    # gradient descent
    m, n = X.shape
    theta = np.zeros((n, 1))
    learning_rate = 0.01
    iterations = 5000

    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * (X.T @ errors)
        theta -= learning_rate * gradient

    # prepare test vector
    for d in district_cols:
        x_test[f"{d}_x_surface"] = x_test[d] * x_test["surface_num"]
        x_test[f"{d}_x_rooms"] = x_test[d] * x_test["no_of_rooms"]

    x_vec = np.array([x_test[c] for c in feature_cols]).reshape(1, -1)
    x_vec[:, numeric_idx] = (x_vec[:, numeric_idx] - X_mean) / X_std

    return (x_vec @ theta).item()


# create example input
if __name__ == "__main__":
    x_test = {c: 0 for c in columns_prep.df.columns if c != "price_num"}
    x_test["surface_num"] = 820
    x_test["no_of_rooms"] = 7
    x_test["bias"] = 1
    x_test["district_Żoliborz"] = 1  # set chosen district

    predicted_price = predict_price(x_test)
    print(f"Predicted price (Gradient Descent): {predicted_price:,.2f} PLN")
