import columns_prep
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# copy dataset
df = columns_prep.df.copy()
district_cols = [c for c in df.columns if c.startswith("district_")]

# create interaction features between districts and numeric variables
for d in district_cols:
    df[f"{d}_x_surface"] = df[d] * df["surface_num"]
    df[f"{d}_x_rooms"] = df[d] * df["no_of_rooms"]

feature_cols = [c for c in df.columns if c != "price_num"]
X = df[feature_cols].astype(float).values
y = df["price_num"].values.reshape(-1, 1).astype(float)

# select numeric columns for normalization
numeric_cols = ["surface_num", "no_of_rooms"] + \
               [f"{d}_x_surface" for d in district_cols] + \
               [f"{d}_x_rooms" for d in district_cols]
numeric_idx = [feature_cols.index(c) for c in numeric_cols]

# normalize numeric columns
X_mean = np.mean(X[:, numeric_idx], axis=0)
X_std = np.std(X[:, numeric_idx], axis=0)
X_std[X_std == 0] = 0.01  # avoid division by zero
X[:, numeric_idx] = (X[:, numeric_idx] - X_mean) / X_std

# training
m, n = X.shape
theta = np.zeros((n, 1))
learning_rate = 0.01
iterations = 5000
cost_history = []

for i in range(iterations):
    predictions = X @ theta
    errors = predictions - y
    gradient = (1/m) * (X.T @ errors)
    theta -= learning_rate * gradient
    cost_history.append(np.mean(errors ** 2))

#  plot cost over iterations
plt.plot(range(iterations), cost_history)
plt.title("Error change during training")
plt.xlabel("Iteration")
plt.ylabel("Error (MSE)")

# predicting price
def predict_price(x_test):
    for d in district_cols:
        x_test[f"{d}_x_surface"] = x_test[d] * x_test["surface_num"]
        x_test[f"{d}_x_rooms"] = x_test[d] * x_test["no_of_rooms"]

    x_vec = np.array([x_test[c] for c in feature_cols]).reshape(1, -1)
    x_vec[:, numeric_idx] = (x_vec[:, numeric_idx] - X_mean) / X_std

    return float((x_vec @ theta)[0, 0])


# plot in file only
if __name__ == "__main__":
    x_test = {c: 0 for c in df.columns if c != "price_num"}
    x_test["surface_num"] = 82
    x_test["no_of_rooms"] = 3
    x_test["district_Ursynów"] = 1

    price = predict_price(x_test)
    print(f"Predicted price for {x_test['no_of_rooms']} room, {x_test['surface_num']} m² apartment: {price:.2f}")
    plt.show()
