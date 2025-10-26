import columns_prep
import numpy as np

# dane z poprzedniego etapu
df = columns_prep.df

district_cols = [c for c in df.columns if c.startswith("district_")]

# interakcje
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

# standaryzacja
X_mean = np.mean(X[:, numeric_idx], axis=0)
X_std = np.std(X[:, numeric_idx], axis=0)
X_std[X_std == 0] = 0.01
X[:, numeric_idx] = (X[:, numeric_idx] - X_mean) / X_std

# inicjalizacja
m, n = X.shape
theta = np.zeros((n, 1))
learning_rate = 0.01
iterations = 50000

# gradient descent
for i in range(iterations):
    predictions = X @ theta
    errors = predictions - y
    gradient = (1/m) * (X.T @ errors)
    theta -= learning_rate * gradient


print("\nWspółczynniki regresji (gradient descent):")
for col, coef in zip(feature_cols, theta.flatten()):
    print(f"{col}: {coef}")

# przykładowa predykcja
x_test = {c: 0 for c in feature_cols}
x_test["surface_num"] = 820
x_test["no_of_rooms"] = 7
x_test["bias"] = 1

for col in district_cols:
    if "Żoliborz" in col:
        x_test[col] = 1

for d in district_cols:
    x_test[f"{d}_x_surface"] = x_test[d] * x_test["surface_num"]
    x_test[f"{d}_x_rooms"] = x_test[d] * x_test["no_of_rooms"]

x_vec = np.array([x_test[c] for c in feature_cols]).reshape(1, -1)
x_vec[:, numeric_idx] = (x_vec[:, numeric_idx] - X_mean) / X_std

predicted_price = (x_vec @ theta).item()
print(f"\nSzacowana cena: {predicted_price:,.2f} zł")
