import columns_prep
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# dane
df = columns_prep.df
district_cols = [c for c in df.columns if c.startswith("district_")]

for d in district_cols:
    df[f"{d}_x_surface"] = df[d] * df["surface_num"]
    df[f"{d}_x_rooms"] = df[d] * df["no_of_rooms"]

feature_cols = [c for c in df.columns if c != "price_num"]
X = df[feature_cols].astype(float).values
y = df["price_num"].values

numeric_cols = ["surface_num", "no_of_rooms"] + [f"{d}_x_surface" for d in district_cols] + [f"{d}_x_rooms" for d in district_cols]
numeric_idx = [feature_cols.index(c) for c in numeric_cols]

# standaryzacja
scaler = StandardScaler()
X[:, numeric_idx] = scaler.fit_transform(X[:, numeric_idx])

# regresja
model = LinearRegression()
model.fit(X, y)

# dane testowe
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
x_vec[:, numeric_idx] = scaler.transform(x_vec[:, numeric_idx])

pred = model.predict(x_vec)[0]
district_name = next(d for d in district_cols if x_test[d] == 1).replace("district_", "")

print(f"Szacowana cena mieszkania ({x_test['surface_num']}m², {x_test['no_of_rooms']} pokoje, {district_name}): {pred:,.2f} zł")
