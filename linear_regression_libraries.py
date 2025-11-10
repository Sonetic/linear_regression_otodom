import columns_prep
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

df = columns_prep.df.copy()
district_cols = [c for c in df.columns if c.startswith("district_")]

# base features
base_features = ["surface_num", "no_of_rooms"] + district_cols
X_base = df[base_features].astype(float).values
y = df["price_num"].values

# polynomial Features (interactions only, no squares of individual features)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
X_poly = poly.fit_transform(X_base)

# standardize numeric features only
numeric_idx = [i for i, f in enumerate(poly.get_feature_names_out(base_features))
               if "surface_num" in f or "no_of_rooms" in f]
scaler = StandardScaler()
X_poly[:, numeric_idx] = scaler.fit_transform(X_poly[:, numeric_idx])

# train model
model = LinearRegression()
model.fit(X_poly, y)

# prediction function
def predict_price(x_test: dict) -> float:
    # prepare base feature vector
    x_vec_base = np.array([x_test.get(f, 0) for f in base_features]).reshape(1, -1)
    x_vec_poly = poly.transform(x_vec_base)
    x_vec_poly[:, numeric_idx] = scaler.transform(x_vec_poly[:, numeric_idx])
    return model.predict(x_vec_poly)[0]

# in file example
if __name__ == "__main__":
    x_test = {f: 0 for f in base_features}
    x_test["surface_num"] = 82
    x_test["no_of_rooms"] = 3
    x_test["district_Żoliborz"] = 1

    pred = predict_price(x_test)
    print(f"Predicted price (Sklearn LinearRegression + PolynomialFeatures): {pred:,.2f} PLN")
