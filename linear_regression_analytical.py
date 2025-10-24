import sys
sys.path.append(r"C:\Users\Krzysztof\PycharmProjects\cv_project.py\pythonProject6\columns_prep.py")
import columns_prep
import numpy as np

# import previously prepared dataframe from columns_prep.py
df = columns_prep.df

# creating interactions (correlations between variables)

# find all columns representing districts
district_cols = [c for c in df.columns if c.startswith("district_")]

# create interactions between district and surface (e.g. district_Wilanow * surface)
for d in district_cols:
    df[f"{d}_x_surface"] = df[d] * df["surface_num"]

# create interactions between district and number of rooms
for d in district_cols:
    df[f"{d}_x_rooms"] = df[d] * df["no_of_rooms"]


# preparing input data X and y

feature_cols = [c for c in df.columns if c != "price_num"]

X = df[feature_cols].astype(float).values  # all columns except price
y = df["price_num"].values.reshape(-1, 1).astype(float)  # target vector with price

# standardization of continuous numerical variables to reduce the impact of variables with different scales,
# e.g. so that price in millions does not dominate the influence of room count which is small in magnitude

numeric_cols = ["surface_num", "no_of_rooms"]

# add interaction columns to the numeric list
for d in district_cols:
    numeric_cols.append(f"{d}_x_surface")
    numeric_cols.append(f"{d}_x_rooms")

# find indices of numeric columns in X matrix
numeric_idx = [feature_cols.index(c) for c in numeric_cols]

# create a copy of X for normalization
X_norm = np.copy(X)

# calculate mean and standard deviation for each numeric column
X_mean = np.mean(X[:, numeric_idx], axis=0)
X_std = np.std(X[:, numeric_idx], axis=0)
X_std[X_std == 0] = 0.01  # avoid division by zero

# normalize numeric columns only
X_norm[:, numeric_idx] = (X[:, numeric_idx] - X_mean) / X_std


# compute regression coefficients (normal equation): theta = (XᵀX)⁻¹ Xᵀy

XtX = X_norm.T @ X_norm  # matrix multiplication
XtX_inv = np.linalg.pinv(XtX)
theta = XtX_inv @ X_norm.T @ y


# print regression coefficients
print("regression coefficients:")
for col, coef in zip(feature_cols, theta.flatten()):
    print(f"{col}: {coef}")

# preparing test data (for prediction)
# create an empty row with all columns

x_test = {c: 0 for c in feature_cols}

# set surface and number of rooms
x_test["surface_num"] = 820
x_test["no_of_rooms"] = 7
x_test["bias"] = 1  # intercept term

# set chosen district (here: Zoliborz)
for col in district_cols:
    if "Żoliborz" in col:
        x_test[col] = 1

# create interactions in test data
for d in district_cols:
    x_test[f"{d}_x_surface"] = x_test[d] * x_test["surface_num"]
    x_test[f"{d}_x_rooms"] = x_test[d] * x_test["no_of_rooms"]

# convert to numpy vector in the same order as X
x_vec = np.array([x_test[c] for c in feature_cols]).reshape(1, -1)

# normalize the same numeric columns as before
x_vec[:, numeric_idx] = (x_vec[:, numeric_idx] - X_mean) / X_std

# predict price
predicted_price = (x_vec @ theta).item()

# prediction summary based on input data
district_name = next(d for d in district_cols if x_test[d] == 1).replace("district_", "")
print(f"\nestimated apartment price ({x_test['surface_num']}m², "
      f"{x_test['no_of_rooms']} rooms, {district_name}): {predicted_price:,.2f} zł")
