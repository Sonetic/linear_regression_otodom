import sys
sys.path.append(r"C:\Users\Krzysztof\PycharmProjects\cv_project.py\pythonProject6\columns_prep.py")
import columns_prep
import numpy as np

df = columns_prep.df

# --------------------------
# 🔹 Tworzenie interakcji
# --------------------------
district_cols = [c for c in df.columns if c.startswith("district_")]

# Interakcje z powierzchnią
for d in district_cols:
    df[f"{d}_x_surface"] = df[d] * df["surface_num"]

# Interakcje z liczbą pokoi
for d in district_cols:
    df[f"{d}_x_rooms"] = df[d] * df["no_of_rooms"]

# Dodanie biasu
df["bias"] = 1

# --------------------------
# 🔹 Przygotowanie macierzy X i y
# --------------------------
feature_cols = [c for c in df.columns if c != "price_num"]
X = df[feature_cols].astype(float).values
y = df["price_num"].values.reshape(-1, 1).astype(float)

# --------------------------
# 🔹 Standaryzacja tylko zmiennych ciągłych
# --------------------------
numeric_cols = ["surface_num", "no_of_rooms"]
for d in district_cols:
    numeric_cols.append(f"{d}_x_surface")
    numeric_cols.append(f"{d}_x_rooms")

# indeksy kolumn do standaryzacji
numeric_idx = [feature_cols.index(c) for c in numeric_cols]

X_norm = np.copy(X)
X_mean = np.mean(X[:, numeric_idx], axis=0)
X_std = np.std(X[:, numeric_idx], axis=0)
X_std[X_std == 0] = 1  # uniknięcie dzielenia przez 0
X_norm[:, numeric_idx] = (X[:, numeric_idx] - X_mean) / X_std

# --------------------------
# 🔹 Równanie normalne
# --------------------------
XtX = X_norm.T @ X_norm
XtX_inv = np.linalg.pinv(XtX)
theta = XtX_inv @ X_norm.T @ y

# --------------------------
# 🔹 Wypisanie współczynników
# --------------------------
print("Współczynniki regresji:")
for col, coef in zip(feature_cols, theta.flatten()):
    print(f"{col}: {coef}")

# --------------------------
# 🔹 Predykcja
# --------------------------




x_test = {c: 0 for c in feature_cols}
x_test["surface_num"] = 82
x_test["no_of_rooms"] = 3
x_test["bias"] = 1

# ustawiamy dzielnicę
for col in district_cols:
    if "Żoliborz" in col:
        x_test[col] = 1

# dodajemy interakcje w testowym wierszu
for d in district_cols:
    x_test[f"{d}_x_surface"] = x_test[d] * x_test["surface_num"]
    x_test[f"{d}_x_rooms"] = x_test[d] * x_test["no_of_rooms"]

# konwersja do wektora numpy
x_vec = np.array([x_test[c] for c in feature_cols]).reshape(1, -1)

# normalizacja tylko kolumn numerycznych
x_vec[:, numeric_idx] = (x_vec[:, numeric_idx] - X_mean) / X_std

# predykcja
predicted_price = float(x_vec @ theta)
print(f"\nSzacowana cena mieszkania (100m², 5 pokoi, Śródmieście): {predicted_price:,.2f} zł")
