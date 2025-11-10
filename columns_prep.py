import pandas as pd
import numpy as np

# uploading otodom_cleaned.csv
df = pd.read_csv("otodom_cleaned.csv")

# estimetaed boundries of districts in Warsaw
warszawa_districts = {
    "Śródmieście": {"lat": (52.21, 52.25), "lon": (21.00, 21.05)},
    "Mokotów": {"lat": (52.15, 52.22), "lon": (21.00, 21.10)},
    "Wola": {"lat": (52.22, 52.25), "lon": (20.95, 21.00)},
    "Praga-Północ": {"lat": (52.25, 52.28), "lon": (21.00, 21.10)},
    "Praga-Południe": {"lat": (52.23, 52.27), "lon": (21.05, 21.15)},
    "Ursynów": {"lat": (52.09, 52.15), "lon": (21.00, 21.10)},
    "Bielany": {"lat": (52.30, 52.36), "lon": (20.90, 21.00)},
    "Żoliborz": {"lat": (52.26, 52.30), "lon": (20.95, 21.00)},
    "Wawer": {"lat": (52.17, 52.25), "lon": (21.15, 21.30)},
    "Ochota": {"lat": (52.20, 52.23), "lon": (20.98, 21.05)},
    "Targówek": {"lat": (52.25, 52.30), "lon": (21.05, 21.15)},
    "Rembertów": {"lat": (52.22, 52.27), "lon": (21.15, 21.25)},
    "Ursus": {"lat": (52.20, 52.23), "lon": (20.87, 20.95)},
    "Włochy": {"lat": (52.18, 52.22), "lon": (20.90, 21.00)},
    "Bemowo": {"lat": (52.25, 52.30), "lon": (20.85, 20.95)}
}

# applying districts
def apply_district(lat, lon):
    for dzielnica, bounds in warszawa_districts.items():
        if bounds["lat"][0] <= lat <= bounds["lat"][1] and bounds["lon"][0] <= lon <= bounds["lon"][1]:
            return dzielnica
    return None  # mieszkania poza Warszawą będą odrzucone

# adding district column
df["district"] = df.apply(lambda row: apply_district(row["Latitude"], row["Longitude"]), axis=1)

# dropping rows from other longitude and latitude
df = df.dropna(subset=["district"])

# cleaning
df["no_of_rooms"] = pd.to_numeric(df["no_of_rooms"], errors="coerce")
df["surface_num"] = pd.to_numeric(df["surface_num"], errors="coerce")
df["price_num"] = pd.to_numeric(df["price_num"], errors="coerce")
df = df.dropna(subset=["no_of_rooms", "surface_num", "price_num"])

# filters to drop extrem values
df = df[
    (df["price_num"] >= 200000) &
    (df["price_num"] <= 3000000) &

    (df["surface_num"] >= 20)
]

# creating dummie columns, and dropping first column, so it could be the one that we will refer to
district_dummies = pd.get_dummies(df["district"], prefix="district", drop_first=True)
df = pd.concat([df, district_dummies], axis=1)
df = df.drop("district", axis=1)


# setting the order of columns for the model
df = df[["surface_num", "price_num", "no_of_rooms"] + list(district_dummies.columns)]
df["bias"] = 1
