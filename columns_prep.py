import pandas as pd

df = pd.read_csv(r"C:\Users\Krzysztof\GitHub\linear_regression_otodom\otodom_cleaned.csv")

# przybliżone granice dzielnic wwa
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

def apply_district(lat, lon):
    for dzielnica, bounds in warszawa_districts.items():
        if bounds["lat"][0] <= lat <= bounds["lat"][1] and bounds["lon"][0] <= lon <= bounds["lon"][1]:
            return dzielnica
    return "Inna"

# dodanie kolumny z dzielnicą i poprawienie innych danych(price, no_of_rooms, surface)
df["district"] = df.apply(lambda row: apply_district(row["Latitude"], row["Longitude"]), axis=1)



df["no_of_rooms_num"] = pd.to_numeric(df["no_of_rooms"], errors="coerce")
df = df.dropna(subset=["no_of_rooms_num"])
df = df.dropna(subset=["surface_num"])
df = df.dropna(subset=["price_num"])
df = df.dropna(subset=["district"])





#sprawdzenie danych
#print(df["surface_num"].unique())
#print(df["price_num"].unique())
#print(df["no_of_rooms_num"].unique())
#print(df["district"].unique())

# tworzymy kolumny dummy od razu z drop_first=True, w celu ustawienia bazowej dzielnicy
district_dummies = pd.get_dummies(df["district"], prefix="district", drop_first=True)

# dodajemy do df
df = pd.concat([df, district_dummies], axis=1)

# usuwamy oryginalną kolumnę
df = df.drop("district", axis=1)


df = df[["surface_num", "price_num", "no_of_rooms"] + list(district_dummies.columns)]
df["bias"] = 1  # klasyczny bias = 1