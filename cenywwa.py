import pandas as pd
import json

# ścieżka do pliku CSV
csv_path = r"C:\Users\Krzysztof\GitHub\linear_regression_otodom\1.csv"
df = pd.read_csv(csv_path, header=None, names=["json_data"])

# rozpakowanie danych JSON
df_expanded = pd.json_normalize(df["json_data"].apply(json.loads))

# funkcja do czyszczenia cen i konwersji walut
def clean_price(price, euro_to_pln=4.25, usd_to_pln=3.66):
    if pd.isna(price):
        return None
    price = price.replace("\xa0","").replace(",","").strip()
    try:
        if "€" in price:
            return float(price.replace("€","")) * euro_to_pln
        elif "$" in price:
            return float(price.replace("$","")) * usd_to_pln
        else:  # PLN lub brak symbolu
            return float(price.replace("PLN",""))
    except:
        return None

df_expanded["price_num"] = df_expanded["price"].apply(clean_price)

# funkcja do czyszczenia powierzchni
def clean_surface(surface):
    if pd.isna(surface):
        return None
    surface = surface.replace(" m²","").replace(",",".")
    try:
        return float(surface)
    except:
        return None

df_expanded["surface_num"] = df_expanded["surface"].apply(clean_surface)

# wyciąganie współrzędnych
df_expanded[["Longitude","Latitude"]] = df_expanded["location"].str.extract(
    r"Longitude: ([\d\.]+) \| Latitude: ([\d\.]+)"
).astype(float)

# zapis do CSV
output_path = r"C:\Users\Krzysztof\OneDrive\Pulpit\projekt\otodom_cleaned.csv"
df_expanded.to_csv(output_path, index=False)

print("Plik zapisany:", output_path)
print(df_expanded[["price_num","surface_num","Longitude","Latitude"]].head())
