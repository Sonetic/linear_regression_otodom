import pandas as pd
import json

csv_path = r"C:\Users\Krzysztof\OneDrive\Pulpit\projekt\1.csv"
df = pd.read_csv(csv_path, header=None, names=["json_data"])

df_expanded = pd.json_normalize(df["json_data"].apply(json.loads))

# czyszczenie ceny
def clean_price(price):
    if pd.isna(price):
        return None
    price = price.replace("PLN","").replace("€","").replace("\xa0","").replace(",","")
    try:
        return float(price)
    except:
        return None

df_expanded["price_num"] = df_expanded["price"].apply(clean_price)

# czyszczenie powierzchni (zamiana przecinka na kropkę)
def clean_surface(surface):
    if pd.isna(surface):
        return None
    surface = surface.replace(" m²","").replace(",",".")
    try:
        return float(surface)
    except:
        return None

df_expanded["surface_num"] = df_expanded["surface"].apply(clean_surface)

# współrzędne
df_expanded[["Longitude","Latitude"]] = df_expanded["location"].str.extract(
    r"Longitude: ([\d\.]+) \| Latitude: ([\d\.]+)"
).astype(float)

# zapis do CSV
output_path = r"C:\Users\Krzysztof\OneDrive\Pulpit\projekt\otodom_clean.csv"
df_expanded.to_csv(output_path, index=False)

print("Plik zapisany:", output_path)
print(df_expanded[["price_num","surface_num","Longitude","Latitude"]].head())
