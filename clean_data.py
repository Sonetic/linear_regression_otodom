import pandas as pd
import json

# path to the CSV file
csv_path = r"C:\Users\Krzysztof\GitHub\linear_regression_otodom\1.csv"
df = pd.read_csv(csv_path, header=None, names=["json_data"])

# unpack JSON data
df_expanded = pd.json_normalize(df["json_data"].apply(json.loads))

# function to clean price and convert currencies
def clean_price(price, euro_to_pln=4.25, usd_to_pln=3.66):
    if pd.isna(price):
        return None
    price = price.replace("\xa0","").replace(",","").strip()
    try:
        if "€" in price:
            return float(price.replace("€","")) * euro_to_pln
        elif "$" in price:
            return float(price.replace("$","")) * usd_to_pln
        else:  # PLN or no symbol
            return float(price.replace("PLN",""))
    except:
        return None

df_expanded["price_num"] = df_expanded["price"].apply(clean_price)

# function to clean surface area
def clean_surface(surface):
    if pd.isna(surface):
        return None
    surface = surface.replace(" m²","").replace(",",".")
    try:
        return float(surface)
    except:
        return None

df_expanded["surface_num"] = df_expanded["surface"].apply(clean_surface)

# extract coordinates
df_expanded[["Longitude","Latitude"]] = df_expanded["location"].str.extract(
    r"Longitude: ([\d\.]+) \| Latitude: ([\d\.]+)"
).astype(float)

# save to CSV
output_path = r"C:\Users\Krzysztof\GitHub\linear_regression_otodom\otodom_cleaned.csv"
df_expanded.to_csv(output_path, index=False)

print("file saved:", output_path)
