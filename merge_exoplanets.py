import pandas as pd

# File paths
file_toi = "TOI_2025.10.04_08.53.58.csv"
file_cumulative = "cumulative_2025.10.04_08.53.52.csv"
file_k2pandc = "k2pandc_2025.10.04_08.54.03.csv"

# Read CSVs
df_toi = pd.read_csv(file_toi, comment="#")
df_cumulative = pd.read_csv(file_cumulative, comment="#")
df_k2pandc = pd.read_csv(file_k2pandc, comment="#")

# Standardize column names to lowercase
df_toi.columns = df_toi.columns.str.lower()
df_cumulative.columns = df_cumulative.columns.str.lower()
df_k2pandc.columns = df_k2pandc.columns.str.lower()

# Rename disposition columns to "pl_disposition"
rename_map = {
    "disposition": "pl_disposition",
    "koi_disposition": "pl_disposition"
}
df_toi = df_toi.rename(columns=rename_map)
df_cumulative = df_cumulative.rename(columns=rename_map)
df_k2pandc = df_k2pandc.rename(columns=rename_map)

# Merge datasets
df_all = pd.concat([df_toi, df_cumulative, df_k2pandc], ignore_index=True)

# Required columns
required_columns = [
    "pl_orbper",     # Orbital period
    "pl_rade",       # Planet radius
    "pl_bmassj",     # Planet mass
    "st_teff",       # Star temperature
    "st_rad",        # Star radius
    "st_mass",       # Star mass
    "pl_disposition" # Unified disposition column
]

# Keep only available columns
available = [col for col in required_columns if col in df_all.columns]
df_all = df_all[available]

# Drop rows with missing disposition
df_all = df_all.dropna(subset=["pl_disposition"])

# Add fake exoplanets
fake_planets = pd.DataFrame([
    {"pl_orbper": 0.01, "pl_rade": 0.05, "pl_bmassj": 0.001,
     "st_teff": 1000, "st_rad": 0.1, "st_mass": 0.1, "pl_disposition": "FALSE POSITIVE"},
    {"pl_orbper": 99999, "pl_rade": 50, "pl_bmassj": 20,
     "st_teff": 20000, "st_rad": 100, "st_mass": 50, "pl_disposition": "FALSE POSITIVE"},
    {"pl_orbper": 365, "pl_rade": 1, "pl_bmassj": 0.003,
     "st_teff": 5778, "st_rad": 1, "st_mass": 1, "pl_disposition": "CONFIRMED"}
])

# Ensure fake data only includes available columns
fake_planets = fake_planets[available]

# Append
df_all = pd.concat([df_all, fake_planets], ignore_index=True)

# Save
output_file = "data/merged_exoplanets.csv"
df_all.to_csv(output_file, index=False)

print(f"✅ Merged dataset saved as {output_file} with {len(df_all)} rows")
print("Columns in final dataset:", df_all.columns.tolist())
