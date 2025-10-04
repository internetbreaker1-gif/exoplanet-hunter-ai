import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(file_path):
    # Read CSV
    df = pd.read_csv(file_path)

    # If 'pl_disposition' exists, use it; else create a dummy target
    if 'pl_disposition' in df.columns:
        target_column = 'pl_disposition'
    else:
        print("Warning: 'pl_disposition' not found. Creating dummy target.")
        df['pl_disposition'] = 'CONFIRMED'
        target_column = 'pl_disposition'

    # Fill missing values for 6 features
    for col in ['pl_orbper', 'pl_rade', 'pl_bmassj', 'st_teff', 'st_rad', 'st_mass']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            print(f"Warning: '{col}' not found. Filling with 0")
            df[col] = 0.0

    # Encode target
    le = LabelEncoder()
    df['target'] = le.fit_transform(df[target_column])

    # Prepare features and labels
    X = df[['pl_orbper', 'pl_rade', 'pl_bmassj', 'st_teff', 'st_rad', 'st_mass']]
    y = df['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le

