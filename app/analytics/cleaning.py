import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df = df.dropna(subset=['revenue'])
    df = df[df['revenue'] >= 0]
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(1)
    else:
        df['quantity'] = 1
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        date_ratio = df['sale_date'].notna().mean()
        if date_ratio > 0.5:
            df = df.dropna(subset=['sale_date'])
    return df.reset_index(drop=True)
