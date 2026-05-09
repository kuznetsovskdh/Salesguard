import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['month'] = df['sale_date'].dt.month
    df['year'] = df['sale_date'].dt.year
    df['avg_check'] = df['revenue'] / df['quantity'].replace(0, 1)
    return df
