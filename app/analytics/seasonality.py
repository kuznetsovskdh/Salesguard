import pandas as pd

def seasonality(df: pd.DataFrame) -> pd.DataFrame:
    if 'month' not in df.columns:
        return pd.DataFrame()
    return (df.groupby(['year', 'month'])['revenue']
              .sum().reset_index()
              .sort_values(['year', 'month']))
