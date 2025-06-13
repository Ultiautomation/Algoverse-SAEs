import pandas as pd

def load_sampled_dataframe(path):
    df = pd.read_csv(path)
    print(df.head())
    return df