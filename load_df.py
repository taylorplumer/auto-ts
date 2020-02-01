import pandas as pd

def load_df(filepath_):
    """
    data transformation for time series analysis
    """
    df = pd.read_csv(filepath_)
    month = pd.date_range('19490131', periods=144, freq='M')
    df['datestamp'] = month
    df = df.rename({'#Passengers': 'passengers'}, axis=1)
    df.set_index('datestamp', inplace=True)
    df = df.drop(columns=['Month'], axis=1)

    return df
