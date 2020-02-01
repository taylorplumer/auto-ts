from load_df import load_df
import pmdarima as pm
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from sklearn import metrics
import scipy
import joblib


df = load_df('Data/Input/AirPassengers.csv')

train, test = df.iloc[:100].copy(), df[100:].copy()


Arima_model = pm.auto_arima(train, start_p=0, start_q=0, max_p=8, max_q=8,
                            start_P=0, start_Q=0, max_P=8, max_Q=8,
                           m=12, seasonal=True,trace=True,d=1,D=1,
                           error_action='warn', suppress_warnings=True,
                            stepwise=True, random_state=20, n_fits=30)


joblib.dump(Arima_model, 'Model/arima.pkl')

prediction = pd.DataFrame(Arima_model.predict(n_periods=44), index=test.index)

prediction.columns = ['predicted_passengers']
prediction.to_csv('Data/Output/prediction_df.csv', index=False)
