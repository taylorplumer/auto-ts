import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objs as go
from sklearn import metrics
import scipy
import plotly.express as px
import joblib
import pmdarima as pm

@st.cache
def load_df(filename, date_format):
    """
        data transformation for time series analysis
        """
    custom_date_parser = lambda dates: pd.datetime.strptime(dates, date_format) # http://nmmarcelnv.pythonanywhere.com/blogs/timeseries
    df = pd.read_csv(filename,
                          parse_dates=['date'],
                          index_col = 'date',
                          date_parser= custom_date_parser)
    return df

@st.cache(allow_output_mutation=True)
def run_auto_arima(train, m):
    arima_model = pm.auto_arima(train, start_p=0, start_q=0, max_p=8, max_q=8,
                            start_P=0, start_Q=0, max_P=8, max_Q=8,
                           m=m, seasonal=True,trace=True,d=1,D=1,
                           error_action='warn', suppress_warnings=True,
                            stepwise=True, random_state=42, return_valid_fits=True, n_fits=30)
    return arima_model

@st.cache
def best_result_df(grid_search_df):
    best_result_df = grid_search_df[grid_search_df.aic == grid_search_df.aic.min()]

    return best_result_df


def make_predictions(Arima_models, best_results_df_index_value, n_periods):
    preds =Arima_models[best_results_df_index_value].predict(n_periods=n_periods)
    return preds


def calc_errors(test):
    return test['value'] - test['preds']

def time_series_plot(df):
    time_series_plot = go.Figure(
        data=[go.Scatter(x=df.index,
                         y=df['value'],
                         mode='lines')],
        layout={'title': "Time Series",
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Value'}}
    )
    return time_series_plot


def trend_plot(df, decomposition):
    trend_plot = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.trend.tolist(),
                        mode='lines')],
        layout_title_text="Trend Plot"
    )
    return trend_plot

def seasonal_plot(df, decomposition):
    seasonal_plot = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.seasonal.tolist(),
                        mode='lines')],
        layout_title_text="Seasonality Plot"
    )
    return seasonal_plot

def residual_plot(df, decomposition):
    residual_plot = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.resid.tolist(),
                        mode='lines')],
        layout_title_text="Residual Plot"
    )
    return residual_plot


def display_plots(df, decomposition):
    figs = [time_series_plot(df),
            residual_plot(df, decomposition),
            seasonal_plot(df, decomposition),
            trend_plot(df, decomposition)]
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)

def train_test_predict_plot(train, test, title):
    fig = go.Figure(
                                    data=[go.Scatter(x=train.index,
                                               y = train.iloc[:, 0].values.tolist(),
                                               mode='lines',
                                               name = 'train'),
                                        go.Scatter(x=test.index,
                                               y = test.iloc[:, 0].values.tolist(),
                                               mode='lines',
                                               name = 'test'),
                                        go.Scatter(x=test.index,
                                               y = test.iloc[:, 1].values.tolist(),
                                               mode='lines',
                                               name = 'predicted')
                                        ],
                                    layout = {'title': title,
                                                'xaxis': {'title': 'Date'},
                                                'yaxis': {'title': 'Value'}}
                    )
    return fig

def metrics_table(test):
    metrics_dict = {'Mean Absolute Error': [round(metrics.mean_absolute_error(test.value, test.preds), 2)],
                    'Mean Squared Error': [round(metrics.mean_squared_error(test.value, test.preds), 2)],
                    'Median Absolute Error': [round(metrics.median_absolute_error(test.value, test.preds), 2)]
                    }
    metrics_table = pd.DataFrame.from_dict(metrics_dict)

    return metrics_table
def prediction_error_plot(test):
    fig = go.Figure(data=[go.Scatter(x=test.index, y=test['error'], mode='lines')],
              layout={'title': 'Error Over Time',
                      'xaxis': {'title': 'Date', },
                      'yaxis': {'title': 'Error'}})
    return fig


def probability_plot(test):
    probplot = np.array(scipy.stats.probplot(test.error), dtype='object')
    fig = px.scatter(x=probplot[0][0],
               y=probplot[0][1],
               trendline="ols",
               title='Probability Plot',
               labels={'x': 'Theoretical Quantiles',
                       'y': 'Sample Quantiles'
                       }
               )
    return fig
