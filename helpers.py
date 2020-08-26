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
from datetime import datetime

@st.cache
def load_df(filename, date_format):
    """
    Load data from flat file

    Args:
        filename (str): path where flat csv file is located
        date_format (str): corresponding format string

    Returns:
        df: pandas dataframe

    """
    custom_date_parser = lambda dates: datetime.strptime(dates, date_format) # http://nmmarcelnv.pythonanywhere.com/blogs/timeseries
    df = pd.read_csv(filename,
                          parse_dates=['date'],
                          index_col = 'date',
                          date_parser= custom_date_parser)
    return df

@st.cache(allow_output_mutation=True)
def run_auto_arima(train, m):
    """
    Performs grid search to discover optimal order for an ARIMA model
    based on Akaike Information Criterion ("AIC")

    Args:
        train: pandas dataframe training set
        m (int): Period for seasonal differencing; 4 for quarterly,
                12 for monthly, or 1 for annual (non-seasonal) data

    Returns:
        arima_models: list of all valid ARIMA estimator fits

    """
    arima_models = pm.auto_arima(train, start_p=0, start_q=0, max_p=8, max_q=8,
                            start_P=0, start_Q=0, max_P=8, max_Q=8,
                           m=m, seasonal=True,trace=True,d=1,D=1,
                           error_action='warn', suppress_warnings=True,
                           stepwise=True, random_state=42,
                           return_valid_fits=True, n_fits=30)
    return arima_models

@st.cache
def best_result_df(grid_search_df):
    """
    Select optimal order for ARIMA model from dataframe of valid fits

    Args:
        grid_search_df: pandas dataframe of valid ARIMA estimators' parameters

    Returns:
        best_result_df: pandas dataframe of row with optimal order based on AIC

    """
    best_result_df = grid_search_df[grid_search_df.aic == grid_search_df.aic.min()]

    return best_result_df


def make_predictions(Arima_models, best_results_df_index_value, n_periods):
    """
    Make predictions over specified number of periods with ARIMA model

    Args:
        Arima_models: list of valid ARIMA estimator fits
        best_results_df_index_value (int): integer based on index value of
                    best_results_df to reference correct element from Arima_models
        n_periods: number of periods for testing

    Returns:
        preds: ndarray of predicted values

    """
    preds =Arima_models[best_results_df_index_value].predict(n_periods=n_periods)
    return preds


def calc_errors(test):
    """
    Calculate difference between actual test values and predictions

    Args:
        test: pandas dataframe of test set

    Returns: pandas series result of operation

    """
    return test['value'] - test['preds']

def time_series_plot(df):
    """
    Plot time series of data

    Args:
        df: pandas dataframe of time series data containing datetimeindex and
            value series

    Returns:
        trend_plot: plotly Figure graph object

    """
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
    """
    Plot trend component of data

    Args:
        df: pandas dataframe
        decomposition: statsmodels object object with additive or multiplicative model

    Returns:
        trend_plot: plotly Figure graph object

    """
    trend_plot = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.trend.tolist(),
                        mode='lines')],
        layout_title_text="Trend Plot"
    )
    return trend_plot

def seasonal_plot(df, decomposition):
    """
    Plot seasonal component of data

    Args:
        df: pandas dataframe
        decomposition: statsmodels object with additive or multiplicative model

    Returns:
        seasonal_plot: plotly Figure graph object

    """
    seasonal_plot = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.seasonal.tolist(),
                        mode='lines')],
        layout_title_text="Seasonality Plot"
    )
    return seasonal_plot

def residual_plot(df, decomposition):
    """
    Plot residual component of data

    Args:
        df: pandas dataframe
        decomposition: statsmodels object with additive or multiplicative model

    Returns:
        residual_plot: plotly Figure graph object

    """
    residual_plot = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.resid.tolist(),
                        mode='lines')],
        layout_title_text="Residual Plot"
    )
    return residual_plot


def display_plots(df, decomposition):
    """
    Display plotly charts of time series decomposition

    """
    figs = [time_series_plot(df),
            residual_plot(df, decomposition),
            seasonal_plot(df, decomposition),
            trend_plot(df, decomposition)]
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)

def train_test_predict_plot(train, test, title):
    """
    Plot validation plot comparing train, test, and prediction values

    Args:
        train: pandas dataframe of training set
        test: pandas dataframe of test set containing actual and predicted values
        title (str): optional title for plot

    Returns:
        plotly Figure graph object

    """
    fig = go.Figure(data=[go.Scatter(x=train.index,
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
                                'yaxis': {'title': 'Value'}})
    return fig

def metrics_table(test):
    """
    Create dataframe of time series metrics (Mean Absolute Error,
    Mean Squared Error, and Median Aboslute Error)

    Args:
        test: pandas dataframe of test set

    Returns:
        metrics_table: pandas dataframe

    """
    metrics_dict = {'Mean Absolute Error': [round(metrics.mean_absolute_error(test.value, test.preds), 2)],
                    'Mean Squared Error': [round(metrics.mean_squared_error(test.value, test.preds), 2)],
                    'Median Absolute Error': [round(metrics.median_absolute_error(test.value, test.preds), 2)]
                    }
    metrics_table = pd.DataFrame.from_dict(metrics_dict)

    return metrics_table

def prediction_error_plot(test):
    """
    Plot of prediction error that shows actual values against predicted values
    generated by model

    Args:
        test: pandas dataframe of test set

    Returns:
        fig: plotly Figure graph object

    """
    fig = go.Figure(data=[go.Scatter(x=test.index, y=test['error'], mode='lines')],
              layout={'title': 'Error Over Time',
                      'xaxis': {'title': 'Date', },
                      'yaxis': {'title': 'Error'}})
    return fig


def probability_plot(test):
    """
    Plot a probability plot of sample data against theoretical quantiles with
    ols trendline

    Args:
        test: pandas dataframe of test set

    Returns:
        fig: plotly Figure graph object
        
    """
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
