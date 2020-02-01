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

app = dash.Dash(__name__)
server = app.server


train, test = df.iloc[:100].copy(), df[100:].copy()


additive_decomposition = seasonal_decompose(df, model= 'additive')
multiplicative_decomposition = seasonal_decompose(df, model= 'multiplicative')

adf_test = pm.arima.ADFTest(alpha=0.05)

with open ("Data/Output/out.txt", "r") as myfile:
    out_file=myfile.readlines()

for i in range(len(out_file)):
    out_file[i] = out_file[i].split(', Fit time', 1)[0]

revised_out_file = [x for x in out_file if "Near non-" not in x]
revised_out_file = [x for x in revised_out_file if "Total" not in x]

Arima_model = joblib.load('Model/arima.pkl')

#prediction = pd.DataFrame(Arima_model.predict(n_periods=44), index=test.index)
prediction = pd.read_csv('Data/Output/prediction_df.csv')
prediction.columns = ['predicted_passengers']

test['predicted_passengers'] = prediction['predicted_passengers'].values
test['error'] = test['passengers'] - test['predicted_passengers']
probplot = np.array(scipy.stats.probplot(test.error))


app.layout = html.Div([
            html.Img(src=app.get_asset_url('auto-ts-logo.png'), style={'height': '10%', 'width': '25%'}),
            dcc.Tabs([
                dcc.Tab(label='Time Series Decomposition', children = [
                    html.H2(children='Decomposition Plots'),
                    dcc.RadioItems(
                    id = 'decomposition_model-picker',
                    options = [
                    {'label': 'Additive', 'value': 'Additive'},
                    {'label': 'Multiplicative', 'value': 'Multiplicative'}
                    ],
                    value='Additive',
                    labelStyle={'display': 'inline-block'}
                    ),
                    dcc.Graph(id = 'feature-graphic',
                    figure = go.Figure(
                        data=[go.Scatter(x= df.index,
                                        y = df['passengers'],
                                        mode='lines')],
                        #layout_title_text="Air Passengers Time Series",
                        layout = {'title': "Air Passengers Time Series",
                                    #'margin': {'l': 50, 'r': 50, 'b':50, 't':100},
                                    #'height': 225
                        },
                    )),
                    dcc.Graph(id = 'trend-graphic'),
                    dcc.Graph(id = 'seasonal-graphic'),
                    dcc.Graph(id= 'resid-graphic'),
                    html.H2(children='Stationarity'),
                    html.P(children= 'ADF Test Results: {}'.format(adf_test.should_diff(df))),
                ]),
                dcc.Tab(label='Model Selection', children = [
                    html.H2(children='Auto-Arima Grid Search'),
                    dcc.Graph(id = 'arima_table-graphic',
                    figure = go.Figure(
                        data=[go.Table(header=dict(values=['Results']),
                                        cells=dict(values=[revised_out_file]))
                     ])),
                    html.H2(children='Best Fit Model'),
                    dcc.Graph(id='best_model-graphic',
                    figure = go.Figure(
                        data=[go.Table(header=dict(values=['Order', 'Seasonal_Order', 'AIC', 'BIC']),
                                        cells=dict(values=[str(Arima_model.order), str(Arima_model.seasonal_order), round(Arima_model.aic(), 3), round(Arima_model.bic(),3)]))
                     ])),
                ]),
                dcc.Tab(label='Model Evaluation', children = [
                    dcc.Graph(id='train_test_predict-graphic',
                    figure = go.Figure(
                        data=[go.Scatter(x=train.index,
                                   y = train.passengers,
                                   mode='lines',
                                   name = 'train'),
                            go.Scatter(x=test.index,
                                   y = test.passengers,
                                   mode='lines',
                                   name = 'test'),
                            go.Scatter(x=test.index,
                                   y = test.predicted_passengers,
                                   mode='lines',
                                   name = 'predicted')
                            ],
                        layout_title_text = 'Train Test Predict Plot'
                    )),
                    html.Div([
                    dcc.Graph(id = 'error_measures-graphic',
                    figure = go.Figure(data=[go.Table(header=dict(values=['mean_absolute_error', 'mean_squared_error', 'median_absolute_error']),
                                cells=dict(values=[round(metrics.mean_absolute_error(test.passengers, test.predicted_passengers), 2),
                                    round(metrics.mean_squared_error(test.passengers, test.predicted_passengers), 2),
                                    round(metrics.median_absolute_error(test.passengers, test.predicted_passengers), 2)]))],
                     #layout = {'margin': {'l': 0, 'r': 0, 'b':0, 't':0}}
                     ))]
                     #, style = {'width': '50%', 'height': '25'}
                     ), # html.div
                     html.Div([
                     html.Div([
                    dcc.Graph(id='prediction_error-graphic',
                    figure = go.Figure(data = [go.Scatter(x= test.index, y = test['error'], mode='lines')],
                                        layout = {'title': 'Error Distribution'})
                    )], className="six columns", style = {'width': '48%', 'display': 'inline-block'}),
                    html.Div([
                    dcc.Graph(id='probablity_plot-graphic',
                    figure = px.scatter(x = probplot[0][0],
                                        y = probplot[0][1],
                                        trendline="ols",
                                        title= 'Probability Plot',
                                        labels ={'x': 'Theoretical Quantiles',
                                                'y':'Sample Quantiles'}
                    ))], className="six columns", style = {'width': '48%', 'display': 'inline-block'}) # dcc.Graph and html div
                    ], className="row") # html.div
                    ]) # dcc.tab


                    ]) # dcc.tabs
], style = {'height': '50%', 'width': '70%'}) # html.div

@app.callback(Output('trend-graphic', 'figure'),
                [Input('decomposition_model-picker', 'value')])

def update_trend(decomposition_model):
    if decomposition_model == 'Additive':
        decomposition = additive_decomposition
    elif decomposition_model == 'Multiplicative':
        decomposition = multiplicative_decomposition


    figure = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.trend['passengers'].values.tolist(),
                        mode='lines')],
        layout_title_text="Trend Plot"
    )

    return figure

@app.callback(Output('seasonal-graphic', 'figure'),
                [Input('decomposition_model-picker', 'value')])

def update_seasonal(decomposition_model):
    if decomposition_model == 'Additive':
        decomposition = additive_decomposition
    elif decomposition_model == 'Multiplicative':
        decomposition = multiplicative_decomposition


    figure = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.seasonal['passengers'].values.tolist(),
                        mode='lines')],
        layout_title_text="Seasonality Plot"
    )

    return figure


@app.callback(Output('resid-graphic', 'figure'),
                [Input('decomposition_model-picker', 'value')])

def update_resid(decomposition_model):
    if decomposition_model == 'Additive':
        decomposition = additive_decomposition
    elif decomposition_model == 'Multiplicative':
        decomposition = multiplicative_decomposition


    figure = go.Figure(
        data=[go.Scatter(x= df.index,
                        y = decomposition.resid['passengers'].values.tolist(),
                        mode='lines')],
        layout_title_text="Residual Plot"
    )

    return figure


if __name__ == '__main__':
    app.run_server()
