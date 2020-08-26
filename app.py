from helpers import *
from autots.autots import Dashboard

df = load_df('Data/AirPassengers.csv', '%Y-%m')
app = Dashboard(df = df, image='assets/auto-ts-logo.png', title="Air Passengers", m=12)
app.preprocess()
app.main()
